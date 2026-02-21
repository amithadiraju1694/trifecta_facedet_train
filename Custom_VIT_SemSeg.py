import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    ViTModel,
    SegformerModel
                          )
from typing import Tuple, Dict
from peft import LoraConfig, get_peft_model, TaskType

from radar_layers import RADAR
from helpers import (
    cxcywh_to_xyxy,
    nms_xyxy
)


def _hf_set_gradient_checkpointing(backbone: nn.Module, enabled: bool = True, **kwargs) -> None:
    method_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    candidates = [backbone]
    base_model = getattr(backbone, "base_model", None)
    if base_model is not None and base_model is not backbone:
        candidates.append(base_model)

    for candidate in candidates:
        method = getattr(candidate, method_name, None)
        if callable(method):
            if enabled:
                method(**kwargs)
            else:
                method()
            return

    raise AttributeError(f"{type(backbone).__name__} does not support {method_name}()")



# ----- Baseline Semantic Segmentation ------ #
class ViTLoRA_SemSeg(nn.Module):
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 num_out_classes:int = 10,
                 r = 16,
                 lora_alpha = 16,
                 lora_dropout = 0.05,
                 target_module = "attn_min",
                 transpose_convolutions = False
                 ):

        super().__init__()

        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.hidden = self.vit.config.hidden_size
        self.transpose_conv = transpose_convolutions
        

        # Choose which linear layers to adapt (canonical = Q/V only)
        if target_module == "attn_min":
            target_layers = ["query", "value"]                 # LoRA paper default
        elif target_module == "attn_full":
            target_layers = ["query", "key", "value", "output.dense"]
        elif target_module == "attn_mlp":
            target_layers = ["query", "key", "value",
                              "attention.output.dense",
                              "intermediate.dense", "output.dense"]
        else:
            raise ValueError(target_module)
        

        lconf = LoraConfig(
            # This is for extracting features for downstream task
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_layers
        )

        self.vit = get_peft_model(self.vit, lconf)

        # Freeze non-LoRA backbone params; keep LoRA params trainable
        for n, p in self.vit.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        
        
        if self.transpose_conv:
            self.seg_head = nn.Sequential(
                # B,D,14,14 -> B, 256, 56,56
                nn.ConvTranspose2d(in_channels = self.hidden, out_channels = 256, kernel_size = 4, stride = 4),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),

                # B,256,56,56 -> B,128,112,112
                nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True),

                # B,128,112,112 -> B,64,224,224
                nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),

                # B,64,224,224 -> B,nC, 224,224
                nn.Conv2d(in_channels = 64, out_channels = num_out_classes, kernel_size = 1)

                                                )
        else:
            # simple 1×1 + upsample head
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden, num_out_classes, kernel_size=1),
            )

    def gradient_checkpoint_enabled(self, enabled: bool = True, **kwargs):
        _hf_set_gradient_checkpointing(self.vit, enabled=enabled, **kwargs)
        return self

    def gradient_checkpointing_enable(self, **kwargs):
        return self.gradient_checkpoint_enabled(True, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpoint_enabled(False)

    
    def forward(self, pixel_values, train_mode:bool = True):
        
        base = self.vit.base_model # peft.PeftModel -> underlying ViTModel
        sequence_output = base(pixel_values=pixel_values, return_dict=True)
        
        # We need to re-create image of original shape, so taking single CLS token doesn't work.
        all_tokens = sequence_output.last_hidden_state[:, 1:, :]          # (B,196,D)
        B, N, D = all_tokens.shape
        H = W = int(N ** 0.5)
        
        # reshape to (B,D,H,W) with H=W=14
        all_tokens = all_tokens.transpose(1, 2).reshape(B, D, H, W)
        
        # batch size, num_segment maps, height , width
        # could be bs,C,14,14 or bs,C,224,224 depends
        logits = self.seg_head(all_tokens) 
        
        # Use BiLinear interpolation if transpose convolutions are not used
        if not self.transpose_conv:

            # explode image into original size with bilinear filling.
            # This could also be transpose convolution operation
            logits = F.interpolate(logits,
                                size=(224, 224),
                                mode="bilinear",
                                align_corners=False
                                )
            
        return logits  # (B,C,224,224)

class ViTWithSegFormer(nn.Module):
    """
    Frozen SegFormer backbone (nvidia/segformer-b0-finetuned-ade-512-512)
    + custom segmentation head for 4-way semantic segmentation.

    - Inputs:  pixel_values of shape (B, 3, 224, 224), already preprocessed
               (resize + normalize done outside, same for all models).
    - Backbone: completely frozen.
    - Head: small Conv/Deconv stack defined here (no HF decoder/head).
    """

    def __init__(
        self,
        pretrained_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_out_classes: int = 4,
        transpose_convolutions:bool = False
    ):
        super().__init__()

        self.segformer = SegformerModel.from_pretrained(pretrained_model_name)

        # freeze backbone
        for p in self.segformer.parameters():
            p.requires_grad = False
        self.segformer.eval()

        cfg = self.segformer.config
        # last encoder feature dim
        self.hidden = cfg.hidden_sizes[-1] if hasattr(cfg, "hidden_sizes") else cfg.hidden_size
        self.num_out_classes = num_out_classes
        self.transpose_conv = transpose_convolutions

        
        # Task-Specific Semantic Segmentation layers
        if self.transpose_conv:
            self.seg_head = nn.Sequential(
                # B,256,7,7 -> B, 256, 56,56
                nn.ConvTranspose2d(in_channels = self.hidden, out_channels = 256, kernel_size = 8, stride = 8),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),

                # B,256,56,56 -> B,128,112,112
                nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True),

                # B,128,112,112 -> B,64,224,224
                nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),

                # B,64,224,224 -> B,nC, 224,224
                nn.Conv2d(in_channels = 64, out_channels = num_out_classes, kernel_size = 1)

                                                )
        else:
            # simple 1×1 + upsample head
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden, num_out_classes, kernel_size=1),
            )

    def gradient_checkpoint_enabled(self, enabled: bool = True, **kwargs):
        _hf_set_gradient_checkpointing(self.segformer, enabled=enabled, **kwargs)
        return self

    def gradient_checkpointing_enable(self, **kwargs):
        return self.gradient_checkpoint_enabled(True, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpoint_enabled(False)

    def __get_segformer_feats(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run frozen SegFormer backbone and return last feature map.

        Args:
            pixel_values: (B, 3, H, W), already resized/normalized.

        Returns:
            feats: (B, C, H', W') feature map from the last stage.
        """
        outputs = self.segformer(pixel_values=pixel_values, return_dict=True)
        # (B, C, H', W')
        feats = outputs.last_hidden_state
        return feats

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224)

        Returns:
            logits: (B, num_out_classes, 224, 224)
        """
        
        with torch.no_grad():
            sequence_output = self.__get_segformer_feats(pixel_values)  # (B, C, H', W')

        # batch size, num_segment maps, height , width
        # SegFormer doesn't use CLS token, hence not removed
        # could be bs,C,14,14 or bs,C,224,224 depends
        logits = self.seg_head(sequence_output)
        
        # Use BiLinear interpolation if transpose convolutions are not used
        if not self.transpose_conv:

            # explode image into original size with bilinear filling.
            # This could also be transpose convolution operation
            logits = F.interpolate(logits,
                                size=(224, 224),
                                mode="bilinear",
                                align_corners=False
                                )
            
        return logits  # (B,C,224,224)

# ----- Custom Models Semantic Segmentation ------ #
# v1 contains both s and b when injecting into patch embeddings.
class ViTRADAR_SoftAnchor_v1_SemSeg(nn.Module):
    """This class implements the ViT-RADAR model with Soft Anchor distances of patch embeddings: this computes important sequences in patch embeddings space
    and modulates those sequences with projected features derived from distances between soft anchors and patch embeddings.

    This version uses exact FiLM style injection: of injecting both 's' and 'b' into patch embeddings.
    Goal of this modulation is to better guide MHSA towards "important sequences" adding global awareness to each patch embedding.
    """
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 distance_metric='euclidean',
                 aggregate_method='norm_softmax',
                 seq_select_method = 'weighted_sum',
                 num_out_classes = 10,
                 add_coordinates = False,
                 K = 3,
                 aggregate_dim = 2,
                 norm_type = 2,
                 return_anchors = True,
                 perc_ape: float = 0.5,
                 transpose_convolutions:bool = False
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.vit.eval()
        self.perc_ape = perc_ape # Percentage of Absolute Positional Encoding to be used in Forward pass
        self.transpose_conv = transpose_convolutions
        self.hidden = self.vit.config.hidden_size

        self.radar_layer = RADAR(
                                    token_hid_dim = self.hidden,
                                    K = K,
                                    add_coordinates = add_coordinates,
                                    distance_metric = distance_metric,
                                    aggregate_method = aggregate_method,
                                    seq_select_method=seq_select_method,
                                    aggregate_dim=aggregate_dim,
                                    norm_type=norm_type,
                                    return_anchors=return_anchors,
                                    corrupt_imp_weights=False,
                                )

        # Task-Specific Semantic Segmentation layers
        if self.transpose_conv:
            self.seg_head = nn.Sequential(
                # B,D,14,14 -> B, 256, 56,56
                nn.ConvTranspose2d(in_channels = self.hidden, out_channels = 256, kernel_size = 4, stride = 4),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),

                # B,256,56,56 -> B,128,112,112
                nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True),

                # B,128,112,112 -> B,64,224,224
                nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),

                # B,64,224,224 -> B,nC, 224,224
                nn.Conv2d(in_channels = 64, out_channels = num_out_classes, kernel_size = 1)

                                                )
        else:
            # simple 1×1 + upsample head
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden, num_out_classes, kernel_size=1),
            )

    def gradient_checkpoint_enabled(self, enabled: bool = True, **kwargs):
        _hf_set_gradient_checkpointing(self.vit, enabled=enabled, **kwargs)
        return self

    def gradient_checkpointing_enable(self, **kwargs):
        return self.gradient_checkpoint_enabled(True, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpoint_enabled(False)

    def __get_vit_peout(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Private method that applies patch embeddings given raw image pixels, loads cls token and static positional encodings; doesn't apply positional encoding or 
        cls token to patch embeddings.
        
        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (batch_size, num_channels, height, width).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - ViT_stat_pos_emb (torch.Tensor): Static positional embeddings of shape (1, seq_len+1, hidden_size).
                - patch_emb_output (torch.Tensor): Patch embeddings of shape (batch_size, seq_len, hidden_size).
                - cls_token (torch.Tensor): Class token expanded to batch size of shape (batch_size, 1, hidden_size).
        """

        # Get patch embeddings - (bts, seq_len, hidden_size)
        patch_emb_output = self.vit.embeddings.patch_embeddings(pixel_values)

        # Add class token
        cls_token = self.vit.embeddings.cls_token.expand(
            patch_emb_output.shape[0], -1, -1)

        # Get static positional embeddings
        ViT_stat_pos_emb = self.vit.embeddings.position_embeddings  # (1, seq_len+1, hidden_size)

        return (ViT_stat_pos_emb, patch_emb_output, cls_token)

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = True):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """
        # TODO: Pre-compute first epoch of patch embeddings and cache
        with torch.no_grad():
            # patch_emb_out - > (batch_size, seq_len, hidden_size)
            ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        modulated_patch_embeddings = self.radar_layer(patch_emb_output)
        patch_emb_output = torch.cat([cls_token, modulated_patch_embeddings], dim = 1)

        # A Scalar percentage for using Absolute PEs is added to facilitate comparison with PEG.
        pos_encoded = patch_emb_output + ( self.perc_ape * ViT_stat_pos_emb )
        position_encoded = self.vit.embeddings.dropout(pos_encoded) # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(position_encoded, return_dict = True)  # It outputs only one item
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len+1, hidden_size)

        # We need to re-create image of original shape, so taking single CLS token doesn't work.
        all_tokens = sequence_output[:, 1:, :]  #(bs, seqlen, hidden_size)
        B, N, D = all_tokens.shape
        H = W = int(N ** 0.5)
        
        # reshape to (B,D,H,W) with H=W=14
        all_tokens = all_tokens.transpose(1, 2).reshape(B, D, H, W) # BS, FTRDIM, H,W(N)

        # batch size, num_segment maps, height , width
        # could be bs,C,14,14 or bs,C,224,224 depends
        logits = self.seg_head(all_tokens) 
        
        # Use BiLinear interpolation if transpose convolutions are not used
        if not self.transpose_conv:

            # explode image into original size with bilinear filling.
            # This could also be transpose convolution operation
            logits = F.interpolate(logits,
                                size=(224, 224),
                                mode="bilinear",
                                align_corners=False
                                )
            
        return logits  # (B,C,224,224)


# -------- Face Detection ----------- #
class ViTFaceDetectorPlain(nn.Module):
    def __init__(self, image_size: int = 224, patch: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch = patch
        self.grid = image_size // patch  # 14 for 224/16

        # Google ViT-Small backbone (HF checkpoint name)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        hidden = self.vit.config.hidden_size  # e.g. 384

        # Dense heads on token grid
        # objectness logits: (B,1,grid,grid)
        self.obj_head = nn.Conv2d(in_channels = hidden,
                                  out_channels=1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0
                                  )
        
        # box head predicts normalized cx,cy,w,h in [0,1] (we'll sigmoid)
        self.box_head = nn.Conv2d(in_channels = hidden,
                                  out_channels=4,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0
                                  )

    def gradient_checkpoint_enabled(self, enabled: bool = True, **kwargs):
        _hf_set_gradient_checkpointing(self.vit, enabled=enabled, **kwargs)
        return self

    def gradient_checkpointing_enable(self, **kwargs):
        return self.gradient_checkpoint_enabled(True, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpoint_enabled(False)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.vit(pixel_values, return_dict=True)
        tokens = out.last_hidden_state  # (B, 1+N, D)

        # remove CLS
        tokens = tokens[:, 1:, :]  # (B, N, D), N=grid*grid
        B, _, D = tokens.shape

        # reshape to grid
        tokens = tokens.reshape(B, self.grid, self.grid, D) # (B,g,g,D)
        tokens = tokens.permute(0, 3, 1, 2).contiguous() # (B,D,g,g)

        # obj logits = does any of 14x14 grid contain a face in centre or not
        # box norm/logits = if there's face output cx,cy,w,h else gaussian x,y,w,h
        obj_logits = self.obj_head(tokens)  # (B,1,g,g)
        box_raw = self.box_head(tokens)  # (B,4,g,g)
        box_norm = torch.sigmoid(box_raw) # (B,4,g,g) in [0,1]

        return {"obj_logits": obj_logits, "box_norm": box_norm}

    @torch.no_grad()
    def infer(self,
              images: torch.Tensor,
              score_thresh: float = 0.5,
              iou_thresh: float = 0.4
              ):
        
       
        pred = self.forward(images)
        
        obj = torch.sigmoid(pred["obj_logits"])   # (B,1,g,g)
        box = pred["box_norm"]                    # (B,4,g,g)

        B = images.shape[0]
        results = []
        for b in range(B):
            scores_map = obj[b, 0]                # (g,g)
            boxes_map = box[b]                    # (4,g,g)

            ys, xs = torch.where(scores_map >= score_thresh)
            if ys.numel() == 0:
                results.append({"boxes_xyxy": torch.zeros((0,4)),
                                "scores": torch.zeros((0,))})
                continue

            scores = scores_map[ys, xs]           # (K,)
            # gather boxes at those cells: (K,4)
            boxes_cxcywh = boxes_map[:, ys, xs].permute(1, 0).contiguous()

            # boxes are normalized to [0,1] over image size
            # convert to pixel xyxy
            boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2] * self.image_size
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2] * self.image_size

            # clamp
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, self.image_size - 1)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, self.image_size - 1)

            keep = nms_xyxy(boxes_xyxy, scores, iou_thresh=iou_thresh)
            results.append({"boxes_xyxy": boxes_xyxy[keep],
                            "scores": scores[keep]}
                          )
        return results


class ViTFaceDetectorRADAR(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch: int = 16,
                 distance_metric='euclidean',
                 aggregate_method='norm_softmax',
                 seq_select_method = 'weighted_sum',
                 add_coordinates = False,
                 K = 3,
                 aggregate_dim = 2,
                 norm_type = 2,
                 return_anchors = True
                 ):
        super().__init__()
        self.image_size = image_size
        self.patch = patch
        self.grid = image_size // patch  # 14 for 224/16

        # Google ViT-Small backbone (HF checkpoint name)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        hidden = self.vit.config.hidden_size  # e.g. 384

        self.radar_layer = RADAR(
                                    token_hid_dim = hidden,
                                    K = K,
                                    add_coordinates = add_coordinates,
                                    distance_metric = distance_metric,
                                    aggregate_method = aggregate_method,
                                    seq_select_method=seq_select_method,
                                    aggregate_dim=aggregate_dim,
                                    norm_type=norm_type,
                                    return_anchors=return_anchors
                                )

        # Dense heads on token grid
        # objectness logits: (B,1,grid,grid)
        self.obj_head = nn.Conv2d(in_channels = hidden, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        # box head predicts normalized cx,cy,w,h in [0,1] (we'll sigmoid)
        self.box_head = nn.Conv2d(in_channels = hidden, out_channels=4, kernel_size=1, stride=1, padding=0)

    def gradient_checkpoint_enabled(self, enabled: bool = True, **kwargs):
        _hf_set_gradient_checkpointing(self.vit, enabled=enabled, **kwargs)
        return self

    def gradient_checkpointing_enable(self, **kwargs):
        return self.gradient_checkpoint_enabled(True, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpoint_enabled(False)

    def __get_vit_peout(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Private method that applies patch embeddings given raw image pixels, loads cls token and static positional encodings; doesn't apply positional encoding or 
        cls token to patch embeddings.
        
        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (batch_size, num_channels, height, width).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - ViT_stat_pos_emb (torch.Tensor): Static positional embeddings of shape (1, seq_len+1, hidden_size).
                - patch_emb_output (torch.Tensor): Patch embeddings of shape (batch_size, seq_len, hidden_size).
                - cls_token (torch.Tensor): Class token expanded to batch size of shape (batch_size, 1, hidden_size).
        """

        # Get patch embeddings - (bts, seq_len, hidden_size)
        patch_emb_output = self.vit.embeddings.patch_embeddings(pixel_values)

        # Add class token
        cls_token = self.vit.embeddings.cls_token.expand(
            patch_emb_output.shape[0], -1, -1)

        # Get static positional embeddings
        ViT_stat_pos_emb = self.vit.embeddings.position_embeddings  # (1, seq_len+1, hidden_size)

        return (ViT_stat_pos_emb, patch_emb_output, cls_token)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict:
           obj_logits = Conv1x1(tokens) -> (B,1,grid,grid)
           box_raw    = Conv1x1(tokens) -> (B,4,grid,grid)
           box_norm   = sigmoid(box_raw) so outputs in [0,1]
        """
        
        B = pixel_values.shape[0]

        with torch.no_grad():
            # patch_emb_out - > (batch_size, seq_len, hidden_size)
            ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)
        
        modulated_patch_embeddings = self.radar_layer(patch_emb_output)
        patch_emb_output = torch.cat([cls_token, modulated_patch_embeddings], dim = 1)

        pos_encoded = patch_emb_output + ViT_stat_pos_emb
        position_encoded = self.vit.embeddings.dropout(pos_encoded) # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(position_encoded, return_dict = True)  # It outputs only one item
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len+1, hidden_size)

        # We need to re-create image of original shape, so taking single CLS token doesn't work.
        all_tokens = sequence_output[:, 1:, :]  #(bs, seqlen, hidden_size)
        
        # reshape to (B,D,H,W) with H=W=14
        D = all_tokens.shape[-1]
        all_tokens = all_tokens.reshape(B, self.grid, self.grid, D)         # (B,g,g,D)
        all_tokens = all_tokens.permute(0, 3, 1, 2).contiguous()            # (B,D,g,g)

        obj_logits = self.obj_head(all_tokens)                          # (B,1,g,g)
        box_raw = self.box_head(all_tokens)                             # (B,4,g,g)
        box_norm = torch.sigmoid(box_raw)                           # (B,4,g,g) in [0,1]

        return {"obj_logits": obj_logits, "box_norm": box_norm}

    @torch.no_grad()
    def infer(self, images: torch.Tensor, score_thresh: float = 0.5, iou_thresh: float = 0.4):
        
       
        pred = self.forward(images)
        
        obj = torch.sigmoid(pred["obj_logits"])   # (B,1,g,g)
        box = pred["box_norm"]                    # (B,4,g,g)

        B = images.shape[0]
        results = []
        for b in range(B):
            scores_map = obj[b, 0]                # (g,g)
            boxes_map = box[b]                    # (4,g,g)

            ys, xs = torch.where(scores_map >= score_thresh)
            if ys.numel() == 0:
                results.append({"boxes_xyxy": torch.zeros((0,4)), "scores": torch.zeros((0,))})
                continue

            scores = scores_map[ys, xs]           # (K,)
            # gather boxes at those cells: (K,4)
            boxes_cxcywh = boxes_map[:, ys, xs].permute(1, 0).contiguous()

            # boxes are normalized to [0,1] over image size
            # convert to pixel xyxy
            boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2] * self.image_size
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2] * self.image_size

            # clamp
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, self.image_size - 1)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, self.image_size - 1)

            keep = nms_xyxy(boxes_xyxy, scores, iou_thresh=iou_thresh)
            results.append({"boxes_xyxy": boxes_xyxy[keep], "scores": scores[keep]})
        return results


