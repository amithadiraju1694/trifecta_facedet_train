import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    ViTModel,
    Mask2FormerModel,
    SegformerModel
                          )
import math
from typing import Tuple, Optional, Union
from peft import LoraConfig, get_peft_model, TaskType


from helpers import compute_single_patch_phi

from Custom_VIT import (
    PEG,
    AggregateSequenceGrading,
    
)



class GPSA(nn.Module):
    """
    Gated Positional Self-Attention (ConViT-style) operating on patch tokens.
    Combines content attention with a learnable locality prior over 2D positions:
        Attn = softmax( λ * (QK^T / sqrt(dh)) + (1-λ) * P )
    where P encodes relative 2D distances (Gaussian kernel with learnable scale).
    """
    def __init__(self, dim, num_heads=6, locality_strength=1.0, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.dh = dim // num_heads

        # Content projections
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Gating between content vs locality (λ in (0,1))
        self.logit_lambda = nn.Parameter(torch.zeros(1))      # λ = sigmoid(...)
        # Locality scale (Gaussian width)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(locality_strength)))

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    @staticmethod
    def _rel_pos_kernel(H, W, device, sigma):
        # (N,2) coordinate grid
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        coords = torch.stack([ys, xs], dim=-1).view(-1, 2).float()  # (N,2)
        # Pairwise squared distances
        diff = coords[:, None, :] - coords[None, :, :]               # (N,N,2)
        dist2 = (diff ** 2).sum(-1)                                  # (N,N)
        # Gaussian kernel
        P = torch.exp(-dist2 / (2 * sigma * sigma))                  # (N,N)
        return P

    def forward(self, x, H, W):
        """
        x: (B, N, D) patch tokens (NO CLS)
        H,W: grid size (N = H*W)
        """
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)    # (B,h,N,dh)
        k = self.k(x).view(B, N, self.h, self.dh).transpose(1, 2)    # (B,h,N,dh)
        v = self.v(x).view(B, N, self.h, self.dh).transpose(1, 2)    # (B,h,N,dh)

        # Content logits
        content = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B,h,N,N)

        # Locality prior P shared across heads (compute once per device/grid)
        sigma = F.softplus(self.log_sigma) + 1e-6
        P = self._rel_pos_kernel(H, W, x.device, sigma)              # (N,N)
        P = P.unsqueeze(0).unsqueeze(0).expand(B, self.h, N, N)      # (B,h,N,N)

        # Gating
        lam = torch.sigmoid(self.logit_lambda)                       # scalar
        logits = lam * content + (1.0 - lam) * P

        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                                  # (B,h,N,dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)         # (B,N,D)
        out = self.proj_drop(self.proj(out))
        return out

class ConViTHead(nn.Module):
    """
    Minimal ConViT-style head: LN -> GPSA -> LN -> MLP, width-preserving.
    """
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.gpsa = GPSA(dim, num_heads=num_heads, dropout=drop)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x, H, W):
        x = x + self.gpsa(self.ln1(x), H, W)
        x = x + self.mlp(self.ln2(x))
        return x

class ViTWithConvGPSAHead(nn.Module):
    """
    Canonical, compute-matched baseline:
      - Frozen ViT (google/vit-base-patch16-224) provides static-PE tokens.
      - A trainable ConViT-style GPSA block refines patch tokens BEFORE ViT encoder.
      - Pass refined tokens through the frozen ViT encoder; classify with a trainable head.
    """
    def __init__(
                self,
                pretrained_model_name="google/vit-base-patch16-224",
                num_out_classes=10,
                convit_heads=6,
                mlp_ratio=4.0,
                drop=0.0
                ):

        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        for p in self.vit.parameters(): p.requires_grad = False
        self.vit.eval()

        d = self.vit.config.hidden_size
        self.convit = ConViTHead(d, num_heads=convit_heads, mlp_ratio=mlp_ratio, drop=drop)
        self.classifier = nn.Linear(d, num_out_classes)

    def __get_vit_peout(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Includes patch embed + add static pos + cls
        return self.vit.embeddings(pixel_values)  # (B, 1+N, D)

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = True):
        with torch.no_grad():
            tok = self.__get_vit_peout(pixel_values)                # (B,1+N,D)

        # Split CLS / patches
        cls, patches = tok[:, :1, :], tok[:, 1:, :]                 # (B,1,D), (B,N,D)

        # Infer H,W from N (supports square grids)
        B, N, D = patches.shape
        HW = int(math.sqrt(N))
        assert HW * HW == N, "Non-square grid not supported in this minimal baseline."
        # ConViT refinement on patches (trainable)
        patches = self.convit(patches, HW, HW)                      # (B,N,D)

        # Reattach CLS and run frozen encoder
        x = torch.cat([cls, patches], dim=1)
        with torch.no_grad():
            enc = self.vit.encoder(x, return_dict=True)
            seq = self.vit.layernorm(enc.last_hidden_state)         # (B,1+N,D)

        # Classification (trainable)
        cls_tok = seq[:, 0]
        logits = self.classifier(cls_tok)
        return logits



# ----- Baseline Semantic Segmentation ------ #
class ViTWithPEG_SemSeg(nn.Module):
    """
    CPVT-like: Single-PEG once before the encoder
    Uses the standard single 'encoder' (stack of L transformer blocks).
    """
    def __init__(self,
                 base_ckpt="google/vit-base-patch16-224",
                 num_labels=10,
                 perc_ape: float = 1.0,
                 k: int = 3,
                 transpose_convolutions = False
                 ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(base_ckpt)
        self.num_labels = num_labels
        self.perc_ape = perc_ape
        self.k = k

        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.vit.eval()
        self.hidden = self.vit.config.hidden_size
        self.transpose_conv = transpose_convolutions
        
        ftr_dim = self.vit.config.hidden_size
        self.peg = PEG(ftr_dim, k=self.k)

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

                # B,128,56,56 -> B,64,224,224
                nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),

                # B,64,224,224 -> B,nC, 224,224
                nn.Conv2d(in_channels = 64, out_channels = num_labels, kernel_size = 1)

                                                )
        else:
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden, num_labels, kernel_size=1),
            )
    
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

    def forward(self, pixel_values: torch.Tensor, train_mode:bool = True, alignment_mode: bool = False, necs_mode: bool = False):
        
        
        with torch.no_grad():
            # patch_emb_output - (bs, seqlen, ftrdim)
            ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        _, seqlen, _ = patch_emb_output.shape
        H = W = int(math.sqrt(seqlen))  # assumes square grids (ViT 224/16 => 14x14)
        assert H*W == seqlen, "Non-square or unexpected grid; compute H,W from image/patch sizes."

        # When PEG is used, patch embeddings themselves contain positional + patch information. So, can take this as-is.
        # This Pos_idx-1 PEG variant, where injection is before any transformer layer, without multple PEG blocks. "Canonical Remedy"
        patch_emb_output = self.peg(x=patch_emb_output, H=H, W=W)  # (bs, seqlen, ftrdim)
        patch_emb_output = torch.cat([cls_token, patch_emb_output], dim=1) # (bs,seqlen+1,ftrdim)

        # No absolute position embeddings in canonical CPVT
        pos_encoded = patch_emb_output + ( self.perc_ape * ViT_stat_pos_emb )  # (bs, seqlen+1, ftrdim)
        position_encoded = self.vit.embeddings.dropout(pos_encoded)

        # Single encoder (stack of L transformer blocks)
        encoder_outputs = self.vit.encoder(position_encoded, return_dict=True)
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state)              # final LN

        # We need to get image of original shape, so taking first token alone doesn't help
        all_tokens = sequence_output[:, 1:, :]  #(bs, seqlen, hidden_size)
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

                # B,128,56,56 -> B,64,224,224
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
 
    def forward(self, pixel_values, train_mode:bool = True, alignment_mode: bool = False, necs_mode: bool = False):
        
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

class ViTWithMask2FormerSeg(nn.Module):
    """
    Assumes `pixel_values` are already preprocessed (or use `self.processor`).
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/mask2former-swin-base-ade-semantic",
        num_out_classes: int = 4,
        transpose_convolutions:bool = False
    ):
        super().__init__()

        # ---- Backbone: bare Mask2FormerModel (no head) ----
        self.m2f = Mask2FormerModel.from_pretrained(pretrained_model_name)
        
        for p in self.m2f.parameters():
            p.requires_grad = False
        self.m2f.eval()

        self.transpose_conv = transpose_convolutions
        self.hidden = self.m2f.config.feature_size

        # Task-Specific Semantic Segmentation layers
        if self.transpose_conv:
            self.seg_head = nn.Sequential(
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

    def __get_m2f_encout(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run Mask2Former in eval mode and return a (B, C, H, W) feature map.
        Assumes pixel_values are already preprocessed (e.g., resized to 224x224, normalized).
        """
        outputs = self.m2f(pixel_values=pixel_values, return_dict=True)
        feat = outputs.pixel_decoder_last_hidden_state
        if feat is None:
            # Fallback: use encoder feature map if pixel decoder isn't returned
            feat = outputs.encoder_last_hidden_state
        return feat
       
    def forward(self, pixel_values: torch.Tensor, train_mode: bool = False, alignment_mode: bool = False, necs_mode: bool = False) -> torch.Tensor:
        """
        pixel_values: (B, 3, 224, 224) tensors, already resized/normalized outside.
        Returns: logits (B, num_out_classes, 224, 224)
        """
        
        with torch.no_grad():
            sequence_output = self.__get_m2f_encout(pixel_values)
        
        B, FTR_DIM, H, W = sequence_output.shape # BS, FTRDIM, H,W(N)
        # For 224x224 inputs and common_stride=4, expect 56x56 feature map.
        # M2F output doesn't have CLS token, hence no slicing & reshaping needed.
        # (If you change input size / config, this assert may need updating.)
        assert H == 56 and W == 56, f"Unexpected feature map size: {(H, W)}"

        # M2F doesn't use CLS , so need not remove sequences
        # batch size, num_segment maps, height , width
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

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = False, alignment_mode: bool = False, necs_mode: bool = False) -> torch.Tensor:
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
class ViTRADAR_SoftDegrade_SemSeg(nn.Module):

    """This class implements the ViT-PFIM model with soft degrade/upgrade of patch embeddings: which computes important sequences in patch embeddings space
    and modulates those sequences with distances from patch embeddings directly, without projecting those distance into 
    polynomial features like in ViTRADAR-SoftAnchor model. Goal of this modulation is to better guide MHSA towards "important sequences"
    adding global awareness to each patch embedding.
    """
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 distance_metric='euclidean',
                 aggregate_method='norm',
                 seq_select_method = 'weighted_sum',
                 aggregate_dim = 2,
                 norm_type = 2,
                 return_anchors = False,
                 perc_ape = 0.75,
                 corrupt_imp_weights=False,
                 num_out_classes = 10,
                 transpose_convolutions:bool = False
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        # Ensuring strict eval mode of backbone
        self.vit.eval()

        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method
        self.seq_select_method= seq_select_method
        self.aggregate_dim=aggregate_dim
        self.norm_type = norm_type
        self.return_anchors = return_anchors
        self.corrupt_imp_weights=corrupt_imp_weights
        self.transpose_conv = transpose_convolutions
        self.hidden = self.vit.config.hidden_size

        # This is to check how much of custom positional encoding to be added to the original positional encoding
        # This is not a trainable parameter
        self.perc_ape = perc_ape

        # norm is L2 Norm
        self.custom_pos_encoding = AggregateSequenceGrading(
            distance_metric=self.distance_metric,
            aggregate_method=self.aggregate_method, # l2- Norm, entropy
            seq_select_method = self.seq_select_method,
            aggregate_dim = self.aggregate_dim,
            norm_type = self.norm_type,
            return_anchors = self.return_anchors,
            corrupt_imp_weights = self.corrupt_imp_weights
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

                # B,128,56,56 -> B,64,224,224
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

    def forward(self, pixel_values: torch.Tensor, train_mode:bool = True, alignment_mode: bool = False, necs_mode: bool = False):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        with torch.no_grad():
            # patch_emb_output -> (batch_size, seq_len, hidden_size)
            ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        
        if self.return_anchors:
            # Returns tuple if return_anchors=True, else just one
            _, custom_pos_encodings, _ = self.custom_pos_encoding(patch_emb_output) # (bs, seqlen, ftrdim)
        else:
            custom_pos_encodings = self.custom_pos_encoding(patch_emb_output) # (bs, seqlen, ftrdim)
        
        custom_pos_encodings = torch.cat([cls_token, custom_pos_encodings], dim = 1)
        
        # Add both positional encodings
        position_encoded = custom_pos_encodings + (self.perc_ape * ViT_stat_pos_emb) # (batch_size, seq_len, hidden_size)
        
        position_encoded = self.vit.embeddings.dropout(position_encoded) # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(position_encoded, return_dict = True)  # It outputs only one item
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len+1, hidden_size)

        # We need to re-create image of original shape, so taking single CLS token doesn't work.
        all_tokens = sequence_output[:, 1:, :]  #(bs, seqlen, hidden_size)
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
                 corrupt_imp_weights: bool = False,
                 transpose_convolutions:bool = False
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.vit.eval()

        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method
        self.seq_select_method = seq_select_method
        self.add_coordinates = add_coordinates
        self.K = K
        self.aggregate_dim = aggregate_dim
        self.norm_type = norm_type
        self.return_anchors = return_anchors
        self.add_coordinates = add_coordinates # Whether to get co-ordinates output when dealing with phi vector computation
        self.perc_ape = perc_ape # Percentage of Absolute Positional Encoding to be used in Forward pass
        self.corrupt_imp_weights = corrupt_imp_weights
        self.transpose_conv = transpose_convolutions
        self.hidden = self.vit.config.hidden_size

        # These are learnable params that govern how much % of s,b will be added to patch embeddings before sending to encoder.
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype = torch.float32))  # Trainable scalar parameter
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype = torch.float32))  # Trainable scalar parameter

        # Custom aggregation on top of patch embeddings.
        self.custom_pos_encoding = AggregateSequenceGrading(
            distance_metric=self.distance_metric,
            aggregate_method=self.aggregate_method,
            seq_select_method = self.seq_select_method,
            aggregate_dim=self.aggregate_dim,
            norm_type=self.norm_type,
            return_anchors=self.return_anchors,
            corrupt_imp_weights = self.corrupt_imp_weights
        )

        # Computing output dimensions of Phi features
        if self.add_coordinates:
            # co-ordinates = 4, non-coords = 3 = 7
            # 6K
            phi_out_features = 7 + 6 * K
        else:
            phi_out_features = 3 + 2 * K

        # Linear sequential layers to project phi onto new dimensions
        self.projection_phi = nn.Sequential(
            nn.Linear(in_features = phi_out_features, out_features=128),
            nn.Linear(in_features = 128, out_features = self.vit.config.hidden_size*2)
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

                # B,128,56,56 -> B,64,224,224
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

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = True, alignment_mode: bool = False, necs_mode: bool = False):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        with torch.no_grad():
            # patch_emb_out - > (batch_size, seq_len, hidden_size)
            ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)


        # (bts, seq_len, 1) or (bts, seq_len, feature_dim)
        _, anchor_values, weights = self.custom_pos_encoding(patch_emb_output) # distances, anchor values, weights

        # (bs, seqlen, num_phi_ftr)
        phi_offset = compute_single_patch_phi(anchor_values=anchor_values,
                                            x = patch_emb_output,
                                            K = self.K,
                                            add_coords=self.add_coordinates,
                                            weights_sequences = weights
                                                        )
        
        custom_pos_encodings = self.projection_phi(phi_offset) # (bs, seqlen, ftrdim*2)

        s = custom_pos_encodings[:, :, :self.vit.config.hidden_size] # (bs, seqlen, ftrdim)
        b = custom_pos_encodings[:, :, self.vit.config.hidden_size: ] # (bs, seqlen, ftrdim)

        patch_emb_output = patch_emb_output * (1 + self.alpha * s) + self.gamma * b
        patch_emb_output = torch.cat([cls_token, patch_emb_output], dim = 1)

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


        

        


