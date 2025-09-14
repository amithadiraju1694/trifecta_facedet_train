
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from transformers import ViTModel
import math
from typing import Tuple, Optional, Union
from helpers import get_vector_agg, compute_single_patch_phi, safe_pow_gate

from Custom_VIT import AggregateSequenceGrading, PEG

class ViTFiLM_RandNoise(nn.Module):
    """This class implements FiLM approach of modulating patch embeddings with s & b vectors 
    but with random Gaussian noise for both."""
    def __init__(self,
                 exp_seed,
                 pretrained_model_name="google/vit-base-patch16-224",
                 num_out_classes = 10,
                 use_both = False,
                 perc_ape = 1.0
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.use_both = use_both
        self.num_out_classes = num_out_classes
        self.perc_ape = perc_ape
        
        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.eval()

        # These are learnable params that govern how much % of s,b will be added to patch embeddings before sending to encoder.
        self.alpha = 1.0
        self.noise_std_s = 1.0

        if self.use_both:
            self.gamma = 1.0
            self.noise_std_b = 1.0

        # Using generator for seeding manually, so that random noise are reproducible across runs
        set_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = torch.Generator(device = set_device).manual_seed(exp_seed)

        # Classifier layer for the output
        self.classifier = nn.Linear(self.vit.config.hidden_size,
                                    num_out_classes)

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

    def generate_gaussian_noise(self, x: torch.Tensor,train_mode: bool):
        """Function that creates gaussian noise tensors 's' and 'b' during training and zeros during eval mode."""
        
        # Create Gaussian Noise S and B Tensors during training and zeros otherwise
        if train_mode:
            s = torch.empty_like(x).normal_(mean=0.0, std=self.noise_std_s, generator=self.generator)
            if self.use_both: b = torch.empty_like(x).normal_(mean=0.0, std=self.noise_std_b, generator=self.generator)
        else:
            shape_tuple = x.shape
            s = torch.zeros(shape_tuple)
            if self.use_both: b = torch.zeros(shape_tuple)

        
        if torch.cuda.is_available():
            s = s.to('cuda', non_blocking = True)
            if self.use_both:
                b = b.to('cuda', non_blocking = True)
            
        
        if self.use_both: return s,b
        return s

    def forward(self, pixel_values, train_mode = True):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        with torch.no_grad():
            # patch_emb_out - > (batch_size, seq_len, hidden_size)
            ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

            # Whether to use both S and B vectors for patch embedding modulation
            if self.use_both:
                # We always use s,b = 0 to show clear performance drop , as-good as baseline without distance features
                s,b = self.generate_gaussian_noise( patch_emb_output,False)
                patch_emb_output = patch_emb_output * (1 + self.alpha * s) + self.gamma * b
            else:
                s = self.generate_gaussian_noise(patch_emb_output,False)
                patch_emb_output = patch_emb_output * (1 + self.alpha * s)

            patch_emb_output = torch.cat([cls_token, patch_emb_output], dim = 1)
            pos_encoded = patch_emb_output + ( self.perc_ape * ViT_stat_pos_emb )

            position_encoded = self.vit.embeddings.dropout(pos_encoded) # (batch_size, seq_len+1, hidden_size)

            # Continue with the encoder
            encoder_outputs = self.vit.encoder(position_encoded, return_dict = True)  # It outputs only one item
            sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len, hidden_size)

            # Use [CLS] token for classification, since it attends to all future tokens
            req_token = sequence_output[:, 0]  #(bs, seqlen)


        logits = self.classifier(req_token)  #(bs, num_labels)

        return logits


class ViTWithPEG(nn.Module):
    """
    CPVT-like: Single-PEG once before the encoder
    Uses the standard single 'encoder' (stack of L transformer blocks).
    """
    def __init__(self,
                 base_ckpt="google/vit-base-patch16-224",
                 num_labels=10,
                 perc_ape: float = 1.0,
                 k: int = 3
                 ):
        super().__init__()
        # backbone without classification head
        self.vit = ViTModel.from_pretrained(base_ckpt)
        self.num_labels = num_labels
        self.perc_ape = perc_ape
        self.k = k

        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.vit.eval()
        
        ftr_dim = self.vit.config.hidden_size
        self.peg = PEG(ftr_dim, k=self.k)

        self.classifier = nn.Linear(ftr_dim, num_labels)
    
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

    def forward(self, pixel_values: torch.Tensor, alignment_mode: bool = False, necs_mode: bool = False):
        
        
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

        req_token = sequence_output[:, 0]  #(bs, seqlen)
        logits = self.classifier(req_token)  #(bs, num_labels)
        
        return logits


# v1 contains both s and b when injecting into patch embeddings.
class ViTRADAR_SoftAnchor_v1(nn.Module):
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
                 corrupt_imp_weights: bool = False
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

        # Classifier layer for the output
        self.classifier = nn.Linear(self.vit.config.hidden_size,
                                    num_out_classes)

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

    def forward(self, pixel_values: torch.Tensor, alignment_mode: bool = False, necs_mode: bool = False):

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
        
        
        # Random Permutations of Sj and Bj to show clear drop in performance
        if alignment_mode:
            perm = torch.randperm(patch_emb_output.shape[1], device=s.device)
            s = s[:, perm, :]
            b = b[:, perm, :]
        
        # Sj, Bj are zero-ed out, Necessity test
        # both randomness and necessity shouldn't be part of same test
        if necs_mode:
            s = torch.zeros(s.shape, device = s.device)
            b = torch.zeros(b.shape, device = b.device)


        patch_emb_output = patch_emb_output * (1 + self.alpha * s) + self.gamma * b
        patch_emb_output = torch.cat([cls_token, patch_emb_output], dim = 1)

        # A Scalar percentage for using Absolute PEs is added to facilitate comparison with PEG.
        pos_encoded = patch_emb_output + ( self.perc_ape * ViT_stat_pos_emb )

        position_encoded = self.vit.embeddings.dropout(pos_encoded) # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(position_encoded, return_dict = True)  # It outputs only one item
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len, hidden_size)

        req_token = sequence_output[:, 0]  #(bs, seqlen)
        logits = self.classifier(req_token)  #(bs, num_labels)

        return logits


class ViTRADAR_SoftDegrade(nn.Module):

    """This class implements the ViT-RADAR model with soft degrade/upgrade of patch embeddings: which computes important sequences in patch embeddings space
    and modulates those sequences with distances from patch embeddings directly, without projecting those distance into 
    polynomial features like in ViTRADAR-SoftAnchor model. Goal of this modulation is to better guide MHSA towards "important sequences"
    adding global awareness to each patch embedding.
    """
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 distance_metric='NA',
                 aggregate_method='entropy',
                 seq_select_method = 'safe_pow_gate',
                 aggregate_dim = 2,
                 norm_type = 2,
                 return_anchors = False,
                 perc_ape = 0.0,
                 corrupt_imp_weights=False,
                 num_out_classes = 10
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
        self.seq_select_method= seq_select_method
        self.aggregate_dim=aggregate_dim
        self.norm_type = norm_type
        self.return_anchors = return_anchors
        self.corrupt_imp_weights=corrupt_imp_weights

        # This is to check how much of custom positional encoding to be added to the original positional encoding
        # This is not a trainable parameter
        self.perc_ape = perc_ape

        # Classifier layer for the output
        self.classifier = nn.Linear(self.vit.config.hidden_size,
                                    num_out_classes)

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

    def forward(self, pixel_values: torch.Tensor, alignment_mode:bool = False, necs_mode: bool = False):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        with torch.no_grad():
            # patch_emb_output -> (batch_size, seq_len, hidden_size)
            ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        N = patch_emb_output.shape[1]
        
        # Aggregate vectors of original image/ patch embeddings. ex: l2 norm, entropy-based, softmax, max-min etc
        # (bs, seqlen)
        vector_values = get_vector_agg(aggregate_method=self.aggregate_method,
                       x=patch_emb_output,
                       req_dim = self.aggregate_dim,
                       norm_type = self.norm_type,
                       smooth_topk=False,
                       topk_val= None
                       ) # (batch_size, seq_len)
        
        # Alignment mode is always permuting importance values, irrespective of methods used
        # alignment_mode and zeor_mode should never be used together
        if alignment_mode:
            perm = torch.randperm(N, device=vector_values.device)
            vector_values = vector_values[:, perm]

        # Setting mix to zero ensures that modulated patch embeddings are not added
        # Which shows the effect of modulations and drop in test accuracy
        if necs_mode:
            custom_pos_encodings = safe_pow_gate(x = patch_emb_output, s = vector_values, mix = 0.0)
        if not necs_mode:
            # Leaving default values as-is , should be used during training
            custom_pos_encodings = safe_pow_gate(x = patch_emb_output, s = vector_values)
        
        custom_pos_encodings = torch.cat([cls_token, custom_pos_encodings], dim = 1)
        
        # Add both positional encodings
        position_encoded = custom_pos_encodings + (self.perc_ape * ViT_stat_pos_emb) # (batch_size, seq_len, hidden_size)
        position_encoded = self.vit.embeddings.dropout(position_encoded) # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(position_encoded, return_dict = True)  # It outputs only one item
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len, hidden_size)

        req_token = sequence_output[:, 0]  #(bs, seqlen)
        logits = self.classifier(req_token)  #(bs, num_labels)

        return logits
