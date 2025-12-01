import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import math
from typing import Tuple, Optional, Union
from peft import LoraConfig, get_peft_model, TaskType

from helpers import (
    get_vector_agg,
    get_anchor_vectors,
    compute_single_patch_phi
)

class AggregateSequenceGrading(nn.Module):
    """
    This class aggregates sequences based on various methods and computes distances from the aggregated vector.
    It can return distances, anchor vectors, and importance weights based on configuration.
    """

    def __init__(self,
                 distance_metric='euclidean',
                 aggregate_method='norm',
                 seq_select_method = 'weighted_sum',
                 aggregate_dim = 2,
                 norm_type = None,
                 return_anchors = True,
                 corrupt_imp_weights: bool = False
                 ):
        """
        Initialize the custom positional encoding module.

        Args:
            distance_metric: Method to compute distances ('euclidean', 'cosine', 'manhattan')
            aggregate_method: Method to determine the "maximum" vector ('norm', 'sum', 'max_elem', 'entropy', 'softmax')
            seq_select_method: Method to select sequences, whether one of all sequences ( argmin or argmax ) or weighted sequences by importance
            (soft_weighted_sum, loo_weighsum_loop) etc.
            aggregate_dim: Which dimension to aggregate vectors by.
            norm_type: Whether to perform L1 or L2 norm, only used when aggregate_method == 'norm'
            return_anchors: Whether to get back weighted anchors (improtance * sequences ) as output.
        """
        
        super(AggregateSequenceGrading, self).__init__()
        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method
        self.seq_select_method = seq_select_method
        self.aggregate_dim = aggregate_dim
        self.norm_type = norm_type
        self.return_anchors = return_anchors
        self.corrupt_imp_weights = corrupt_imp_weights

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            Positional encodings as distances from max vector (batch_size, seq_len, 1)
        """
        batch_size = x.shape[0]

        # Extract the maximum vectors
        batch_indices = torch.arange(batch_size, device=x.device)

        # Aggregate vectors of original image/ patch embeddings. ex: l2 norm, entropy-based, softmax, max-min etc
        # (bs, seqlen)
        vector_values = get_vector_agg(aggregate_method=self.aggregate_method,
                       x=x,
                       req_dim = self.aggregate_dim,
                       norm_type = self.norm_type,
                       smooth_topk=self.corrupt_imp_weights,
                       topk_val= 98 if self.corrupt_imp_weights else None
                       ) # (batch_size, seq_len)
        
        if self.aggregate_method == 'entropy':
            # Doing this negation to ensure high entropy values will be scaled down aggresively
            # and low entropy values will be scaled up , both in PFIM and RADAR LOOSA
            # Small epsilon to cover for inclusive 0,1 boundary values
            vector_values = (1. - vector_values) + 1e-8

        # Single anchor or group of anchors with soft selection
        # These anchor vectors are weighted by importance weights computed above with orginal patch embeddings
        anchor_vectors = get_anchor_vectors(seq_select_method = self.seq_select_method,
                                            vector_values = vector_values,
                                            x = x,
                                            batch_indices = batch_indices)
        
        # Compute distances from the maximum vector
        if self.distance_metric == 'euclidean':

            # TODO: May need to change this to element wise difference
            #  and keep difference asa vector
            distances = torch.norm(x - anchor_vectors, p=2, dim=2, keepdim=True)
        
        elif self.distance_metric == 'cosine':
        
            x_norm = torch.norm(x, p=2, dim=2, keepdim=True) + 1e-8
            max_norm = torch.norm(anchor_vectors, p=2, dim=2, keepdim=True) + 1e-8

            x_normalized = x / x_norm
            max_normalized = anchor_vectors / max_norm

            cosine_sim = torch.sum(x_normalized * max_normalized, dim=2, keepdim=True)
            distances = 1 - cosine_sim
        
        elif self.distance_metric == 'manhattan':
            distances = torch.sum(torch.abs(x - anchor_vectors), dim=2, keepdim=True)
        
        elif self.distance_metric == 'per_feat_manhattan':
            distances = torch.abs(x - anchor_vectors) # (bs, seqlen, feature_dim)
        
        elif self.distance_metric == 'NA':
            # User will compute custom distances with anchor values found, hence distances are None
            distances = None

        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")


        if self.return_anchors:
            return(distances, anchor_vectors, vector_values)
        
        return distances

class DecompSequenceGrading(nn.Module):
    """
        This class implements sequence grading based on various decomposition methods like QR, SVD, Eigen decomposition.
        It can return transformed sequences based on top-k decomposed vectors and specified strategy.
        It's currently functional for only Eigen decomposition.
    """

    def __init__(self,
                 decomp_algo: str = 'qr',
                 decomp_strategy: Optional[str] = "project", # project or "importance"
                 top_k_seqfeat: Optional[int] = 64,
                 alpha: Union[float, torch.Tensor] = 0.5
                 ):
        """
        Initialize the custom positional encoding module.

        Args:
            decomp_algo: Type of decomposition to use ('qr', 'svd')
            top_k_seqfeat: Number of top sequences to keep after decomposition
            keep_all_decomposed: If True, keep all decomposed vectors, otherwise only top_k.
            This normalizes importance of each sequence based on the decomposition and multiplies
            this normalized importance with the original sequence, effectively embedding which sequences are more relevant.
        """
        
        super(DecompSequenceGrading, self).__init__()
        self.top_k_seqfeat = top_k_seqfeat

        self.decomp_algo = decomp_algo
        self.decomp_strategy = decomp_strategy

        self.epsilon = 1e-7
        self.alpha = alpha

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            transformed_sequences: Tensor of shape (batch_size, seq_len, feature_dim) after decomposition and transformation.
        """

        batch_size, seq_len, feature_dim = x.shape
        identity_stability = torch.eye(feature_dim) * self.epsilon

        # CHOOSE AN APPROPRIATE DECOMPOSITION METHOD FROM AVAILABLE OPTIONS
        if self.decomp_algo == 'qr':

            
            # Q - (bts,seqlen, seqlen), R - (bts,seqlen, ftrdim)
            decomp_vectors, magnitude_values = torch.linalg.qr(x,
                                    mode='reduced') # (bts, seq_len, seq_len)

            # Aggregated magntitude values, for each sequence
            magnitude_values = torch.norm(magnitude_values, p=2, dim=2)  # (bts, seqlen)

        if self.decomp_algo == 'svd':
            #  U(sl, sl), S, Vh
            decomp_vectors, maginute_S_vectors\
                , magnitude_V_vectors = torch.linalg.svd(x,
                      full_matrices=False) # U - (bts, seq_len, seq_len)

            # (bts, seq_len, ftr_dim)
            # magnitude_values = maginute_S_vectors @ magnitude_V_vectors.transpose(1, 2)

            # Aggregated magntitude values, for each sequence
            # magnitude_values = torch.norm(magnitude_values, p=2, dim=2)  # (bts, seqlen)


        if self.decomp_algo == 'eig':
            
            # Ensuring Eigen value & vectors computation is gradient-free.
            with torch.no_grad():
                x_mean = x.mean(dim = 1, keepdim=True) # (bts, 1, ftrdim)
                x_centered = x - x_mean # (bts, seq_len, ftr_dim)
                #TODO: Should denominator be ftrdim ?
                x_covar = (x_centered.transpose(1,2) @ x_centered ) / (seq_len - 1) # (bts, ftrdim, ftrdim)

                # mag_vals - (bts, ftrdim), decomp_vectors - (bts, ftr_dim, ftr_dim). w,V
                eigen_values, eigen_vectors = torch.linalg.eigh(x_covar + identity_stability.unsqueeze(0)) # comes in ascending order

                # Sort eigen vectors by their eigen value norms
                idx = eigen_values.argsort(dim=-1, descending=True)
                eigen_vectors = eigen_vectors.gather(-1, idx.unsqueeze(-2).expand_as(eigen_vectors)) # (bs,ftrdim,ftrdim)
                
                # Z = Xc @ V
                Z = x_centered @ eigen_vectors # (bts, seqlen, ftrdim)

        if self.decomp_strategy == 'recon_topk':
            
            # Top-K Reconstruction: x_topk (bs, seqlen, ftrdim) = (bs, seqlen, K) @ (bs,K, ftrdim)
            decomposed_vectors = Z[:, :, :self.top_k_seqfeat] @ eigen_vectors[:, :, :self.top_k_seqfeat].transpose(1,2) + x_mean # (bts, seqlen, ftrdim)
        

        if self.decomp_strategy == 'scalar_saliency_topk':

            # top-k energy
            s_topk = (Z[..., :self.top_k_seqfeat]**2).sum(dim=-1) # (bts,seqlen)

            # normalize (per image)
            with torch.no_grad():
                decomposed_vectors = torch.softmax(s_topk, dim=-1) #(bts, seqlen)
        

        if self.decomp_strategy == 'saliency_topk':

            x_topk = Z[:, :, :self.top_k_seqfeat] @ eigen_vectors[:, :, :self.top_k_seqfeat].transpose(1,2) # (bts, seqlen, ftrdim)

            # top-k energy
            s_topk = (Z[..., :self.top_k_seqfeat]**2).sum(dim=-1) # (bts,seqlen)

            # normalize (per image)
            with torch.no_grad():
                s = torch.softmax(s_topk, dim=-1) #(bts, seqlen)
            
            decomposed_vectors = s.unsqueeze(-1) * x_topk # (bts, seqlen, ftrdim)


        return decomposed_vectors


class PEG(nn.Module):
    """Position Encoding Generator (CPVT): depth-wise kxk conv on patch grid."""
    def __init__(self, dim, k=3):
        super().__init__()
        pad = k // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=pad, groups=dim, bias=True)

    def forward(self, x, H, W):
        # x: (B, N, D) where N = H*W (patch tokens only, no CLS)
        B, N, D = x.shape
        x2d = x.transpose(1, 2).contiguous().view(B, D, H, W)   # (B,D,H,W)
        pe = self.dwconv(x2d)                                   # (B,D,H,W)
        pe = pe.view(B, D, H*W).transpose(1, 2).contiguous()    # (B,N,D)
        return x + pe                                           # residual PEG (canonical)

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

    def forward(self, pixel_values: torch.Tensor, train_mode:bool = True):
        
        
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

class ViTRADAR_SoftDegrade(nn.Module):

    """This class implements the ViT-RADAR model with soft degrade/upgrade of patch embeddings: which computes important sequences in patch embeddings space
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

    def forward(self, pixel_values: torch.Tensor, train_mode:bool = True):

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
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len, hidden_size)

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

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = True):

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
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len, hidden_size)

        req_token = sequence_output[:, 0]  #(bs, seqlen)
        logits = self.classifier(req_token)  #(bs, num_labels)

        return logits

# v2 doesn't contain both s and b when injecting into patch embeddings.
# Just 's' - like injection
class ViTRADAR_SoftAnchor_v2(nn.Module):
    """This class implements the ViT-RADAR model with Soft Anchor distances of patch embeddings: this computes important sequences in patch embeddings space
    and modulates those sequences with projected features derived from distances between soft anchors and patch embeddings.

    This version uses only the 's' part of FiLM style injection into patch embeddings.
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
                 corrupt_imp_weights=False
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
            nn.Linear(in_features = 128, out_features = self.vit.config.hidden_size)
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

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = True):

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
        
        custom_pos_encodings = self.projection_phi(phi_offset) # (bs, seqlen, ftrdim)
        patch_emb_output = patch_emb_output * (1 + self.alpha * custom_pos_encodings)

        patch_emb_output = torch.cat([cls_token, patch_emb_output], dim = 1)
        
        # A Scalar percentage for using Absolute PEs are added to facilitate comparison with PEG.
        pos_encoded = patch_emb_output + ( self.perc_ape * ViT_stat_pos_emb )

        position_encoded = self.vit.embeddings.dropout(pos_encoded) # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(position_encoded, return_dict = True)  # It outputs only one item
        sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len, hidden_size)

        req_token = sequence_output[:, 0]  #(bs, seqlen)
        logits = self.classifier(req_token)  #(bs, num_labels)

        return logits


# TODO: This needs bug fixes
class ViTWithDecompSequenceGrading(nn.Module):
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 decomp_algo = 'qr',  # or qr or svd
                 decomp_strategy = 'project', # project or importance
                 top_k_seqfeat = 32,
                 num_out_classes = 10,
                 alpha = 0.85
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        self.decomp_algo = decomp_algo
        self.decomp_strategy = decomp_strategy
        self.top_k_seqfeat = top_k_seqfeat 
        self.alpha = alpha

        # norm is L2 Norm
        self.custom_pos_encoding = DecompSequenceGrading(
            decomp_algo = self.decomp_algo,
            decomp_strategy = self.decomp_strategy, # project or "importance"
            top_k_seqfeat = self.top_k_seqfeat,
            alpha = self.alpha
                                    )

        # Classifier layer for the output
        self.classifier = nn.Linear(self.vit.config.hidden_size,
                                    num_out_classes)

    def __get_vit_peout(self, pixel_values):

        # Get patch embeddings - (bts, seq_len, hidden_size)
        patch_emb_output = self.vit.embeddings.patch_embeddings(pixel_values)

        # Add class token
        cls_token = self.vit.embeddings.cls_token.expand(
            patch_emb_output.shape[0], -1, -1)

        # Apply dropout
        patch_emb_output = self.vit.embeddings.dropout(patch_emb_output)  # (batch_size, seq_len+1, hidden_size)

        # Get static positional embeddings
        ViT_stat_pos_emb = self.vit.embeddings.position_embeddings  # (1, seq_len+1, hidden_size)

        # Detaching to ensure gradients don't follow back on to pre-trained Graph
        patch_emb_output = patch_emb_output.detach()
        ViT_stat_pos_emb = ViT_stat_pos_emb.detach()
        cls_token = cls_token.detach()

        return (ViT_stat_pos_emb, patch_emb_output, cls_token)

    def forward(self, pixel_values, train_mode = True):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        # patch_embed_out -> (batch_size, seq_len, hidden_size)
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        # Calculate custom positional encoding on sequence with class token
        # Shape varies by sequence type
        custom_pos_encodings = self.custom_pos_encoding(patch_emb_output)

        # Add both positional encodings
        if self.decomp_strategy == 'recon_topk':
            custom_pos_en_out = torch.cat([cls_token, custom_pos_encodings], dim = 1) # (bs, seqlen+1, ftrdim)

        # If importance, then sequence importances are given between 0-1
        elif self.decomp_strategy =='scalar_saliency_topk':
            alpha = torch.tensor([0.5], dtype = torch.float32)
            custom_pos_en_out = patch_emb_output * (1 + alpha * custom_pos_encodings.unsqueeze(-1))
            custom_pos_en_out = torch.cat([cls_token, custom_pos_en_out], dim = 1) # (bs, seqlen+1, ftrdim)
        
        elif self.decomp_strategy == 'saliency_topk':
            beta = torch.tensor([0.5], dtype = torch.float32)
            custom_pos_en_out = patch_emb_output + ( beta * custom_pos_encodings )
            custom_pos_en_out = torch.cat([cls_token, custom_pos_en_out], dim = 1) # (bs, seqlen+1, ftrdim)

        pos_encoded = custom_pos_en_out + ViT_stat_pos_emb # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(pos_encoded) # It outputs only one item
        sequence_output = encoder_outputs[0] # (batchsize,seq_len,hidden_size)

        req_token = sequence_output[:, 0]  #(bs, seqlen)
        logits = self.classifier(req_token)  #(bs, num_labels)

        return logits


class ViTWithStaticPositionalEncoding(nn.Module):
    def __init__(self,
                    pretrained_model_name="google/vit-base-patch16-224",
                    num_out_classes=10):
        
        """
        Class that implements Vanilla ViT with only static positional encoding from pre-trained ViT.
        This freezes all layers of ViT and only trains the final classifier layer; this implementation is CANONICAL to HuggingFace's ViTModel class.

        Args:
            pretrained_model_name (str): Name of the pre-trained ViT model from HuggingFace.
            num_out_classes (int): Number of output classes for classification.
        
        Returns:
            logits ( in forward pass ) (torch.Tensor): Output logits of shape (batch_size, num_out_classes).
        """
        super().__init__()
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Setting only the backbone to be frozen
        self.vit.eval()
    
        # Classifier layer for the output
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_out_classes)

    def __get_vit_peout(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Private method that applies patch embeddings given raw image pixels, loads cls token and static positional encodings from pre-trained ViT;
        and returns the positionally encoded patch embeddings to be sent to the encoder.
        
        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (batch_size, num_channels, height, width).
        
        Returns:
            position_encoded (torch.Tensor): Positionally encoded patch embeddings of shape (batch_size, seq_len+1, hidden_size).
        """

        # Get patch embeddings - (bts, seq_len, hidden_size)
        patch_emb_output = self.vit.embeddings.patch_embeddings(pixel_values)

        # Add class token
        cls_token = self.vit.embeddings.cls_token.expand(
            patch_emb_output.shape[0], -1, -1)
        
        # (batch_size, seq_len+1, hidden_size). This is right order for concatenating with CLS in baseline, but not for custom approaches.
        patch_emb_output = torch.cat((cls_token, patch_emb_output), dim=1)
        absolute_pe = self.vit.embeddings.position_embeddings # (1, seqlen+1, ftrdim)

        position_encoded = patch_emb_output + absolute_pe # (bs, seqlen+1, ftrdim)
        position_encoded = self.vit.embeddings.dropout(position_encoded)  # (batch_size, seq_len+1, hidden_size)
    

        return position_encoded

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = True):
        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        with torch.no_grad():

            # (batch_size, seq_len+1, hidden_size), BASIC STATIS POS ENC FROM ViT
            position_encoded_output = self.__get_vit_peout(pixel_values)

            # Continue with the encoder
            encoder_outputs = self.vit.encoder(position_encoded_output, return_dict = True)  # It outputs only one item
            sequence_output = self.vit.layernorm(encoder_outputs.last_hidden_state) # (batchsize, seq_len, hidden_size)

            # Use [CLS] token for classification, since it attends to all future tokens
            cls_token = sequence_output[:, 0]  # (batch_size, hidden_size)

        # Compute gradients only for last layer for fine tuning
        logits = self.classifier(cls_token)  # (batch_size, num_out_classes)

        return logits


class ViTLoRAClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 num_out_classes:int = 10,
                 r = 16,
                 lora_alpha = 16,
                 lora_dropout = 0.05,
                 target_module = "attn_min"):

        super().__init__()

        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        

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
        
        # Task head (train this with LoRA)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_out_classes)
    

    def forward(self, pixel_values, train_mode:bool = True):
        
        # IMPORTANT: call the *base model* to bypass PEFT’s kwargs plumbing
        # LoRA is injected into modules, so it still applies.
        base = self.vit.base_model            # peft.PeftModel -> underlying ViTModel
        out = base(pixel_values=pixel_values, return_dict=True)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits


# -------- Set Transformer (Lee et al., 2019) blocks --------
class rFF(nn.Module):
    def __init__(self, d, hidden_mult=4, drop=0.0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d, hidden_mult * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_mult * d, d),
        )
    def forward(self, x): return self.ff(x)

class MAB(nn.Module):
    """Multihead Attention Block: LN(x + MHA(x, y)) -> LN(res + rFF)"""
    def __init__(self, d, n_heads=4, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, n_heads, dropout=drop, batch_first=True)
        self.ff  = rFF(d, hidden_mult=4, drop=drop)
    def forward(self, x, y):
        h = self.ln1(x + self.mha(x, y, y, need_weights=False)[0])
        return self.ln2(h + self.ff(h))

class SAB(nn.Module):
    """Self-Attention Block: MAB(X, X)"""
    def __init__(self, d, n_heads=4, drop=0.0):
        super().__init__()
        self.mab = MAB(d, n_heads, drop)
    def forward(self, x): return self.mab(x, x)

class ISAB(nn.Module):
    """Induced Self-Attention Block with m inducing points."""
    def __init__(self, d, m=16, n_heads=4, drop=0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) / (d ** 0.5))
        self.mab1 = MAB(d, n_heads, drop)
        self.mab2 = MAB(d, n_heads, drop)
    def forward(self, x):
        B = x.size(0)
        I = self.I.expand(B, -1, -1)        # (B, m, d)
        H = self.mab1(I, x)                  # Inducing -> set
        return self.mab2(x, H)               # Set attends to induced

class PMA(nn.Module):
    """Pooling by Multihead Attention with k seeds (k=1 for classification)."""
    def __init__(self, d, k=1, n_heads=4, drop=0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d) / (d ** 0.5))
        self.mab = MAB(d, n_heads, drop)
    def forward(self, x):
        B = x.size(0)
        S = self.S.expand(B, -1, -1)        # (B, k, d)
        return self.mab(S, x)                # (B, k, d)

class SetTransformerHead(nn.Module):
    """
    Canonical paper-style head: ISAB -> ISAB -> PMA(1).
    Width-preserving (d stays ViT hidden size).
    """
    def __init__(self, d, m=16, n_heads=4, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ISAB(d, m=m, n_heads=n_heads, drop=drop),
            ISAB(d, m=m, n_heads=n_heads, drop=drop),
        )
        self.pma = PMA(d, k=1, n_heads=n_heads, drop=drop)
    def forward(self, x):
        x = self.block(x)         # (B, N, d)
        z = self.pma(x)           # (B, 1, d)
        return z.squeeze(1)       # (B, d)

# -------- ViT + Static PE (frozen) + Set Transformer head + classifier --------
class ViTWithSetTransformerHead(nn.Module):
    """
    Canonical baseline: Frozen ViT (google/vit-base-patch16-224) provides token embeddings
    with static positional encodings. Patch tokens are pooled by a Set Transformer head,
    then classified.
    """
    def __init__(
        self,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        num_out_classes: int = 10,
        m_inducing: int = 16,
        n_heads: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()
        # Load & freeze ViT backbone
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        d = self.vit.config.hidden_size  # 768 for base
        self.set_head = SetTransformerHead(d=d, m=m_inducing, n_heads=n_heads, drop=drop)
        self.classifier = nn.Linear(d, num_out_classes)

    def __get_vit_peout(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Return ViT embeddings with static positional encodings (includes CLS),
        without updating backbone (frozen).
        """
        # ViT handles patchify + pos-embed inside embeddings()
        return self.vit.embeddings(pixel_values)  # (B, 1+N, d)

    def forward(self, pixel_values: torch.Tensor, train_mode: bool = True):
        with torch.no_grad():
            # (B, 1+N, d) with static PE
            position_encoded = self.__get_vit_peout(pixel_values)
            # Run frozen encoder
            enc = self.vit.encoder(position_encoded, return_dict=True)
            seq = self.vit.layernorm(enc.last_hidden_state)   # (B, 1+N, d)

        # Use **patch tokens only** for set processing (canonical Set Transformer pooling)
        patches = seq[:, 1:, :]   # drop [CLS], shape (B, N, d)

        # Trainable head (Set Transformer) + classifier
        z = self.set_head(patches)         # (B, d)
        logits = self.classifier(z)        # (B, num_out_classes)
        return logits


# ---------- ConViT-style GPSA front-end (trainable), ViT frozen ----------
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

# ---------- ViT (frozen) + ConViT front-end + classifier ----------
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
