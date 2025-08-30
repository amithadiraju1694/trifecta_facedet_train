import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from transformers import ViTModel
from typing import Optional,Union

from helpers import (
    get_vector_agg,
    get_anchor_vectors,
    compute_single_patch_phi
)

class AggregateSequenceGrading(nn.Module):
    """
    Custom positional encoding based on distances from the maximum vector in a sequence.
    """

    def __init__(self,
                 distance_metric='euclidean',
                 aggregate_method='norm',
                 seq_select_method = 'weighted_sum',
                 aggregate_dim = 2,
                 norm_type = None,
                 return_anchors = True
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
                       norm_type = self.norm_type) # (batch_size, seq_len)
        
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
        Custom positional encoding based on Eigen value decomposition of sequences
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


# BELOW THIS IS FIXED
class ViTWithAggPositionalEncoding_PF(nn.Module):
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 distance_metric='euclidean',
                 aggregate_method='norm',
                 seq_select_method = 'weighted_sum',
                 aggregate_dim = 2,
                 norm_type = 2,
                 return_anchors = False,
                 alpha: Union[float, torch.Tensor] = 0.75,
                 num_out_classes = 10,
                 use_both = False
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method
        self.seq_select_method= seq_select_method
        self.aggregate_dim=aggregate_dim
        self.norm_type = norm_type,
        self.return_anchors = return_anchors

        # This is to check how much of custom positional encoding to be added to the original positional encoding
        # This is not a trainable parameter
        self.alpha = torch.tensor( [alpha] , dtype = torch.float32)

        self.use_both = use_both

        # norm is L2 Norm
        self.custom_pos_encoding = AggregateSequenceGrading(
            distance_metric=self.distance_metric,
            aggregate_method=self.aggregate_method, # l2- Norm, entropy
            seq_select_method = self.seq_select_method,
            aggregate_dim = self.aggregate_dim,
            norm_type = self.norm_type,
            return_anchors = self.return_anchors
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

    def forward(self, pixel_values):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        # patch_emb_output -> (batch_size, seq_len, hidden_size)
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        # Calculate custom positional encoding on sequence
        # This already contains patch embedding output included in it
        if self.return_anchors:
            # Returns tuple if return_anchors=True, else just one
            _, custom_pos_encodings, _ = self.custom_pos_encoding(patch_emb_output) # (bs, seqlen, ftrdim)
        else:
            custom_pos_encodings = self.custom_pos_encoding(patch_emb_output) # (bs, seqlen, ftrdim)
        
        
        custom_pos_encodings = torch.cat([cls_token, custom_pos_encodings], dim = 1)
        

        # Add both positional encodings
        if self.use_both:
            pos_encoded = custom_pos_encodings + ViT_stat_pos_emb

        else:
            pos_encoded = custom_pos_encodings + (self.alpha * ViT_stat_pos_emb) # (batch_size, seq_len, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(pos_encoded) # It outputs only one item

        sequence_output = encoder_outputs[0] # (batchsize,seq_len,hidden_size)

        # Use [CLS] token for classification
        cls_token = sequence_output[:, 0]  # (batch_size, hidden_size)

        logits = self.classifier(cls_token) # (batch_size, num_out_classes)

        return logits


# ViT which has custom positional encoding with injection on pre-LN
# anchor vector is still a single patch, projected to linear space before injection
class ViTWithAggPositionalEncoding_SP(nn.Module):
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
                 return_anchors = True
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method
        self.seq_select_method = seq_select_method
        self.add_coordinates = add_coordinates
        self.K = K
        self.aggregate_dim = aggregate_dim
        self.norm_type = norm_type
        self.return_anchors = return_anchors
        self.add_coordinates = add_coordinates # Whether to get co-ordinates output when dealing with phi vector computation

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
            return_anchors=self.return_anchors
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

    def forward(self, pixel_values):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        # patch_emb_out - > (batch_size, seq_len, hidden_size)
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        # Calculate custom positional encoding on sequence with class token
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

        # TODO: Not touching CLS Token for now. Need better approach, for finding sj,bj for CLS token
        # Below equation injection style is directly inspired from FiLM or CVPR style papers
        patch_emb_output = patch_emb_output * (1 + self.alpha * s) + self.gamma * b
        patch_emb_output = torch.cat([cls_token, patch_emb_output], dim = 1)
        pos_encoded = patch_emb_output + ViT_stat_pos_emb


        # Continue with the encoder
        encoder_outputs = self.vit.encoder(pos_encoded) # It outputs only one item
        sequence_output = encoder_outputs[0] # (batchsize,seq_len,hidden_size)

        # Use [CLS] token for classification
        cls_token = sequence_output[:, 0]  # (batch_size, hidden_size)
        logits = self.classifier(cls_token) # (batch_size, num_out_classes)

        return logits


class ViTWithAggPositionalEncoding_RandNoise(nn.Module):
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 num_out_classes = 10
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False

        # These are learnable params that govern how much % of s,b will be added to patch embeddings before sending to encoder.
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype = torch.float32))  # Trainable scalar parameter
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype = torch.float32))  # Trainable scalar parameter

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

    def forward(self, pixel_values):

        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """

        # patch_emb_out - > (batch_size, seq_len, hidden_size)
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        bs, seqlen, ftrdim = patch_emb_output.shape

        # This is best way for cmparing against custom approaches, per-batch random s and b
        # but requires multiple runs
        s = torch.randn((bs, seqlen, ftrdim))
        b = torch.randn((bs, seqlen, ftrdim))

        #TODO: These s and b needs to be modified to 1,0 in testing and random during train
        if torch.cuda.is_available():
            s = s.to('cuda', non_blocking = True)
            b = b.to('cuda', non_blocking = True)

        # TODO: Not touching CLS Token for now. Need better approach, for finding sj,bj for CLS token
        # Below equation injection style is directly inspired from FiLM or CVPR style papers
        patch_emb_output = patch_emb_output * (1 + self.alpha * s) + self.gamma * b
        patch_emb_output = torch.cat([cls_token, patch_emb_output], dim = 1)
        pos_encoded = patch_emb_output + ViT_stat_pos_emb


        # Continue with the encoder
        encoder_outputs = self.vit.encoder(pos_encoded) # It outputs only one item
        sequence_output = encoder_outputs[0] # (batchsize,seq_len,hidden_size)

        # Use [CLS] token for classification
        cls_token = sequence_output[:, 0]  # (batch_size, hidden_size)
        logits = self.classifier(cls_token) # (batch_size, num_out_classes)

        return logits


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

    def forward(self, pixel_values):

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

        # Use [CLS] token for classification
        cls_token = sequence_output[:, 0]  # (batch_size, hidden_size)

        logits = self.classifier(cls_token) # (batch_size, num_out_classes)

        return logits


class ViTWithStaticPositionalEncoding(nn.Module):
    def __init__(self,
                    pretrained_model_name="google/vit-base-patch16-224",
                    num_out_classes=10):
        super().__init__()
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Ensuring ViT layers are frozen
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Classifier layer for the output
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_out_classes)

    def __get_vit_peout(self, pixel_values):

        # Get patch embeddings - (bts, seq_len, hidden_size)
        patch_emb_output = self.vit.embeddings.patch_embeddings(pixel_values)

        # Add class token
        cls_token = self.vit.embeddings.cls_token.expand(
            patch_emb_output.shape[0], -1, -1)
        
        # (batch_size, seq_len+1, hidden_size)
        patch_emb_output = torch.cat((cls_token, patch_emb_output), dim=1)

        # Apply dropout
        patch_emb_output = self.vit.embeddings.dropout(patch_emb_output)  # (batch_size, seq_len+1, hidden_size)

        # Get static positional embeddings
        ViT_stat_pos_emb = self.vit.embeddings.position_embeddings  # (1, seq_len+1, hidden_size)

        return (ViT_stat_pos_emb, patch_emb_output, cls_token)

    def forward(self, pixel_values):
        """
        pixel_values is a 4D image tensor of shape ( batch_size, num_channels, height, width )
        """
        # (batch_size, seq_len+1, hidden_size), BASIC STATIS POS ENC FROM ViT
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        # Add static positional embeddings
        pos_encoded = patch_emb_output + ViT_stat_pos_emb  # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(pos_encoded)  # It outputs only one item

        sequence_output = encoder_outputs[0]  # (batchsize, seq_len, hidden_size)

        # Use [CLS] token for classification, since it attends to all future tokens
        cls_token = sequence_output[:, 0]  # (batch_size, hidden_size)

        logits = self.classifier(cls_token)  # (batch_size, num_out_classes)

        return logits