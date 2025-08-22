import torch
import torch.nn as nn
from transformers import ViTModel
from typing import Optional,Union

class AggregateSequenceGrading(nn.Module):
    """
    Custom positional encoding based on distances from the maximum vector in a sequence.
    """

    def __init__(self,
                 distance_metric='euclidean',
                 aggregate_method='norm',
                 seq_select_method = 'weighted_avg'
                 ):
        """
        Initialize the custom positional encoding module.

        Args:
            distance_metric: Method to compute distances ('euclidean', 'cosine', 'manhattan')
            aggregate_method: Method to determine the "maximum" vector ('norm', 'sum', 'max_elem')
        """
        super(AggregateSequenceGrading, self).__init__()
        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            Positional encodings as distances from max vector (batch_size, seq_len, 1)
        """
        batch_size = x.shape[0]

        # Find the maximum vector in each sequence
        if self.aggregate_method == 'norm':
            vector_values = torch.norm(x, p=2, dim=2)  # (batch_size, seq_len)
        elif self.aggregate_method == 'sum':
            vector_values = torch.sum(x, dim=2)  # (batch_size, seq_len)
        elif self.aggregate_method == 'max_elem':
            vector_values, _ = torch.max(x, dim=2)  # (batch_size, seq_len)
        elif self.aggregate_method == 'entropy':
            
            # Find max feature among `feature_dim` in each sequence for each image
            max_feat_seq = x.max(axis = 2, keepdim = True).values # (batch_size, seq_len, 1)
            min_feat_seq = x.min(axis = 2, keepdim = True).values # (batch_size, seq_len, 1)

            # normalizing patch embeddings to be between [0-1], for entropy calculation
            normed_patches = torch.subtract(x, max_feat_seq)
            normed_patches /= (torch.subtract(max_feat_seq, min_feat_seq) + 1e-8) # (batch_size, seqlen, ftr_dim)

            probs = normed_patches / (normed_patches.sum(axis = 2, keepdim = True) + 1e-8) # (batch_size, seqlen, ftr_dim)
            vector_values = - torch.sum(probs * torch.log(probs + 1e-8), axis = 2) # (batch_size, seq_len)

        # TODO: create a argument to class called 'seq_select_method', whose values are either max or min
        # Get indices of maximum vectors in each batch
        max_indices = torch.argmax(vector_values, dim=1)  # (batch_size)

        # Extract the maximum vectors
        batch_indices = torch.arange(batch_size, device=x.device)
        max_vectors = x[batch_indices, max_indices, :].unsqueeze(1)  # (batch_size, 1, feature_dim)

        # Compute distances from the maximum vector
        if self.distance_metric == 'euclidean':

            # TODO: May need to change this to element wise difference
            #  and keep difference asa vector
            distances = torch.norm(x - max_vectors, p=2, dim=2, keepdim=True)
        
        elif self.distance_metric == 'cosine':
        
            x_norm = torch.norm(x, p=2, dim=2, keepdim=True) + 1e-8
            max_norm = torch.norm(max_vectors, p=2, dim=2, keepdim=True) + 1e-8

            x_normalized = x / x_norm
            max_normalized = max_vectors / max_norm

            cosine_sim = torch.sum(x_normalized * max_normalized, dim=2, keepdim=True)
            distances = 1 - cosine_sim
        
        elif self.distance_metric == 'manhattan':
            distances = torch.sum(torch.abs(x - max_vectors), dim=2, keepdim=True)
        
        elif self.distance_metric == 'per_feat_manhattan':
            distances = torch.abs(x - max_vectors) # (bs, seqlen, feature_dim)

        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")


        return distances


class DecompSequenceGrading(nn.Module):
    """
        Custom positional encoding based on Eigen value decomposition of sequences
    """

    def __init__(self,
                 decomp_algo: str = 'qr',
                 decomp_strategy: Optional[str] = "project", # project or "importance"
                 top_k_sequences: Optional[int] = 64,
                 alpha: Union[float, torch.Tensor] = 0.5,
                 ):
        """
        Initialize the custom positional encoding module.

        Args:
            decomp_algo: Type of decomposition to use ('qr', 'svd')
            top_k_sequences: Number of top sequences to keep after decomposition
            keep_all_decomposed: If True, keep all decomposed vectors, otherwise only top_k.
            This normalizes importance of each sequence based on the decomposition and multiplies
            this normalized importance with the original sequence, effectively embedding which sequences are more relevant.
        """
        
        super(DecompSequenceGrading, self).__init__()
        self.top_k_sequences = top_k_sequences

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

        #TODO: Think of a normalization strategy after 'projection' through either of methods.
        batch_size, seq_len, feature_dim = x.shape
        identity_stability = torch.eye(seq_len) * self.epsilon

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


        # CHECK IF THE DECOMPOSED VECTORS returned any NaNs in them, if so;
        # fall back to default decomposed vectors
        if self.decomp_algo == 'eig':
        
            x_mean = x.mean(dim = 2, keepdim=True) # (bts, seq_len, 1)
            x_centered = x - x_mean # (bts, seq_len, ftr_dim)
            x_covar = x_centered @ x_centered.transpose(1,2) # (bts, seqlen, seqlen)

            # mag_vals - (bts, seqlen), decomp_vectors - (bts, seqlen, seqlen)
            magnitude_values, decomp_vectors = torch.linalg.eigh(x_covar + identity_stability.unsqueeze(0))

        # Choose an appropriate decomposition strategy
        if self.decomp_strategy == 'project':
            xt = x.permute(0,2,1) # (bts, ftr_dim, seqlen)
            decomposed_vectors = xt @ decomp_vectors
            decomposed_vectors = decomposed_vectors.permute(0,2,1) # (bts, seqlen, ftr_dim)

        if self.decomp_strategy == 'importance':

            # normalizing magnitude values to reflect importances from 0 - 1
            scaled_magnitude_norms = magnitude_values / torch.sum(magnitude_values,
                                                                  dim = 1,
                                                                  keepdim = True) # (bts, seqlen)

            # residual blending of importance scores with original vectors
            # decomposed_vectors = x + self.alpha * (scaled_magnitude_norms.unsqueeze(2) * x ) # (bts, seqlen, ftr_dim)
            
            # completely shut down certain sequences
            decomposed_vectors = scaled_magnitude_norms.unsqueeze(2) * x # (bts, seqlen, ftr_dim)

        return decomposed_vectors


class ViTWithAggPositionalEncoding(nn.Module):
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 distance_metric='euclidean',
                 aggregate_method='norm',
                 alpha: Union[float, torch.Tensor] = 0.5,
                 num_out_classes = 10
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method

        # This is to check how much of custom positional encoding to be added to the original positional encoding
        self.alpha = nn.Parameter(torch.Tensor([alpha]))

        # norm is L2 Norm
        self.custom_pos_encoding = AggregateSequenceGrading(
            distance_metric=self.distance_metric,
            aggregate_method=self.aggregate_method
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

        # (batch_size, seq_len+1, hidden_size)
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        # Calculate custom positional encoding on sequence with class token
        # (bts, seq_len, 1) or (bts, seq_len, feature_dim)
        custom_pos_encodings = self.custom_pos_encoding(patch_emb_output)

        # Expand custom positional encodings if needed
        if custom_pos_encodings.size(1) != ViT_stat_pos_emb.size(1):
            custom_pos_encodings = torch.cat((cls_token, custom_pos_encodings), dim=1)  # (batch_size, seq_len+1, 1)

        # Add both positional encodings
        pos_encoded = patch_emb_output * ( self.alpha * custom_pos_encodings ) + ViT_stat_pos_emb # (batch_size, seq_len+1, hidden_size)

        # Continue with the encoder
        encoder_outputs = self.vit.encoder(pos_encoded) # It outputs only one item

        sequence_output = encoder_outputs[0] # (batchsize,seq_len,hidden_size)

        # Use [CLS] token for classification
        cls_token = sequence_output[:, 0]  # (batch_size, hidden_size)

        logits = self.classifier(cls_token) # (batch_size, num_out_classes)

        return logits


class ViTWithAggPositionalEncoding_PF(nn.Module):
    def __init__(self,
                 pretrained_model_name="google/vit-base-patch16-224",
                 distance_metric='euclidean',
                 aggregate_method='norm',
                 alpha: Union[float, torch.Tensor] = 0.5,
                 num_out_classes = 10,
                 use_both = False
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.distance_metric = distance_metric
        self.aggregate_method = aggregate_method

        # This is to check how much of custom positional encoding to be added to the original positional encoding
        self.alpha = alpha

        self.use_both = use_both

        # norm is L2 Norm
        self.custom_pos_encoding = AggregateSequenceGrading(
            distance_metric=self.distance_metric,
            aggregate_method=self.aggregate_method
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

        # (batch_size, seq_len+1, hidden_size)
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        # Calculate custom positional encoding on sequence with class token
        # (bts, seq_len, 1) or (bts, seq_len, feature_dim)
        custom_pos_encodings = self.custom_pos_encoding(patch_emb_output)

        # Expand custom positional encodings if needed
        if custom_pos_encodings.size(1) != ViT_stat_pos_emb.size(1):
            custom_pos_encodings = torch.cat((cls_token, custom_pos_encodings), dim=1)  # (batch_size, seq_len+1, 1)

        # Add both positional encodings
        if self.use_both:
            pos_encoded = patch_emb_output + custom_pos_encodings + ViT_stat_pos_emb
        else:
            pos_encoded = patch_emb_output + self.alpha * custom_pos_encodings + (1-self.alpha) * ViT_stat_pos_emb # (batch_size, seq_len+1, hidden_size)

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
                 num_out_classes = 10,
                 alpha = 0.85
                 ):
        super().__init__()

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.decomp_algo = decomp_algo
        self.decomp_strategy = decomp_strategy
        self.alpha = alpha

        # norm is L2 Norm
        self.custom_pos_encoding = DecompSequenceGrading(
            decomp_algo = self.decomp_algo,
            decomp_strategy = self.decomp_strategy,  # project or "importance"
            alpha = alpha
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

        # (batch_size, seq_len+1, hidden_size)
        ViT_stat_pos_emb, patch_emb_output, cls_token = self.__get_vit_peout(pixel_values)

        # Calculate custom positional encoding on sequence with class token
        # (bts, seq_len, feature_dim)
        custom_pos_encodings = self.custom_pos_encoding(patch_emb_output)

        # Expand custom positional encodings if needed
        if custom_pos_encodings.size(1) != ViT_stat_pos_emb.size(1):
            custom_pos_encodings = torch.cat((cls_token, custom_pos_encodings), dim=1)  # (batch_size, seq_len+1, 1)

        # Add both positional encodings
        if self.decomp_strategy == 'project':
            pos_encoded = custom_pos_encodings + ViT_stat_pos_emb # (batch_size, seq_len+1, hidden_size)
        
        else:
            pos_encoded = patch_emb_output * ( self.alpha * custom_pos_encodings ) + ViT_stat_pos_emb

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