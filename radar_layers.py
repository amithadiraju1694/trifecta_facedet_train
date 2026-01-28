import math
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    
    def __get_vector_agg(self,x: torch.Tensor,smooth_topk = False, topk_val = None) -> torch.Tensor:

        """
        Compute some aggregate of a tensor 'x' using different methods (norm, sum, max_elem, entropy) along a required dimension 'req_dim'. Shape of x will be collapsed on req_dim.

        Parameters:
            x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
            smooth_topk: Optional, bool ->
            topk_val: Optional, bool ->
        Returns:
            vector_values -> Tensor of shape same as 'x' except req_dim is removed. For this usecase it is (bs, seqlen).
        """

        if self.aggregate_method in ('norm', 'norm_softmax'):
            p = 2 if self.norm_type is None else self.norm_type
            scores = torch.linalg.vector_norm(x, ord=p, dim=self.aggregate_dim)
            scores = self.__stabilize_scores(scores)
            if self.aggregate_method == 'norm_softmax':
                with torch.no_grad():
                    scores = torch.softmax(scores, dim=-1)
            vector_values = scores
        elif self.aggregate_method == 'sum':
            vector_values = torch.sum(x, dim=self.aggregate_dim)
        elif self.aggregate_method == 'max_elem':
            vector_values = torch.amax(x, dim=self.aggregate_dim)
        elif self.aggregate_method == 'entropy':
            vector_values = self.__entropy_vec(x=x)
        else:
            raise ValueError(
                "Aggregate method must be one of: 'norm', 'norm_softmax', 'sum', 'max_elem', 'entropy'."
            )

        if smooth_topk:
            vector_values = self.__smooth_topk(vector_values, topk_val=topk_val)

        return vector_values

    @staticmethod
    def __stabilize_scores(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # This is to ensure values don't blow up
        scores = scores - scores.amax(dim=-1, keepdim=True)
        denom = scores.std(dim=-1, keepdim=True).clamp_min(eps)
        return scores / denom

    @staticmethod
    def __smooth_topk(scores: torch.Tensor, topk_val: int, eps: float = 1e-8) -> torch.Tensor:
        if topk_val is None:
            raise ValueError("topk_val must be provided if smooth_topk is True.")
        k = min(topk_val, scores.size(-1))
        with torch.no_grad():
            topk_idx = scores.topk(k, dim=-1, largest=True).indices
            mask = torch.zeros_like(scores)
            mask.scatter_(1, topk_idx, 1.0)
            scores = scores * mask
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(eps)
        return scores

    def __get_anchor_vectors(self,vector_values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        """
        Compute "anchor vectors" of a tensor 'x' using different methods (max, min, weighted_sum,
        loo_weightedsum, safe_pow_gate). Depending on 'seq_select_method' resulting anchor shape may change.

        Parameters:
            vector_values: Required -> a tensor with aggregate values computed from patch embeddings.
            x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
        Returns:
            anchor_vectors -> Computed anchor values which can be used to compute distances with patch emeddings or modulate.
        """

        if self.seq_select_method == 'argmax':
            batch_indices = torch.arange(x.size(0), device=x.device)
            agg_indices = torch.argmax(vector_values, dim=1)  # (batch_size)
            anchor_vectors = x[batch_indices, agg_indices, :].unsqueeze(1)  # (batch_size, 1, feature_dim)

        elif self.seq_select_method == 'argmin':
            batch_indices = torch.arange(x.size(0), device=x.device)
            agg_indices = torch.argmin(vector_values, dim = 1)
            anchor_vectors = x[batch_indices, agg_indices, :].unsqueeze(1)  # (batch_size, 1, feature_dim)

        elif self.seq_select_method == 'weighted_sum':
            anchor_vectors = self.__soft_weighted_sum(weights = vector_values, x = x, sum_dim = 1) # (batch_size, 1, ftrdim)
        
        elif self.seq_select_method == 'loo_weighsum':

            # Here at each seqlen index, a weighted sum is computed, leaving ith seqlen outside weighted sum
            # so that it doesnt dilute distances when computed with original patch embeddings.
            anchor_vectors = self.__loo_weighsum_vector(weights = vector_values, x = x) # (batch_size, seqlen, ftrdim)
        
        elif self.seq_select_method == 'safe_pow_gate':
            # In this case, anchor vectors contain both patch embeddings and importance vector values embeded into one matrix.
            anchor_vectors = self.__safe_pow_gate(x = x, s = vector_values) # (bs, seqlen, ftrdim)
        
        else:
            raise ValueError(f"Unsupported sequence selection method: {self.seq_select_method}.")
        
        return anchor_vectors

    def __compute_distances(self, x: torch.Tensor, anchor_vectors: torch.Tensor):
        if self.distance_metric == 'euclidean':
            # TODO: May need to change this to element wise difference and keep difference as a vector
            return torch.linalg.vector_norm(x - anchor_vectors, ord=2, dim=-1, keepdim=True)
        if self.distance_metric == 'cosine':
            cosine_sim = F.cosine_similarity(x, anchor_vectors, dim=-1, eps=1e-8)
            return (1 - cosine_sim).unsqueeze(-1)
        if self.distance_metric == 'manhattan':
            return (x - anchor_vectors).abs().sum(dim=-1, keepdim=True)
        if self.distance_metric == 'per_feat_manhattan':
            return (x - anchor_vectors).abs()  # (bs, seqlen, feature_dim)
        if self.distance_metric == 'NA':
            # User will compute custom distances with anchor values found, hence distances are None
            return None
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            Positional encodings as distances from max vector (batch_size, seq_len, 1)
        """
        # Do grading/softmax in fp32 for stability (especially in fp16), then cast outputs back.
        x_fp32 = x if x.dtype == torch.float32 else x.float()

        # Aggregate vectors of original image/ patch embeddings. ex: l2 norm, entropy-based, softmax, max-min etc
        # (bs, seqlen)
        vector_values = self.__get_vector_agg(x=x_fp32,
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
        anchor_vectors = self.__get_anchor_vectors(vector_values = vector_values,
                                                 x = x_fp32
                                                 )
        
        # Compute distances from the maximum vector
        distances = self.__compute_distances(x_fp32, anchor_vectors)

        vector_values = vector_values.to(x.dtype)
        anchor_vectors = anchor_vectors.to(x.dtype)
        if distances is not None:
            distances = distances.to(x.dtype)

        if self.return_anchors:
            return(distances, anchor_vectors, vector_values)
        
        return distances
    
    def __entropy_vec(self,x: torch.Tensor, scale_entropy = True) -> torch.Tensor:
        """
        Computes standard shannon entropy of tensor provided. It uses log softmax for stability when  computing probabilities.

        Parameters:
            x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
            req_dim: Required -> dimension on which entropy is to be computed. For this usecase it is 2, assuming it's a feature dimension.
        
        Returns:
            vector_values -> Tensor of shape 'x' except in the req_dim. For this usecase it is (bs, seqlen).

        """
        # This is safest and non-Nan or non-Inf entropy
        with torch.no_grad():
            log_probs = torch.nn.functional.log_softmax(x.float(), dim=self.aggregate_dim)
            probs = log_probs.exp()

        # shannon entropy
        vector_values = -torch.sum(probs * log_probs, dim=self.aggregate_dim)

        if scale_entropy:
            # scaling entropy as H_j / log(K), where K = D, dimension we took entropy on
            denom_scaling = torch.log(torch.tensor(x.shape[self.aggregate_dim], device=x.device))
            vector_values = vector_values / (denom_scaling + 1e-8)

        return vector_values # (bs, seqlen)

    @staticmethod
    def __soft_weighted_sum(weights: torch.Tensor, x: torch.Tensor, sum_dim: int = 1) -> torch.Tensor:
        """
        Compute "soft anchor" from weights of sequences and original patch embeddings. Soft "weighted sum" of tensor and weights.

        Parameters:
            weights: Required -> A tensor containining some type of importance scores per sequence.
            x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions.

        Returns:
            soft_anchor: Soft Anchor computed from patch embeddings.
        """
        
        if len(weights.shape) == 2:
            weights = weights.unsqueeze(-1) # (bs, seqlen, 1)
        
        weights = weights.float()
        x = x.float()
        return torch.sum(weights * x, dim=sum_dim, keepdim=True)

    @staticmethod
    def __loo_weighsum_vector(weights: torch.Tensor, x: torch.Tensor, eps: float=1e-8):
        
        """
        Computes "Leave-One-Out Soft Anchor" from weights of sequences and original patch embeddings. This skips i-th vector when computing
        "soft anchor" and comes up with a vector for each sequence ( 196 different vectors of shape ftr_dim in this usecase ).

        Args:
            weights: Required -> A tensor containining some type of importance scores per sequence in a image. Shape must be (batch_size, seq_len)
            x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions.

        Returns:
            loo_anchor: Soft Anchor computed from patch embeddings ( uses all-but-one sequences weighted by importance score ).
        
        """
        weights = weights.float()
        x = x.float()

        # totals over all tokens
        # This sum ensures all [0-1] probs are added together making 1, suitable for denominator.
        sum_w   = weights.sum(dim=1, keepdim=True)                      # (B,1)
        ws_all  = torch.bmm(weights.unsqueeze(1), x).squeeze(1)   # (B,D)

        # leave-one-out numerator: subtract each token's own contribution
        contrib = weights.unsqueeze(-1) * x                       # (B,T,D)
        numer   = ws_all.unsqueeze(1) - contrib                         # (B,T, D)

        # leave-one-out denominator: sum_w - weight_j  (clamped)
        denom   = torch.clamp(sum_w - weights, min=eps).unsqueeze(-1)   # (B,T,1)

        loo_anchor = numer / denom                                             # (B,T,D)
        return loo_anchor

    @staticmethod
    def __safe_pow_gate(x: torch.Tensor,
                      s: torch.Tensor, *,  # x: (B,N,D), s∈[0,1]: importance per patch
                      tau: float =2.0,
                      alpha_lo: float=0.85,
                      alpha_hi: float=1.15,
                      eps: float=1e-3, mix: float=0.3
                      ) -> torch.Tensor:
        """
        Scales values of tensor 'x' ( up or down ) based on importance scores provided in s. Applies sign-aware absolute scaling. If embeddings are passed, ideally value of embeddings must be reduced for low importance scores and increased for high.
        

        Args:
            x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
            s: Required -> importance scores for each sequence / token in patch embeddings space in range [0,1]. Shape must be same as 'x' except in last dimension. For this usecase it is (bs, seqlen).
            tau: Optional -> temperature for tanh scaling. Default is 2.0.
            alpha_lo: Optional -> lower bound of power to use for scaling. Default is 0.85.
            alpha_hi: Optional -> upper bound of power to use for scaling. Default is 1.15.
            eps: Optional -> small value to avoid numerical issues. Default is 1e-3.
            mix: Optional -> mixing factor for residual connection with original tensor. Default is 0.3.
        
        Returns:
            residual_mixed_x -> Tensor of same shape as 'x' after scaling and residual mixing. For this usecase it is (bs, seqlen, ftrdim).
        """

        # No Grad for speed boost
        with torch.no_grad():
            # Bound magnitudes between (0-1) so pow>1 always shrinks
            x_b = torch.tanh(x / tau)            # |x_b| ≤ 1

            # Map importance→exponent: low s ⇒ α>1 (shrink), high s ⇒ α<1 (expand)
            # [..., None] = unsqueeze(-1)
            α = alpha_lo + (alpha_hi - alpha_lo) * (1 - s)[..., None]  # (B,N,1)
            α = torch.clamp(α, min=0.75, max=1.25) # Clamp a single alpha for power 

            # Signed, numerically-safe power
            x_pow = x_b.sign() * (x_b.abs() + eps).pow(α)

            # Gentle residual mix to avoid over-suppression
            residual_mixed_x = (1 - mix) * x + mix * x_pow
            return residual_mixed_x


class RADAR(nn.Module):
    """
    RADAR module:
      1) uses AggregateSequenceGrading on ViT patch embeddings to get (anchor_values, weights)
      2) computes phi offsets with anchor values and weights
      3) projects phi to s and b
      4) returns modulated patch embeddings: vit_pe * (1 + alpha * s) + gamma * b (with CLS re-attached) FiLM-style
    """
    
    def __init__(
        self,
        token_hid_dim: int,
        K: int = 3,
        add_coordinates: bool = False,
        distance_metric: str = "euclidean",
        aggregate_method: str = "norm_softmax",
        seq_select_method: str = "weighted_sum",
        aggregate_dim: int = 2,
        norm_type: int = 2,
        return_anchors: bool = True,
        corrupt_imp_weights: bool = False,
    ):
        super().__init__()
        self.token_hid_dim = token_hid_dim

        self.K = K
        self.add_coordinates = add_coordinates

        # aggregator to get anchors and weights
        self.aggregator = AggregateSequenceGrading(
            distance_metric=distance_metric,
            aggregate_method=aggregate_method,
            seq_select_method=seq_select_method,
            aggregate_dim=aggregate_dim,
            norm_type=norm_type,
            return_anchors=return_anchors,
            corrupt_imp_weights=corrupt_imp_weights,
        )

        # phi feature size (same logic as in Custom_VIT)
        if self.add_coordinates:
            phi_out_features = 7 + 6 * self.K
        else:
            phi_out_features = 3 + 2 * self.K

        # projection to produce both s and b (concatenated)
        self.projection_phi = nn.Sequential(
            nn.Linear(phi_out_features, 128),
            nn.Linear(128, self.token_hid_dim * 2)
        )

        # FiLM-like scale & shift params
        self.alpha = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self._coords_cache = {}

    def __compute_single_patch_phi(self,
                                anchor_values: torch.Tensor,
                                 x: torch.Tensor,
                                 weights_sequences = None) -> torch.Tensor:
        """
        Computes RADAR phi offsets using vectorized norms, cached coordinates, and
        numerically stable statistics.

         anchor_vectors: Required -> Anchor values from 'get_anchor_vectors' which can be used to compute distances with patch emeddings or modulate them.
            Shape is either (bs, 1, ftr_dim) or (bs, seqlen, ftr_dim); they provide some aggregated view of global important patches in the patch embedding space.
        x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
            K: Required -> No. of harmonics to use for computing polynomial features. Each harmonic adds 2 features (sin, cos) per distance type.
            add_coords: Optional -> Whether to compute distances between anchor vectors and co-ordinates of patches as well. Default is False.
            weights_sequences: Optional -> These weights are used to compute weighted sum with co-orodinate grid locations, so that approximate location of "important patch/sequence "
            is found from original image, this is similar to anchor_vectors, but in original imag space rather than patch space. Shape must be (bs, seqlen)

        """
    
        bs, seqlen, _ = x.shape

        # stable offset sum computation
        delta = 1e-3
        diff = x.float() - anchor_values.float()
        offset = torch.sqrt( diff.pow(2) + delta ** 2) - delta
        offset_sum = offset.sum(dim=-1).clamp_min(1e-8)


        d = torch.sqrt(offset_sum)
        dlog = torch.log1p(d)

        # std and mean come from single call reducing FLOPs
        std, mean = torch.std_mean(d, dim=-1, keepdim=True)
        dhat = (d - mean) / (std + 1e-8)
        
        #TODO: Add Cosine Similarity of (each sequence vector of x, with anchor_values) as additional dimensions in phi
        phi = [d, dhat.pow(2), dlog]

        dx = dy = rad = None
        if self.add_coordinates:
            if weights_sequences is None:
                raise ValueError("Weight values must be provided if co-ordinates are required.")
            if weights_sequences.ndim == 2:
                weights_sequences = weights_sequences.unsqueeze(-1)

            # Extract or compute grid, saves FLOPs if extracted, torch.meshgrid+arange is slow
            # normalized grid coords in [0,1]
            coords = self._get_coordinate_grid(seqlen, x.device)
            coords = coords.unsqueeze(0).expand(bs, -1, -1)

            weights_sequences = weights_sequences.to(device=x.device, dtype=coords.dtype)
            offset_pos = torch.sum(coords * weights_sequences, dim=1, keepdim=True)

            r = coords - offset_pos
            dx, dy = r.unbind(dim=-1)
            rad = torch.hypot(dx, dy).clamp_min(1e-8)
            phi.extend([dx, dy, rad, rad.pow(2)])

        for k in range(1, self.K + 1):
            sigma = 2 * math.pi * k
            sin_dhat = torch.sin(sigma * dhat)
            cos_dhat = torch.cos(sigma * dhat)
            phi.extend([sin_dhat, cos_dhat])

            if self.add_coordinates:
                phi.extend(
                    [
                        torch.sin(sigma * dx),
                        torch.cos(sigma * dx),
                        torch.sin(sigma * dy),
                        torch.cos(sigma * dy),
                    ]
                )
        
        phi = torch.stack(phi, dim = - 1)

        return phi

    def _get_coordinate_grid(self, seqlen: int, device: torch.device) -> torch.Tensor:

        """
        Computes normalized grid co-ordinates betwenn [0,1] and caches them in dictionary to be reused
        """
        coords = self._coords_cache.get(seqlen)
        if coords is None:
            h = int(math.sqrt(seqlen))
            h = max(h, 1)
            while seqlen % h != 0 and h > 1:
                h -= 1
            w = seqlen // h
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing='ij'
            )
            coordsi = torch.stack([(xx + 0.5) / w, (yy + 0.5) / h], dim=-1).reshape(-1, 2)
            coords = coordsi
            self._coords_cache[seqlen] = coords
        return coords.to(device)

    def forward(self, token_patch_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_patch_emb: (B, SEQLEN, HIDDIM)
        Returns:
            position-modulated tokens
        """

        # 1) aggregate -> (_, anchor_values, weights)
        _, anchor_values, weights = self.aggregator(token_patch_emb)

        # 2) compute phi offsets
        phi_offset = self.__compute_single_patch_phi(
            anchor_values=anchor_values,
            x=token_patch_emb,
            weights_sequences=weights
        )  # (B, N, phi_out_features)

        # 3) project to s and b
        proj = self.projection_phi(phi_offset)  # (B, N, 2*D)
        s = proj[:, :, :self.token_hid_dim]
        b = proj[:, :, self.token_hid_dim:]

        # 4) scale & shift
        modulated_patch_embeddings = token_patch_emb * (1.0 + self.alpha * s) + self.gamma * b  # (B, N, D)

        return  modulated_patch_embeddings


class PFIM(nn.Module):

    def __init__(self, 
                 distance_metric: str = "euclidean",
                 aggregate_method: str = "norm_softmax",
                 seq_select_method: str = "weighted_sum",
                 aggregate_dim: int = 2,
                 norm_type: int = 2,
                 return_anchors: bool = True,
                 corrupt_imp_weights: bool = False
                 ):
        super().__init__()
        self.return_anchors = return_anchors
        
        # aggregator to get anchors and weights
        self.aggregator = AggregateSequenceGrading(
            distance_metric=distance_metric,
            aggregate_method=aggregate_method,
            seq_select_method=seq_select_method,
            aggregate_dim=aggregate_dim,
            norm_type=norm_type,
            return_anchors=return_anchors,
            corrupt_imp_weights=corrupt_imp_weights,
        )

    def forward(self, token_patch_emb: torch.Tensor):
        

        if self.return_anchors:
            # Returns tuple if return_anchors=True, else just one
            _, scaled_patch_embeddings, _ = self.aggregator(token_patch_emb) # (bs, seqlen, ftrdim)
        else:
            scaled_patch_embeddings = self.aggregator(token_patch_emb) # (bs, seqlen, ftrdim)
        
        return scaled_patch_embeddings
