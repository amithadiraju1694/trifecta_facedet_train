import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2 as transforms
from torchvision import datasets
from typing import Optional, Tuple, Union
import os
from helpers_profiling import *


def entropy_vec(x: torch.Tensor, req_dim: int = 2) -> torch.Tensor:
    """
    Function that computes standard shannon entropy of tensor provided. It uses log softmax for stability when 
    computing probabilities. Shape of x will be collapsed on req_dim.

    Args:
        x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
        req_dim: Required -> dimension on which entropy is to be computed. For this usecase it is 2, assuming it's a feature dimension.
    
    Returns:
        vector_values -> Tensor of shape 'x' except in the req_dim. For this usecase it is (bs, seqlen).

    """
    # This is safest and non-Nan or non-Inf entropy
    with torch.no_grad():
        log_probs = torch.nn.functional.log_softmax(x.float(), dim = req_dim) # (bts, seqlen, ftrdim)
        probs = log_probs.exp() # (bts, seqlen, ftrdim)

    # shannon entropy
    vector_values = -torch.sum(probs * log_probs, axis = req_dim) # (batch_size, seq_len)

    assert not torch.isnan(vector_values).all()
    assert torch.isfinite(vector_values).all()

    return vector_values # (bs, seqlen)


def get_vector_agg(aggregate_method: str, x: torch.Tensor, req_dim: int, norm_type = None, smooth_topk = False, topk_val = None) -> torch.Tensor:

    """
    Function that computes some aggregate of a tensor 'x' using different methods (norm, sum, max_elem, entropy) 
    along a required dimension 'req_dim'. Shape of x will be collapsed on req_dim.

    Goal is to get a aggregate information for each sequence/token in patch embeddings space so that it can be used to select 'important sequences/tokens'.

    Args:
        aggregate_method: Required -> method to use for aggregation. One of 'norm', 'sum' , 'max_elem','entropy','norm_softmax'.
        x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
        req_dim: Required -> dimension on which entropy is to be computed. For this usecase it is 2, assuming it's a feature dimension.
        norm_type: Optional -> type of norm to use if aggregate_method is 'norm' or 'norm_softmax'. Default is None, which implies L2 norm.
    
    Returns:
        vector_values -> Tensor of shape same as 'x' except req_dim is removed. For this usecase it is (bs, seqlen).
    """

    if aggregate_method == 'norm':
        vector_values = torch.norm(x, p=norm_type, dim=req_dim)  # (batch_size, seq_len)

        # This is to ensure values don't blow up
        agg_vector_values = ( agg_vector_values - agg_vector_values.max(dim = -1, keepdim = True).values )  / (agg_vector_values.std(dim = -1, keepdim = True) + 1e-6) # (batch_size, seq_len)

    elif aggregate_method == 'sum':
        vector_values = torch.sum(x, dim=req_dim)  # (batch_size, seq_len)

    elif aggregate_method == 'max_elem':
        vector_values, _ = torch.max(x, dim=req_dim)  # (batch_size, seq_len)

    elif aggregate_method == 'entropy':
        vector_values = entropy_vec(x=x, req_dim=req_dim) # (batch_size, seq_len)
    
    elif aggregate_method == 'norm_softmax':
        agg_vector_values = torch.norm(x, p=norm_type, dim=req_dim)  # (batch_size, seq_len)
        
        # This is to ensure values don't blow up
        agg_vector_values = ( agg_vector_values - agg_vector_values.max(dim = -1, keepdim = True).values )  / (agg_vector_values.std(dim = -1, keepdim = True) + 1e-6) # (batch_size, seq_len)
        
        with torch.no_grad():
            vector_values = torch.nn.functional.softmax(agg_vector_values, dim = -1) # (batch_size, seq_len)
    
    else:
        raise ValueError(f"Aggregate method: {aggregate_method} is not supported. Please provide one of : 'norm', 'sum' , 'max_elem','entropy")

    if smooth_topk:
        print("WARNING: Smoothing top-k values in aggregate vector, ensure this is not training phase.")
        if topk_val is None:
            raise ValueError("topk_val must be provided if smooth_topk is True.")
        
        alpha = 0.8 # Higher alpha would favor lowering the effect of topk values aggressively
        
        # Smooth top-k by zeroing out all but top-k values, then re-normalizing
        with torch.no_grad():
            idx = vector_values.topk(topk_val).indices
            vector_values[idx] = (1-alpha) * vector_values[idx] + alpha * ( vector_values[idx]  / topk_val )

    return vector_values


def safe_pow_gate(x: torch.Tensor, s: torch.Tensor, *,  # x: (B,N,D), s∈[0,1]: importance per patch
                  tau: float =2.0, alpha_lo: float=0.85, alpha_hi: float=1.15,
                  eps: float=1e-3, mix: float=0.3) -> torch.Tensor:
    """
    Function that scales values of tensor 'x' ( up or down ) based on importance scores provided in s.
    Since sign of embedding values becomes critical for scaling, this function applies absolute scaling
    and then applies sign post that. If embeddings are passed, ideally value of embeddings must be reduced 
    for low importance scores and increased for high.
    
    It scales using power operation with a specific range power "safe alpha power" and residually mixes 
    original tensor ( patch embeddings in this case) based on importance scores.

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

        assert torch.isfinite(residual_mixed_x).all()
        assert not torch.isnan(residual_mixed_x).all()

        return residual_mixed_x



def get_anchor_vectors(seq_select_method: str, vector_values: torch.Tensor, x: torch.Tensor, batch_indices) -> torch.Tensor:

    """
    Function that computes "anchor vectors" of a tensor 'x' using different methods (max, min, weighted_sum, loo_weightedsum, safe_pow_gate) 
    .Shape of anchor_vectors may be same as x or different, depending on the `seq_select_method` used.

    Goal is to get a representative vector (1 per sequence or one for all sequences/image in the batch) in patch embeddings space; so that it can be used to 
    compute distances with original patch embeddings. These distances can be used to modulate patch embeddings, which might better guide MH-self-attention or 
    other blocks in the transformer.

    Args:
        seq_select_method: Required -> method to use for selecting anchor vectors. One of 'argmax', 'argmin', 'weighted_sum', 'loo_weighsum', 'safe_pow_gate'.
        vector_values: Required -> a tensor with aggregate values computed from patch embeddings using get_vector_agg function. These values could be between 0-1 or unbounded,
        but they must represent some varying scale of importance of each sequence/token in patch embeddings space. Must have at least 2 dimensions, for this usecase it is (bs, seqlen).
        x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
        batch_indices: Required -> dimension on which entropy is to be computed. For this usecase it is 2, assuming it's a feature dimension.
        
    Returns:
        anchor_vectors -> Computed anchor values which can be used to compute distances with patch emeddings or modulate them. Shape depends on `seq_select_method` used.
    """

    if seq_select_method == 'argmax':
        agg_indices = torch.argmax(vector_values, dim=1)  # (batch_size)
        anchor_vectors = x[batch_indices, agg_indices, :].unsqueeze(1)  # (batch_size, 1, feature_dim)

    elif seq_select_method == 'argmin':
        agg_indices = torch.argmin(vector_values, dim = 1)
        anchor_vectors = x[batch_indices, agg_indices, :].unsqueeze(1)  # (batch_size, 1, feature_dim)

    elif seq_select_method == 'weighted_sum':
        anchor_vectors = soft_weighted_sum(weights = vector_values, x = x, sum_dim = 1) # (batch_size, 1, ftrdim)
    
    elif seq_select_method == 'loo_weighsum':

        # Here at each seqlen index, a weighted sum is computed, leaving ith seqlen outside weighted sum
        # so that it doesnt dilute distances when computed with original patch embeddings.
        anchor_vectors = loo_weighsum_vector(weights = vector_values, x = x) # (batch_size, seqlen, ftrdim)
    
    elif seq_select_method == 'safe_pow_gate':
        # In this case, anchor vectors contain both patch embeddings and importance vector values embeded into one matrix.
        anchor_vectors = safe_pow_gate(x = x, s = vector_values) # (bs, seqlen, ftrdim)
    
    else:
        raise ValueError(f"Unsupported sequence selection method: {seq_select_method}.")
    
    return anchor_vectors



def compute_single_patch_phi(anchor_values: torch.Tensor, x: torch.Tensor, K: int, add_coords: bool = False, weights_sequences = None) -> torch.Tensor:
    """
    Function that uses patch embeddings (x) and anchor vectors to compute distances and turn those distances into polynomial features similar to phi features 
    computed in FiLM paper. Similar to FiLM-style, co-ordinates are computed and used to compute distances with anchor vectors in addition to regular polynomial features.

    FiLM paper uses these phi vectors to project into linear space, simialr to hidden size of ViT and use them for injection, but these phi vectors can also be used as-is
    for other purposes.

    Args:
        anchor_vectors: Required -> Anchor values from 'get_anchor_vectors' which can be used to compute distances with patch emeddings or modulate them.
        Shape is either (bs, 1, ftr_dim) or (bs, seqlen, ftr_dim); they provide some aggregated view of global important patches in the patch embedding space.
        x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).
        K: Required -> No. of harmonics to use for computing polynomial features. Each harmonic adds 2 features (sin, cos) per distance type.
        add_coords: Optional -> Whether to compute distances between anchor vectors and co-ordinates of patches as well. Default is False.
        weights_sequences: Optional -> These weights are used to compute weighted sum with co-orodinate grid locations, so that approximate location of "important patch/sequence "
        is found from original image, this is similar to anchor_vectors, but in original imag space rather than patch space. Shape must be (bs, seqlen)

    Returns:
        phi -> Polynomial features computed from distances between Anchor vectors and original patch embeddings, may or maynot included co-ordinate distance features
        depending on `add_coords` flag. Shape is (bs, seqlen, num_ftrs); where num_ftrs = 3 + 2K when add_coords = False. Else = 7 + 6K
    """
    
    bs, seqlen, _ = x.shape


    # (bs, seqlen, ftrdim)
    # This is safe and best offset for FiLM style injection rather than using absolute distances
    delta = 1e-3
    offset = torch.sqrt((x.float()-anchor_values.float()).pow(2) + delta**2) - delta

    # Normalizing distances
    offset_sum = torch.sum( offset, dim = -1 ) + 1e-8
    
    d = torch.sqrt(offset_sum ) # (bs, seqlen)
    dlog = torch.log(1+d+1e-8)
    
    # Further Normalizing distances for stability
    std = torch.std(d, dim = -1, keepdim = True) + 1e-8
    meand = torch.mean(d, dim = -1, keepdim = True)
    dhat = torch.subtract(d, meand) / std # (bs, seqlen)
    
    #TODO: Add Cosine Similarity of (each sequence vector of x, with anchor_values) as additional dimensions in phi
    phi = [d, dhat**2,dlog]

    
    if add_coords:

        if weights_sequences == None: raise ValueError("Weight values must be provided if co-ordinates are to be added to phi vectors")
        if len(weights_sequences.shape) == 2: weights_sequences = weights_sequences.unsqueeze(-1)

        Nseq = seqlen # Total num values or sequences in this batch of images.
        
        # infer grid if not provided. Computing Height and Width of grid. Ex: if Ngrid is 196; h,w should ideally be 14x14
        h = int(Nseq**0.5)
        while Nseq % h != 0: h -= 1
        w = Nseq // h

        # normalized grid coords in [0,1]
        yy, xx = torch.meshgrid(
        torch.arange(h).float(),
        torch.arange(w).float(),
        indexing='ij'
        )

        coordsi = torch.stack([(xx+0.5)/w, (yy+0.5)/h], dim=-1).view(-1, 2)  # (Nseq,2)
        coords = coordsi.unsqueeze(0).expand(bs, -1, -1)  # (B,Nseq,2)

        # each pair of co-ordinates weighted by their importance, reflecting approximate location of important patch in original image.
        offset_pos = torch.sum( coords * weights_sequences, dim = 1, keepdim = True) # (bs, 1, 2)

        r = coords - offset_pos # (bs, seqlen, 2)
        dx, dy = r[..., 0], r[..., 1]
        rad = torch.sqrt(dx*dx + dy*dy + 1e-8)

        for it in [dx, dy, rad, rad**2]:
            phi.append(it)

    for k in range(1, K+1):

        sigma = 2 * torch.pi * k

        # Adds 2*K features
        feats_to_add = [torch.sin(sigma * dhat), torch.cos(sigma * dhat)]

        if add_coords:
            for it in [torch.sin(sigma*dx), torch.cos(sigma*dx),torch.sin(sigma*dy), torch.cos(sigma*dy)]:
                feats_to_add.append(it)
        
        phi += feats_to_add
    
    phi = torch.stack(phi, dim = - 1)

    assert torch.isfinite(phi).all()
    assert not torch.isnan(phi).all()
    
    return phi


def soft_weighted_sum(weights: torch.Tensor, x: torch.Tensor, sum_dim: int = 1) -> torch.Tensor:
    """
    Function that computes "soft anchor" from weights of sequences and original patch embeddings. This uses all sequences to compute "soft anchor" 
    and comes up with a single representative vector for each image in the batch. This is similar to attention mechanism where weights are attention scores
    and patch embeddings are values. Weights should ideally contain improtance information of sequences between 0-1, but can be unbounded as well,
    with some information on importance of sequence.

    Args:

        weights: Required -> A tensor containining some type of importance scores per sequence in a image. Shape must be (batch_size, seq_len)
        x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).

    Returns:
        soft_anchor: Soft Anchor computed from patch embeddings ( uses all sequences per image weighted by importance score ) space per image in the batch. Shape will be (bs, 1, ftr_dim).
    """
    
    if len(weights.shape) == 2:
        weights = weights.unsqueeze(-1) # (bs, seqlen, 1)
    
    weights = weights.float()
    x = x.float()

    assert torch.isfinite(weights).all()
    assert torch.isfinite(x).all()

    assert not torch.isnan(weights).all()
    assert not torch.isnan(x).all()
    
    weighted_values = weights * x # (bs, seqlen, ftrdim)
    soft_anchor = torch.sum(weighted_values, dim = sum_dim, keepdim = True) # (bs, 1, ftrdim)

    return soft_anchor


def loo_weighsum_loop(weights, original_vectors, sum_dim = 1):
    """
    weights: (batch_size, seq_len)
    original_vectors: (batch_size, seq_len, feature_dim)

    Returns:
        loo_soft_anchors: (batch_size, seq_len, feature_dim)
    """
    
    list_soft_anchors = []
    _, seqlen, _ = original_vectors.shape

    for j in range(seqlen):

        alpha_j = weights[:, j]
        denom = 1. - alpha_j # (bs, )
        denom = denom.unsqueeze(-1).unsqueeze(-1) # (bs, 1, 1)

        # Weights leaving current index out
        weights_j = torch.cat( [ weights[:, :j ], weights[:, j+1: ] ] , dim = -1 ) # (bs, seqlen-1)
        weights_j = weights_j.unsqueeze(-1) # (bs, seqlen-1, 1)


        # original vector leaving current index out
        original_vectors_j = torch.cat( [ original_vectors[:, :j, : ],
                                         original_vectors[:, j+1:, :] ] ,
                                        dim = sum_dim 
                                        ) # (bs, seqlen-1, ftrdim)
        

        weighted_values_j = weights_j * original_vectors_j # (bs, seqlen-1, ftrdim)

        weighted_sum_j = torch.sum(weighted_values_j, dim = sum_dim, keepdim=True) # (bs, 1, ftrdim)
        soft_anchor_j = torch.divide(weighted_sum_j, denom) # (bs, 1, ftrdim) if summed on sequences

        list_soft_anchors.append(
            soft_anchor_j
        )
    
    
    loo_soft_anchors = torch.cat(list_soft_anchors, dim = sum_dim) # (bs, seqlen, ftrdim)

    return loo_soft_anchors


def loo_weighsum_vector(weights: torch.Tensor, x: torch.Tensor, eps: float=1e-8):
    """
    
    Function that computes "Leave-One-Out Soft Anchor" from weights of sequences and original patch embeddings. This skips i-th vector when computing
    "soft anchor" and comes up with a vector for each sequence ( 196 different vectors of shape ftr_dim in this usecase ) for each image in the batch.
    This is similar to attention mechanism where weights are attention scores and patch embeddings are values.
    
    Weights should ideally contain improtance information of sequences between 0-1, but can be unbounded as well, with some information on importance of sequence.

    Args:
        weights: Required -> A tensor containining some type of importance scores per sequence in a image. Shape must be (batch_size, seq_len)
        x: Required -> a tensor in patch embedding space, without CLS TOKEN; which must have at least 2 dimensions. For this usecase it is (bs, seqlen, ftrdim).

    Returns:
        loo_anchor: Soft Anchor computed from patch embeddings ( uses all-but-one sequences per image weighted by importance score ) space per image in the batch. Shape will be SAME as x.
    
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
    # Optional strict check (debug)
    assert torch.isfinite(loo_anchor).all()
    return loo_anchor

def get_cifar10_loaders_optimized(data_dir: str='./data') -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    """Function that loads CIFAR10 data set from torchvision and applies transforms including resize to 224x224, normalization and augmentations.
        Normalization uses mean and std computed from CIFAR-10 dataset.
    """

    # CIFAR-10 mean and std for normalization. computed from training set.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms with augmentation and resize to 224x224
    # TODO: Use Deterministic Horizontal Flip from paper, and may be extend to Deterministic Affine Transform as well.
    train_transform = transforms.Compose([
                            transforms.ToImage(),                                       # PIL -> tensor (uint8)
                            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.ToDtype(torch.float32, scale=True),              # now [0,1]
                            transforms.Normalize(mean, std),
                                        ])

    # Test transforms with resize to 224x224
    test_transform = transforms.Compose([
        transforms.ToImage(), # PIL -> Tensor fast path
        transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),  # Resize to ViT's expected input size
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset

def save_cached_split(ds, path: str, batch_size: int=512, num_workers: int=8, dtype=torch.float16) -> None:

    """Saves tensor data in specified path with the raw torch data set provided."""
    
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True)
    Xs, Ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            Xs.append(xb.to(dtype).contiguous().cpu())
            Ys.append(yb.cpu())
    
    X = torch.cat(Xs); Y = torch.cat(Ys)
    torch.save({"images": X, "labels": Y}, path)

def split_pt_file_stratified(src_path,
                             split1_path="validation_ablations.pt",
                             split2_path="test_ablations.pt",
                             split1_frac=0.7,
                             seed=108,
                             save_both_splits = True
                             ):
    
    """ Splits a single tensor file into two tensor files and writes them to disk as tensors.
    It assumes that single tensor file contains features and labels with anmes "images", "labels".

    Useful for splitting validation set of original experiment into val and test for ablations only.
    """

    d = torch.load(src_path, map_location= "cpu")

    X, Y = d["images"], d["labels"]
    assert X.shape[0] == Y.shape[0], "Mismatched X/Y lengths"
    g = torch.Generator().manual_seed(seed)

    split1_idx, split2_idx = [], []
    # Stratified, balanced sampling
    for c in torch.unique(Y).tolist():
        
        idx = torch.where(Y == c)[0]
        perm = idx[torch.randperm(idx.numel(), generator=g)]

        n_val = max(1, min(idx.numel() - 1, int(idx.numel() * split1_frac)))
        
        split1_idx.append(perm[:n_val])
        split2_idx.append(perm[n_val:])

    split1_idx = torch.cat(split1_idx)
    split2_idx = torch.cat(split2_idx)

    # optional: shuffle within splits
    split1_idx = split1_idx[torch.randperm(split1_idx.numel(), generator=g)]
    split2_idx = split2_idx[torch.randperm(split2_idx.numel(), generator=g)]

    split1 = {"images": X[split1_idx].contiguous(), "labels": Y[split1_idx].contiguous()}
    split1_dataset = TensorDataset(split1["images"], split1["labels"])
    

    device_dtype = torch.float32
    if torch.cuda.is_available():
        device_dtype = torch.float16

    save_cached_split(split1_dataset, split1_path, dtype=device_dtype)
    print("Written Split 1 dataset ")
    

    if save_both_splits:
        split2 = {"images": X[split2_idx].contiguous(), "labels": Y[split2_idx].contiguous()}
        split2_dataset = TensorDataset(split2["images"], split2["labels"])
        
        save_cached_split(split2_dataset,   split2_path, dtype = device_dtype)
        print("Written Split 2 dataset ")


def make_cached_loader(path: str, batch_size: int=512, shuffle: bool =True, num_workers:int =8):

    """Function that loads pre-cached tensor data from specified path and returns a dataloader."""

    if not os.path.exists(path):
        raise ValueError(f"Cached data not found in path: {path}. Please run prepare_cached_datasets function first.")
    
    blob = torch.load(path, map_location="cpu")
    
    ds = TensorDataset(blob["images"], blob["labels"])
    
    pin_memory = False; persistent_workers = False
    if torch.cuda.is_available():
        pin_memory = True
        persistent_workers = True

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=persistent_workers)

def prepare_cached_datasets(cached_data_path: str) -> dict:
    """ Function that prepares tensors from cifar10 data set , with specified transforms and stores them in current project space.
    Uses torch.float16 if cuda is available, else torch.float32 on saved tensors.
    Cached path should end with / .

    Args:
        cached_data_path: Required -> Path to store cached tensors. Should end with / .
    Returns:
        data_paths: Dictionary containing paths to train, val and test tensor data.
    """
    
    if os.path.exists(cached_data_path + "train.pt") and os.path.exists(cached_data_path + "val.pt") and os.path.exists(cached_data_path + "test.pt"):
        print("Cached data already exists. Skipping caching step.")
        data_paths = {'train_data': cached_data_path + "train.pt", 'val_data': cached_data_path + "val.pt", 'test_data': cached_data_path + "test.pt"}
        return data_paths
    
    # Get images data which are transformed with affines, resized to 224x224
    train_raw , test_raw = get_cifar10_loaders_optimized(data_dir='./data')
    
    idx = torch.randperm(len(train_raw))
    cut = int(0.7 * len(train_raw))

    # Split train to train and validation sets
    train_ds = torch.utils.data.Subset(train_raw, idx[:cut])
    val_ds   = torch.utils.data.Subset(train_raw, idx[cut:])

    device_dtype = torch.float32
    if torch.cuda.is_available():
        device_dtype = torch.float16

    
    # Save pre-transformed image features into tensors
    os.makedirs(cached_data_path, exist_ok=True)

    save_cached_split(train_ds, cached_data_path + "train.pt", dtype=device_dtype)
    print("Written Train dataset ")
    save_cached_split(val_ds,   cached_data_path + "val.pt", dtype = device_dtype)
    print("written validation dataset")
    save_cached_split(test_raw, cached_data_path + "test.pt", dtype=device_dtype)
    print("written test dataset")

    data_paths = {'train_data': cached_data_path + "train.pt", 'val_data': cached_data_path + "val.pt", 'test_data': cached_data_path + "test.pt"}
    return data_paths


def profile_models(model, example_input, total_tr_rows, batch_size, num_epochs):

    """
    Function is a helper to compute FLOPs and Trainable parameters for overall and by-module of a model.
    """
    
    imp_modules = ['classifier', 'custom_pos_encoding', 'projection_phi', 'peg']
    imp_flop_metrics = ['forward_total_per_sample', 'forward_trainable_per_sample','train_per_sample',
                        'forward_per_step','train_per_step','train_per_epoch',
                        'train_full', 'num_train_samples','batch_size',
                        'num_epochs']

    profile_metrics = {}

    all_ops_model = get_all_inline_ops(model, example_input)
    dict_flops_model = flops_breakdown(model, example_input, all_ops_model, total_tr_rows, batch_size, num_epochs)
    flops_by_mod = dict_flops_model['by_module']

    # Extract FLOPS by module from flops breakdown
    for module_name in imp_modules:
        if module_name in flops_by_mod:
            profile_metrics[module_name] = flops_by_mod[module_name]

    
    # Extract other metrics from flops breakdown
    for module_name in imp_flop_metrics:
        if module_name in dict_flops_model:
            profile_metrics[module_name] = dict_flops_model[module_name]
    
    # compute and extract trainable other metrics from model
    # int, float, dict
    total_tr_params, total_tr_params_mb, tr_params_by_mod = count_parameters(model)
    
    profile_metrics['total_trainable_params'] = total_tr_params
    profile_metrics['total_trainable_params_mb'] = total_tr_params_mb
    profile_metrics['trainable_params_by_mod'] = tr_params_by_mod

    return profile_metrics
