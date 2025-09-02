import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2 as transforms
from torchvision import datasets
import os

def entropy_vec(x, req_dim = 2):
    """
    x is 3D (bs, seqlen, ftrdim)
    """
    # This is safest and non-Nan or non-Inf entropy
    with torch.no_grad():
        log_probs = torch.nn.functional.log_softmax(x.float(), dim = req_dim) # (bts, seqlen, ftrdim)
        probs = log_probs.exp() # (bts, seqlen, ftrdim)

    # This is entropy
    vector_values = - torch.sum(probs * log_probs, axis = req_dim) # (batch_size, seq_len)

    assert not torch.isnan(vector_values).all()
    assert torch.isfinite(vector_values).all()

    return vector_values # (bs, seqlen)


def get_vector_agg(aggregate_method, x, req_dim, norm_type = None):

    """
    x is (bs, seqlen, ftrdim)
    """

    # Find the maximum vector in each sequence
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

    return vector_values


def safe_pow_gate(x, s, *,  # x: (B,N,D), s∈[0,1]: importance per patch
                  tau=2.0, alpha_lo=0.85, alpha_hi=1.15,
                  eps=1e-3, mix=0.3):
    """Goal of this function is to take importance score per sequence for each patch embedding. And then safely apply those improtance scores to:
        ideally reduce embedding power of tokens that has low importance scores and increase power of embeddings that has higher improtance scores.
        It chooses a safe alpha power and residually mixes original patch embeddings base don improtance scores.

        x is 3D: (bs, seqlen, ftrdim)
        s is 2D: ( bs, seqlen)
    """

    with torch.no_grad():
        # 1) Bound magnitudes between (0-1) so pow>1 always shrinks
        x_b = torch.tanh(x / tau)            # |x_b| ≤ 1

        # 2) Map importance→exponent: low s ⇒ α>1 (shrink), high s ⇒ α<1 (expand)
        # [..., None] = unsqueeze(-1)
        α = alpha_lo + (alpha_hi - alpha_lo) * (1 - s)[..., None]  # (B,N,1)
        α = torch.clamp(α, min=0.75, max=1.25) # Clamp a single alpha for power 

        # 3) Signed, numerically-safe power
        x_pow = x_b.sign() * (x_b.abs() + eps).pow(α)

        # 4) Gentle residual mix to avoid over-suppression
        return (1 - mix) * x + mix * x_pow



def get_anchor_vectors(seq_select_method, vector_values, x, batch_indices):

    # Get indices of maximum vectors in each batch
    if seq_select_method == 'argmax':
        agg_indices = torch.argmax(vector_values, dim=1)  # (batch_size)
        anchor_vectors = x[batch_indices, agg_indices, :].unsqueeze(1)  # (batch_size, 1, feature_dim)

    elif seq_select_method == 'argmin':
        agg_indices = torch.argmin(vector_values, dim = 1)
        anchor_vectors = x[batch_indices, agg_indices, :].unsqueeze(1)  # (batch_size, 1, feature_dim)

    elif seq_select_method == 'weighted_sum':
        anchor_vectors = soft_weighted_sum(weights = vector_values, original_vectors = x, sum_dim = 1) # (batch_size, 1, ftrdim)
    
    elif seq_select_method == 'loo_weighsum':

        # Here at each seqlen index, a weighted sum is computed, leaving that seqlen outside weighted sum
        # so that it doesnt dilute distances when compute with original patch embeddings
        anchor_vectors = loo_weighsum_vector(weights = vector_values, original_vectors = x) # (batch_size, seqlen, ftrdim)
    
    elif seq_select_method == 'safe_pow_gate':
        # In this case, anchor vectors contain both patch embeddings and importance vector values embeed into one matrix.
        anchor_vectors = safe_pow_gate(x = x, s = vector_values) # (bs, seqlen, ftrdim)
    
    else:
        raise ValueError(f"Unsupported sequence selection method: {seq_select_method}.")
    
    return anchor_vectors



def compute_single_patch_phi(anchor_values, x, K, add_coords = False, weights_sequences = None):
    """
    anchor_values can be (bs, seqlen, ftrdim) or (bs, 1, ftrdim). These are weights * original vector values ( patch embeddings in this case).
    x is (bs, seqlen, ftrdim)
    K is no. of sine and cosine terms to add 
    weights_sequences is (bs, seqlen)

    Returns:
    phi - (bs, seqlen, num_ftrs)
    num_ftrs = 3 + 2K when add_coords = False. Else = 7 + 6K
    """
    
    bs, seqlen, _ = x.shape


    # (bs, seqlen, ftrdim)
    # This is safe and best offset for FiLM style injection
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
        # Makes sense to add sin, cos for co-ordinates, but may be not for offset with anchor distances, experiment with others.
        feats_to_add = [torch.sin(sigma * dhat), torch.cos(sigma * dhat)]

        if add_coords:
            for it in [torch.sin(sigma*dx), torch.cos(sigma*dx),torch.sin(sigma*dy), torch.cos(sigma*dy)]:
                feats_to_add.append(it)
        
        phi += feats_to_add
    
    phi = torch.stack(phi, dim = - 1)

    assert torch.isfinite(phi).all()
    assert not torch.isnan(phi).all()
    
    return phi


def soft_weighted_sum(weights, original_vectors, sum_dim = 1):
    """

    Returns weighted sum of vectors, given weights and original vectors to be weighted with, on the dimension provided.

    weights: (batch_size, seq_len)
    original_vectors: (batch_size, seq_len, feature_dim)

    Returns:
        soft_anchor: (bs, 1, ftr_dim), as if a scalar is returned, it won't be smoothed out from LayerNorm.
    """
    
    if len(weights.shape) == 2:
        weights = weights.unsqueeze(-1) # (bs, seqlen, 1)
    
    weights = weights.float()
    original_vectors = original_vectors.float()

    assert torch.isfinite(weights).all()
    assert torch.isfinite(original_vectors).all()

    assert not torch.isnan(weights).all()
    assert not torch.isnan(original_vectors).all()
    
    weighted_values = weights * original_vectors # (bs, seqlen, ftrdim)
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


def loo_weighsum_vector(weights, original_vectors, eps=1e-8):
    """
    weights: (B, T)         # ideally softmax-normalized over T
    vectors: (B, T, D)      # patch tokens (exclude CLS before calling)
    returns: (B, T, D)      # LOO anchors per token
    """
    weights = weights.float()
    original_vectors = original_vectors.float()

    # totals over all tokens
    # This sum ensures all [0-1] probs are added together making 1, suitable for denominator.
    sum_w   = weights.sum(dim=1, keepdim=True)                      # (B,1)
    ws_all  = torch.bmm(weights.unsqueeze(1), original_vectors).squeeze(1)   # (B,D)

    # leave-one-out numerator: subtract each token's own contribution
    contrib = weights.unsqueeze(-1) * original_vectors                       # (B,T,D)
    numer   = ws_all.unsqueeze(1) - contrib                         # (B,T, D)

    # leave-one-out denominator: sum_w - weight_j  (clamped)
    denom   = torch.clamp(sum_w - weights, min=eps).unsqueeze(-1)   # (B,T,1)

    loo = numer / denom                                             # (B,T,D)
    # Optional strict check (debug)
    assert torch.isfinite(loo).all()
    return loo

def get_cifar10_loaders_optimized(data_dir='./data'):

    # CIFAR-10 mean and std for normalization
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

def save_cached_split(ds, path, batch_size=512, num_workers=8, dtype=torch.float16):

    """Saves tesnor data in specified path with the raw torch data set provided."""
    
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

def make_cached_loader(path, batch_size=512, shuffle=True, num_workers=8):

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

def prepare_cached_datasets(cached_data_path):
    """Cached path should end with / """
    
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