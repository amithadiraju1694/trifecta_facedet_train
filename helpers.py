import torch

def entropy_vec(x, req_dim = 2):
    """
    x is 3D (bs, seqlen, ftrdim)
    """
    # Find max feature among `feature_dim` in each sequence for each image
    max_feat_seq = x.max(axis = req_dim, keepdim = True).values # (batch_size, seq_len, 1)
    min_feat_seq = x.min(axis = req_dim, keepdim = True).values # (batch_size, seq_len, 1)

    # normalizing patch embeddings to be between [0-1], for entropy calculation
    normed_patches = torch.subtract(x, max_feat_seq)
    normed_patches /= (torch.subtract(max_feat_seq, min_feat_seq) + 1e-8) # (batch_size, seqlen, ftr_dim)

    probs = normed_patches / (normed_patches.sum(axis = req_dim, keepdim = True) + 1e-8) # (batch_size, seqlen, ftr_dim)
    vector_values = - torch.sum(probs * torch.log(probs + 1e-8), axis = req_dim) # (batch_size, seq_len)

    return vector_values # (bs, seqlen)


def get_vector_agg(aggregate_method, x, req_dim, norm_type = None):

    """
    x is (bs, seqlen, ftrdim)
    """

    # Find the maximum vector in each sequence
    if aggregate_method == 'norm':
        vector_values = torch.norm(x, p=norm_type, dim=req_dim)  # (batch_size, seq_len)

    elif aggregate_method == 'sum':
        vector_values = torch.sum(x, dim=req_dim)  # (batch_size, seq_len)

    elif aggregate_method == 'max_elem':
        vector_values, _ = torch.max(x, dim=req_dim)  # (batch_size, seq_len)

    elif aggregate_method == 'entropy':
        #TODO: Ensure that it's scaled like pre-softmax norms
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