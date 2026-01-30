import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

