import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

"""XCiT in the style of lucidrains """

def exists(x):
    return x is not None

def default(val,d):
    return val if exists(val) else d 

def l2norm(t):
    return F.normalize(t,dim = -1)

class LayerNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeroes(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shapee[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    def forward(self,x): 
                x,gate = x.chunk(2,dim = -1)
                return F.gelu(gate) * x 

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q, k = map(l2norm, (q, k))

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        q = q / (C // self.num_heads) ** 0.5
        k = k / (C // self.num_heads) ** 0.5

        attn = torch.einsum('bhnd,bhkd->bhnk', q, k)
        attn = attn / self.temperature
        attn = attn.softmax(dim=-1)

        if mask is not None:
            attn = attn * mask

        attn = self.attn_drop(attn)

        out = torch.einsum('bhnk,bhkd->bhnd', attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out
