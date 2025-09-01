# Copyright 2025 SIE

"""
Custom ViT cell implementation for Autoplay in Pytorch.

Contact:
    | Pierluigi.Vito.Amadori@sony.com
    | Timothy.Bradley@Sony.com
    | Ryan.Spick@Sony.com
    | Guy.Moss@sony.com
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
import math


def apply_rope(x, theta):
    """Applies Rotary Positional Encoding (RoPE) to queries and keys"""
    seq_len, batch, num_heads, head_dim = x.shape  # Ensure correct dimensions

    theta = theta[: head_dim // 2]

    cos_theta = torch.cos(
        theta * torch.arange(seq_len, device=x.device).float().unsqueeze(-1)
    )
    sin_theta = torch.sin(
        theta * torch.arange(seq_len, device=x.device).float().unsqueeze(-1)
    )

    x_reshaped = x.contiguous().view(seq_len, batch, num_heads, -1, 2)

    x_real, x_imag = x_reshaped[..., 0], x_reshaped[..., 1]

    cos_theta = cos_theta.unsqueeze(1).unsqueeze(2)  # (seq_len, 1, 1, head_dim//2)
    sin_theta = sin_theta.unsqueeze(1).unsqueeze(2)

    x_rotated = torch.stack(
        [
            x_real * cos_theta - x_imag * sin_theta,
            x_real * sin_theta + x_imag * cos_theta,
        ],
        dim=-1,
    ).reshape(seq_len, batch, num_heads, head_dim)

    return x_rotated


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GatedFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, x = self.fc1(x).chunk(2, dim=-1)
        x = F.gelu(gate) * x
        return self.dropout(self.fc2(x))


class AttentionBlock(nn.Module):
    def __init__(
        self, embed_dim, hidden_dim, num_heads, dropout=0.1, drop_path_prob=0.05
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        # values from RoPE paper
        theta = 10000 ** (
            -2 * (torch.arange(self.head_dim // 2).float() / self.head_dim)
        )
        self.register_buffer("theta", theta)

        self.ffn = GatedFFN(embed_dim, hidden_dim, dropout)

        # might want to remove drop_path for shallower models, and switch
        # to dropout. Left it in as it prevents overfitting and we're trying to
        # generalise
        self.drop_path = DropPath(drop_prob=drop_path_prob)

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape  # N=seq_len, c=embed_dim

        # pre-norm, deliberate choice
        x_ln = self.layer_norm_1(x)

        qkv = (
            self.qkv(x_ln)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = apply_rope(q.transpose(0, 2), self.theta).transpose(0, 2)
        k = apply_rope(k.transpose(0, 2), self.theta).transpose(0, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)

        x = x + self.drop_path(self.out(attn_output))

        x_ln_ffn = self.layer_norm_2(x)
        x = x + self.drop_path(self.ffn(x_ln_ffn))

        return x


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        # broadcast to batch dim
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.attn(query, x, x)
        return attn_output.squeeze(1)


class SimpleViT(nn.Module):
    def __init__(
        self,
        output_size=128,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        patch_size=16,
        input_size=384,
        num_channels=3,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            num_channels, output_size, kernel_size=patch_size, stride=patch_size
        )
        n_patches = (input_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, output_size))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                AttentionBlock(output_size, hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(output_size)
        self.attn_pooling = AttentionPooling(output_size, num_heads)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embedding
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.attn_pooling(
            x
        )  # Apply attention-based pooling instead of mean pooling
        return x
