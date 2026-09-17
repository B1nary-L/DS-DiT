import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

#################################################################################
#                           Basic Attention & MLP                               #
#################################################################################

def attention(q, k, v, heads):
    """
    Scaled dot-product attention (no mask needed for dual M2).

    Args:
        q, k, v: [B, N, D] query, key, value
        heads: number of attention heads
    """
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
            bias=bias, dtype=dtype, device=device,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0,
    scaling_factor=None, offset=None,
):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Blocks                               #
#################################################################################

def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        rmsnorm: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x


class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        **block_kwargs,
    ):
        super().__init__()
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device,
            )
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias,
            pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm,
            dtype=dtype, device=device,
        )

        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(
                    hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device,
                )
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(
                    in_features=hidden_size, hidden_features=mlp_hidden_dim,
                    act_layer=nn.GELU(approximate="tanh"), dtype=dtype, device=device,
                )
            else:
                self.mlp = SwiGLUFeedForward(
                    dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256
                )

        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(c).chunk(6, dim=1)
                )
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


#################################################################################
#                         Patch-Level Weighting                                    #
#################################################################################

class PatchLevelWeighting(nn.Module):
    def __init__(self, hidden_size, dtype=None, device=None):
        super().__init__()
        # Gate MLP: 3*hidden_size -> 3*hidden_size//8 -> 3*hidden_size//64 -> 2
        input_dim = hidden_size * 3
        mid1 = input_dim // 8
        mid2 = mid1 // 8
        self.gate = nn.Sequential(
            nn.Linear(input_dim, mid1, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(mid1, mid2, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(mid2, 2, dtype=dtype, device=device),
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.zero_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, lr_out, ref_out, x_out):
        gate_input = torch.cat([lr_out, ref_out, x_out], dim=-1)  # [B, N, 3D]

        logits = self.gate(gate_input)  # [B, N, 2]
        weights = torch.softmax(logits, dim=-1)  # [B, N, 2]

        alpha_lr = weights[..., 0:1]   # [B, N, 1]
        alpha_ref = weights[..., 1:2]  # [B, N, 1]

        fused = alpha_lr * lr_out + alpha_ref * ref_out  # [B, N, D]
        fused_proj = self.zero_proj(fused)  # [B, N, D]

        return fused_proj


class DualJointBlock(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs.pop("pre_only", None)
        qk_norm = kwargs.pop("qk_norm", None)

        hidden_size = args[0]

        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)
        self.lr_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)
        self.ref_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)

        self.weighting = PatchLevelWeighting(hidden_size, dtype=kwargs.get('dtype'), device=kwargs.get('device'))

    def forward(self, x, lr, ref, c, ref_scale=1.0):
        x_qkv, x_inter = self.x_block.pre_attention(x, c)
        lr_qkv, lr_inter = self.lr_block.pre_attention(lr, c)
        ref_qkv, ref_inter = self.ref_block.pre_attention(ref, c)

        n_x = x_qkv[0].shape[1]
        q_lr, k_lr, v_lr = tuple(
            torch.cat([x_qkv[i], lr_qkv[i]], dim=1) for i in range(3)
        )
        attn_lr = attention(q_lr, k_lr, v_lr, self.x_block.attn.num_heads)

        x_attn_lr = attn_lr[:, :n_x]
        lr_attn = attn_lr[:, n_x:]

        q_ref, k_ref, v_ref = tuple(
            torch.cat([x_qkv[i], ref_qkv[i]], dim=1) for i in range(3)
        )
        attn_ref = attention(q_ref, k_ref, v_ref, self.x_block.attn.num_heads)

        x_attn_ref = attn_ref[:, :n_x]
        ref_attn = attn_ref[:, n_x:]

        x_saved, gate_msa, shift_mlp, scale_mlp, gate_mlp = x_inter

        x_attn_lr_proj = self.x_block.attn.post_attention(x_attn_lr)
        x_attn_ref_proj = self.x_block.attn.post_attention(x_attn_ref)

        x = x_saved + gate_msa.unsqueeze(1) * (x_attn_lr_proj + ref_scale * x_attn_ref_proj)

        x = x + gate_mlp.unsqueeze(1) * self.x_block.mlp(
            modulate(self.x_block.norm2(x), shift_mlp, scale_mlp)
        )

        lr = self.lr_block.post_attention(lr_attn, *lr_inter)
        ref = self.ref_block.post_attention(ref_attn, *ref_inter)

        fused_proj = self.weighting(lr, ref, x)
        x = x + fused_proj

        return x, lr, ref


#################################################################################
#                              Final Layer                                      #
#################################################################################

class FinalLayer(nn.Module):
    """The final layer of DiT."""
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        total_out_channels: Optional[int] = None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
            if (total_out_channels is None)
            else nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DSDiT(nn.Module):

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        depth: int = 24,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        register_length: int = 0,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches=None,
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        dtype=None,
        device=None,
        verbose=False,
    ):
        super().__init__()
        if verbose:
            print(f"Initializing DSDiT with: {input_size=}, {patch_size=}, {in_channels=}, {depth=}")

        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size

        # Magic: head_size = 64
        hidden_size = 64 * depth
        num_heads = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Three PatchEmbed embedders: HR, LR, Ref
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True,
            strict_img_size=self.pos_embed_max_size is None,
            dtype=dtype, device=device,
        )
        self.lr_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True,
            strict_img_size=self.pos_embed_max_size is None,
            dtype=dtype, device=device,
        )
        self.ref_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True,
            strict_img_size=self.pos_embed_max_size is None,
            dtype=dtype, device=device,
        )

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype, device=device))

        # Positional embedding
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device),
            )
        else:
            self.pos_embed = None

        self.joint_blocks = nn.ModuleList([
            DualJointBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                rmsnorm=rmsnorm, scale_mod_only=scale_mod_only,
                swiglu=swiglu, qk_norm=qk_norm,
                dtype=dtype, device=device,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)

    def _initialize_from_sd3(self, sd3_state_dict, log_file=None):
        loaded_weights = []
        copied_weights = []

        # Load x_embedder
        self.x_embedder.proj.weight.data.copy_(sd3_state_dict['x_embedder.proj.weight'])
        self.x_embedder.proj.bias.data.copy_(sd3_state_dict['x_embedder.proj.bias'])
        loaded_weights.extend(['x_embedder.proj.weight', 'x_embedder.proj.bias'])

        # Copy lr_embedder and ref_embedder from x_embedder
        self.lr_embedder.proj.weight.data.copy_(self.x_embedder.proj.weight.data)
        self.lr_embedder.proj.bias.data.copy_(self.x_embedder.proj.bias.data)
        self.ref_embedder.proj.weight.data.copy_(self.x_embedder.proj.weight.data)
        self.ref_embedder.proj.bias.data.copy_(self.x_embedder.proj.bias.data)
        copied_weights.extend([
            'lr_embedder <- x_embedder',
            'ref_embedder <- x_embedder'
        ])

        # Load t_embedder
        self.t_embedder.mlp[0].weight.data.copy_(sd3_state_dict['t_embedder.mlp.0.weight'])
        self.t_embedder.mlp[0].bias.data.copy_(sd3_state_dict['t_embedder.mlp.0.bias'])
        self.t_embedder.mlp[2].weight.data.copy_(sd3_state_dict['t_embedder.mlp.2.weight'])
        self.t_embedder.mlp[2].bias.data.copy_(sd3_state_dict['t_embedder.mlp.2.bias'])
        loaded_weights.extend([
            't_embedder.mlp.0.weight', 't_embedder.mlp.0.bias',
            't_embedder.mlp.2.weight', 't_embedder.mlp.2.bias'
        ])

        if 'pos_embed' in sd3_state_dict and self.pos_embed is not None:
            self.pos_embed.data.copy_(sd3_state_dict['pos_embed'])
            loaded_weights.append('pos_embed')

        # Load joint_blocks
        for i, block in enumerate(self.joint_blocks):
            prefix = f'joint_blocks.{i}'

            # Load x_block from SD3
            x_loaded = self._load_dismantled_block(block.x_block, sd3_state_dict, f'{prefix}.x_block')
            loaded_weights.extend([f'{prefix}.x_block.{w}' for w in x_loaded])

            # Initialize lr_block by copying from x_block
            lr_copied = self._init_condition_block_from_x_block(block.lr_block, block.x_block)
            copied_weights.extend([f'{prefix}.lr_block.{w}' for w in lr_copied])

            # Initialize ref_block by copying from x_block
            ref_copied = self._init_condition_block_from_x_block(block.ref_block, block.x_block)
            copied_weights.extend([f'{prefix}.ref_block.{w}' for w in ref_copied])

            copied_weights.append(f'{prefix}.weighting <- zero_init')

        # Load final_layer
        self.final_layer.linear.weight.data.copy_(sd3_state_dict['final_layer.linear.weight'])
        self.final_layer.linear.bias.data.copy_(sd3_state_dict['final_layer.linear.bias'])
        self.final_layer.adaLN_modulation[1].weight.data.copy_(sd3_state_dict['final_layer.adaLN_modulation.1.weight'])
        self.final_layer.adaLN_modulation[1].bias.data.copy_(sd3_state_dict['final_layer.adaLN_modulation.1.bias'])
        loaded_weights.extend([
            'final_layer.linear.weight', 'final_layer.linear.bias',
            'final_layer.adaLN_modulation.1.weight', 'final_layer.adaLN_modulation.1.bias'
        ])

        # Write logs
        if log_file:
            with open(log_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("DSDiT Weight Initialization Log\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"[LOADED FROM SD3] Total: {len(loaded_weights)} weights\n")
                f.write("-" * 80 + "\n")
                for w in loaded_weights:
                    f.write(f"  + {w}\n")

                f.write(f"\n[COPIED/INITIALIZED] Total: {len(copied_weights)} weights\n")
                f.write("-" * 80 + "\n")
                for w in copied_weights:
                    f.write(f"  -> {w}\n")

                f.write(f"\n" + "=" * 80 + "\n")
                f.write(f"Summary: {len(loaded_weights)} loaded + {len(copied_weights)} copied\n")
                f.write("=" * 80 + "\n")

        return loaded_weights, copied_weights

    def _load_dismantled_block(self, block, state_dict, prefix):
        """Load a DismantledBlock from state dict"""
        loaded_weights = []

        # Attention
        block.attn.qkv.weight.data.copy_(state_dict[f'{prefix}.attn.qkv.weight'])
        loaded_weights.append('attn.qkv.weight')
        if block.attn.qkv.bias is not None:
            block.attn.qkv.bias.data.copy_(state_dict[f'{prefix}.attn.qkv.bias'])
            loaded_weights.append('attn.qkv.bias')
        if hasattr(block.attn, 'proj') and block.attn.proj is not None:
            block.attn.proj.weight.data.copy_(state_dict[f'{prefix}.attn.proj.weight'])
            block.attn.proj.bias.data.copy_(state_dict[f'{prefix}.attn.proj.bias'])
            loaded_weights.extend(['attn.proj.weight', 'attn.proj.bias'])

        if f'{prefix}.attn.ln_q.weight' in state_dict:
            if hasattr(block.attn, 'ln_q') and hasattr(block.attn.ln_q, 'weight'):
                block.attn.ln_q.weight.data.copy_(state_dict[f'{prefix}.attn.ln_q.weight'])
                loaded_weights.append('attn.ln_q.weight')
        if f'{prefix}.attn.ln_k.weight' in state_dict:
            if hasattr(block.attn, 'ln_k') and hasattr(block.attn.ln_k, 'weight'):
                block.attn.ln_k.weight.data.copy_(state_dict[f'{prefix}.attn.ln_k.weight'])
                loaded_weights.append('attn.ln_k.weight')

        # MLP
        if hasattr(block, 'mlp') and block.mlp is not None:
            block.mlp.fc1.weight.data.copy_(state_dict[f'{prefix}.mlp.fc1.weight'])
            block.mlp.fc1.bias.data.copy_(state_dict[f'{prefix}.mlp.fc1.bias'])
            block.mlp.fc2.weight.data.copy_(state_dict[f'{prefix}.mlp.fc2.weight'])
            block.mlp.fc2.bias.data.copy_(state_dict[f'{prefix}.mlp.fc2.bias'])
            loaded_weights.extend([
                'mlp.fc1.weight', 'mlp.fc1.bias',
                'mlp.fc2.weight', 'mlp.fc2.bias'
            ])

        # adaLN_modulation
        block.adaLN_modulation[1].weight.data.copy_(state_dict[f'{prefix}.adaLN_modulation.1.weight'])
        block.adaLN_modulation[1].bias.data.copy_(state_dict[f'{prefix}.adaLN_modulation.1.bias'])
        loaded_weights.extend(['adaLN_modulation.1.weight', 'adaLN_modulation.1.bias'])

        return loaded_weights

    def _init_condition_block_from_x_block(self, cond_block, x_block):
        """
        Initialize condition block (lr_block or ref_block) from x_block.

        Strategy:
        - Copy all weights from x_block (QKV, proj, MLP, adaLN)
        - This allows utilizing pretrained features
        """
        copied_weights = []

        # Copy attention QKV
        cond_block.attn.qkv.weight.data.copy_(x_block.attn.qkv.weight.data)
        copied_weights.append('attn.qkv.weight <- x_block')
        if cond_block.attn.qkv.bias is not None:
            cond_block.attn.qkv.bias.data.copy_(x_block.attn.qkv.bias.data)
            copied_weights.append('attn.qkv.bias <- x_block')

        # Copy attention proj
        if hasattr(cond_block.attn, 'proj') and cond_block.attn.proj is not None:
            cond_block.attn.proj.weight.data.copy_(x_block.attn.proj.weight.data)
            cond_block.attn.proj.bias.data.copy_(x_block.attn.proj.bias.data)
            copied_weights.extend(['attn.proj.weight <- x_block', 'attn.proj.bias <- x_block'])

        if hasattr(cond_block.attn, 'ln_q') and hasattr(cond_block.attn.ln_q, 'weight'):
            if hasattr(x_block.attn.ln_q, 'weight'):
                cond_block.attn.ln_q.weight.data.copy_(x_block.attn.ln_q.weight.data)
                copied_weights.append('attn.ln_q.weight <- x_block')
        if hasattr(cond_block.attn, 'ln_k') and hasattr(cond_block.attn.ln_k, 'weight'):
            if hasattr(x_block.attn.ln_k, 'weight'):
                cond_block.attn.ln_k.weight.data.copy_(x_block.attn.ln_k.weight.data)
                copied_weights.append('attn.ln_k.weight <- x_block')

        # Copy MLP
        if hasattr(cond_block, 'mlp') and cond_block.mlp is not None:
            cond_block.mlp.fc1.weight.data.copy_(x_block.mlp.fc1.weight.data)
            cond_block.mlp.fc1.bias.data.copy_(x_block.mlp.fc1.bias.data)
            cond_block.mlp.fc2.weight.data.copy_(x_block.mlp.fc2.weight.data)
            cond_block.mlp.fc2.bias.data.copy_(x_block.mlp.fc2.bias.data)
            copied_weights.extend([
                'mlp.fc1 <- x_block', 'mlp.fc2 <- x_block'
            ])

        cond_out_dim = cond_block.adaLN_modulation[1].weight.shape[0]
        cond_block.adaLN_modulation[1].weight.data.copy_(
            x_block.adaLN_modulation[1].weight.data[:cond_out_dim]
        )
        cond_block.adaLN_modulation[1].bias.data.copy_(
            x_block.adaLN_modulation[1].bias.data[:cond_out_dim]
        )
        copied_weights.extend(['adaLN_modulation <- x_block'])

        return copied_weights

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size and w <= self.pos_embed_max_size
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed, "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size, w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(
        self,
        hr_noisy: torch.Tensor,
        lr: torch.Tensor,
        ref: torch.Tensor,
        t: torch.Tensor,
        ref_scale: float = 1.0,
    ) -> torch.Tensor:

        hw = hr_noisy.shape[-2:]

        x = self.x_embedder(hr_noisy) + self.cropped_pos_embed(hw)
        lr_tokens = self.lr_embedder(lr) + self.cropped_pos_embed(hw)
        ref_tokens = self.ref_embedder(ref) + self.cropped_pos_embed(hw)

        c = self.t_embedder(t, dtype=x.dtype)

        if self.register_length > 0:
            lr_tokens = torch.cat((
                repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                lr_tokens,
            ), 1)

        for block in self.joint_blocks:
            x, lr_tokens, ref_tokens = block(x, lr_tokens, ref_tokens, c=c, ref_scale=ref_scale)

        x = self.final_layer(x, c)
        x = self.unpatchify(x, hw=hw)

        return x