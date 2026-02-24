import torch
import torch.nn as nn

try:
    import xformers
    import xformers.ops
except ImportError:
    XFORMERS_IS_AVAILBLE = False

from src.models.components.embedding.rotary_emb import apply_rotary_emb


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_lora=False,
        attention_mode="math",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(0)  # makge torchscript happy (cannot use tensor as tuple)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis, head_first=False)
            assert (
                q.shape == q.shape and k.shape == k.shape
            ), f"img_kk: {q.shape}, img_q: {q.shape}, img_kk: {k.shape}, img_k: {k.shape}"

        if self.attention_mode == "xformers":  # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1, 2).contiguous()
            k_xf = k.transpose(1, 2).contiguous()
            v_xf = v.transpose(1, 2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

        elif self.attention_mode == "flash":
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(
                    B, N, C
                )  # require pytorch 2.0

        elif self.attention_mode == "math":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplementedError

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
