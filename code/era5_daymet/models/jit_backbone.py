#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
jit_backbone.py — JiT: 像素空间扩散 Transformer (x-prediction 去噪主体)
============================================================================
移植自 JiT (Li & He, arXiv:2511.13720) 官方实现, 改动限于三点:

  1. 矩形网格: 原实现假定方形 (unpatchify 的 h=w、RoPE/APE 单轴复用), 此处 patch 网格
     (gh, gw) 全程分开, 支持 720x1440 一类整幅输入;
  2. 条件方式: 类别 embedding / in-context class token / CFG 全部移除, 改为把条件场与
     噪声目标按通道 concat 后进同一个 patch 嵌入; adaLN 的条件向量只含时间嵌入;
  3. MoE 挂点: 按 JiTMoE (arXiv:2512.01252) 的口径, 可把奇数索引块(第 0 块恒稠密)的
     FFN 换成 moe_ffn.DSMoE。

保留的原实现细节(数值口径, 勿随手"优化"):
  - patch 嵌入走 bottleneck: p x p 卷积(无 bias)先压到低维再 1x1 升到 hidden;
  - 位置编码两者并用: 固定 2D sincos 加性编码 + 每层 q/k 的 2D 轴向 RoPE;
  - 注意力: q/k 逐 head RMSNorm; QK^T 与 softmax 固定 fp32(局部关闭 autocast),
    概率矩阵乘 V 回到环境精度;
  - FFN 为 SwiGLU, 稠密块中间宽度取 int(4*hidden*2/3); RMSNorm(eps=1e-6) 全程;
  - adaLN-zero 逐块调制(零初始化), 输出层线性零初始化 -> 初始 x 预测恒为 0;
  - dropout 只作用于中段块 (depth//4 <= i < 3*depth//4)。
============================================================================
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from era5_daymet.models.moe_ffn import DSMoE
from era5_daymet.models.vit import get_2d_sincos_pos_embed


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dt)


class TimestepEmbedder(nn.Module):
    """标量 t -> 正弦频率向量 -> MLP。t 取值 [0,1](t=1 为数据端)。"""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32,
                                                          device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


def rotate_half(x):
    """相邻两维成对旋转: (x1, x2) -> (-x2, x1)。"""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


class Rope2D(nn.Module):
    """2D 轴向 RoPE(矩形网格): head_dim 的前一半编码行相位、后一半编码列相位。
    频率取语言模型惯用的 theta=10000 幂律。缓冲区不入 checkpoint(按尺寸重建)。"""

    def __init__(self, head_dim, gh, gw, theta=10000.0):
        super().__init__()
        assert head_dim % 4 == 0, f"2D RoPE 需 head_dim%4==0, 得到 {head_dim}"
        d4 = head_dim // 4
        freqs = 1.0 / (theta ** (torch.arange(0, d4).float() / d4))
        fh = torch.einsum("i,j->ij", torch.arange(gh).float(), freqs)
        fw = torch.einsum("i,j->ij", torch.arange(gw).float(), freqs)
        fh = fh.repeat_interleave(2, dim=-1)                     # (gh, head_dim/2)
        fw = fw.repeat_interleave(2, dim=-1)                     # (gw, head_dim/2)
        full = torch.cat([fh[:, None, :].expand(gh, gw, -1),
                          fw[None, :, :].expand(gh, gw, -1)], dim=-1)
        self.register_buffer("cos", full.cos().reshape(gh * gw, head_dim),
                             persistent=False)
        self.register_buffer("sin", full.sin().reshape(gh * gw, head_dim),
                             persistent=False)

    def forward(self, x):                                        # x: (B, heads, N, head_dim)
        return x * self.cos + rotate_half(x) * self.sin


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        hd = dim // num_heads
        self.q_norm = RMSNorm(hd) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(hd) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        # QK^T 与 softmax 固定 fp32; 概率矩阵乘 V 交还环境精度(autocast 下自动回 bf16)
        with torch.autocast(device_type=x.device.type, enabled=False):
            aw = q.float() @ k.float().transpose(-2, -1) * (q.shape[-1] ** -0.5)
        aw = torch.softmax(aw, dim=-1)
        aw = F.dropout(aw, self.attn_drop, training=self.training)
        x = (aw @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    """稠密块的 SwiGLU: 中间宽度先乘 2/3(与门控支路合计持平常规 4x FFN 参数量)。"""

    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class JiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
                 moe_config=None):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        if moe_config is not None:
            self.mlp = DSMoE(
                num_experts=moe_config["num_experts"], dim=dim,
                moe_inter=moe_config["moe_intermediate_size"],
                num_experts_per_tok=moe_config["num_experts_per_tok"],
                n_group=moe_config["n_group"], topk_group=moe_config["topk_group"],
                routed_scaling_factor=moe_config["routed_scaling_factor"],
                use_shared_expert=moe_config["use_shared_expert"],
                proj_drop=moe_config["proj_drop"])
        else:
            self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, rope):
        sa, ca, ga, sm, cm, gm = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + ga.unsqueeze(1) * self.attn(modulate(self.norm1(x), sa, ca), rope)
        x = x + gm.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sm, cm))
        return x


class BottleneckPatchEmbed(nn.Module):
    """patch 嵌入的低秩重参数化: p x p 卷积(无 bias)压到 bottleneck 维, 1x1 升到 hidden。"""

    def __init__(self, hw, patch, in_ch, bottleneck, hidden, bias=True):
        super().__init__()
        H, W = hw
        assert H % patch == 0 and W % patch == 0, f"{hw} 不可被 patch={patch} 整除"
        self.gh, self.gw = H // patch, W // patch
        self.proj1 = nn.Conv2d(in_ch, bottleneck, patch, stride=patch, bias=False)
        self.proj2 = nn.Conv2d(bottleneck, hidden, 1, bias=bias)

    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)   # (B, gh*gw, hidden)


class FinalLayer(nn.Module):
    def __init__(self, hidden, patch, out_ch):
        super().__init__()
        self.norm_final = RMSNorm(hidden)
        self.linear = nn.Linear(hidden, patch * patch * out_ch)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class JiT(nn.Module):
    """条件式 JiT 去噪器: net(z, t, cond) -> x 预测。

    z    : (B, out_ch, H, W) 当前噪声化目标
    t    : (B,) 时间, t=1 为数据端
    cond : (B, cond_ch, H, W) 条件场, 与 z 按通道 concat 进 patch 嵌入
    moe_config: None 为全稠密; dict 时按 interleave 把 FFN 换成 DSMoE
                (True: 奇数索引块; False: 全部块)。
    """

    def __init__(self, hw=(720, 1440), patch=16, cond_ch=20, out_ch=1,
                 hidden=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 bottleneck=128, attn_drop=0.0, proj_drop=0.0, moe_config=None):
        super().__init__()
        self.hw, self.patch, self.out_ch = tuple(hw), patch, out_ch
        self.hidden, self.depth = hidden, depth
        self.t_embedder = TimestepEmbedder(hidden)
        self.x_embedder = BottleneckPatchEmbed(hw, patch, cond_ch + out_ch,
                                               bottleneck, hidden)
        gh, gw = self.x_embedder.gh, self.x_embedder.gw
        self.register_buffer("pos_embed",
                             get_2d_sincos_pos_embed(hidden, gh, gw).float(),
                             persistent=False)
        self.rope = Rope2D(hidden // num_heads, gh, gw)
        if moe_config is not None:
            use_moe = [(i % 2 == 1) if moe_config.get("interleave", True) else True
                       for i in range(depth)]
        else:
            use_moe = [False] * depth
        mid = lambda i: (depth // 4 * 3 > i >= depth // 4)       # dropout 只在中段块
        self.blocks = nn.ModuleList([
            JiTBlock(hidden, num_heads, mlp_ratio,
                     attn_drop=attn_drop if mid(i) else 0.0,
                     proj_drop=proj_drop if mid(i) else 0.0,
                     moe_config=moe_config if use_moe[i] else None)
            for i in range(depth)])
        self.final_layer = FinalLayer(hidden, patch, out_ch)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)                       # 注: TopkRouter.weight 是裸 Parameter,
        # 不在此覆盖, 保持其构造时的 kaiming 初始化
        for w in (self.x_embedder.proj1.weight, self.x_embedder.proj2.weight):
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """(B, gh*gw, p*p*C) -> (B, C, gh*p, gw*p), 矩形网格。"""
        B = x.shape[0]
        gh, gw, p, c = self.x_embedder.gh, self.x_embedder.gw, self.patch, self.out_ch
        x = x.reshape(B, gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(B, c, gh * p, gw * p)

    def forward(self, z, t, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.x_embedder(x)
        x = x + self.pos_embed.to(x.dtype)
        c = self.t_embedder(t)
        for blk in self.blocks:
            x = blk(x, c, self.rope)
        return self.unpatchify(self.final_layer(x, c))

    def moe_layers(self):
        return [m for m in self.modules() if isinstance(m, DSMoE)]

    def param_counts(self):
        """总参数与"激活参数"(路由专家按 K/E 折算)。"""
        total = sum(p.numel() for p in self.parameters())
        routed = act_routed = 0
        for m in self.moe_layers():
            r = sum(p.numel() for p in m.experts.parameters())
            routed += r
            act_routed += r * m.top_k // m.n_experts
        return {"total": total, "routed_experts": routed,
                "activated": total - routed + act_routed}
