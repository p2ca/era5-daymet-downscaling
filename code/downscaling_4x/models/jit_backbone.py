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
     FFN 换成 moe_ffn.DSMoE; 路由为 D-EC 时主体额外持有难度头(DifficultyHead), 并把
     token 域掩膜与难度先验沿块透传给路由(见 moe_ffn 模块开头)。

可选的两条流结构(arch 配置, 每一项独立开关, 全关时与上述单流模型逐字节相同):
  - 分流: 细流 = 噪声目标 + 静态场 + 年内相位(+ 其余非 ERA5 通道), 粗流 = ERA5 动态通道
    (+ 年内相位), 两条流同一切块网格、同一随机起点, 各自一套 patch 嵌入;
  - 粗流自处理(CoarseStream): token 网格上的 3x3 卷积 + 若干个"自注意力 + SwiGLU"块, 不接 t,
    一条采样轨迹只算一次(precompute), 结果作为每层交叉注意力的 K/V 与加工站的第二张卡;
  - 交叉注意力(CrossAttention): Q 来自细 token, K/V 来自粗 token; ERA5 无数据的粗 token 不参与;
    K 的位置可按该 token 的日均风推移 τ 小时(Lagrangian), 每个注意力头各自的风层与 τ;
  - 位置单位: grid = 行列序号; km = 以一个南北 token 间距(111.2 km)为单位, 东西向乘 cos(纬度),
    风程 u·τ 换算成同一单位后才能加到位置上;
  - 加工站读两张卡: 细 token 与正上方粗 token 各投影后拼接, 作为稠密 FFN / 路由专家的输入;
  - 地形键(TerrainKey): 该块静态场单独嵌成的小向量, 进路由的专家分; 科室键(粗 token 投影)进科室分。
  数据侧常量(风通道的均值/标准差、ERA5 有效掩膜)由 set_data_constants 写入持久 buffer,
  随 checkpoint 保存, 采样侧从 checkpoint 直接恢复。

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from downscaling_4x.models.moe_ffn import DSMoE
from downscaling_4x import contract as C
from downscaling_4x.contract import DEFAULT_MODE, cond_channels, cond_layout
from downscaling_4x.models.pos_embed import get_2d_sincos_pos_embed


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
    频率取语言模型惯用的 theta=10000 幂律。缓冲区不入 checkpoint(按尺寸重建)。

    缺省按行列序号预生成 cos/sin 表; tables() 可按任意坐标(如公里制、或按风推移后的
    位置)现算同一套频率下的表, 供 forward 显式传入。"""

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
        self.register_buffer("freqs", freqs, persistent=False)

    def tables(self, y, x):
        """按坐标现算 (cos, sin): y, x 形状 (..., N), 单位与行列序号同尺度(1 = 一个 token 间距),
        返回各 (..., N, head_dim)。行相位用 y, 列相位用 x, 与预生成表同一套频率。"""
        fh = (y.float()[..., None] * self.freqs).repeat_interleave(2, dim=-1)
        fw = (x.float()[..., None] * self.freqs).repeat_interleave(2, dim=-1)
        full = torch.cat([fh, fw], dim=-1)
        return full.cos(), full.sin()

    def forward(self, x, tabs=None):                             # x: (B, heads, N, head_dim)
        if tabs is None:
            return x * self.cos + rotate_half(x) * self.sin
        cos, sin = tabs
        return x * cos + rotate_half(x) * sin


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

    def forward(self, x, rope, tabs=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = rope(self.q_norm(q), tabs)
        k = rope(self.k_norm(k), tabs)
        # QK^T 与 softmax 固定 fp32; 概率矩阵乘 V 交还环境精度(autocast 下自动回 bf16)
        with torch.autocast(device_type=x.device.type, enabled=False):
            aw = q.float() @ k.float().transpose(-2, -1) * (q.shape[-1] ** -0.5)
        aw = torch.softmax(aw, dim=-1)
        aw = F.dropout(aw, self.attn_drop, training=self.training)
        x = (aw @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    """稠密块的 SwiGLU: 中间宽度先乘 2/3(与门控支路合计持平常规 4x FFN 参数量)。
    in_dim 缺省等于 dim; 加工站读两张卡时输入宽度由拼接后的卡决定, 输出仍是 dim。"""

    def __init__(self, dim, hidden_dim, drop=0.0, bias=True, in_dim=None):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(in_dim or dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RefineHead(nn.Module):
    """输出端全分辨率卷积精修: x̂ ← x̂ + Head(concat[x̂, 静态通道切片])。

    末层零初始化 ⇒ 初始为恒等映射, 打开开关的瞬间训练动力学连续, 也不破坏扩散版
    "输出层零初始化、初始 x 预测恒为 0"的既有契约。静态通道(Δz/高程/地表/海陆)直连
    输出端, 供地形锁定纹理的全分辨率渲染 —— token 线性解码是全网唯一的逐块合成瓶颈,
    本模块是绕开它的最后一公里。刻意不喂完整条件场, 压缩捷径容量以防主干惰化。
    """

    def __init__(self, out_ch, n_static, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(out_ch + n_static, width, 3, padding=1)
        self.n1 = nn.GroupNorm(8, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.n2 = nn.GroupNorm(8, width)
        self.conv3 = nn.Conv2d(width, out_ch, 3, padding=1)
        nn.init.constant_(self.conv3.weight, 0)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, xhat, statics):
        h = torch.cat([xhat, statics.to(xhat.dtype)], dim=1)
        h = F.silu(self.n1(self.conv1(h)))
        h = F.silu(self.n2(self.conv2(h)))
        return xhat + self.conv3(h)


class CrossAttention(nn.Module):
    """细 token 向粗 token 的问询: Q 来自细 token, K/V 来自粗 token。

    q_tabs / k_tabs 是 Rope2D.tables 给出的 (cos, sin): Q 用细 token 自己的位置, K 用粗 token 的
    位置或按风推移后的位置, k_tabs 可逐头不同(形状 (B, heads, N, head_dim))。key_mask (B, Nk)
    为 False 的粗 token 匹配分置 -inf, 不参与作答。数值口径与 Attention 相同: QK^T 与 softmax
    固定 fp32, 概率矩阵乘 V 回到环境精度。"""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        hd = dim // num_heads
        self.q_norm = RMSNorm(hd) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(hd) if qk_norm else nn.Identity()
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xkv, rope, q_tabs=None, k_tabs=None, key_mask=None):
        B, Nq, C = xq.shape
        Nk = xkv.shape[1]
        h, hd = self.num_heads, C // self.num_heads
        q = self.q(xq).reshape(B, Nq, h, hd).permute(0, 2, 1, 3)
        kv = self.kv(xkv).reshape(B, Nk, 2, h, hd).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = rope(self.q_norm(q), q_tabs)
        k = rope(self.k_norm(k), k_tabs)
        with torch.autocast(device_type=xq.device.type, enabled=False):
            aw = q.float() @ k.float().transpose(-2, -1) * (hd ** -0.5)
            if key_mask is not None:
                aw = aw.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        aw = torch.softmax(aw, dim=-1)
        aw = F.dropout(aw, self.attn_drop, training=self.training)
        x = (aw @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.proj_drop(self.proj(x))


class PlainBlock(nn.Module):
    """粗流的块: pre-norm 自注意力 + SwiGLU, 无 adaLN(粗流只描述干净的条件场, 不看 t)。"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, rope, tabs=None):
        x = x + self.attn(self.norm1(x), rope, tabs)
        return x + self.mlp(self.norm2(x))


class CoarseStream(nn.Module):
    """粗流: ERA5 条件通道自己的 patch 嵌入 -> token 网格上的 3x3 卷积 -> 若干 PlainBlock -> RMSNorm。

    卷积做邻格差分(梯度、散度、涡度一类), 块里的 MLP 做格内乘积(相对湿度、平流、水汽通量一类),
    自注意力取远处形势。输出的粗 token 供交叉注意力当 K/V, 也是加工站的第二张卡。"""

    def __init__(self, grid_hw, patch, in_ch, bottleneck, hidden, num_heads, mlp_ratio=4.0,
                 conv=True, blocks=3, margin=0):
        super().__init__()
        self.embed = BottleneckPatchEmbed(grid_hw, patch, in_ch, bottleneck, hidden, margin=margin)
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1) if conv else None
        self.blocks = nn.ModuleList([PlainBlock(hidden, num_heads, mlp_ratio) for _ in range(int(blocks))])
        self.norm_out = RMSNorm(hidden, eps=1e-6)

    def forward(self, xin, pos_embed, rope, tabs=None):
        x = self.embed(xin) + pos_embed.to(xin.dtype)
        if self.conv is not None:
            B, N, C = x.shape
            g = x.transpose(1, 2).reshape(B, C, self.embed.gh, self.embed.gw)
            x = x + F.silu(self.conv(g)).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x, rope, tabs)
        return self.norm_out(x)


class TerrainKey(nn.Module):
    """地形键: 每个 token 的静态场(Δz / 高程 / 地表覆盖 / 掩膜)单独嵌成一个小向量, 恒定不随天变。
    读取窗口 window >= patch, 步长仍为 patch, 与 patch 嵌入同一种"读多写少"的几何; 反射补边。"""

    def __init__(self, patch, n_static, dim=64, window=20):
        super().__init__()
        window = int(window)
        assert window >= patch and (window - patch) % 2 == 0, f"地形键窗口 {window} 须 >= patch 且同奇偶"
        margin = (window - patch) // 2
        self.conv = nn.Conv2d(n_static, dim, window, stride=patch, padding=margin,
                              padding_mode="reflect" if margin > 0 else "zeros")
        self.norm = RMSNorm(dim)

    def forward(self, statics):                    # (B, n_static, Hp, Wp) -> (B, N, dim)
        return self.norm(self.conv(statics).flatten(2).transpose(1, 2))


class StreamContext:
    """一次前向里沿块透传的两条流上下文(precompute 的产物)。"""

    __slots__ = ("q_tabs", "k_tabs", "xc", "coarse_valid", "terrain_key", "offset", "pos")

    def __init__(self, q_tabs=None, k_tabs=None, xc=None, coarse_valid=None, terrain_key=None, offset=(0, 0)):
        self.q_tabs, self.k_tabs, self.xc = q_tabs, k_tabs, xc
        self.coarse_valid, self.terrain_key, self.offset = coarse_valid, terrain_key, offset
        self.pos = None


class JiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
                 moe_config=None, cross=False, cross_modulate=True, two_card=None):
        super().__init__()
        # two_card: None(单流) 或 {"dims": (a, b), "moe": bool, "dense": bool, "ward_key": bool};
        # a、b 是细 token 与粗 token 投到加工站入口前的宽度, 拼接后是 FFN / 专家的输入宽度
        tc = dict(two_card or {})
        card_dim = (int(tc["dims"][0]) + int(tc["dims"][1])) if tc else None
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
                proj_drop=moe_config["proj_drop"],
                router_mode=moe_config.get("router_mode", "tc"),
                gating_mode=moe_config.get("gating_mode", "norm"),
                dec_config=moe_config.get("dec"),
                card_dim=card_dim if tc.get("moe") else None,
                ward_key_dim=(int(tc["dims"][1]) if tc.get("ward_key") else 0),
                terrain_key_dim=int(moe_config.get("terrain_key_dim", 0) or 0),
                drop_outside=bool(moe_config.get("drop_outside", False)))
        else:
            self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio), drop=proj_drop,
                                 in_dim=(card_dim if tc.get("dense") else None))
        self.cross = None
        self.cross_modulate = bool(cross_modulate)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, (9 if cross else 6) * dim))
        if cross:
            self.norm_x = RMSNorm(dim, eps=1e-6)
            self.cross = CrossAttention(dim, num_heads, qkv_bias=True, qk_norm=True,
                                        attn_drop=attn_drop, proj_drop=proj_drop)
        is_moe = isinstance(self.mlp, DSMoE)
        self.card_moe = bool(tc.get("moe")) and is_moe
        self.card_dense = bool(tc.get("dense")) and not is_moe
        self.ward_key = bool(tc.get("ward_key")) and is_moe
        self.card_f = self.card_c = None
        if self.card_moe or self.card_dense:
            self.card_f = nn.Linear(dim, int(tc["dims"][0]))
            self.card_c = nn.Linear(dim, int(tc["dims"][1]))
        elif self.ward_key:
            self.card_c = nn.Linear(dim, int(tc["dims"][1]))

    def forward(self, x, c, rope, tok_mask=None, prior=None, ctx=None):
        q_tabs = ctx.q_tabs if ctx is not None else None
        mod = self.adaLN_modulation(c)
        if self.cross is not None:
            sa, ca, ga, sm, cm, gm, sx, cx, gx = mod.chunk(9, dim=-1)
        else:
            sa, ca, ga, sm, cm, gm = mod.chunk(6, dim=-1)
        x = x + ga.unsqueeze(1) * self.attn(modulate(self.norm1(x), sa, ca), rope, q_tabs)
        if self.cross is not None:
            hx = self.norm_x(x)
            if self.cross_modulate:
                hx = modulate(hx, sx, cx)
            x = x + gx.unsqueeze(1) * self.cross(hx, ctx.xc, rope, q_tabs, ctx.k_tabs, ctx.coarse_valid)
        h = modulate(self.norm2(x), sm, cm)
        cc = self.card_c(ctx.xc) if self.card_c is not None else None
        card = torch.cat([self.card_f(h), cc], dim=-1) if self.card_f is not None else None
        if isinstance(self.mlp, DSMoE):
            # token 域掩膜与难度先验由 D-EC(或 TC 的域外舍弃)消费; 两张卡与键只在对应开关打开时传入
            tkey = ctx.terrain_key if (ctx is not None and self.mlp.terrain_proj is not None) else None
            y = self.mlp(h, tok_mask, prior, card=(card if self.card_moe else None),
                         ward_key=(cc if self.ward_key else None), terrain_key=tkey)
        else:
            y = self.mlp(card if self.card_dense else h)
        x = x + gm.unsqueeze(1) * y
        return x


def draw_patch_offset(patch, device, generator=None):
    """抽切块网格的起点 (dy, dx) ∈ [0, patch)²。

    必须从调用方给定的随机流里抽, 而不是全局流: 训练时它与 (t, 噪声) 共用那条随断点
    入盘的专属流, 续训才逐位连续; 验证时它来自逐批固定种子的临时流, 起点因而逐 epoch
    固定, val 数值跨 epoch 可比。
    """
    if not patch or patch <= 1:
        return (0, 0)
    r = torch.randint(0, int(patch), (2,), device=device, generator=generator)
    return (int(r[0]), int(r[1]))


def token_pool(field, patch, offset, grid_hw):
    """(B,1,H,W) -> (B, gh·gw): 与主干同一个切块网格(同 offset、同补边尺寸), 逐块取均值。

    补边用零而不是主干输入的反射: 补边区的输出会被裁掉、不承担任何监督, 镜像进去的域内
    像素会把纯补边 token 算成域内、把边缘 token 的误差重复计数。掩膜、误差图与任何要落到
    token 网格上的量都必须经这一处: 另写一份 pad/offset 逻辑与路由看到的 token 错一格
    不会报错, 只会把地形喂成别的 token 的难度。
    """
    B, _, H, W = field.shape
    Hp, Wp = grid_hw
    dy, dx = int(offset[0]) % patch, int(offset[1]) % patch
    xp = F.pad(field.float(), (dx, Wp - W - dx, dy, Hp - H - dy), mode="constant", value=0.0)
    return F.avg_pool2d(xp, patch).flatten(1)


def token_domain_mask(domain_mask, patch, offset, grid_hw):
    """像素有效域掩膜 (B,1,H,W) -> token 域掩膜 (B, gh·gw) bool: 块内任一像素在域内即域内。
    沿海只含少数域内像素的 token 仍受损失监督, 因此必须可被路由。"""
    return token_pool(domain_mask, patch, offset, grid_hw) > 0


class DifficultyHead(nn.Module):
    """D-EC 的难度头: 从早层 token 隐状态与 adaLN 条件向量预测该 token 本次前向的误差幅度。

    输入由调用方 detach, 输出只作为路由的选择先验(不进门控), 因此主干与本头之间没有任何
    梯度路径; 头由训练侧用本次前向的逐 token 损失做监督。全程 fp32, 与 router 同精度。
    """

    def __init__(self, dim, width=64):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc_x = nn.Linear(dim, width)
        self.fc_c = nn.Linear(dim, width, bias=False)
        self.fc_out = nn.Linear(width, 1)

    def forward(self, x, c):                       # x (B,N,D), c (B,D) -> (B,N)
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = F.silu(self.fc_x(self.norm(x.float())) + self.fc_c(c.float())[:, None, :])
            return self.fc_out(h).squeeze(-1)


class BottleneckPatchEmbed(nn.Module):
    """patch 嵌入的低秩重参数化: 卷积(无 bias)压到 bottleneck 维, 1x1 升到 hidden。

    margin > 0 时卷积核放大到 patch + 2*margin 而**步长仍为 patch**: 每块的*读取*窗口
    与邻居重叠 margin 像素, 但*写出*范围不变(仍是自己那 patch x patch)。因此块数、拼接
    方式与输出形状全都不变, 也不需要任何融合权重 —— 没有两块画到同一像素上。
    padding=margin 恰好抵消核变大带来的尺寸缩水, 用反射而非补零, 免得在域边引入人造零值。

    不重叠时每块只看得见自己那一格, 相邻两块在共享边界上无从对齐; 重叠让它们看到彼此。
    """

    def __init__(self, hw, patch, in_ch, bottleneck, hidden, bias=True, margin=0):
        super().__init__()
        H, W = hw
        assert H % patch == 0 and W % patch == 0, f"{hw} 不可被 patch={patch} 整除"
        self.gh, self.gw = H // patch, W // patch
        self.margin = int(margin)
        self.proj1 = nn.Conv2d(in_ch, bottleneck, patch + 2 * self.margin, stride=patch,
                               padding=self.margin, padding_mode="reflect", bias=False)
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


def parse_tau_spec(spec, num_heads):
    """交叉问询各头的 (风层, τ 小时) 表。spec 形如 "0,0,850:6,850:12,500:12,500:24": 逗号分隔、
    每头一项, "0" 表示不推(τ=0), "层:小时" 表示按该层日均风推该小时数; 项数须等于头数。"""
    if isinstance(spec, (list, tuple)):
        items = [str(v) for v in spec]
    else:
        items = [v.strip() for v in str(spec or "").split(",") if v.strip()]
    if not items:
        items = ["0"] * int(num_heads)
    if len(items) != int(num_heads):
        raise ValueError(f"tau_spec 有 {len(items)} 项, 须等于注意力头数 {num_heads}")
    out = []
    for it in items:
        if ":" in it:
            level, hours = it.split(":", 1)
            level, hours = level.strip(), float(hours)
            if level not in ("850", "500"):
                raise ValueError(f"tau_spec 的风层只能是 850 或 500, 得到 {level!r}")
            if hours < 0:
                raise ValueError(f"tau_spec 的小时数须 >= 0, 得到 {hours}")
            out.append((level, hours))
        else:
            if float(it) != 0.0:
                raise ValueError(f"tau_spec 不带风层的项只能是 0, 得到 {it!r}")
            out.append(("850", 0.0))
    return out


def sincos_from_coords(dim, y, x):
    """按坐标现算的 2D sin-cos 绝对位置编码, 与 pos_embed.get_2d_sincos_pos_embed 同一套频率与拼接
    顺序(y 轴一半、x 轴一半, 各 sin/cos 拼接); y, x 形状 (N,), 返回 (N, dim)。"""
    d4 = dim // 4
    omega = torch.arange(d4, device=y.device, dtype=torch.float32) / d4
    omega = 1.0 / (10000.0 ** omega)
    ey = y.float()[:, None] * omega
    ex = x.float()[:, None] * omega
    return torch.cat([ey.sin(), ey.cos(), ex.sin(), ex.cos()], dim=1)


class JiT(nn.Module):
    """条件式 JiT 去噪器: net(z, t, cond) -> x 预测。

    z    : (B, out_ch, H, W) 当前噪声化目标
    t    : (B,) 时间, t=1 为数据端
    cond : (B, cond_ch, H, W) 条件场; 单流时与 z 按通道 concat 进 patch 嵌入, 两条流时按通道名
           拆成细流(非 ERA5 通道)与粗流(ERA5 动态通道)
    moe_config: None 为全稠密; dict 时按 interleave 把 FFN 换成 DSMoE
                (True: 奇数索引块; False: 全部块)。
    arch : None 为单流模型; dict 为两条流的开关表(键见 training.train_downscale.ARCH_DEFAULTS),
           由 jit_stream_config 校验过依赖关系后传入。
    """

    WIND_VARS = ("u_component_of_wind_850", "v_component_of_wind_850",
                 "u_component_of_wind_500", "v_component_of_wind_500")

    def __init__(self, hw=(720, 1440), patch=32, cond_ch=cond_channels(DEFAULT_MODE), out_ch=1,
                 hidden=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 bottleneck=128, attn_drop=0.0, proj_drop=0.0, moe_config=None,
                 patch_margin=0, refine_head=0, refine_static_idx=None, arch=None, mode=DEFAULT_MODE):
        super().__init__()
        self.hw, self.patch, self.out_ch = tuple(hw), patch, out_ch
        self.hidden, self.depth = hidden, depth
        # 切块网格补到"不小于 H+patch-1 的最小可整除尺寸": 这样任意起点 (dy,dx)∈[0,patch)²
        # 都放得下, 且**形状恒定** —— 位置编码与 RoPE 是按固定网格预生成的, 形状一变就得重建。
        self.grid_hw = (-(-(self.hw[0] + patch - 1) // patch) * patch,
                        -(-(self.hw[1] + patch - 1) // patch) * patch)
        A = dict(arch) if arch else {}
        self.arch = A or None
        self.two_stream = bool(A.get("two_stream", 0))
        self.rope_units = str(A.get("rope_units", "grid"))
        self.drop_outside = bool(A.get("drop_outside", 0))
        assert self.rope_units in ("grid", "km"), f"未知位置单位 {self.rope_units!r}"
        # ---- 条件通道按名字分流: ERA5 动态通道(含历史)进粗流, 其余(静态、年内相位、μ 等)进细流 ----
        layout = list(cond_layout(mode))
        names = layout + [f"extra_{i}" for i in range(len(layout), cond_ch)]
        is_era5 = [(n in C.ERA5_IN) or n.startswith(C.HISTORY_PREFIX) for n in names]
        self.idx_era5 = [i for i, f in enumerate(is_era5) if f]
        self.idx_fine = [i for i, f in enumerate(is_era5) if not f]
        self.idx_doy = [names.index(n) for n in C.TIME_ORDER if n in names]
        self.idx_static = [names.index(n) for n in C.STATIC_ORDER if n in names]
        self.idx_wind = [names.index(n) for n in self.WIND_VARS if n in names]
        fine_in = out_ch + (len(self.idx_fine) if self.two_stream else cond_ch)
        # 地理常量: 行 0 在域的南边界, 一个 token 的南北跨度换算成公里
        self.lat0 = float(C.DOMAIN_LAT[0])
        self.dlat = (float(C.DOMAIN_LAT[1]) - float(C.DOMAIN_LAT[0])) / float(self.hw[0])
        self.token_km = patch * self.dlat * 111.2

        self.t_embedder = TimestepEmbedder(hidden)
        self.x_embedder = BottleneckPatchEmbed(self.grid_hw, patch, fine_in,
                                               bottleneck, hidden, margin=patch_margin)
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
        cross = self.two_stream and bool(A.get("cross_attn", 1))
        two_card = None
        if self.two_stream:
            dims = A.get("two_card_dims", (192, 192))
            two_card = {"dims": (int(dims[0]), int(dims[1])), "moe": bool(A.get("expert_two_card", 0)),
                        "dense": bool(A.get("dense_two_card", 0)), "ward_key": bool(A.get("ward_key", 0))}
        self.blocks = nn.ModuleList([
            JiTBlock(hidden, num_heads, mlp_ratio,
                     attn_drop=attn_drop if mid(i) else 0.0,
                     proj_drop=proj_drop if mid(i) else 0.0,
                     moe_config=moe_config if use_moe[i] else None,
                     cross=cross, cross_modulate=bool(A.get("cross_modulate", 1)), two_card=two_card)
            for i in range(depth)])
        self.final_layer = FinalLayer(hidden, patch, out_ch)
        self.refine = None
        self.initialize_weights()
        # 精修头在 initialize_weights **之后**构造: 该函数会再消耗全局随机流(proj1/2 的
        # xavier 重初始化), 头若先构造会移动随机流, 令关/开两侧的主干参数不再逐位相同。
        # 关闭时不构造子模块: 参数数量/遍历顺序/随机数消耗与无此参数的版本逐字节相同,
        # 旧 checkpoint 原样可载, 已跑实验自动成为消融矩阵的"关"格。
        if int(refine_head):
            assert refine_static_idx, "refine_head=1 需要静态通道切片索引"
            self.refine_static_idx = [int(i) for i in refine_static_idx]
            self.refine = RefineHead(out_ch, len(self.refine_static_idx))
        # D-EC: 三个成分开关随 moe_config["dec"] 进 DSMoE; 难度头只在 prior 成分开启时存在,
        # 与精修头同理在 initialize_weights 之后构造, 主干参数与关侧逐位相同。
        self.dec, self.dec_head, self.dec_lambda = None, None, 0.0
        dec = moe_config.get("dec") if moe_config else None
        if dec:
            first_moe = next((i for i, u in enumerate(use_moe) if u), None)
            if first_moe is None:
                raise ValueError("D-EC 需要至少一个 MoE 块")
            head_layer = int(dec.get("head_layer", 1))
            if dec["prior"] and not (0 <= head_layer <= first_moe):
                raise ValueError(f"dec head_layer={head_layer} 须在 [0, 首个 MoE 块序号 {first_moe}] 内, "
                                 "否则更早的 MoE 块拿不到先验")
            self.dec = {"pool": bool(dec["pool"]), "drop": bool(dec["drop"]),
                        "prior": bool(dec["prior"]), "head_layer": head_layer,
                        "prior_max": float(dec.get("prior_max", 1.0)),
                        "prior_ramp": float(dec.get("prior_ramp", 0.1))}
            if self.dec["prior"]:
                self.dec_head = DifficultyHead(hidden)
            self.set_dec_progress(0.0)
        # ---- 两条流的模块同样在 initialize_weights 之后构造, 单流模型的随机流不受影响 ----
        self.coarse = self.terrain = None
        self.cross_on = self.mask_outside = self.lagrangian = False
        self.tau_spec = []
        if self.two_stream:
            self.coarse_doy = bool(A.get("coarse_doy", 1))
            coarse_in = len(self.idx_era5) + (len(self.idx_doy) if self.coarse_doy else 0)
            self.coarse = CoarseStream(self.grid_hw, patch, coarse_in, bottleneck, hidden, num_heads,
                                       mlp_ratio, conv=bool(A.get("coarse_conv", 1)),
                                       blocks=int(A.get("coarse_blocks", 3)), margin=patch_margin)
            self._init_stream_weights(self.coarse)
            self.cross_on = cross
            self.mask_outside = cross and bool(A.get("cross_mask_outside", 1))
            self.lagrangian = cross and bool(A.get("lagrangian", 0))
            # τ 表只在按风推移时才解析与校验; 不推时全部头用原位置
            self.tau_spec = (parse_tau_spec(A.get("tau_spec", ""), num_heads) if self.lagrangian
                             else [("850", 0.0)] * int(num_heads))
            if self.lagrangian and self.rope_units != "km":
                raise ValueError("按风推位置章(lagrangian)要求 rope_units=km")
            if self.lagrangian and len(self.idx_wind) != 4:
                raise ValueError("按风推位置章需要 850/500 hPa 的 u、v 四个通道在条件布局里")
            # 数据侧常量: 风通道的均值/标准差(还原 m/s)与 ERA5 有效掩膜(粗 token 域外判定);
            # 持久 buffer, 由 set_data_constants 写入, 随 checkpoint 保存
            self.register_buffer("wind_mean", torch.zeros(4))
            self.register_buffer("wind_std", torch.ones(4))
            self.register_buffer("era5_valid", torch.zeros(self.hw, dtype=torch.uint8))
            self.register_buffer("consts_ready", torch.zeros((), dtype=torch.uint8))
        if int(A.get("terrain_key", 0) or 0):
            self.terrain = TerrainKey(patch, len(self.idx_static), int(A.get("terrain_key_dim", 64)),
                                      int(A.get("terrain_key_window", 20)))
            self._init_stream_weights(self.terrain)

    # ------------------------------------------------------------------ 开关与常量
    @property
    def needs_domain_mask(self):
        """前向是否需要 domain_mask: D-EC 的 drop 成分或 TC 的域外舍弃要据此算 token 域掩膜。"""
        return (self.dec is not None and self.dec["drop"]) or self.drop_outside

    @property
    def needs_ctx(self):
        return self.two_stream or self.rope_units == "km" or self.terrain is not None

    def set_data_constants(self, wind_mean, wind_std, era5_valid):
        """写入风通道统计量(顺序同 WIND_VARS)与 ERA5 有效掩膜 (H, W)。单流模型无此需要, 直接返回。"""
        if not self.two_stream:
            return
        self.wind_mean.copy_(torch.as_tensor(wind_mean, dtype=torch.float32).reshape(4))
        self.wind_std.copy_(torch.as_tensor(wind_std, dtype=torch.float32).reshape(4))
        ev = torch.as_tensor(np.asarray(era5_valid)).reshape(self.hw)
        self.era5_valid.copy_((ev > 0).to(torch.uint8))
        self.consts_ready.fill_(1)

    def set_dec_progress(self, frac):
        """按训练进度 frac = samples/duration 设难度先验系数 λ: 从 0 线性升到 prior_max,
        历时 prior_ramp 比例的预算; 采样侧按 checkpoint 记录的进度复原。返回当前 λ。"""
        if self.dec is None:
            return 0.0
        ramp = self.dec["prior_ramp"]
        frac = min(1.0, max(0.0, float(frac)))
        lam = self.dec["prior_max"] * (min(1.0, frac / ramp) if ramp > 0 else 1.0)
        self.dec_lambda = lam
        for m in self.moe_layers():
            m.dec_lambda = lam
        return lam

    def set_dec_eval(self, mode="frame", capacity=1.0):
        """推理期路由规则(见 DSMoE.set_dec_eval); 非 D-EC 模型上是无操作。"""
        for m in self.moe_layers():
            m.set_dec_eval(mode, capacity)

    def pop_dec_stats(self):
        """各 MoE 层的 D-EC 统计向量 (L, E+4), 读取并清零; 非 D-EC 模型返回 None。"""
        if self.dec is None:
            return None
        return torch.stack([m.pop_dec_stats() for m in self.moe_layers()])

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
            if blk.cross is not None:
                # 交叉问询是 ERA5 进细流的唯一通道, 其门控初值取 1 而非 adaLN-zero 的 0:
                # 零门控会让模型开局对天气失明
                blk.adaLN_modulation[-1].bias.data[8 * self.hidden:9 * self.hidden].fill_(1.0)
            if isinstance(blk.mlp, DSMoE):
                blk.mlp.zero_key_projections()
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @staticmethod
    def _init_stream_weights(module):
        """粗流 / 地形键的初始化: 线性层 xavier, patch 嵌入的卷积按展平矩阵 xavier(同主干)。"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        emb = getattr(module, "embed", None)
        if emb is not None:
            for w in (emb.proj1.weight, emb.proj2.weight):
                nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.constant_(emb.proj2.bias, 0)

    # ------------------------------------------------------------------ 位置与粗流
    def coords(self, offset, device):
        """token 中心的坐标 (y, x), 各 (N,): grid = 行列序号; km = 以一个南北 token 间距为单位,
        东西向乘该行纬度的余弦(纬度由当前切块起点决定)。"""
        gh, gw = self.x_embedder.gh, self.x_embedder.gw
        i = torch.arange(gh, device=device, dtype=torch.float32)
        j = torch.arange(gw, device=device, dtype=torch.float32)
        y = i[:, None].expand(gh, gw)
        x = j[None, :].expand(gh, gw)
        if self.rope_units == "km":
            dy = int(offset[0]) % self.patch
            lat = self.lat0 + ((i * self.patch + self.patch / 2.0) - dy) * self.dlat
            x = x * torch.cos(torch.deg2rad(lat))[:, None]
        return y.reshape(-1), x.reshape(-1)

    def _wind_tokens(self, cond, offset):
        """粗 token 上的日均风 (m/s): 4 个风通道各在块内的图像像素上取均值(补边像素不计入),
        再用记录的统计量还原; 纯补边 token 的风为 0。返回 {层: (u, v)}。"""
        frac = token_pool(torch.ones_like(cond[:, :1]), self.patch, offset, self.grid_hw)   # 块内图像像素占比
        w = []
        for k, idx in enumerate(self.idx_wind):
            z = token_pool(cond[:, idx:idx + 1], self.patch, offset, self.grid_hw) / frac.clamp_min(1e-6)
            w.append(z * self.wind_std[k] + self.wind_mean[k])
        return {"850": (w[0], w[1]), "500": (w[2], w[3])}

    def _k_tabs(self, cond, offset, y, x, B):
        """交叉问询里 K 的位置表 (cos, sin), 各 (B, heads, N, head_dim): 每个头按各自的 (风层, τ)
        把粗 token 的位置推到"其空气 τ 小时后到达的位置"; τ=0 的头用原位置。"""
        wind = self._wind_tokens(cond, offset) if self.lagrangian else None
        ys, xs = [], []
        for level, hours in self.tau_spec:
            if wind is None or hours <= 0:
                ys.append(y[None].expand(B, -1))
                xs.append(x[None].expand(B, -1))
            else:
                u, v = wind[level]
                s = float(hours) * 3.6 / self.token_km              # m/s·h -> km -> token 间距
                ys.append(y[None] + v * s)
                xs.append(x[None] + u * s)
        return self.rope.tables(torch.stack(ys, dim=1), torch.stack(xs, dim=1))

    def precompute(self, cond, offset=(0, 0)):
        """只依赖条件场与切块起点的那部分: 位置表、粗流、K 的推移位置、粗 token 掩膜、地形键。
        与 t 和噪声无关, 采样时一条轨迹算一次即可; 训练时每步随起点重算。"""
        if not self.needs_ctx:
            return None
        B = cond.shape[0]
        H, W = self.hw
        Hp, Wp = self.grid_hw
        dy, dx = int(offset[0]) % self.patch, int(offset[1]) % self.patch
        pad = lambda a: F.pad(a, (dx, Wp - W - dx, dy, Hp - H - dy), mode="reflect")
        ctx = StreamContext(offset=(dy, dx))
        y, x = self.coords((dy, dx), cond.device)
        if self.rope_units == "km":
            ctx.q_tabs = self.rope.tables(y, x)
            pos = sincos_from_coords(self.hidden, y, x)
        else:
            pos = self.pos_embed
        ctx.pos = pos
        if self.two_stream:
            if (self.lagrangian or self.mask_outside) and int(self.consts_ready) == 0:
                raise RuntimeError("两条流模型缺数据侧常量(风统计量 / ERA5 有效掩膜): 训练前调 set_data_constants, "
                                   "采样时应从 checkpoint 恢复")
            ch = self.idx_era5 + (self.idx_doy if self.coarse_doy else [])
            ctx.xc = self.coarse(pad(cond[:, ch]), pos, self.rope, ctx.q_tabs)
            if self.mask_outside:
                valid = self.era5_valid.to(cond.dtype)[None, None]
                ctx.coarse_valid = (token_pool(valid, self.patch, (dy, dx), self.grid_hw) > 0).expand(B, -1)
            if self.cross_on:
                if self.lagrangian:
                    ctx.k_tabs = self._k_tabs(cond, (dy, dx), y, x, B)
                else:
                    ctx.k_tabs = ctx.q_tabs
        if self.terrain is not None:
            ctx.terrain_key = self.terrain(pad(cond[:, self.idx_static]))
        return ctx

    def unpatchify(self, x):
        """(B, gh*gw, p*p*C) -> (B, C, gh*p, gw*p), 矩形网格。"""
        B = x.shape[0]
        gh, gw, p, c = self.x_embedder.gh, self.x_embedder.gw, self.patch, self.out_ch
        x = x.reshape(B, gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(B, c, gh * p, gw * p)

    def forward(self, z, t, cond, offset=(0, 0), domain_mask=None, return_dec=False, coarse_cache=None):
        """offset=(dy,dx) 是切块网格的起点, 取值 [0,patch)。

        固定起点时同一个像素永远落在块内同一位置, 而逐像素损失从不惩罚"相邻块在边界上
        对不上", 于是接缝是免费的, 会在固定位置累积成网格。每步换起点后, 同一像素这次
        在边界、下次在块内, 任何位置特异的偏置在平均意义上都要挨罚 —— 接缝因此变得有代价。

        domain_mask (B,1,H,W): 像素有效域掩膜, D-EC 的 drop 成分与 TC 的域外舍弃据此算 token
        域掩膜, 其余情况忽略。return_dec=True 时额外返回难度头预测与 token 域掩膜, 供训练侧算
        辅助损失; 其中 pred 是 (B,N) 的 fp32, tok_mask 是 (B,N) bool 或 None。
        coarse_cache: precompute() 的产物; 给定时必须与本次 offset 一致, 采样时一条轨迹复用。
        """
        H, W = self.hw
        Hp, Wp = self.grid_hw
        dy, dx = int(offset[0]) % self.patch, int(offset[1]) % self.patch
        x = torch.cat([z, cond[:, self.idx_fine] if self.two_stream else cond], dim=1)
        x = F.pad(x, (dx, Wp - W - dx, dy, Hp - H - dy), mode="reflect")
        x = self.x_embedder(x)
        if coarse_cache is not None:
            if tuple(coarse_cache.offset) != (dy, dx):
                raise RuntimeError(f"coarse_cache 的切块起点 {coarse_cache.offset} 与本次 {(dy, dx)} 不一致")
            ctx = coarse_cache
        else:
            ctx = self.precompute(cond, (dy, dx))
        pos = ctx.pos if ctx is not None else self.pos_embed
        x = x + pos.to(x.dtype)
        c = self.t_embedder(t)
        tok_mask = prior = pred = None
        if self.needs_domain_mask:
            if domain_mask is None:
                raise RuntimeError("该模型的路由需要 domain_mask (B,1,H,W)")
            tok_mask = token_domain_mask(domain_mask, self.patch, (dy, dx),
                                         self.grid_hw).reshape(-1)
        for i, blk in enumerate(self.blocks):
            if self.dec_head is not None and i == self.dec["head_layer"]:
                pred = self.dec_head(x.detach(), c.detach())            # (B, N) fp32
                prior = pred.reshape(-1)
            x = blk(x, c, self.rope, tok_mask, prior, ctx)
        out = self.unpatchify(self.final_layer(x, c))
        out = out[..., dy:dy + H, dx:dx + W]
        if self.refine is not None:
            # 用未 pad 的原始 cond 取静态切片: pad/crop 循环之外, 对齐无歧义
            out = self.refine(out, cond[:, self.refine_static_idx])
        if return_dec:
            return out, {"pred": pred, "offset": (dy, dx),
                         "tok_mask": (tok_mask.view(z.shape[0], -1)
                                      if tok_mask is not None else None)}
        return out

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
