#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
vit.py — SETR 风格 Vision Transformer 降尺度主体
============================================================================
ERA5 -> Daymet 降尺度的 Transformer 主体: patch 切 token -> 前置归一化 Transformer 块
-> 上采样回像素 -> 3 层卷积细化头。细化头是必需的, 没有它 patch 边界会留下规则格子。

位置编码两种:
  sincos  固定的二维正弦编码, 按 (gh, gw) 惰性生成并缓存, 不是参数 —— 因此同一份权重
          能吃任意尺寸输入, 整幅推理与裁块推理走同一条路径。
  learned 可学习参数, 形状绑定训练时的 token 数, 换尺寸即失效; 仅为兼容早期 checkpoint。

上采样头两种: pixelshuffle(无棋盘伪影, 默认) 与 convt(ConvTranspose, 兼容早期 checkpoint)。

序列并行(SP)只改变注意力的计算分布, 不改变权重与数值: sp_group=None 时用 PyTorch 的
nn.MultiheadAttention, 非 None 时换成 SPSelfAttention 把 O(N^2) 拆给 SP 组。两者数值
等价, 故 SP 训练出的 checkpoint 可以在单进程下直接加载评测(见 set_sp_group)。
============================================================================
"""
import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: 训练时按样本随机丢整条残差分支(推理/eval 恒等)。"""
    def __init__(self, p=0.0):
        super().__init__(); self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep

class TBlock(nn.Module):
    """Pre-norm Transformer block: LN+MHSA(残差) -> LN+MLP(残差)。dropout/drop_path 抗过拟合。
    sp_group=None: 原 nn.MultiheadAttention(默认, 路径不变);非 None: 序列并行全局注意力
    (SP.SPSelfAttention, 与 MHA 数值等价, 把 O(N²) 拆给 SP 组, 见 seq_parallel_attn.py)。"""
    def __init__(self, dim, heads, mlp, dropout=0.0, drop_path=0.0, sp_group=None):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.sp_group = sp_group
        self._use_mha = sp_group is None                 # 模块类型固定于构造(set_sp_group 不改它)
        if sp_group is None:
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        else:
            from era5_daymet.models import seq_parallel_attn as SP
            self.attn = SP.SPSelfAttention(dim, heads, dropout=dropout, sp_group=sp_group)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(h, dim), nn.Dropout(dropout))
        self.dp = DropPath(drop_path)

    def forward(self, x):
        y = self.n1(x)
        # 调用约定按模块类型(而非当前 sp_group): SPSelfAttention(x) 内部处理 sp_group=None(退化本地全注意力)
        a = self.attn(y, y, y, need_weights=False)[0] if self._use_mha else self.attn(y)
        x = x + self.dp(a)
        return x + self.dp(self.mlp(self.n2(x)))

def get_2d_sincos_pos_embed(dim, gh, gw, device=None, dtype=None):
    """2D sin-cos 固定位置编码 (MAE 风格): 无参数、任意分辨率通用 -> 整幅训练不再需要
    绑尺寸的可学习 pos (整幅 259200 token 的可学习位置编码裸参数达 ~1 亿)。
    dim 一半编码 h 轴、一半编码 w 轴; 每轴 dim/4 个几何频率的 sin/cos 拼接。要求 dim%4==0。
    token 顺序与 embed 后 flatten(2) 一致(h 外 w 内, 即 idx=h*gw+w)。返回 (1, gh*gw, dim)。"""
    assert dim % 4 == 0, f"sincos pos embed 需 dim%4==0, 得到 dim={dim}"
    d4 = dim // 4
    omega = torch.arange(d4, device=device, dtype=torch.float32) / d4
    omega = 1.0 / (10000.0 ** omega)                                   # (d4,) 频率
    y = torch.arange(gh, device=device, dtype=torch.float32)
    x = torch.arange(gw, device=device, dtype=torch.float32)
    ey = torch.einsum("i,j->ij", y, omega)                             # (gh, d4)
    ex = torch.einsum("i,j->ij", x, omega)                             # (gw, d4)
    ey = torch.cat([ey.sin(), ey.cos()], dim=1)                        # (gh, dim/2)
    ex = torch.cat([ex.sin(), ex.cos()], dim=1)                        # (gw, dim/2)
    ey = ey[:, None, :].expand(gh, gw, dim // 2)
    ex = ex[None, :, :].expand(gh, gw, dim // 2)
    emb = torch.cat([ey, ex], dim=2).reshape(1, gh * gw, dim)          # (1, gh*gw, dim)
    return emb.to(dtype) if dtype is not None else emb

class ViT(nn.Module):
    """SETR 风格 ViT: patch->token->Transformer->像素, 末端卷积细化头消格子伪影。
    pos_type: 'sincos'(固定, 任意尺寸 -> 整幅) 或 'learned'(绑 crop 尺寸, 兼容旧 ckpt)。
    head_up : 'pixelshuffle'(无棋盘) 或 'convt'(ConvTranspose, 兼容旧 ckpt)。"""
    def __init__(self, in_ch, out_ch, img=64, patch=2, dim=384, depth=8, heads=6, mlp=4.0,
                 dropout=0.0, drop_path=0.0, pos_type="sincos", head_up="pixelshuffle",
                 full_frame=False, sp_group=None, sp_size=1, sp_rank=0):
        super().__init__()
        assert pos_type in ("sincos", "learned"); assert head_up in ("pixelshuffle", "convt")
        self.patch = patch; self.dim = dim
        self.pos_type = pos_type; self.head_up = head_up; self.full_frame = full_frame
        self.sp_group = sp_group; self.sp_size = sp_size; self.sp_rank = sp_rank  # 序列并行(默认关)
        g = img // patch
        self.n = g * g
        self.embed = nn.Conv2d(in_ch, dim, patch, stride=patch)            # 像素 -> token
        if pos_type == "learned":
            self.pos = nn.Parameter(torch.zeros(1, self.n, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self._pos_cache = {}                                          # (gh,gw)->tensor 惰性缓存(非参数)
        self.pos_drop = nn.Dropout(dropout)
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]     # 线性 0->drop_path
        self.blocks = nn.ModuleList([TBlock(dim, heads, mlp, dropout, dpr[i], sp_group=sp_group)
                                     for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        if head_up == "convt":
            self.unembed = nn.ConvTranspose2d(dim, dim // 2, patch, stride=patch)  # token -> 像素(棋盘源)
        else:
            self.proj = nn.Conv2d(dim, (dim // 2) * patch * patch, 1)      # PixelShuffle 上采样: 无棋盘
            self.pixel_shuffle = nn.PixelShuffle(patch)
        self.head = nn.Sequential(                                          # ★卷积细化(消 patch 缝)
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim // 2, out_ch, 3, padding=1))

    def _get_pos(self, gh, gw, device, dtype):
        key = (gh, gw); p = self._pos_cache.get(key)
        if p is None or p.device != device or p.dtype != dtype:
            p = get_2d_sincos_pos_embed(self.dim, gh, gw, device, dtype)
            self._pos_cache[key] = p
        return p

    def forward(self, x):
        B = x.shape[0]
        t = self.embed(x)
        gh, gw = t.shape[-2], t.shape[-1]
        t = t.flatten(2).transpose(1, 2)
        pos = self.pos[:, :gh * gw] if self.pos_type == "learned" else self._get_pos(gh, gw, t.device, t.dtype)
        t = self.pos_drop(t + pos)                          # [B, N, dim] 完整序列
        if self.sp_group is not None:                       # 序列并行: 切本 rank 的 token 段
            from era5_daymet.models import seq_parallel_attn as SP
            t = SP.shard_seq(t, self.sp_rank, self.sp_size)  # [B, N/sp, dim]
        for blk in self.blocks:
            t = blk(t)                                       # SP 注意力在组内聚 K/V, 算精确全局注意力
        t = self.norm(t)                                     # per-token, 在(分片的)本地段上
        if self.sp_group is not None:                        # 聚回完整序列供整幅卷积头
            from era5_daymet.models import seq_parallel_attn as SP
            t = SP.gather_seq_tokens(t, self.sp_group)        # [B, N, dim]
        t = t.transpose(1, 2).reshape(B, -1, gh, gw)
        u = self.unembed(t) if self.head_up == "convt" else self.pixel_shuffle(self.proj(t))
        return self.head(u)

    def sp_param_groups(self):
        """SP 训练梯度同步用: 返回 (sharded_params, redundant_params)。
        redundant = gather 后的整幅卷积头(proj/unembed + head, 每 SP rank 各算完整一份、梯度相同, 不可 SP-SUM);
        sharded = 其余(embed/blocks/norm, 各 rank 只算本段 token, 需 SP-SUM)。
        ★两个列表必须是**确定性顺序**(各 rank 一致)——梯度会被 flatten 后按序 all_reduce,
        用 list(set(...)) 会因 tensor id 哈希序在各 rank 不同 -> 归约错位。故用模块迭代序 + id 做成员判断。"""
        red_mods = [self.head, self.unembed if self.head_up == "convt" else self.proj]
        redundant = [p for m in red_mods for p in m.parameters()]      # 确定性顺序
        red_ids = set(id(p) for p in redundant)
        sharded = [p for p in self.parameters() if id(p) not in red_ids]  # 确定性顺序
        return sharded, redundant

    def set_sp_group(self, sp_group, sp_size=1, sp_rank=0):
        """切换序列并行组(供 rank0 末端整幅评测时关掉 SP -> 单进程无集合通信, 免挂)。
        SPSelfAttention(sp_group=None) 退化为本地全注意力, 权重不变。"""
        self.sp_group, self.sp_size, self.sp_rank = sp_group, sp_size, sp_rank
        for blk in self.blocks:
            blk.sp_group = sp_group
            if hasattr(blk.attn, "sp_group"):
                blk.attn.sp_group = sp_group
