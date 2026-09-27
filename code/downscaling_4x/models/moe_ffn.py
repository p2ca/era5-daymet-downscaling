#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
moe_ffn.py — DeepSeek 风格稀疏 FFN(共享专家 + sigmoid top-K 路由)
============================================================================
移植自 EfficientMoE (Liu et al., arXiv:2512.01252) 的 JiTMoE 实现, 语义逐项对齐其发布
代码而非论文正文, 两者不一致处以代码为准:

  - 亲和分 s = sigmoid(x @ W_router^T), router 全程 fp32;
  - top-K 选择用 s + b(逐专家偏置, 仅参与选择), 门控权重用**原始** s 在被选专家上归一,
    再乘 routed_scaling_factor(发布代码为 2.5, 论文正文未写);
  - 专家与共享专家均为 SwiGLU, 且不做稠密 FFN 惯用的 2/3 中间宽度缩放;
  - dropless: 无容量因子, 所有被选 token 都送达专家;
  - 分组路由(node-limited routing)按 DeepSeek-V3 结构保留, 默认 n_group=topk_group=2
    即选满所有组, 等价于无操作;
  - 模块输出 = 路由输出 + 共享专家(x); 残差与 adaLN 门控由外层 Transformer 块提供。
    关闭共享专家时输出 = 路由输出 + x(发布代码的行为, 会与外层残差叠加)。

与发布代码的两处有意偏离:
  - router 权重补 kaiming_normal_ 初始化(其 JiTMoE 版遗漏初始化, DSMoE 版有);
  - 偏置 b 的免辅助损失更新(DeepSeek-V3: 过载减、欠载增)实现为 update_bias(),
    默认不调用即 b 恒零(与发布代码一致); 逐层负载计数常开, 供塌缩监控与专家分析。

选择方向有三档(router_mode):
  - tc  每 token 挑 top-K 专家(DeepSeek);
  - ec  每专家在帧内挑 top-C token(Expert-Choice, Zhou 2022), C=⌈N·K/E⌉ 使 FLOPs 与 tc 对齐;
  - dec 难度感知的池化专家选择(D-EC): 在 ec 的选择方向上叠三个可独立开关的成分
      pool   专家在本次前向的全部帧上挑 token(跨帧池), 关则退回帧内池;
      drop   有效域外 token 不参与路由专家的挑选, 只走共享专家与外层残差;
             容量按域内 token 数计, 每个域内 token 平均仍是 K 个专家;
      prior  选择分加难度先验 λ·σ_pool(s)·z(ĥ): ĥ 由外部难度头逐 token 给出, z 为池内
             域内 token 上的标准化并截到 ±3, σ_pool(s) 是池内亲和分的标准差, 使 λ 无量纲;
             先验只进选择, 门控权重仍用原始 s, 因此主损失没有任何梯度路径进难度头。
    推理期(eval 模式)不再依赖同批: 缺省按帧内取 top-C 乘 dec_eval_capacity, 逐帧决策与
    batch 组成无关; "pool" 档保留给需要同批的诊断。

tc 路由的两条流扩展(各自独立开关, 全关时与上述 tc 逐位相同):
  - 两张卡(card_dim): 路由专家读"细 token 投影 ‖ 正上方粗 token 投影"的拼接, 路由分与共享专家
    仍读 token 本身;
  - 两级分诊: 组 = 科室(n_group 个), 第一级按组分选 topk_group 个科室, 组分 = 组内前两名专家分之和
    + 科室键(ward_key_dim, 零初始化投影); 第二级在科室内按专家分取 top-K;
  - 地形键(terrain_key_dim): 零初始化投影后加进专家 logits(sigmoid 之前);
  - 域外舍弃(drop_outside): 域外 token 不派发给路由专家, 只走共享专家与外层残差。
============================================================================
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """专家用 SwiGLU: w12 = d->2S 合并投影, w3 = S->d。S 不做 2/3 缩放。
    in_dim 缺省等于 dim; 专家读两张卡时输入宽度是拼接后的卡宽, 输出仍回到 dim。"""

    def __init__(self, dim, inter, drop=0.0, bias=True, in_dim=None):
        super().__init__()
        self.w12 = nn.Linear(in_dim or dim, 2 * inter, bias=bias)
        self.w3 = nn.Linear(inter, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


class TopkRouter(nn.Module):
    """线性亲和路由器(无 bias 项), logits 全程 fp32。

    e_score_correction_bias 是逐专家的选择偏置(buffer, 非参数): 只加进 top-K 选择的
    分数, 不进入门控权重。它必须随 checkpoint 保存, 且导出 EMA 权重采样时须一并携带。
    """

    def __init__(self, dim, n_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.kaiming_normal_(self.weight)
        self.register_buffer("e_score_correction_bias", torch.zeros(n_experts))

    def forward(self, x_flat):
        return F.linear(x_flat.float(), self.weight.float())


def dec_pools(B, N, pool_batch):
    """池的切片表: 批级池把 B·N 个 token 放进一池, 否则每帧各一池。
    路由端的选择与损失端的难度目标都从这里取池, 两边的池定义不可能分叉。"""
    if pool_batch:
        return [slice(0, B * N)]
    return [slice(b * N, (b + 1) * N) for b in range(B)]


def dec_standardize(values, valid, pools, clip=3.0):
    """逐池在有效 token 上把标量标准化(零均值单位方差)并截断到 ±clip; 无效 token 处为 0。

    values (T,), valid (T,) bool。有效 token 少于 2 个的池整段为 0(标准差无定义)。
    """
    v = values.float()
    out = torch.zeros_like(v)
    for sl in pools:
        m = valid[sl]
        if int(m.sum()) < 2:
            continue
        x = v[sl][m]
        z = (v[sl] - x.mean()) / (x.std(unbiased=False) + 1e-6)
        out[sl] = z.clamp(-clip, clip) * m.to(z.dtype)
    return out


class DSMoE(nn.Module):
    """共享专家 + 路由专家的稀疏 FFN 层。

    forward 输入 (B, N, d), 输出同形。同时把本次各专家命中 token 数累加进 load_acc
    (buffer, 不入 checkpoint), 由训练侧读取/清零, 用于负载监控与免辅助损失偏置更新。
    """

    def __init__(self, num_experts, dim, moe_inter, num_experts_per_tok=2,
                 n_group=2, topk_group=2, norm_topk_prob=True,
                 routed_scaling_factor=2.5, use_shared_expert=True, proj_drop=0.0,
                 router_mode="tc", gating_mode="norm", dec_config=None,
                 card_dim=None, ward_key_dim=0, terrain_key_dim=0, drop_outside=False):
        super().__init__()
        assert num_experts % n_group == 0
        # 两条流下 TC 路由的扩展(其余路由不接受, 传了就拒绝):
        #   card_dim        路由专家读"两张卡"拼接后的输入宽度(None = 读 token 本身, 与共享专家同源);
        #   ward_key_dim    科室键(粗 token 投影)的宽度, 经零初始化矩阵加进科室分, 只在分组选择时起作用;
        #   terrain_key_dim 地形键的宽度, 经零初始化矩阵加进专家 logits(sigmoid 之前);
        #   drop_outside    域外 token 不进路由专家, 只走共享专家与外层残差(D-EC 之外的路由也可用)。
        if router_mode != "tc" and (card_dim or ward_key_dim or terrain_key_dim or drop_outside):
            raise ValueError("card_dim / ward_key_dim / terrain_key_dim / drop_outside 只支持 router_mode='tc'")
        self.card_dim = int(card_dim) if card_dim else None
        self.drop_outside = bool(drop_outside)
        # router_mode: tc = 每 token 挑 top-K 专家(DeepSeek); ec = 每专家挑 top-C token
        #   (Expert-Choice 选择方向, Zhou 2022), C=⌈N·K/E⌉ 使 FLOPs 与 tc 严格对齐,
        #   池按样本内(帧内)取 —— 整幅非因果推理下训练与推理的池构成天然一致;
        #   dec = D-EC, 见模块开头。
        # gating_mode: norm = 当选专家上归一(家法, tc 固定此档); raw = sigmoid 原值
        #   (仅 ec/dec 允许, 保留"多专家 token 更大幅度"的通道; 均值仍 ≈ norm 档)。
        # 组合约束(tc+raw 拒绝等)由 train_downscale.jit_moe_config 单点执法。
        assert router_mode in ("tc", "ec", "dec") and gating_mode in ("norm", "raw")
        self.router_mode, self.gating_mode = router_mode, gating_mode
        self.n_experts, self.top_k = num_experts, num_experts_per_tok
        self.n_group, self.topk_group = n_group, topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.use_shared_expert = use_shared_expert
        self.experts = nn.ModuleList(
            [SwiGLUExpert(dim, moe_inter, proj_drop, in_dim=self.card_dim) for _ in range(num_experts)])
        self.gate = TopkRouter(dim, num_experts)
        if use_shared_expert:
            self.shared_experts = SwiGLUExpert(dim, moe_inter, proj_drop)
        # 键的投影零初始化: 键接进来的瞬间路由与不带键时逐位相同, 训练中再学会用
        self.ward_proj = nn.Linear(int(ward_key_dim), n_group, bias=False) if ward_key_dim else None
        self.terrain_proj = nn.Linear(int(terrain_key_dim), num_experts, bias=False) if terrain_key_dim else None
        self.zero_key_projections()
        # 负载计数是逐 rank 的本地统计, 存普通张量而非 buffer: DDP 的 broadcast_buffers
        # 会在每次前向把 rank0 的 buffer 覆盖到所有 rank, 会静默污染各 rank 的计数
        self.load_acc = torch.zeros(num_experts)
        # ---- D-EC 状态: 三个成分开关与推理期规则; 全部是普通属性, 不进 checkpoint ----
        if router_mode == "dec":
            if not dec_config:
                raise ValueError("router_mode='dec' 需要 dec_config(pool/drop/prior)")
            self.dec_pool = bool(dec_config["pool"])
            self.dec_drop = bool(dec_config["drop"])
            self.dec_prior = bool(dec_config["prior"])
            if not (self.dec_pool or self.dec_drop or self.dec_prior):
                raise ValueError("D-EC 三个成分全关等价于 ec, 请用 router_mode='ec'")
        else:
            self.dec_pool = self.dec_drop = self.dec_prior = False
        self.dec_lambda = 0.0                     # 难度先验系数, 训练侧按进度设置
        self.dec_eval_mode, self.dec_eval_capacity = "frame", 1.0
        # D-EC 统计(本地普通张量, 同 load_acc): [0..E] 域内 token 被几个专家选中的直方图,
        # [E+1] 域内 token 数, [E+2] 批级池下逐帧容量占比的相对标准差之和, [E+3] 其计数
        self.dec_acc = torch.zeros(num_experts + 4)

    def set_dec_eval(self, mode="frame", capacity=1.0):
        """推理期规则: frame = 帧内 top-C·capacity(缺省, 与 batch 组成无关);
        pool = 同批帧共池(只供诊断)。capacity 只在 eval 模式生效, 训练恒为 1。"""
        if mode not in ("frame", "pool"):
            raise ValueError(f"未知 dec_eval_mode {mode!r}")
        if not (capacity > 0):
            raise ValueError(f"dec_eval_capacity 必须为正, 得到 {capacity}")
        self.dec_eval_mode, self.dec_eval_capacity = mode, float(capacity)

    @torch.no_grad()
    def zero_key_projections(self):
        """科室键与地形键的投影矩阵置零(主干初始化会给全部线性层 xavier, 之后要再调一次)。"""
        for m in (self.ward_proj, self.terrain_proj):
            if m is not None:
                m.weight.zero_()

    def route(self, scores, ward_bias=None):
        """scores: (T, E) 的 sigmoid 亲和分 -> (top-K 专家索引, 门控权重)。
        ward_bias: (T, n_group) 加进组分的科室键分, None 时组分只由专家分决定。"""
        for_choice = scores + self.gate.e_score_correction_bias
        # 分组路由: 每组取前 2 名分数之和作为组分, 选 topk_group 个组, 组外专家不参选。
        # 默认配置选满所有组, 此段为无操作, 保留 DeepSeek-V3 的结构; 两级分诊(科室 = 组)
        # 时 topk_group < n_group, 第一级按组分选科室, 第二级在科室内按专家分取 top-K。
        gsize = self.n_experts // self.n_group
        group_scores = for_choice.view(-1, self.n_group, gsize).topk(
            min(2, gsize), dim=-1)[0].sum(dim=-1)
        if ward_bias is not None:
            group_scores = group_scores + ward_bias
        gidx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        gmask = torch.zeros_like(group_scores).scatter_(1, gidx, 1)
        smask = gmask.unsqueeze(-1).expand(-1, self.n_group, gsize).reshape(
            -1, self.n_experts)
        for_choice = for_choice.masked_fill(~smask.bool(), 0.0)
        topk_idx = torch.topk(for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_w = scores.gather(1, topk_idx)          # 门控用原始亲和分, 偏置只影响选择
        if self.top_k > 1 and self.norm_topk_prob:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_w * self.routed_scaling_factor

    def route_ec(self, scores, B, N):
        """EC 选择: 每专家在**样本内**挑 top-C token, C=⌈N·K/E⌉。

        返回 (选择掩膜 (B*N, E) bool, 门控权重 (B*N, E))。偏置不参与: 给整列加常数
        不改变列内 top-C 排序, 在 EC 下它数学上是无操作。未被任何专家选中的 token
        路由输出为 0(共享专家与外层残差兜底)。
        """
        cap = (N * self.top_k + self.n_experts - 1) // self.n_experts
        sc = scores.view(B, N, self.n_experts)
        top_tok = torch.topk(sc, k=cap, dim=1, sorted=False)[1]       # (B, C, E)
        sel = torch.zeros_like(sc, dtype=torch.bool).scatter_(1, top_tok, True)
        w = sc.masked_fill(~sel, 0.0)
        if self.gating_mode == "norm":
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)      # 全零行安全得 0
        return sel.view(-1, self.n_experts), (w * self.routed_scaling_factor).view(
            -1, self.n_experts)

    def route_dec(self, scores, B, N, tok_mask=None, prior=None):
        """D-EC 选择: 返回与 route_ec 同形的 (选择掩膜, 门控权重)。

        scores (T,E) fp32 sigmoid 亲和分; tok_mask (T,) bool, True 为有效域内 token;
        prior (T,) 难度头的逐 token 预测(未标准化)。三个成分全关时与 route_ec 等价。
        容量 C = ⌈n_valid·K·capacity/E⌉ 按池内可选 token 数计; 域外 token 的选择分置 -inf,
        永不入选; 池内可选 token 少于 2 个时先验无定义, 按无先验处理。
        """
        E, K = self.n_experts, self.top_k
        T = B * N
        if self.dec_drop and tok_mask is not None:
            valid = tok_mask.to(torch.bool)
        else:
            valid = torch.ones(T, dtype=torch.bool, device=scores.device)
        pool_batch = self.dec_pool if self.training else (self.dec_eval_mode == "pool")
        cap = 1.0 if self.training else self.dec_eval_capacity
        lam = float(self.dec_lambda) if (self.dec_prior and prior is not None) else 0.0
        pools = dec_pools(B, N, pool_batch)
        sel_score = scores
        if lam > 0:
            z = dec_standardize(prior, valid, pools)
            sel_score = scores.clone()
            for sl in pools:
                m = valid[sl]
                if int(m.sum()) < 2:
                    continue
                sigma = scores[sl][m].std(unbiased=False)
                sel_score[sl] = scores[sl] + (lam * sigma) * z[sl][:, None]
        if self.dec_drop:
            sel_score = sel_score.masked_fill(~valid[:, None], float("-inf"))
        sel = torch.zeros(T, E, dtype=torch.bool, device=scores.device)
        for sl in pools:
            n_valid = int(valid[sl].sum())
            c = min(n_valid, math.ceil(n_valid * K * cap / E))
            if c <= 0:
                continue
            top = torch.topk(sel_score[sl], k=c, dim=0, sorted=False)[1]   # (c, E)
            sel[sl] = sel[sl].scatter(0, top, True)
        w = scores.masked_fill(~sel, 0.0)
        if self.gating_mode == "norm":
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        with torch.no_grad():
            if self.dec_acc.device != scores.device:
                self.dec_acc = self.dec_acc.to(scores.device)
            kv = sel[valid].sum(dim=1)
            self.dec_acc[:E + 1] += torch.bincount(kv, minlength=E + 1)[:E + 1].float()
            self.dec_acc[E + 1] += float(valid.sum())
            if pool_batch and B > 1:
                per_frame = sel.view(B, N, E).sum(dim=(1, 2)).float()
                share = per_frame / per_frame.sum().clamp_min(1.0)
                self.dec_acc[E + 2] += float(share.std(unbiased=False)) * B
                self.dec_acc[E + 3] += 1.0
        return sel, w * self.routed_scaling_factor

    def forward(self, x, tok_mask=None, prior=None, card=None, ward_key=None, terrain_key=None):
        """x (B,N,d) 是路由与共享专家读的 token; card (B,N,card_dim) 给定时路由专家改读它(两张卡拼接);
        ward_key (B,N,·) 进科室分, terrain_key (B,N,·) 进专家 logits; tok_mask (B*N,) bool 为域内 token。"""
        if self.router_mode != "dec" and prior is not None:
            raise RuntimeError(f"router_mode={self.router_mode!r} 不接受 prior, 它只属于 dec")
        if self.router_mode != "dec" and tok_mask is not None and not self.drop_outside:
            raise RuntimeError(f"router_mode={self.router_mode!r} 未开 drop_outside, 不接受 tok_mask")
        if card is not None and self.card_dim is None:
            raise RuntimeError("该层未按两张卡构造(card_dim=None), 不接受 card")
        if card is None and self.card_dim is not None:
            raise RuntimeError("该层按两张卡构造, 前向必须传 card")
        if (ward_key is None) != (self.ward_proj is None):
            raise RuntimeError("ward_key 的有无必须与构造时的 ward_key_dim 一致")
        if (terrain_key is None) != (self.terrain_proj is None):
            raise RuntimeError("terrain_key 的有无必须与构造时的 terrain_key_dim 一致")
        B, N, d = x.shape
        flat = x.reshape(-1, d)
        # 路由固定 fp32: 显式关闭 autocast(否则 bf16 训练下 F.linear 会被打回 bf16,
        # 亲和分与 top-K 选择带上量化噪声)。专家计算仍走环境精度。
        out = torch.zeros_like(flat)
        if self.router_mode in ("ec", "dec"):
            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = torch.sigmoid(self.gate(flat))               # (T, E) fp32
                if self.router_mode == "ec":
                    sel, w = self.route_ec(scores, B, N)
                else:
                    if self.dec_drop and tok_mask is None:
                        raise RuntimeError("D-EC 的 drop 成分需要 tok_mask (B*N,) bool")
                    if self.dec_prior and prior is None:
                        raise RuntimeError("D-EC 的 prior 成分需要 prior (B*N,)")
                    sel, w = self.route_dec(scores, B, N, tok_mask, prior)
            with torch.no_grad():
                if self.load_acc.device != flat.device:
                    self.load_acc = self.load_acc.to(flat.device)
                self.load_acc += sel.sum(0).to(self.load_acc.dtype)
            for e in range(self.n_experts):
                tok = sel[:, e].nonzero().flatten()
                if tok.numel():
                    y = self.experts[e](flat[tok]) * w[tok, e, None].to(flat.dtype)
                    out.index_add_(0, tok, y.to(out.dtype))
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = self.gate(flat)                                # (T, E) fp32
                if terrain_key is not None:
                    logits = logits + self.terrain_proj(terrain_key.reshape(-1, terrain_key.shape[-1]).float())
                scores = torch.sigmoid(logits)
                ward_bias = (self.ward_proj(ward_key.reshape(-1, ward_key.shape[-1]).float())
                             if ward_key is not None else None)
                topk_idx, topk_w = self.route(scores, ward_bias)
            # dropless 派发: 逐命中专家收集其 token, 计算后按门控权重累加回原位。
            hit = F.one_hot(topk_idx, num_classes=self.n_experts)     # (T, K, E)
            if self.drop_outside and tok_mask is not None:
                # 域外 token 不派发: 它们没有真值, 专家不该在它们身上练; 只剩共享专家与外层残差
                hit = hit * tok_mask.to(hit.dtype).view(-1, 1, 1)
            with torch.no_grad():
                if self.load_acc.device != flat.device:
                    self.load_acc = self.load_acc.to(flat.device)
                self.load_acc += hit.sum(dim=(0, 1)).to(self.load_acc.dtype)
            src = card.reshape(-1, card.shape[-1]) if card is not None else flat
            for e in hit.sum(dim=(0, 1)).nonzero().flatten().tolist():
                slot, tok = torch.where(hit[:, :, e].T)               # slot: 第几个被选名额
                y = self.experts[e](src[tok]) * topk_w[tok, slot, None].to(src.dtype)
                out.index_add_(0, tok, y.to(out.dtype))
        out = out.view(B, N, d)
        if self.use_shared_expert:
            return out + self.shared_experts(x)
        return out + x

    @torch.no_grad()
    def update_bias(self, gamma, counts=None):
        """免辅助损失负载均衡(DeepSeek-V3): 过载专家偏置减 gamma, 欠载加 gamma。
        counts 缺省用本层 load_acc; 分布式训练应传入 all-reduce 后的全局计数。"""
        if gamma <= 0:
            return
        c = self.load_acc if counts is None else counts.to(self.load_acc.device)
        self.gate.e_score_correction_bias += gamma * torch.sign(c.mean() - c)

    @torch.no_grad()
    def pop_load(self):
        """读取并清零负载累计。"""
        c = self.load_acc.clone()
        self.load_acc.zero_()
        return c

    @torch.no_grad()
    def pop_dec_stats(self):
        """读取并清零 D-EC 统计向量(布局见 __init__); 分布式下由训练侧 all-reduce 后再汇总。"""
        c = self.dec_acc.clone()
        self.dec_acc.zero_()
        return c


def dec_stats_summary(vec, n_experts):
    """D-EC 统计向量 -> 可读字典: 域内 token 的专家数分布、平均专家数、逐帧容量占比的离散度。"""
    E = n_experts
    hist = vec[:E + 1]
    n_in = float(vec[E + 1])
    tot = float(hist.sum())
    dist = (hist / tot).tolist() if tot > 0 else [0.0] * (E + 1)
    mean_k = float((hist * torch.arange(E + 1, dtype=hist.dtype)).sum() / tot) if tot > 0 else 0.0
    n_pool = float(vec[E + 3])
    return {"k_dist": [round(x, 4) for x in dist], "mean_k": round(mean_k, 4),
            "frac_k0": round(dist[0], 4), "n_in_tokens": int(n_in),
            "frame_share_rel_std": (round(float(vec[E + 2]) / n_pool, 4) if n_pool > 0 else None)}
