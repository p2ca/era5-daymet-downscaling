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
============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """专家用 SwiGLU: w12 = d->2S 合并投影, w3 = S->d。S 不做 2/3 缩放。"""

    def __init__(self, dim, inter, drop=0.0, bias=True):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * inter, bias=bias)
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


class DSMoE(nn.Module):
    """共享专家 + 路由专家的稀疏 FFN 层。

    forward 输入 (B, N, d), 输出同形。同时把本次各专家命中 token 数累加进 load_acc
    (buffer, 不入 checkpoint), 由训练侧读取/清零, 用于负载监控与免辅助损失偏置更新。
    """

    def __init__(self, num_experts, dim, moe_inter, num_experts_per_tok=2,
                 n_group=2, topk_group=2, norm_topk_prob=True,
                 routed_scaling_factor=2.5, use_shared_expert=True, proj_drop=0.0):
        super().__init__()
        assert num_experts % n_group == 0
        self.n_experts, self.top_k = num_experts, num_experts_per_tok
        self.n_group, self.topk_group = n_group, topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.use_shared_expert = use_shared_expert
        self.experts = nn.ModuleList(
            [SwiGLUExpert(dim, moe_inter, proj_drop) for _ in range(num_experts)])
        self.gate = TopkRouter(dim, num_experts)
        if use_shared_expert:
            self.shared_experts = SwiGLUExpert(dim, moe_inter, proj_drop)
        # 负载计数是逐 rank 的本地统计, 存普通张量而非 buffer: DDP 的 broadcast_buffers
        # 会在每次前向把 rank0 的 buffer 覆盖到所有 rank, 会静默污染各 rank 的计数
        self.load_acc = torch.zeros(num_experts)

    def route(self, scores):
        """scores: (T, E) 的 sigmoid 亲和分 -> (top-K 专家索引, 门控权重)。"""
        for_choice = scores + self.gate.e_score_correction_bias
        # 分组路由: 每组取前 2 名分数之和作为组分, 选 topk_group 个组, 组外专家不参选。
        # 默认配置选满所有组, 此段为无操作, 保留 DeepSeek-V3 的结构。
        gsize = self.n_experts // self.n_group
        group_scores = for_choice.view(-1, self.n_group, gsize).topk(
            min(2, gsize), dim=-1)[0].sum(dim=-1)
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

    def forward(self, x):
        B, N, d = x.shape
        flat = x.reshape(-1, d)
        # 路由固定 fp32: 显式关闭 autocast(否则 bf16 训练下 F.linear 会被打回 bf16,
        # 亲和分与 top-K 选择带上量化噪声)。专家计算仍走环境精度。
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.sigmoid(self.gate(flat))                   # (T, E) fp32
            topk_idx, topk_w = self.route(scores)
        with torch.no_grad():
            if self.load_acc.device != flat.device:
                self.load_acc = self.load_acc.to(flat.device)
            self.load_acc += torch.bincount(
                topk_idx.reshape(-1), minlength=self.n_experts).to(self.load_acc.dtype)
        out = torch.zeros_like(flat)
        # dropless 派发: 逐命中专家收集其 token, 计算后按门控权重累加回原位。
        hit = F.one_hot(topk_idx, num_classes=self.n_experts)         # (T, K, E)
        for e in hit.sum(dim=(0, 1)).nonzero().flatten().tolist():
            slot, tok = torch.where(hit[:, :, e].T)                   # slot: 第几个被选名额
            y = self.experts[e](flat[tok]) * topk_w[tok, slot, None].to(flat.dtype)
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
