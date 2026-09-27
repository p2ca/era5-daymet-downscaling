#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
routing_dump.py — JiT-MoE 采样过程中的路由截获、累加与落盘
============================================================================
采样时网络每次前向都会做一次路由决策; 落场只保留输出场, 决策本身随即丢失, 之后任何
"专家在哪里生效"的分析都得重放整条轨迹。本模块在采样进行时把决策记下来:

  RoutingCapture   运行时补丁: 包住 DSMoE 的三种选择函数(tc / ec / dec), 记录每层的选择
                   掩膜、实际施加的门控权重、亲和分、TC 的 top-K 索引与难度先验; 可选地
                   在每个专家上挂前向钩子, 量"专家输出占 token 隐状态的比例"
  RoutingAccumulator  一条采样轨迹(一个成员一天)的累加器: 逐 token 逐专家的被选次数(总计与
                   按噪声水平 t 分档)、专家数、门控权重、亲和分、输出范数、先验
  save / load      每成员每天一个 npz; 读取时还原为 numpy
  tok_to_pixel     按该轨迹的切块起点把 token 量落回像素

补丁只读不写: 它接住原函数的返回值原样交还, 前向的数值与不装补丁时逐位相同。
自检: 前向次数等于采样器的 NFE; TC 下每个 token 每次前向恰好 K 次选中; D-EC 下由截获
反推的域内 token 专家数直方图必须与模型自身累计的 dec_acc 完全一致。
============================================================================
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from downscaling_4x.models import moe_ffn

T_BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])     # 噪声水平 t 的分档边界, t=1 为数据端


def t_bin(t):
    return int(np.searchsorted(T_BINS, float(t), side="right") - 1)


def tok_to_pixel(tok_field, gh, gw, patch, offset, H, W):
    """(…, gh*gw) 的 token 量 -> (…, H, W) 像素量: 按块复制后用本轨迹的起点裁回。"""
    dy, dx = offset
    f = np.asarray(tok_field).reshape(*np.shape(tok_field)[:-1], gh, gw)
    f = np.repeat(np.repeat(f, patch, -2), patch, -1)
    return f[..., dy:dy + H, dx:dx + W]


class RoutingCapture:
    """包住 DSMoE 的选择函数, 把本次前向每层的路由结果留在 GPU 上供累加器读取。

    属性(每次前向刷新, 键为 MoE 层在 net.moe_layers() 里的序号):
      sel[l]     (T, E) bool  被选掩膜
      w[l]       (T, E) float 实际施加的门控权重(含 routed_scaling_factor)
      scores[l]  (T, E) float sigmoid 亲和分
      idx[l]     (T, K) long  仅 TC: top-K 专家索引(专家钩子要按它还原 token 顺序)
      prior      (T,)   float 难度头先验(仅 D-EC 且 prior 成分开启)

    专家输出范数另按噪声档累一份(norm_sum_t), 用来看"专家在哪个噪声水平上出力最多";
    各档之和必须还原 norm_sum, 落盘前对拍。
    """

    def __init__(self, net, norms=False):
        self.layers = net.moe_layers()
        self.lid = {id(m): k for k, m in enumerate(self.layers)}
        self.mode = self.layers[0].router_mode if self.layers else "tc"
        self.sel, self.w, self.scores, self.idx = {}, {}, {}, {}
        self.prior = None
        self.acc = None                      # 当前累加器(None = 只截获不累加)
        self._orig = None
        self._hooks = []
        if norms:
            for l, m in enumerate(self.layers):
                for e, ex in enumerate(m.experts):
                    self._hooks.append(ex.register_forward_hook(self._expert_hook(l, e)))

    # ---- 补丁 ----
    def install(self):
        cap = self
        orig_route, orig_ec, orig_dec = moe_ffn.DSMoE.route, moe_ffn.DSMoE.route_ec, moe_ffn.DSMoE.route_dec

        def spy_route(self_, scores, *extra, **kw):
            idx, w = orig_route(self_, scores, *extra, **kw)
            l = cap.lid.get(id(self_))
            if l is not None:
                sel = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device).scatter_(1, idx, True)
                cap.sel[l] = sel
                cap.w[l] = torch.zeros(scores.shape, dtype=w.dtype, device=scores.device).scatter_(1, idx, w)
                cap.scores[l] = scores
                cap.idx[l] = idx
            return idx, w

        def spy_ec(self_, scores, B, N):
            sel, w = orig_ec(self_, scores, B, N)
            l = cap.lid.get(id(self_))
            if l is not None:
                cap.sel[l], cap.w[l], cap.scores[l] = sel, w, scores
            return sel, w

        def spy_dec(self_, scores, B, N, tok_mask=None, prior=None):
            sel, w = orig_dec(self_, scores, B, N, tok_mask, prior)
            l = cap.lid.get(id(self_))
            if l is not None:
                cap.sel[l], cap.w[l], cap.scores[l] = sel, w, scores
                if prior is not None:
                    cap.prior = prior.detach().float()
            return sel, w

        moe_ffn.DSMoE.route, moe_ffn.DSMoE.route_ec, moe_ffn.DSMoE.route_dec = spy_route, spy_ec, spy_dec
        self._orig = (orig_route, orig_ec, orig_dec)
        return self

    def uninstall(self):
        if self._orig is not None:
            moe_ffn.DSMoE.route, moe_ffn.DSMoE.route_ec, moe_ffn.DSMoE.route_dec = self._orig
            self._orig = None
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()

    # ---- 专家输出范数钩子 ----
    def _expert_hook(self, l, e):
        cap = self

        def hook(_m, inputs, output):
            acc = cap.acc
            if acc is None or acc.norm_cur is None:
                return
            if cap.mode == "tc":
                idx = cap.idx[l]
                hit = F.one_hot(idx, num_classes=acc.E)
                slot, tok = torch.where(hit[:, :, e].T)                 # 与 DSMoE.forward 的派发顺序一致
                w_tok = cap.w[l][tok, e]
            else:
                tok = cap.sel[l][:, e].nonzero().flatten()
                w_tok = cap.w[l][tok, e]
            if tok.numel() != output.shape[0]:
                raise RuntimeError(f"层 {l} 专家 {e}: 截获的 token 数 {tok.numel()} 与专家实际处理的 {output.shape[0]} 不符")
            x = inputs[0].float()
            y = output.float() * w_tok.float()[:, None]
            ratio = y.norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)
            acc.norm_cur[l].index_put_((tok, torch.full_like(tok, e)), ratio, accumulate=True)
        return hook


class RoutingAccumulator:
    """一条轨迹的路由累加器(GPU 上累加, 落盘时转 numpy)。"""

    def __init__(self, L, T, E, device, scores=True, norms=True):
        nb = len(T_BINS) - 1
        self.L, self.T, self.E, self.nb = L, T, E, nb
        self.sel_cnt = torch.zeros(L, T, E, dtype=torch.int32, device=device)
        self.sel_cnt_t = torch.zeros(L, nb, T, E, dtype=torch.int32, device=device)
        self.k_sum = torch.zeros(L, T, dtype=torch.int32, device=device)
        self.k0_cnt = torch.zeros(L, T, dtype=torch.int32, device=device)
        self.gate_sum = torch.zeros(L, T, E, device=device)
        self.score_sum_t = torch.zeros(L, nb, T, E, device=device) if scores else None
        self.norm_sum = torch.zeros(L, T, E, device=device) if norms else None
        # 专家输出范数按噪声档分开: 钩子在前向途中就写, 那时还不知道本次前向的 t,
        # 因此先写进 norm_cur, 由 add_forward 在拿到 t 之后归档并清零
        self.norm_sum_t = torch.zeros(L, nb, T, E, device=device) if norms else None
        self.norm_cur = torch.zeros(L, T, E, device=device) if norms else None
        self.prior_sum_t = torch.zeros(nb, T, device=device)
        self.has_prior = False
        self.nfwd_t = torch.zeros(nb, dtype=torch.int32, device=device)
        self.nfwd = 0

    def add_forward(self, t, cap):
        b = t_bin(t)
        self.nfwd += 1
        self.nfwd_t[b] += 1
        for l in range(self.L):
            sel = cap.sel[l]
            k = sel.sum(1).to(torch.int32)
            self.sel_cnt[l] += sel
            self.sel_cnt_t[l, b] += sel
            self.k_sum[l] += k
            self.k0_cnt[l] += (k == 0).to(torch.int32)
            self.gate_sum[l] += cap.w[l].float()
            if self.score_sum_t is not None:
                self.score_sum_t[l, b] += cap.scores[l].float()
        if self.norm_cur is not None:
            self.norm_sum += self.norm_cur
            self.norm_sum_t[:, b] += self.norm_cur
            self.norm_cur.zero_()
        if cap.prior is not None:
            self.prior_sum_t[b] += cap.prior
            self.has_prior = True

    def check(self, mode, K):
        """TC: 每 token 每次前向恰好 K 次选中。其它路由: 每次前向的选中总数等于容量(由层自检)。"""
        if mode == "tc":
            per_tok = self.sel_cnt.sum(-1)
            if not bool((per_tok == K * self.nfwd).all()):
                raise RuntimeError("TC 路由截获不自洽: 存在 token 的选中次数不等于 K × 前向数")
        if int(self.sel_cnt.max()) > 255 or int(self.k0_cnt.max()) > 255:
            raise RuntimeError("前向次数超过 uint8 可表示范围, 需调整落盘类型")

    def to_numpy(self, offset, tok_in):
        d = {
            "offset": np.asarray(offset, np.int16),
            "tok_in": np.asarray(tok_in, bool),
            "nfwd": np.int32(self.nfwd),
            "nfwd_t": self.nfwd_t.cpu().numpy().astype(np.uint16),
            "sel_cnt": self.sel_cnt.cpu().numpy().astype(np.uint8),
            "sel_cnt_t": self.sel_cnt_t.cpu().numpy().astype(np.uint8),
            "k_sum": self.k_sum.cpu().numpy().astype(np.uint16),
            "k0_cnt": self.k0_cnt.cpu().numpy().astype(np.uint8),
            "gate_sum": self.gate_sum.cpu().numpy().astype(np.float16),
        }
        if self.score_sum_t is not None:
            d["score_sum_t"] = self.score_sum_t.cpu().numpy().astype(np.float16)
        if self.norm_sum is not None:
            gap = float((self.norm_sum - self.norm_sum_t.sum(1)).abs().max())
            tol = 1e-3 * max(float(self.norm_sum.abs().max()), 1.0)
            if gap > tol:
                raise RuntimeError(f"norm_sum 与 norm_sum_t 分档求和不一致(最大差 {gap:.3g} > {tol:.3g}), "
                                   "说明有前向的专家贡献没有被归档到噪声档")
            d["norm_sum"] = self.norm_sum.cpu().numpy().astype(np.float16)
            d["norm_sum_t"] = self.norm_sum_t.cpu().numpy().astype(np.float16)
        if self.has_prior:
            d["prior_sum_t"] = self.prior_sum_t.cpu().numpy().astype(np.float16)
        return d


# ---------------------------------------------------------------- 落盘与读取
def routing_dir(out):
    return Path(out) / "routing"


def routing_path(out, y, day, member):
    return routing_dir(out) / f"{y}_d{day}_m{member}.npz"


def save_routing(out, y, day, member, arrays):
    d = routing_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{y}_d{day}_m{member}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(routing_path(out, y, day, member))


def write_routing_meta(out, info):
    d = routing_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(d / "meta.json", "w"), indent=1, ensure_ascii=False)


def load_routing_meta(out):
    p = routing_dir(out) / "meta.json"
    return json.load(open(p)) if p.exists() else None


def load_routing(out, y, day, member):
    """读一个成员一天的路由; 计数还原为 int, 和还原为 float32。"""
    z = np.load(routing_path(out, y, day, member))
    d = {k: z[k] for k in z.files}
    for k in ("sel_cnt", "sel_cnt_t", "k_sum", "k0_cnt", "nfwd_t"):
        d[k] = d[k].astype(np.int32)
    for k in ("gate_sum", "score_sum_t", "norm_sum", "norm_sum_t", "prior_sum_t"):
        if k in d:
            d[k] = d[k].astype(np.float32)
    d["nfwd"] = int(d["nfwd"])
    return d


def available_routing(out):
    """已落盘的 (year, day, member) 列表。"""
    items = []
    for p in routing_dir(out).glob("*_d*_m*.npz"):
        stem = p.stem
        y, rest = stem.split("_d")
        day, m = rest.split("_m")
        items.append((int(y), int(day), int(m)))
    return sorted(items)
