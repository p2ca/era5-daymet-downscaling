#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
train_jit.py — 整幅像素空间条件扩散 (JiT / JiTMoE)
============================================================================
单目标条件生成 p(y | 21 通道条件场): 不做残差分解、不依赖回归均值模型, 整幅
720x1440 直接训练。范式与超参遵循 JiT (arXiv:2511.13720):

  z = t*y + (1-t)*noise_scale*e,  t ~ sigmoid(N(P_mean, P_std^2)),  t=1 为数据端
  网络输出 x 预测, 损失在 v 空间: || (y - x_hat) / max(1-t, t_eps) ||^2

口径约定:
  - 目标场在海洋置 0 后参与加噪(z 的统计处处良定), 损失默认只在陆地归一化;
    采样侧配合把每步 x 预测的海洋区钳 0(见 models/jit_sampler.py)。
  - lr 全程不衰减: --lr 给绝对值, 否则按 blr * 全局批 / 256 线性缩放;
    --warmup-samples > 0 时先逐步线性升到该值, 为 0 则从第一步起就是目标值。
    AdamW(0.9, 0.95), 无权重衰减, 无梯度裁剪(--grad-clip 默认 1e6 仅作范数监控)。
  - 每步维护两份参数 EMA(采样默认用 ema1)。EMA 只覆盖参数; MoE 路由偏置是 buffer,
    随 model state 保存, 导出 EMA 权重采样时由加载方从 model state 取 buffer。
  - 取帧为全局序号决定的无放回洗牌流(与其余整幅训练一致), 断点续训逐帧精确;
    扩散的 (t, 噪声) 走逐 rank 专属 Generator, --save-rng 时随断点入盘 -> 续训后
    随机流逐位连续, 且不受任何库消耗全局随机流的影响。
  - --moe 时偶数序块(索引 1,3,...)的 FFN 换成 DeepSeek 风格稀疏层; 逐层专家负载
    全程记录进损失曲线; --bias-gamma > 0 启用免辅助损失负载均衡。
  - --router dec 为 D-EC(难度感知的池化专家选择, 语义见 models/moe_ffn.py): 有效域掩膜
    随前向传入路由; 难度头以本步逐 token 训练损失为监督(Huber, 只更新头); 先验系数按
    samples/duration 从 0 线性退火到 --dec-prior-max; 三个成分由 --dec-pool/drop/prior 独立开关。
============================================================================
"""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.models.jit_backbone import JiT, draw_patch_offset, token_pool
from downscaling_4x.models.moe_ffn import dec_pools, dec_standardize, dec_stats_summary
from downscaling_4x import contract as C
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.training import train_downscale as TD
from downscaling_4x.evaluation import input_identity as II


class JitFrameStream(torch.utils.data.Dataset):
    """整幅帧流: 每个样本 = (cond, 单目标 target, land)。

    全局序号 g = index_offset + i 唯一决定帧: 第 g//n 遍数据用只依赖 (seed, 遍数) 的
    置换取第 g%n 帧。跨 rank 分片互不重叠、无放回; deterministic=True 时用固定置换,
    供逐次完全一致的验证。
    """

    def __init__(self, data, fi, target, length, seed, index_offset, deterministic=False,
                 cache=None):
        self.d, self.fi, self.cache, self.target = data, fi, cache, target
        self.ti = C.TARGETS.index(target)
        self.len, self.base_seed = int(length), int(seed)
        self.index_offset, self.deterministic = int(index_offset), deterministic
        self.perm = (np.random.default_rng(self.base_seed).permutation(len(fi))
                     if deterministic else None)
        self._pid, self._pperm = -1, None

    def __len__(self):
        return self.len

    def _perm_for(self, k):
        if self._pid != k:
            self._pid = k
            self._pperm = np.random.default_rng(
                np.random.SeedSequence([self.base_seed, int(k)])).permutation(len(self.fi))
        return self._pperm

    def frame_index(self, i):
        """样本序号 -> 帧集合里的下标。索引数学集中于此, 供分片覆盖检验直接调用。"""
        n = len(self.fi)
        if self.deterministic:
            return int(self.perm[(self.index_offset + int(i)) % n])
        g = self.index_offset + int(i)
        return int(self._perm_for(g // n)[g % n])

    def frame_of(self, i):
        return self.fi.frames[self.frame_index(i)]

    def __getitem__(self, i):
        k = self.frame_index(i)
        y, t = self.fi.frames[k]
        cond, tgt, mask, _ = self.d.full(y, t, self.fi.history_of(k))
        land = (mask[0] > 0.5).astype(np.float32)[None]
        # 阶段 B 才需要 μ; 单阶段返回一个占位标量, 使 batch 结构恒定
        mu = (self.cache.get(self.target, y, t)[None] if self.cache is not None
              else np.zeros(1, np.float32))
        return (torch.from_numpy(cond), torch.from_numpy(tgt[self.ti:self.ti + 1]),
                torch.from_numpy(land), torch.from_numpy(mu))


def make_residual(tgt, mu, land, cond, sigma_r):
    """(已在域外置零的 y, μ, 掩膜, 条件, σ_r) -> (残差目标, 拼上 μ 的条件)。

    r = (y - μ)/σ_r, 其中 μ 先在有效域外钉 0 —— 它在那里从未被监督, 原样进残差会让扩散网
    去拟合一片无意义的值; 掩膜只挡得住 loss, 挡不住残差作为**网络输入**经感受野进入域内
    边缘。钉 0 后残差在域外恒为 0, 语义是"均值已说完, 无附加"。

    σ_r 把残差归一到单位方差: 扩散侧的噪声调度按"数据方差约为 1"调好, 而残差的方差远小
    于 1, 不归一会让整条信噪比曲线偏掉 —— 不报错, 只是训不好。

    提成独立函数是为了能被单独对拍: 尺度、域外置零、μ 进条件这三件事写错都不会报错。
    """
    mu = mu * land
    return (tgt - mu) / sigma_r, torch.cat([cond, mu], dim=1)


def jit_vloss(net, tgt, cond, weight, noise_scale, p_mean, p_std, t_eps, generator=None,
              patch=None, domain_mask=None, dec_aux=False):
    """JiT 训练损失。恒等式 (v - v_pred) = (y - x_hat)/(1-t) 使 v 差可由 x 差直接算出。
    t 与噪声取自 generator(训练用专属流并随断点入盘 -> 续训逐位连续, 且不受其他库
    消耗全局随机流的影响; 验证用逐批固定种子的临时流)。

    domain_mask: D-EC 模型的有效域掩膜 (B,1,H,W), 随前向进路由; 非 D-EC 模型传 None,
    调用与改动前逐字相同。dec_aux=True 时(D-EC 的 prior 成分开着)返回 (扩散损失, 辅助损失,
    统计字典), 两项分开: 记录、验证与 ckpt 判据只用扩散损失, 反传时由调用方相加; 否则只返回损失。"""
    B = tgt.shape[0]
    t = torch.sigmoid(torch.randn(B, device=tgt.device, generator=generator)
                      * p_std + p_mean)
    tb = t.view(B, 1, 1, 1)
    e = torch.randn(tgt.shape, device=tgt.device, dtype=tgt.dtype,
                    generator=generator) * noise_scale
    z = tb * tgt + (1.0 - tb) * e
    off = draw_patch_offset(patch, tgt.device, generator)   # 切块起点与 (t,噪声) 同流
    if dec_aux:
        x_hat, info = net(z, t, cond, offset=off, domain_mask=domain_mask, return_dec=True)
    elif domain_mask is not None:
        x_hat = net(z, t, cond, offset=off, domain_mask=domain_mask)
    else:
        x_hat = net(z, t, cond, offset=off)
    diff = (tgt - x_hat.float()) / (1.0 - tb).clamp_min(t_eps)
    loss = (diff.square() * weight).sum() / weight.sum().clamp_min(1.0)
    if not dec_aux:
        return loss
    aux, extra = dec_aux_loss(net, info, diff.detach(), weight, patch, off)
    return loss, aux, extra


def dec_aux_loss(net, info, diff, weight, patch, offset):
    """难度头的监督项。

    目标 = 本次前向的逐像素训练损失 diff²·weight 按与路由**同一套** token 网格池化成逐 token
    均值, 取 log, 再在路由同一个池(批级或帧内)的域内 token 上标准化; 头的预测 ĥ 与之做
    Huber 回归。头的输入在前向里已 detach, 目标在此 detach, 所以这一项只更新头。
    返回 (辅助损失, {"aux", "corr"}), corr 是本批 ĥ 与目标的 Pearson 相关(头的校准度)。
    """
    m = getattr(net, "module", net)
    pred = info["pred"]
    zero = torch.zeros((), device=diff.device)
    if pred is None:
        return zero, {}
    B, N = pred.shape
    patch = patch or m.patch
    num = token_pool(diff.square() * weight, patch, offset, m.grid_hw)     # (B, N)
    den = token_pool(weight, patch, offset, m.grid_hw)
    valid = (den > 0).reshape(-1)
    if info["tok_mask"] is not None:
        valid = valid & info["tok_mask"].reshape(-1)
    if int(valid.sum()) < 2:
        return zero, {}
    err = (num / den.clamp_min(1e-12)).reshape(-1)
    pool_batch = m.dec["pool"] if m.training else (m.moe_layers()[0].dec_eval_mode == "pool")
    target = dec_standardize(torch.log(err + 1e-6), valid, dec_pools(B, N, pool_batch))
    p = pred.reshape(-1)[valid]
    tgt = target[valid]
    aux = F.huber_loss(p, tgt, delta=1.0)
    with torch.no_grad():
        pc, tc = p - p.mean(), tgt - tgt.mean()
        corr = float((pc * tc).sum() / (pc.norm() * tc.norm() + 1e-12))
    return aux, {"aux": float(aux.detach()), "corr": corr}


def _rng_capture(rank, gen):
    st = {"rank": rank, "gen": gen.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state()
    return st


def _rng_gather(is_dist, world, st):
    if not is_dist:
        return [st]
    lst = [None] * world
    dist.all_gather_object(lst, st)
    return lst


def _fingerprint(net, grads=False):
    """逐参数(或其梯度)与 buffer 的 (sum, |sum|, sq-sum) 指纹, 供跨 rank 一致性自检。
    梯度缺席记为 NaN 三元组 —— 缺席模式本身也必须全 rank 一致, 否则优化器步进会分叉。"""
    vals = []
    for _, p in net.named_parameters():
        t = p.grad if grads else p
        if t is None:
            vals += [float("nan"), float("nan"), float("nan")]
        else:
            td = t.detach().double()
            vals += [float(td.sum()), float(td.abs().sum()), float(td.square().sum())]
    if not grads:
        for _, b in net.named_buffers():
            if b.dtype.is_floating_point:
                td = b.detach().double()
                vals += [float(td.sum()), float(td.abs().sum()),
                         float(td.square().sum())]
    return torch.tensor(vals, dtype=torch.float64)


def _assert_ranks_synced(vec, device, world, what):
    """all_gather 指纹并逐 rank 比对; 不一致说明 DDP 同步失效, 必须立即中止而不是
    带着单卡血统的权重跑完全程。"""
    v = vec.to(device)
    outs = [torch.empty_like(v) for _ in range(world)]
    dist.all_gather(outs, v)
    for k in range(1, world):
        if not torch.allclose(outs[0], outs[k], rtol=0, atol=1e-10, equal_nan=True):
            bad = int((~torch.isclose(outs[0], outs[k], rtol=0, atol=1e-10,
                                      equal_nan=True)).sum())
            raise RuntimeError(f"[jit] rank0 与 rank{k} 的{what}不一致 "
                               f"({bad} 项越界) —— DDP 同步失效, 中止训练")


def _ema_init(net):
    return {n: p.detach().clone().float() for n, p in net.named_parameters()}


@torch.no_grad()
def _ema_update(ema, net, decay):
    for n, p in net.named_parameters():
        ema[n].mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)


def _drain_moe_load(layers, is_dist):
    """取出并清零各 MoE 层的专家命中计数, 分布式下聚成全局计数。返回 (L, E) 或 None。"""
    if not layers:
        return None
    c = torch.stack([m.pop_load() for m in layers])
    if is_dist:
        dist.all_reduce(c)
    return c


def _drain_dec_stats(net, is_dist):
    """取出并清零各 MoE 层的 D-EC 统计, 分布式下聚成全局计数。返回 (L, E+4) 或 None。"""
    v = net.pop_dec_stats()
    if v is None:
        return None
    if is_dist:
        dist.all_reduce(v)
    return v


def build_model(a, hw, data=None):
    """按参数字典构建 JiT(a 取 vars(args) 或 checkpoint 里保存的 args)。
    训练与离线采样/评测共用, 保证从 checkpoint 重建的结构与训练时逐项一致。

    data=(Stats, DownscaleData) 只在训练时给: 两条流模型要把风通道的统计量与 ERA5 有效掩膜写进
    持久 buffer; 采样/评测从 checkpoint 恢复这些 buffer, 不需要 data。"""
    get = a.get if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    moe_config = TD.jit_moe_config(a)
    arch = TD.jit_stream_config(a)
    refine = int(get("refine_head", 0) or 0)
    ridx = None
    if refine:
        layout = C.cond_layout(get("mode", C.DEFAULT_MODE))
        ridx = [layout.index(n) for n in C.STATIC_ORDER]
    net = JiT(hw=hw, patch=a["patch"], cond_ch=a["cond_ch"], out_ch=1,
              patch_margin=a.get("patch_margin", 0),
              hidden=a["hidden"], depth=a["depth"], num_heads=a["heads"],
              mlp_ratio=a["mlp_ratio"], bottleneck=a["bottleneck"],
              attn_drop=a["attn_dropout"], proj_drop=a["proj_dropout"],
              moe_config=moe_config, refine_head=refine, refine_static_idx=ridx,
              arch=arch, mode=get("mode", C.DEFAULT_MODE))
    if data is not None and net.two_stream:
        stats, dd = data
        idx = [stats.in_vars.index(v) for v in JiT.WIND_VARS]
        net.set_data_constants(stats.e_mean[idx], stats.e_std[idx], dd.era5_valid_hr)
    return net


# 续训时必须与 checkpoint 逐项一致的参数。noise_scale / lr / batch 一类标量不改变
# 任何参数形状, 在续训段写错只会静默换口径; 预算类参数(duration / max_seconds)不在此列。
RESUME_PINNED_ARGS = (
    "target", "mode", "mu_cache", "stage_a_ckpt", "residual_scale",
    "era5_dir", "daymet_dir", "train_years", "val_years",
    "cond_ch", "hidden", "depth", "heads", "patch", "patch_margin", "bottleneck",
    "mlp_ratio", "attn_dropout", "proj_dropout",
    "moe", "experts", "experts_per_tok", "moe_intermediate", "routed_scaling",
    "moe_all_layers", "moe_no_shared", "bias_gamma", "refine_head", "router", "gating",
    "dec_pool", "dec_drop", "dec_prior", "dec_prior_max", "dec_prior_ramp", "dec_head_layer",
    *TD.ARCH_DEFAULTS.keys(),
    "p_mean", "p_std", "noise_scale", "t_eps",
    "lr", "blr", "warmup_samples", "wd", "ema1", "ema2", "batch", "grad_clip",
    "seed", "save_rng", "loss_scope", "val_steps",
)


def main(argv=None, data=None):
    p = argparse.ArgumentParser(description="JiT / JiTMoE 整幅条件扩散训练")
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--val-years", type=int, nargs="+", default=M.splits["val"])
    p.add_argument("--mode", default=C.DEFAULT_MODE, choices=sorted(C.MODES))
    # --- 两阶段(阶段 B): 给了 --mu-cache 就训残差, 否则是单阶段 JiT ---
    p.add_argument("--mu-cache", default="", help="阶段 A 的 μ 缓存目录; 给了即进入残差模式")
    p.add_argument("--stage-a-ckpt", default="", help="阶段 A checkpoint, 供 μ 缓存做 SHA 校验")
    p.add_argument("--residual-scale", default="",
                   help="σ_r: 一个数, 或 residual_scale.py 落的 json 路径")
    p.add_argument("--cond-ch", type=int, default=0,
                   help="条件通道数; 默认取数据合同, 只有合成数据自测才需要覆盖")
    # 结构
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch", type=int, default=32,
                   help="切块边长; 网格会补到不小于 H+patch-1 的最小可整除尺寸, 故不要求整除")
    p.add_argument("--patch-margin", type=int, default=8,
                   help="每块向四周多读的像素数(重叠只在读、不在写); 0 为不重叠")
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--proj-dropout", type=float, default=0.0)
    # MoE
    p.add_argument("--moe", action="store_true", help="偶数序块 FFN 换成 DeepSeek 稀疏层")
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--experts-per-tok", type=int, default=2)
    p.add_argument("--moe-intermediate", type=int, default=0, help="0 -> 2*hidden")
    p.add_argument("--routed-scaling", type=float, default=2.5)
    p.add_argument("--moe-all-layers", action="store_true", help="全部块用 MoE(消融)")
    p.add_argument("--moe-no-shared", action="store_true", help="去共享专家(消融)")
    p.add_argument("--bias-gamma", type=float, default=0.0,
                   help="免辅助损失负载均衡的偏置步长; 0 = 偏置恒零(只记录负载); "
                        "ec 路由下必须为 0(构造性均衡)")
    p.add_argument("--router", choices=["tc", "ec", "dec"], default="tc",
                   help="专家选择方向: tc=每 token 挑专家 / ec=每专家帧内挑 token(FLOPs 对齐) / "
                        "dec=D-EC(难度感知的池化专家选择, 成分见 --dec-*)")
    p.add_argument("--gating", choices=["norm", "raw"], default="norm",
                   help="门控权重: norm=当选归一(tc 固定此档) / raw=sigmoid 原值(仅 ec/dec)")
    # D-EC 三成分开关缺省全开; 只在 --router dec 下生效, 其余路由下给非缺省值会被拒绝
    p.add_argument("--dec-pool", type=int, choices=[0, 1], default=1,
                   help="[dec] 批级池: 专家在本 rank 本步的全部帧上挑 token(跨帧); 0=帧内池")
    p.add_argument("--dec-drop", type=int, choices=[0, 1], default=1,
                   help="[dec] 域外丢弃: 有效域外 token 只走共享专家, 容量按域内 token 数计")
    p.add_argument("--dec-prior", type=int, choices=[0, 1], default=1,
                   help="[dec] 难度先验: 难度头的逐 token 预测(标准化后)乘系数进选择分")
    p.add_argument("--dec-prior-max", type=float, default=1.0,
                   help="[dec] 先验系数上限, 单位是池内亲和分的标准差")
    p.add_argument("--dec-prior-ramp", type=float, default=0.1,
                   help="[dec] 系数从 0 线性升满所用的 duration 比例; 0 = 从第一步起就是上限")
    p.add_argument("--dec-head-layer", type=int, default=1,
                   help="[dec] 难度头之前经过的 block 数(1 = 读第 0 块之后的隐状态); 须不大于首个 MoE 块序号")
    p.add_argument("--refine-head", type=int, choices=[0, 1], default=0,
                   help="去噪器输出端全分辨率卷积精修头(静态通道直连), 逐去噪步生效; 0=关")
    # 两条流结构: 每个部件一个开关, 缺省 = 单流基座; 依赖关系见 train_downscale.jit_stream_config
    D = TD.ARCH_DEFAULTS
    p.add_argument("--rope-units", choices=["grid", "km"], default=D["rope_units"],
                   help="位置章单位: grid=行列序号 / km=一个南北 token 间距为 1, 东西向乘 cos(纬度)")
    p.add_argument("--drop-outside", type=int, choices=[0, 1], default=D["drop_outside"],
                   help="[tc] 域外 token 不进路由专家, 只走共享专家")
    p.add_argument("--two-stream", type=int, choices=[0, 1], default=D["two_stream"],
                   help="分流: 细流 = 噪声目标+静态+年内相位, 粗流 = ERA5 动态通道, 各自一套 patch 嵌入")
    p.add_argument("--coarse-doy", type=int, choices=[0, 1], default=D["coarse_doy"],
                   help="[two-stream] 粗流也读年内相位两个通道")
    p.add_argument("--coarse-conv", type=int, choices=[0, 1], default=D["coarse_conv"],
                   help="[two-stream] 粗流 token 网格上的 3x3 卷积(邻格差分)")
    p.add_argument("--coarse-blocks", type=int, default=D["coarse_blocks"],
                   help="[two-stream] 粗流的自注意力+MLP 块数; 0 = 只做嵌入")
    p.add_argument("--cross-attn", type=int, choices=[0, 1], default=D["cross_attn"],
                   help="[two-stream] 每个细流块里向粗 token 的交叉问询(Q 细, K/V 粗)")
    p.add_argument("--cross-modulate", type=int, choices=[0, 1], default=D["cross_modulate"],
                   help="[cross-attn] 交叉问询的输入是否经旋钮平移缩放调制")
    p.add_argument("--cross-mask-outside", type=int, choices=[0, 1], default=D["cross_mask_outside"],
                   help="[cross-attn] ERA5 无数据的粗 token 不参与作答")
    p.add_argument("--lagrangian", type=int, choices=[0, 1], default=D["lagrangian"],
                   help="[cross-attn, km] K 的位置按该粗 token 的日均风推移 τ 小时, 逐头见 --tau-spec")
    p.add_argument("--tau-spec", default=D["tau_spec"],
                   help="[lagrangian] 每头一项, 逗号分隔: 0 = 不推, 层:小时 = 按 850/500 hPa 风推该小时数")
    p.add_argument("--expert-two-card", type=int, choices=[0, 1], default=D["expert_two_card"],
                   help="[two-stream, moe] 路由专家读 [细 token 投影 ‖ 正上方粗 token 投影]")
    p.add_argument("--dense-two-card", type=int, choices=[0, 1], default=D["dense_two_card"],
                   help="[two-stream] 稠密块的 FFN 读同样的两张卡")
    p.add_argument("--two-card-dims", default=D["two_card_dims"],
                   help="[two-card] 细、粗两张卡各投到多少维再拼接, 形如 192,192")
    p.add_argument("--wards", type=int, default=D["wards"],
                   help="[tc] 两级分诊的科室数(须整除专家数); 1 = 不分科室")
    p.add_argument("--ward-topk", type=int, default=D["ward_topk"],
                   help="[wards>1] 第一级选几个科室")
    p.add_argument("--ward-key", type=int, choices=[0, 1], default=D["ward_key"],
                   help="[wards>1, two-stream] 正上方粗 token 的投影经零初始化矩阵进科室分")
    p.add_argument("--terrain-key", type=int, choices=[0, 1], default=D["terrain_key"],
                   help="[tc] 该块静态场单独嵌成的地形键经零初始化矩阵进专家分")
    p.add_argument("--terrain-key-dim", type=int, default=D["terrain_key_dim"])
    p.add_argument("--terrain-key-window", type=int, default=D["terrain_key_window"],
                   help="[terrain-key] 地形键读取的像素窗口边长(>= patch, 与 patch 同奇偶)")
    # 扩散
    p.add_argument("--p-mean", type=float, default=-0.8)
    p.add_argument("--p-std", type=float, default=0.8)
    p.add_argument("--noise-scale", type=float, default=4.0,
                   help="噪声幅度; 参考口径为等效边长/256")
    p.add_argument("--t-eps", type=float, default=0.05)
    # 训练
    p.add_argument("--lr", type=float, default=0.0, help="绝对学习率; 0 -> 用 --blr 缩放")
    p.add_argument("--blr", type=float, default=5e-5, help="lr = blr * 全局批 / 256")
    p.add_argument("--warmup-samples", type=int, default=70_000)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--ema1", type=float, default=0.9999)
    p.add_argument("--ema2", type=float, default=0.9996)
    p.add_argument("--batch", type=int, default=1, help="每 rank 每步帧数")
    p.add_argument("--grad-clip", type=float, default=1e6)
    p.add_argument("--duration", type=int, default=8_000_000, help="processed samples 上限")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="超时保存断点并干净退出; 0 表示只按 duration 停")
    p.add_argument("--ckpt-every", type=int, default=8192)
    p.add_argument("--snap-every", type=int, default=524_288,
                   help="model+ema1 快照 snap_*.pt 的间隔(0 关闭)")
    p.add_argument("--save-rng", type=int, default=1)
    p.add_argument("--val-every", type=int, default=8192)
    p.add_argument("--val-steps", type=int, default=4, help="每 rank 的验证批数")
    p.add_argument("--loss-scope", choices=["land", "all"], default="land")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    II.add_arg(p)
    args = p.parse_args(argv)

    rank, world, local, device, is_dist = TD.setup_ddp()
    is_main = rank == 0
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + rank)          # 全局流只用于权重初始化(DDP 构造时统一)
    gen = torch.Generator(device=device)         # 扩散 (t, 噪声) 的专属流, 逐 rank 独立
    gen.manual_seed(args.seed * 100_003 + rank)

    if data is None:
        stats = Stats(args.era5_dir, args.daymet_dir)
        need, lags = C.pairing_history_days(args.mode), C.history_lags(args.mode)
        avail = sorted(set(args.train_years) | set(args.val_years))
        tr_fi = FrameIndex(args.train_years, avail, need, lags, split='train')
        va_fi = FrameIndex(args.val_years, avail, need, lags, split='val')
        TD.assert_same_frames_across_ranks(tr_fi, device, is_dist, 'jit-train')
        TD.assert_same_frames_across_ranks(va_fi, device, is_dist, 'jit-val')
        yrs = lambda fi: sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})
        tr_years, va_years = yrs(tr_fi), yrs(va_fi)
        tr = DownscaleData(args.era5_dir, args.daymet_dir, tr_years, stats,
                           mode=args.mode, era5_cache_years=len(tr_years))
        va = DownscaleData(args.era5_dir, args.daymet_dir, va_years, stats,
                           mode=args.mode, era5_cache_years=len(va_years))
        # ERA5 年数据在主进程一次性载满, fork 后由写时复制供全部 worker 共享。
        # 不预载的话每个 worker 都会私有地积累全部年份的缓存, 宿主内存按 worker 数放大,
        # 多节点整幅训练必触内核 OOM —— 单 rank 被无声击杀, 其余 rank 卡在集合通信直到
        # 看门狗超时, 日志里只剩一句 Force Terminated。各 rank 从不同年份起步轮转,
        # 避免全体同序挤同一个单 OST 年文件。
        tp = time.time()
        off = rank % max(1, len(tr_years))
        for y in tr_years[off:] + tr_years[:off]:
            tr._era5_year(y)
        off = rank % max(1, len(va_years))
        for y in va_years[off:] + va_years[:off]:
            va._era5_year(y)
        if is_main:
            gb = (len(C.ERA5_IN) * C.DAYS_PER_YEAR
                  * C.LR_SHAPE[0] * C.LR_SHAPE[1] * 4 / 1e9)
            print(f"[jit] 预载 ERA5 train {len(tr_years)} 年 + val {len(va_years)} 年 "
                  f"≈ {(len(tr_years) + len(va_years)) * gb:.1f} GB/rank, "
                  f"{time.time() - tp:.0f}s", flush=True)
    else:
        tr, va = data
    H, W = tr.H, tr.W

    # ---- 两阶段: 残差模式 ----
    mu_cache, sigma_r = None, 1.0
    if args.mu_cache:
        from downscaling_4x.data.mu_cache import MuCache
        mu_cache = MuCache(args.mu_cache, [args.target])
        if args.stage_a_ckpt:
            mu_cache.verify({args.target: args.stage_a_ckpt})
        II.check(mu_cache.manifest.get("era5_dir"), args.era5_dir, f"μ 缓存 {args.mu_cache}",
                 allow=getattr(args, II.ALLOW_DEST, False))
        cm = mu_cache.manifest.get("mode") or C.DEFAULT_MODE
        if cm != args.mode:
            raise SystemExit(f"μ 缓存按 --mode {cm} 建, 与本次 {args.mode} 不符")
        if not args.residual_scale:
            raise SystemExit("残差模式必须给 --residual-scale: 残差方差远小于 1, "
                             "不归一会让噪声调度整体偏掉, 而且不会报错")
        if Path(args.residual_scale).is_file():
            sc = json.loads(Path(args.residual_scale).read_text())
            if sc.get("target") != args.target:
                raise SystemExit(f"σ_r 文件是 {sc.get('target')} 的, 与 --target 不符")
            if not sc.get("complete", True):
                print(f"[jit] ⚠ σ_r 由 stride={sc.get('stride')} 抽样估得, 非完整统计", flush=True)
            sigma_r = float(sc["residual_std"])
        else:
            sigma_r = float(args.residual_scale)
        if not (sigma_r > 0):
            raise SystemExit(f"σ_r 必须为正, 得到 {sigma_r}")
        if is_main:
            print(f"[jit] 残差模式: μ 缓存 {args.mu_cache}  σ_r={sigma_r:.5f}", flush=True)

    want_cin = C.cond_channels(args.mode) + (1 if mu_cache is not None else 0)
    if not args.cond_ch:
        args.cond_ch = want_cin
    elif args.cond_ch != want_cin:
        raise SystemExit(f'--cond-ch {args.cond_ch} 与 --mode {args.mode}'
                         f"{'(+μ)' if mu_cache is not None else ''} 的 {want_cin} 不符")
    net = build_model(vars(args), (H, W), data=((stats, tr) if data is None else None)).to(device)
    # broadcast_buffers=False: 本模型的 buffer 要么恒定(位置编码), 要么由构造保证全
    # rank 一致(路由偏置从全局 all-reduce 计数更新); 默认的逐前向广播只会掩盖潜在的
    # 不一致而非修复它。find_unused_parameters 仅 MoE 需要(专家非每步全命中)。
    model = (DDP(net, device_ids=([local] if device.type == 'cuda' else None),
                 find_unused_parameters=args.moe,
                 broadcast_buffers=False) if is_dist else net)
    moe_layers = net.moe_layers()

    global_batch = world * args.batch
    lr = args.lr if args.lr > 0 else args.blr * global_batch / 256
    if is_main:
        pc = net.param_counts()
        gh, gw = net.x_embedder.gh, net.x_embedder.gw
        print(f"[jit] target={args.target} 参数 {pc['total']:,} "
              f"(激活 {pc['activated']:,} / 路由专家 {pc['routed_experts']:,}) "
              f"token {gh}x{gw}={gh*gw} world={world} batch/rank={args.batch}", flush=True)
        how = ("显式给定" if args.lr > 0
               else f"blr {args.blr:.1e} x 全局批 {global_batch}/256")
        ramp = (f"warmup {args.warmup_samples:,} samples 后恒定"
                if args.warmup_samples > 0 else "全程恒定, 无 warmup")
        print(f"[jit] lr={lr:.2e} ({how}; {ramp}) "
              f"noise_scale={args.noise_scale} P_mean={args.p_mean} moe={args.moe} "
              f"duration={args.duration:,} samples", flush=True)
    if is_main and net.arch:
        print(f"[jit] arch={json.dumps(net.arch, ensure_ascii=False)}", flush=True)
    dec_on = net.dec is not None
    dec_prior_on = bool(dec_on and net.dec["prior"])
    if is_main and dec_on:
        d = net.dec
        print(f"[dec] pool={int(d['pool'])} drop={int(d['drop'])} prior={int(d['prior'])} "
              f"λ_max={d['prior_max']} ramp={d['prior_ramp']} head_layer={d['head_layer']} | "
              f"训练池={'本 rank ' + str(args.batch) + ' 帧' if d['pool'] else '帧内'}; "
              f"推理规则=帧内 top-C", flush=True)
        if d["pool"] and args.batch == 1:
            print("[dec] 警告: batch/rank=1, 批级池退化为帧内池, pool 成分不起作用", flush=True)

    per_step = global_batch
    steps_total = args.duration // per_step
    ck = None
    if args.resume:
        if not Path(args.resume).exists():
            raise SystemExit(f"--resume 指定的断点不存在: {args.resume}")
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        filled = TD.apply_legacy_defaults(ck.get("args"), TD.LEGACY_DEFAULTS_JIT)
        if filled and is_main:
            print(f"[jit] 旧断点缺口径键 {filled}, 按默认档解释", flush=True)
        TD.check_resume_args(ck, args, RESUME_PINNED_ARGS, world=world,
                             path_keys=("era5_dir", "daymet_dir"))
    done_steps = (ck["samples"] // per_step) if ck else 0
    if done_steps >= steps_total:
        raise SystemExit(f"断点已达 {ck['samples']:,} samples >= duration {args.duration:,}")

    ds = JitFrameStream(tr, tr_fi, args.target, (steps_total - done_steps) * args.batch, 1234,
                        (rank * steps_total + done_steps) * args.batch, cache=mu_cache)
    vs = JitFrameStream(va, va_fi, args.target, args.val_steps * args.batch, 987,
                        rank * args.val_steps * args.batch, deterministic=True, cache=mu_cache)
    # prefetch_factor=1: 7 个 worker 仍有 7 个整批在飞, 流水不断; 默认值 2 会让在飞
    # 缓冲翻倍(整幅批很大), 无谓抬高本就紧张的任务内存水位
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                     pin_memory=True, drop_last=True,
                                     prefetch_factor=(1 if args.workers > 0 else None))
    vl = torch.utils.data.DataLoader(vs, batch_size=args.batch,
                                     num_workers=max(1, args.workers // 2),
                                     prefetch_factor=1)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=args.wd)
    ema1, ema2 = _ema_init(net), _ema_init(net)
    if ck is not None:
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema1 = {n: v.to(device) for n, v in ck["ema1"].items()}
        ema2 = {n: v.to(device) for n, v in ck["ema2"].items()}
        if args.save_rng:
            saved = ck.get("rng")
            if saved and len(saved) == world:
                st = saved[rank]
                gen.set_state(st["gen"])
                torch.set_rng_state(st["torch_cpu"])
                if "torch_cuda" in st and torch.cuda.is_available():
                    torch.cuda.set_rng_state(st["torch_cuda"])
                if is_main:
                    print("[jit] RNG 状态已按 rank 恢复, 随机流与断点前连续", flush=True)
            elif is_main:
                print("[jit] 断点无匹配 RNG 状态(缺失或 world 不同), 使用新随机流", flush=True)
        if is_main:
            print(f"[jit] 从 {args.resume} 续训: 已完成 {ck['samples']:,} samples "
                  f"({done_steps:,} 步), 剩余 {steps_total - done_steps:,} 步", flush=True)

    def run_batch(cond, tgt, land, mu, g):
        cond = cond.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        land = land.to(device, non_blocking=True)
        tgt = tgt * land                          # 有效域外无监督, 目标定义为 0
        if mu_cache is not None:
            tgt, cond = make_residual(tgt, mu.to(device, non_blocking=True), land,
                                      cond, sigma_r)
        w = land if args.loss_scope == "land" else torch.ones_like(land)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            res = jit_vloss(model, tgt, cond, w, args.noise_scale,
                            args.p_mean, args.p_std, args.t_eps, generator=g,
                            patch=args.patch, domain_mask=(land if net.needs_domain_mask else None),
                            dec_aux=dec_prior_on)
        return res if dec_prior_on else (res, None, {})

    seen, t0, hist = done_steps * per_step, time.time(), []
    best = float("inf")
    hf = out / "loss_history.json"
    if ck is not None and hf.exists():
        hist = [h for h in json.loads(hf.read_text()) if h["samples"] <= seen]
        if hist:
            best = min(h["val"] for h in hist)
    # 断点字典用完必须释放: 它每 rank 常驻约 payload 大小的宿主内存, 且会被随后 fork 的
    # DataLoader worker 整体继承; 任务级内存本就贴近 cgroup 上限, 多驻留会零星触发内核
    # OOM 击杀单个 rank(表现为无任何应用报错的整作业 Force Terminated)
    del ck
    gc.collect()
    run_sum = run_n = 0.0
    gn_sum, gn_max = 0.0, 0.0
    load_mon = None
    dec_aux_sum = dec_corr_sum = dec_n = 0.0
    # 逐区间的墙钟拆分(本 rank): 取数等待 / 计算(前向+反传+优化器) / 断点写盘 / 验证。
    # 各 rank 每步在集合通信上对齐, 别的 rank 的取数卡顿在本 rank 看起来像计算变慢,
    # 因此取数等待另记全 rank 最大值, 两者对照才能把卡顿归到取数还是计算。
    tm = {"data": 0.0, "comp": 0.0, "ckpt": 0.0, "val": 0.0}
    # 验证批只从数据管线取一次, 之后常驻宿主内存(每 rank val_steps x batch 帧): 验证帧集合与
    # (t, 噪声) 种子都是确定的, 缓存后数值逐位不变; 每次验证重起 worker 并让全部 rank 同一时刻
    # 从 Lustre 突发读几十帧, 正常时就要 40 s, 文件系统被挤时会卡上几分钟, 比训练本身还贵。
    val_cache = None
    alloc_retries_prev = (torch.cuda.memory_stats().get("num_alloc_retries", 0)
                          if device.type == "cuda" else 0)
    t_prev_end = time.time()
    net.train()
    for step, (cond, tgt, land, mu) in enumerate(dl, 1):
        t_arrive = time.time(); tm["data"] += t_arrive - t_prev_end
        cur_lr = lr * min(1.0, (seen + per_step) / max(1, args.warmup_samples))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        if dec_on:
            net.set_dec_progress(seen / max(1, args.duration))
        loss, aux, dec_extra = run_batch(cond, tgt, land, mu, gen)
        if dec_extra:
            dec_aux_sum += dec_extra["aux"]; dec_corr_sum += dec_extra["corr"]; dec_n += 1
        lv = float(loss.detach())
        # 非有限损失必须全 rank 一起判定。只让触发的那个 rank 退出的话, 其余 rank 会一直
        # 等在下一次集合通信上, 直到 NCCL 看门狗超时才连带杀掉作业 —— 整个机时白烧, 而
        # 日志里只留一句超时, 真正的原因要翻遍全部 rank 的 stderr 才找得到。
        nonfinite = torch.tensor([0.0 if math.isfinite(lv) else 1.0], device=device)
        if is_dist:
            dist.all_reduce(nonfinite)
        n_bad = int(nonfinite.item())
        if n_bad:
            raise SystemExit(f"[jit] 损失非有限: {n_bad}/{world} 个 rank 触发 "
                             f"(本 rank {lv}) @ {seen:,} samples")
        opt.zero_grad()
        # 难度头的辅助项只在反传时并入; 记录、验证与 ckpt 判据用的都是纯扩散损失,
        # 否则 D-EC 的曲线与 ec/tc/dense 的曲线不可比
        (loss if aux is None else loss + aux).backward()
        if step == 1 and is_dist:
            _assert_ranks_synced(_fingerprint(net, grads=True), device, world,
                                 "首步梯度")
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip))
        opt.step()
        _ema_update(ema1, net, args.ema1)
        _ema_update(ema2, net, args.ema2)
        if step == 1 and is_dist:
            _assert_ranks_synced(_fingerprint(net), device, world, "首步后参数/buffer")
            if is_main:
                print("[jit] 首步自检通过: 梯度与参数全 rank 一致", flush=True)
        if moe_layers and args.bias_gamma > 0:
            counts = _drain_moe_load(moe_layers, is_dist)
            for m, c in zip(moe_layers, counts):
                m.update_bias(args.bias_gamma, c)
            load_mon = counts if load_mon is None else load_mon + counts
        tm["comp"] += time.time() - t_arrive
        seen += per_step
        run_sum += lv; run_n += 1.0
        gn_sum += gn; gn_max = max(gn_max, gn)

        if seen % args.val_every < per_step:
            # 验证去掉两重采样噪声: 帧确定, (t, 噪声)用逐批固定种子的临时流,
            # 训练专属流不被触碰
            dec_train = _drain_dec_stats(net, is_dist) if dec_on else None
            t_val0 = time.time()
            net.eval(); vt = vn = 0.0; vaux = 0.0
            if val_cache is None:
                val_cache = [tuple(x.clone() for x in b) for b in vl]
            with torch.no_grad():
                for k, (c2, t2, l2, m2) in enumerate(val_cache):
                    vg = torch.Generator(device=device); vg.manual_seed(4242 + k)
                    vl_, vaux_, _ = run_batch(c2, t2, l2, m2, vg)
                    vt += float(vl_); vn += 1
                    vaux += float(vaux_) if vaux_ is not None else 0.0
            net.train()
            dec_val = _drain_dec_stats(net, is_dist) if dec_on else None
            tm["val"] += time.time() - t_val0
            data_max = torch.tensor([tm["data"]], device=device)
            if is_dist:
                dist.all_reduce(data_max, op=dist.ReduceOp.MAX)
            retries = (torch.cuda.memory_stats().get("num_alloc_retries", 0)
                       if device.type == "cuda" else 0)
            timing = {"data_wait": round(tm["data"], 1), "data_wait_max": round(float(data_max), 1),
                      "compute": round(tm["comp"], 1), "ckpt": round(tm["ckpt"], 1),
                      "val": round(tm["val"], 1), "alloc_retries": int(retries - alloc_retries_prev)}
            alloc_retries_prev = retries
            tm = {k: 0.0 for k in tm}
            v, tr_avg = vt / max(vn, 1), run_sum / max(run_n, 1)
            vaux_avg = vaux / max(vn, 1)
            if is_dist:
                q = torch.tensor([v, tr_avg, vaux_avg, 1.0], device=device); dist.all_reduce(q)
                v, tr_avg, vaux_avg = float(q[0] / q[3]), float(q[1] / q[3]), float(q[2] / q[3])
            rec = {"samples": seen, "train": round(tr_avg, 6), "val": round(v, 6),
                   "lr": cur_lr, "gnorm_mean": round(gn_sum / max(run_n, 1), 4),
                   "gnorm_max": round(gn_max, 4), "seconds": round(time.time() - t0, 1),
                   "timing": timing}
            if moe_layers:
                if args.bias_gamma <= 0:
                    c = _drain_moe_load(moe_layers, is_dist)
                    load_mon = c if load_mon is None else load_mon + c
                shares = load_mon / load_mon.sum(dim=1, keepdim=True).clamp_min(1.0)
                rec["moe_load"] = [[round(float(x), 4) for x in row] for row in shares]
                load_mon = None
            if dec_on:
                # 难度先验系数、辅助损失与头的校准度按区间平均; 路由统计分训练池(train)与
                # 推理规则(val, 帧内 top-C)两套, 每层一条
                E = moe_layers[0].n_experts
                rec["dec"] = {"lambda": round(float(net.dec_lambda), 4)}
                if dec_n > 0:
                    rec["dec"]["aux"] = round(dec_aux_sum / dec_n, 6)
                    rec["dec"]["aux_val"] = round(vaux_avg, 6)
                    rec["dec"]["head_corr"] = round(dec_corr_sum / dec_n, 4)
                for tag, vec in (("train", dec_train), ("val", dec_val)):
                    if vec is not None:
                        rec["dec"][tag] = [dec_stats_summary(row.cpu(), E) for row in vec]
                dec_aux_sum = dec_corr_sum = dec_n = 0.0
            hist.append(rec)
            run_sum = run_n = 0.0
            gn_sum, gn_max = 0.0, 0.0
            if is_main:
                print(f"[jit] {seen:>9,} samples  train={tr_avg:.5f}  val={v:.5f}  "
                      f"lr={cur_lr:.2e}  {time.time()-t0:.0f}s", flush=True)
                if v < best:
                    best = v
                    TD._atomic_torch_save(
                        {"model": net.state_dict(), "opt": opt.state_dict(),
                         "ema1": ema1, "ema2": ema2, "samples": seen, "val": v,
                         "args": vars(args)}, out / "ckpt.pt")
                tmp = out / f"loss_history.json.tmp.{os.getpid()}"
                tmp.write_text(json.dumps(hist, indent=1))
                os.replace(tmp, out / "loss_history.json")

        do_ckpt = seen % args.ckpt_every < per_step
        rng_states = None
        if args.save_rng and do_ckpt:
            rng_states = _rng_gather(is_dist, world, _rng_capture(rank, gen))
        t_ck = time.time()
        if is_main and do_ckpt:
            payload = {"model": net.state_dict(), "opt": opt.state_dict(),
                       "ema1": ema1, "ema2": ema2, "samples": seen, "args": vars(args)}
            if rng_states is not None:
                payload["rng"] = rng_states
            TD._atomic_torch_save(payload, out / "last.pt")
        if is_main and args.snap_every > 0 and seen % args.snap_every < per_step:
            TD._atomic_torch_save({"model": net.state_dict(), "ema1": ema1,
                                   "samples": seen}, out / f"snap_{seen:09d}.pt")
        tm["ckpt"] += time.time() - t_ck
        t_prev_end = time.time()
        if seen >= args.duration:
            break
        if args.max_seconds > 0:
            timeup = torch.tensor(
                [1.0 if (rank == 0 and time.time() - t0 >= args.max_seconds) else 0.0],
                device=device)
            if is_dist:
                dist.broadcast(timeup, src=0)
            if timeup.item() > 0:
                if is_main:
                    print(f"[jit] 达到 --max-seconds={args.max_seconds:.0f}s, "
                          f"保存断点后退出 ({seen:,} samples)", flush=True)
                break

    rng_states = (_rng_gather(is_dist, world, _rng_capture(rank, gen))
                  if args.save_rng else None)
    if is_main:
        payload = {"model": net.state_dict(), "opt": opt.state_dict(),
                   "ema1": ema1, "ema2": ema2, "samples": seen, "args": vars(args)}
        if rng_states is not None:
            payload["rng"] = rng_states
        TD._atomic_torch_save(payload, out / "last.pt")
        print(f"[jit] 完成 {seen:,} samples, best val={best:.5f}, "
              f"{time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
