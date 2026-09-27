#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
train_downscale.py — 训练核心: 分布式初始化、取帧、训练循环、断点续训契约
============================================================================
只做整幅训练。裁块与整幅学到的不是同一个函数(GroupNorm 的统计量在 (C/G, H, W) 上算,
依赖空间窗口大小), 所以 `--crop` 只允许配 `--smoke` 用, 正式训练一律整幅。

取帧走 `data.frames.FrameIndex`: 帧集合由真实日历配对决定, 缺历史的帧已被剔除。训练用
"洗牌无放回的连续流"—— 第 k 遍数据用一个只依赖 (base_seed, k) 的置换, 全局序号决定取哪帧,
因此各 rank 的分片必不相同、每帧曝光次数几乎相等, 且不依赖任何可变状态。验证用固定置换 +
固定全局索引, 逐 epoch 完全不变, 使早停与 LR 减半的判据不含采样噪声。

**帧集合一致性是这里最要命的一处**: FrameIndex 的丢帧数取决于传入的可用年份, 若某个 rank
少传了一年, 它算出的帧集合更短 -> 各 rank 步数不同 -> all_reduce/barrier 对不上 -> 挂到
NCCL 看门狗超时被杀, 表现为"作业卡死"而不是报错。启动时把配对签名 all-gather 一遍比对,
把这种失败提前成一句明确的错误。签名同时写进 checkpoint, 续训时一并核对。
============================================================================
"""
import argparse
import json
import math
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from downscaling_4x import contract as C
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.data.frames import FrameIndex
from downscaling_4x.evaluation import input_identity as II
from downscaling_4x.models.corrdiff_unet import CorrDiffUNet
from downscaling_4x.models.jit_backbone import draw_patch_offset
from downscaling_4x.models.jit_regressor import JiTRegressor
from downscaling_4x.models.unet import UNet, masked_mse

_STATE_VERSION = 1
SAMPLER_ID = "frameindex_stream_no_replacement_v1"


# ===========================================================================
# 1. 分布式
# ===========================================================================
def setup_ddp():
    """返回 (rank, world, local, device, is_dist)。

    支持三种启动方式: Slurm srun(SLURM_PROCID, nccl)、torchrun(RANK/WORLD_SIZE)、单进程。
    torchrun 在无 GPU 的 login node 上会退到 gloo, 使 DDP 接线能在不占队列的情况下冒烟。
    """
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        world = int(os.environ["SLURM_NTASKS"]); rank = int(os.environ["SLURM_PROCID"])
        local = int(os.environ["SLURM_LOCALID"])
        if "MASTER_ADDR" not in os.environ:
            try:
                import subprocess
                host = subprocess.check_output(
                    ["scontrol", "show", "hostnames", os.environ.get("SLURM_NODELIST", "")]
                ).decode().split()[0]
            except Exception:
                host = os.environ.get("HOSTNAME", "127.0.0.1")
            os.environ["MASTER_ADDR"] = host
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=timedelta(minutes=30), rank=rank, world_size=world)
        if rank == 0:
            print(f"[DDP] nccl world={world} master={os.environ['MASTER_ADDR']}:"
                  f"{os.environ['MASTER_PORT']}", flush=True)
        return rank, world, local, torch.device(f"cuda:{local}"), True

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:            # torchrun
        cuda = torch.cuda.is_available()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if cuda:
            torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl" if cuda else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if rank == 0:
            print(f"[DDP] {'nccl' if cuda else 'gloo'} world={world}", flush=True)
        return rank, world, local, torch.device(f"cuda:{local}" if cuda else "cpu"), True

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return 0, 1, 0, torch.device(dev), False


def assert_same_frames_across_ranks(fi, device, is_dist, label):
    """各 rank 的帧集合必须逐帧相同, 否则步数不齐会挂到 NCCL 超时而不是报错。"""
    if not is_dist:
        return
    world = dist.get_world_size()
    sig = [None] * world
    dist.all_gather_object(sig, (label, len(fi), fi.signature()))
    bad = [s for s in sig if s != sig[0]]
    if bad:
        raise RuntimeError(
            f"{label} 各 rank 的帧集合不同, 训练会在 all_reduce 处卡死: {sig[:4]}")


def assert_params_identical(module, tag="首步"):
    """要求各 rank 的参数逐元素完全一致; 不一致即 DDP 梯度同步失效。

    做法是对整个参数向量各取一次 all_reduce MAX 与 MIN, 再看逐元素极差 —— 精确、与参数量
    无关地灵敏, 且只需两次 all_reduce(不是 all_gather), 通信量不随 rank 数增长。

    ★ 不要退化成"参数和"一类的标量统计: 带符号求和会让逐参数的差异互相抵消, 而按总量定的
      相对阈值又随模型变大而变松 —— 大模型(尤其带大 embedding 表的)上信号会低于阈值, 检查
      静默放行。这正是它要防的那类失效自己也会犯的错。
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
        return 0.0
    v = torch.cat([p.detach().reshape(-1) for p in module.parameters()]).double()
    hi, lo = v.clone(), v.clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    spread = float((hi - lo).abs().max())
    tol = 1e-12 * max(1.0, float(v.abs().max()))
    if spread > tol:
        raise RuntimeError(
            f"{tag}后各 rank 参数不一致 (逐元素极差 {spread:.3e} > {tol:.3e}) —— "
            "DDP 梯度同步失效; 前向很可能绕过了 DDP 包装体")
    return spread


# ===========================================================================
# 2. 取帧
# ===========================================================================
def ds_worker_init(worker_id):
    """本包的取帧只用全局索引, 不依赖 worker 私有 RNG; 保留钩子只为显式说明这一点。"""
    return


class FrameDS(torch.utils.data.Dataset):
    """按 FrameIndex 取帧。两种模式都只由"全局索引"决定取哪一帧, 不依赖可变状态。

    stream=True(训练): 全局序号 g = epoch*epoch_span + index_offset + i, 第 g//N 遍数据的
        置换只依赖 (base_seed, 遍数) -> 全 rank 一致, 各 rank 靠 index_offset 落在互不重叠
        的分片上, 无放回, 每帧曝光次数几乎相等。
    deterministic=True(验证): 固定置换 + 固定全局索引 -> 逐 epoch 完全不变。
    """

    def __init__(self, data, fi, length, seed=0, deterministic=False,
                 index_offset=0, stream=False, epoch_span=0, crop=0):
        self.d, self.fi, self.len = data, fi, int(length)
        self.base_seed = int(seed)
        self.deterministic = deterministic
        self.stream = stream
        self.index_offset = int(index_offset)
        self.epoch_span = int(epoch_span)
        self.epoch = 0
        self.crop = int(crop)
        if not len(fi):
            raise ValueError("帧集合为空")
        self.perm = (np.random.default_rng(self.base_seed).permutation(len(fi))
                     if deterministic else None)
        self._pass_id, self._pass_perm = -1, None

    def __len__(self):
        return self.len

    def _perm_for_pass(self, k):
        if self._pass_id != k:
            self._pass_id = k
            self._pass_perm = np.random.default_rng(
                np.random.SeedSequence([self.base_seed, int(k)])).permutation(len(self.fi))
        return self._pass_perm

    def _index(self, i):
        n = len(self.fi)
        if self.deterministic:
            return int(self.perm[(self.index_offset + int(i)) % n])
        g = self.epoch * self.epoch_span + self.index_offset + int(i)
        return int(self._perm_for_pass(g // n)[g % n])

    def __getitem__(self, i):
        k = self._index(i)
        y, day = self.fi.frames[k]
        cond, tgt, m, raw = self.d.full(y, day, self.fi.history_of(k))
        if self.crop:
            cond, tgt, m, raw = _land_crop(self.d, cond, tgt, m, raw, self.crop, k)
        return torch.from_numpy(cond), torch.from_numpy(tgt), torch.from_numpy(m)


def _land_crop(data, cond, tgt, m, raw, size, key, min_land=0.3):
    """冒烟用: 确定性地切一块陆地占比够高的窗口。全海窗口会让 masked_mse 变成 0/0。"""
    H, W = data.H, data.W
    ny, nx = (H - size) // C.FACTOR, (W - size) // C.FACTOR
    for t in range(ny * nx):
        j = (key * 7919 + t * 3571) % (ny * nx)
        y0, x0 = (j // nx) * C.FACTOR, (j % nx) * C.FACTOR
        if data.mask[y0:y0 + size, x0:x0 + size].mean() >= min_land:
            return data.crop(cond, tgt, m, raw, y0, x0, size)
    raise RuntimeError("找不到陆地占比足够的窗口")


# ===========================================================================
# 3. 断点续训契约
# ===========================================================================
def save_loss_history(out, history):
    """逐 epoch 的 train/val loss 落成 loss_history.json; 只由 rank0 调用, 每轮覆写。

    训练作业只产出这份数据, 不出曲线图: 绘图统一由评测侧入口负责, 免得同一条曲线在几处
    各画一个样子。
    """
    path = os.path.join(out, "loss_history.json")
    json.dump(history, open(path, "w"), indent=2)
    return path


def _atomic_torch_save(payload, path):
    """先写临时文件再 os.replace, 保证任何时刻目标文件都是完整的上一份。

    墙钟到点可能正好落在 rank0 序列化 AdamW 状态的中途; 直接覆写会留下半个文件, 而它看起来
    仍是一个 checkpoint。
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _contract_value(v):
    if isinstance(v, (list, tuple)):
        return [_contract_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _contract_value(x) for k, x in sorted(v.items())}
    if isinstance(v, np.generic):
        return v.item()
    return v


PINNED = ("mode", "era5_dir", "daymet_dir", "target", "base", "pos_grid", "seed",
          "arch", "model_channels", "channel_mult", "num_blocks", "attn_bottleneck",
          "hidden", "depth", "heads", "mlp_ratio", "bottleneck", "patch", "patch_margin",
          "attn_dropout", "proj_dropout", "moe", "experts", "experts_per_tok",
          "moe_intermediate", "routed_scaling", "moe_all_layers", "moe_no_shared",
          "refine_head", "router", "gating", "rand_offset",
          "train_years", "val_years", "batch", "epoch_frames", "steps_per_epoch",
          "lr", "weight_decay", "lr_patience", "lr_factor", "min_lr",
          "patience", "grad_clip", "amp", "val_steps", "crop",
          "improve_criterion", "improve_tol")

# 后加入合同的口径键: 旧 checkpoint/断点缺这些键时按此表解释(旧 run 按构造即这些档),
# 使守卫不误拒历史断点, 同时对新 run 的真不一致保持分辨力。
LEGACY_DEFAULTS = {"refine_head": 0, "router": "tc", "gating": "norm",
                   "improve_criterion": "abs", "improve_tol": 1e-4, "rand_offset": 0}
# D-EC 子开关只属于阶段 B(train_jit); 旧的阶段 B 断点缺这些键时按"全开"的缺省档回填,
# 但它们只在 router=dec 下被消费, 对 tc/ec 断点没有任何影响。
DEC_DEFAULTS = {"dec_pool": 1, "dec_drop": 1, "dec_prior": 1,
                "dec_prior_max": 1.0, "dec_prior_ramp": 0.1, "dec_head_layer": 1}
# 两条流结构的开关表(train_jit)。缺省值 = 单流模型的行为, 旧断点缺这些键时按此回填即逐位复现。
# 依赖关系由 jit_stream_config 单点执法: two_stream 之下的子开关在单流下必须留缺省。
ARCH_DEFAULTS = {"rope_units": "grid", "drop_outside": 0, "two_stream": 0,
                 "coarse_doy": 1, "coarse_conv": 1, "coarse_blocks": 3,
                 "cross_attn": 1, "cross_modulate": 1, "cross_mask_outside": 1,
                 "lagrangian": 0, "tau_spec": "0,0,850:6,850:12,500:12,500:24",
                 "expert_two_card": 0, "dense_two_card": 0, "two_card_dims": "192,192",
                 "wards": 1, "ward_topk": 1, "ward_key": 0,
                 "terrain_key": 0, "terrain_key_dim": 64, "terrain_key_window": 20}
LEGACY_DEFAULTS_JIT = {**LEGACY_DEFAULTS, **DEC_DEFAULTS, **ARCH_DEFAULTS}


def apply_legacy_defaults(d, table=None):
    """给旧 args/契约值字典就地补默认档; 返回补上的键列表(空=无需回填)。
    table 缺省为两入口共用的 LEGACY_DEFAULTS; 阶段 B 传 LEGACY_DEFAULTS_JIT。"""
    table = LEGACY_DEFAULTS if table is None else table
    if not isinstance(d, dict):
        return []
    missing = [k for k in table if k not in d]
    for k in missing:
        d[k] = table[k]
    return missing


# val "算改善" 的判据, plateau 减半与早停共用。abs 是真实 ERA5 输入的口径: 比 best 低至少 tol。
# 同产品(Daymet-oracle)输入的损失量级只有真实输入的几十分之一, 同一个绝对 tol 会占到 val 的
# 几个到十几个百分点, 把每轮都在创新低的正常缓降判成停滞 —— 学习率提前塌缩、早停提前触发,
# 而作业正常结束、曲线看着正常。rel 按 best 的比例判, 与损失量级无关, 是 oracle 输入的口径。
# 缺省 auto 按输入目录的产品自动选: 真实 ERA5 -> abs, oracle -> rel; 显式给定时以给定为准。
IMPROVE_TOL_DEFAULT = {"abs": 1e-4, "rel": 1e-3}
IMPROVE_BY_PRODUCT = {II.PRODUCT_ERA5: "abs", II.PRODUCT_ORACLE: "rel"}


def resolve_improve_args(args):
    """把判据与阈值落成显式值并写回 args, 使其进入 checkpoint 与续训契约; 返回 (判据, 阈值)。"""
    crit = getattr(args, "improve_criterion", None) or "auto"
    if crit == "auto":
        product = II.describe(getattr(args, "era5_dir", None) or M.ERA5_DIR)["input_product"]
        crit = IMPROVE_BY_PRODUCT[product]
    if crit not in IMPROVE_TOL_DEFAULT:
        raise SystemExit(f"未知改善判据 {crit!r}; 可选 auto / {' / '.join(sorted(IMPROVE_TOL_DEFAULT))}")
    tol = getattr(args, "improve_tol", None)
    if tol is None:
        tol = IMPROVE_TOL_DEFAULT[crit]
    tol = float(tol)
    if not (0.0 < tol < 1.0):
        raise SystemExit(f"--improve-tol 必须在 (0, 1) 内, 得到 {tol}")
    args.improve_criterion, args.improve_tol = crit, tol
    return crit, tol


def patch_offset_seed(seed, ep, step, rank):
    """切块起点专属随机流的种子, 只由 (seed, epoch, 步序, rank) 决定。

    不进 checkpoint: 续训一律从 epoch 边界重入, 同一 (ep, step, rank) 必然抽到同一起点,
    所以断点续训仍逐位复现。验证段传 ep=-1, 起点跨 epoch 恒定, val 数值才跨轮可比 ——
    验证也随机重掷会让 val 自带抖动, 而 plateau 减半与 best 判据都读它, 不会有任何报错。
    各 rank 种子不同: 同一步里各 rank 落在不同起点, 一个全局批覆盖多种网格相位。
    """
    return (int(seed) * 1000003 + (int(ep) + 1) * 100003
            + int(step) * 131 + int(rank) * 17) % (2 ** 31 - 1)


def is_improved(vloss, best, criterion="abs", tol=1e-4):
    """本轮 val 是否算改善: abs 要求比 best 低至少 tol, rel 要求比 best 低至少 tol 倍。"""
    if criterion == "rel":
        return vloss < best * (1.0 - tol)
    return vloss < best - tol


def resume_contract(args, dp_size, steps, tr_span, tr_pair, va_pair):
    """续训时不允许静默改变的字段。

    `epochs` 故意不在其中: 续训时它表示新的**累计**目标。帧配对签名在内 —— 换了可用年份或
    历史窗口会让帧集合变, 而通道数和模型形状都不变, 权重照样装得回去。
    """
    values = {}
    for name in PINNED:
        if not hasattr(args, name):
            continue
        v = getattr(args, name)
        if name in {"era5_dir", "daymet_dir"}:
            v = os.path.abspath(v)
        values[name] = _contract_value(v)
    return {"state_version": _STATE_VERSION, "sampler": SAMPLER_ID,
            "dp_size": int(dp_size), "steps_per_epoch_effective": int(steps),
            "train_span_per_epoch": int(tr_span),
            "train_pairing": tr_pair, "val_pairing": va_pair,
            "cond_layout": C.cond_layout(args.mode), "values": values}


def contract_mismatches(saved, current):
    if not isinstance(saved, dict):
        return ["checkpoint 没有有效的 resume_contract"]
    out = []
    for k in ("state_version", "sampler", "dp_size", "steps_per_epoch_effective",
              "train_span_per_epoch", "cond_layout"):
        if saved.get(k) != current.get(k):
            out.append(f"{k}: checkpoint={saved.get(k)!r}, current={current.get(k)!r}")
    for k in ("train_pairing", "val_pairing"):
        a, b = (saved.get(k) or {}), (current.get(k) or {})
        if a.get("frame_pairing_sha256") != b.get("frame_pairing_sha256"):
            out.append(f"{k}: 帧集合不同 checkpoint={a.get('frame_pairing_sha256', '?')[:12]} "
                       f"current={b.get('frame_pairing_sha256', '?')[:12]}")
    sv, cv = saved.get("values", {}), current.get("values", {})
    for k in sorted(set(sv) | set(cv)):
        if sv.get(k) != cv.get(k):
            out.append(f"{k}: checkpoint={sv.get(k)!r}, current={cv.get(k)!r}")
    return out


def jit_moe_config(cfg):
    """从参数字典/Namespace 取 DSMoE 配置; None 表示全稠密。

    阶段 A(确定性)与阶段 B(扩散)共用这一处定义: MoE 是 FFN 的替换, 与是否扩散正交,
    两处各写一份迟早会分叉。
    """
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    if not get("moe"):
        # 稠密主体没有路由器: 非缺省的路由口径会被静默忽略, run 却会顶着 ec 的名字跑完
        if (get("router", "tc") or "tc") != "tc" or (get("gating", "norm") or "norm") != "norm":
            raise SystemExit("--router / --gating 只在 --moe 下生效; 稠密主体不接受非缺省路由口径")
        dec_config(get, "tc")                      # 非缺省的 dec 子开关同样拒绝
        return None
    # 路由三开关的组合守卫单点执法于此(两入口的必经点), 不给各入口各自为政留缝:
    #   tc+raw 既非基线也非任何文献口径, 只可能是误操作; ec/dec 下逐专家偏置对选择是
    #   数学无操作, 开着 bias_gamma 只会让 buffer 无意义漂移进 checkpoint。
    router = get("router", "tc") or "tc"
    gating = get("gating", "norm") or "norm"
    if router not in ("tc", "ec", "dec") or gating not in ("norm", "raw"):
        raise SystemExit(f"未知路由口径 router={router!r} gating={gating!r}")
    if gating == "raw" and router not in ("ec", "dec"):
        raise SystemExit("--gating raw 仅配 --router ec/dec; tc 的门控口径固定为 norm")
    if router in ("ec", "dec") and float(get("bias_gamma", 0) or 0) > 0:
        raise SystemExit(f"--router {router} 构造性均衡, --bias-gamma 必须为 0")
    # 两级分诊(科室)与键只属于 tc: 科室 = 分组路由的组, wards=1 时保留 DeepSeek 缺省的
    # 2 组选 2 组(数学上无操作), 使既有 tc 断点逐位复现
    wards = int(get("wards", 1) or 1)
    ward_topk = int(get("ward_topk", 1) or 1)
    E = int(get("experts"))
    if wards > 1:
        if router != "tc":
            raise SystemExit("--wards > 1(两级分诊)只在 --router tc 下生效")
        if E % wards != 0:
            raise SystemExit(f"--wards {wards} 须整除专家数 {E}")
        if not (1 <= ward_topk <= wards):
            raise SystemExit(f"--ward-topk {ward_topk} 须在 [1, wards={wards}] 内")
        if ward_topk * (E // wards) < int(get("experts_per_tok")):
            raise SystemExit("选中科室里的专家总数少于 top-K, 无法凑齐每 token 的专家数")
        n_group, topk_group = wards, ward_topk
    else:
        if ward_topk != 1:
            raise SystemExit("--ward-topk 只在 --wards > 1 下生效")
        n_group, topk_group = 2, 2
    terrain = int(get("terrain_key", 0) or 0)
    if terrain and router != "tc":
        raise SystemExit("--terrain-key 只在 --router tc 下生效")
    drop = int(get("drop_outside", 0) or 0)
    if drop and router != "tc":
        raise SystemExit("--drop-outside 只在 --router tc 下生效; ec/dec 的域外处理由各自的路由定义")
    return {"num_experts": E,
            "moe_intermediate_size": get("moe_intermediate") or 2 * get("hidden"),
            "num_experts_per_tok": get("experts_per_tok"),
            "n_group": n_group, "topk_group": topk_group,
            "routed_scaling_factor": get("routed_scaling"),
            "interleave": not get("moe_all_layers"),
            "use_shared_expert": not get("moe_no_shared"),
            "proj_drop": get("proj_dropout", 0.0),
            "router_mode": router, "gating_mode": gating,
            "dec": dec_config(get, router),
            "terrain_key_dim": (int(get("terrain_key_dim", 64) or 64) if terrain else 0),
            "drop_outside": bool(drop)}


def parse_two_card_dims(v):
    """'a,b' 或 (a, b) -> (a, b) 两个正整数。"""
    if isinstance(v, (list, tuple)):
        a, b = v
    else:
        parts = [x.strip() for x in str(v).split(",")]
        if len(parts) != 2:
            raise SystemExit(f"--two-card-dims 须为 'a,b', 得到 {v!r}")
        a, b = parts
    a, b = int(a), int(b)
    if a <= 0 or b <= 0:
        raise SystemExit(f"--two-card-dims 须为正整数, 得到 {(a, b)}")
    return a, b


def jit_stream_config(cfg):
    """两条流结构的开关表 -> JiT 的 arch 字典; 全部处于单流缺省时返回 None。

    依赖关系在此单点执法, 违反即拒绝启动, 而不是静默按别的口径跑:
      - two_stream=0 时 coarse_*/cross_*/lagrangian/two_card/ward_key 必须留缺省;
      - lagrangian 要求 cross_attn=1 且 rope_units=km;
      - cross_mask_outside / cross_modulate 只在 cross_attn=1 下生效;
      - two_stream=1 时至少要开 cross_attn 或某种 two_card, 否则细流看不到天气;
      - ward_key 要求 wards>1 且 two_stream=1; expert_two_card 要求 --moe。
    """
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    vals = {}
    for k, dflt in ARCH_DEFAULTS.items():
        v = get(k, None)
        vals[k] = dflt if v is None else v
    units = str(vals["rope_units"])
    if units not in ("grid", "km"):
        raise SystemExit(f"--rope-units 只能是 grid / km, 得到 {units!r}")
    two = bool(int(vals["two_stream"]))
    sub = ("coarse_doy", "coarse_conv", "coarse_blocks", "cross_attn", "cross_modulate",
           "cross_mask_outside", "lagrangian", "tau_spec", "expert_two_card", "dense_two_card",
           "two_card_dims", "ward_key")
    if not two:
        bad = [k for k in sub if str(vals[k]) != str(ARCH_DEFAULTS[k])]
        if bad:
            raise SystemExit(f"--{bad[0].replace('_', '-')} 只在 --two-stream 1 下生效")
    cross = two and bool(int(vals["cross_attn"]))
    lag = bool(int(vals["lagrangian"]))
    exp_card, den_card = bool(int(vals["expert_two_card"])), bool(int(vals["dense_two_card"]))
    ward_key = bool(int(vals["ward_key"]))
    if two:
        if lag and not cross:
            raise SystemExit("--lagrangian 1 需要 --cross-attn 1: 推移的是交叉问询里 K 的位置")
        if lag and units != "km":
            raise SystemExit("--lagrangian 1 需要 --rope-units km: 风程是公里, 位置章须同单位")
        if not cross and (str(vals["cross_mask_outside"]) != str(ARCH_DEFAULTS["cross_mask_outside"])
                          or str(vals["cross_modulate"]) != str(ARCH_DEFAULTS["cross_modulate"])):
            raise SystemExit("--cross-mask-outside / --cross-modulate 只在 --cross-attn 1 下生效")
        if not cross and not exp_card and not den_card:
            raise SystemExit("--two-stream 1 而交叉问询与两张卡都关: 细流看不到任何天气, 请至少开一项")
        if int(vals["coarse_blocks"]) < 0:
            raise SystemExit("--coarse-blocks 须 >= 0")
        if exp_card and not get("moe"):
            raise SystemExit("--expert-two-card 1 需要 --moe")
        if ward_key and (int(vals["wards"]) <= 1 or not get("moe")):
            raise SystemExit("--ward-key 1 需要 --moe 且 --wards > 1")
        vals["two_card_dims"] = parse_two_card_dims(vals["two_card_dims"])
    if int(vals["terrain_key"]) and not get("moe"):
        raise SystemExit("--terrain-key 1 需要 --moe")
    if int(vals["terrain_key"]):
        w = int(vals["terrain_key_window"])
        if w < 1 or (w - int(get("patch"))) % 2 != 0 or w < int(get("patch")):
            raise SystemExit(f"--terrain-key-window {w} 须 >= patch 且与 patch 同奇偶")
    active = (two or units != "grid" or int(vals["drop_outside"]) or int(vals["terrain_key"])
              or int(vals["wards"]) > 1)
    if not active:
        return None
    return vals


def dec_config(get, router):
    """D-EC 子配置(仅 router=dec 下非 None)。

    三个成分开关 dec_pool / dec_drop / dec_prior 缺省全开; 全关等价于 ec, 拒绝——否则一个
    实验会顶着 D-EC 的名字按 ec 跑完。其余路由下任何非缺省的 dec 子开关同样拒绝, 理由相同。
    """
    vals = {}
    for k, dflt in DEC_DEFAULTS.items():
        v = get(k, None)
        vals[k] = dflt if v is None else v
    if router != "dec":
        bad = [k for k, v in vals.items() if float(v) != float(DEC_DEFAULTS[k])]
        if bad:
            raise SystemExit(f"--{bad[0].replace('_', '-')} 只在 --router dec 下生效")
        return None
    pool, drop, prior = (bool(int(vals[k])) for k in ("dec_pool", "dec_drop", "dec_prior"))
    if not (pool or drop or prior):
        raise SystemExit("D-EC 三个成分全关等价于 ec, 请直接用 --router ec")
    prior_max, ramp = float(vals["dec_prior_max"]), float(vals["dec_prior_ramp"])
    if prior_max < 0 or not (0.0 <= ramp <= 1.0):
        raise SystemExit(f"--dec-prior-max 须 >= 0, --dec-prior-ramp 须在 [0,1]: 得到 {prior_max}, {ramp}")
    return {"pool": pool, "drop": drop, "prior": prior, "prior_max": prior_max,
            "prior_ramp": ramp, "head_layer": int(vals["dec_head_layer"])}


def build_regressor(cin, cout, cfg, temb=0, hw=None):
    """按结构参数构建确定性主体(阶段 A 也走这里)。

    cfg 可以是 argparse Namespace, 也可以是 checkpoint 里存下的 args 字典 —— 训练、评测与
    μ 缓存共用本函数, 保证同一份 checkpoint 在任何入口都被还原成同一个结构。

    hw 是模型将要处理的空间尺寸, 缺省取合同的 HR_SHAPE。UNet 与 CorrDiffUNet 是全卷积,
    对它不敏感; JiT 的位置编码与 RoPE 按固定 token 网格预生成, **尺寸变了就得重建**,
    因此必须按实际输入尺寸构建(冒烟切块时即为块边长)。
    """
    hw = tuple(hw) if hw else C.HR_SHAPE
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    arch = get("arch", "unet") or "unet"
    pos_grid = int(get("pos_grid", 0) or 0)
    if arch == "unet":
        return UNet(cin, cout, base=int(get("base", 64)), temb=temb, pos_grid=pos_grid)
    if arch == "corrdiff":
        return CorrDiffUNet(cin, cout,
                            model_channels=int(get("model_channels", 64)),
                            channel_mult=tuple(get("channel_mult", (1, 2, 2, 2, 2))),
                            num_blocks=int(get("num_blocks", 4)),
                            attn_bottleneck=bool(int(get("attn_bottleneck", 1))),
                            pos_grid=pos_grid, temb=temb)
    if arch == "jit":
        if (get("router", "tc") or "tc") == "dec":
            # D-EC 的掩膜与难度头接线只存在于扩散版 JiT; 回归器若照单构造, 路由会在首次
            # 前向因缺掩膜/先验抛错——这里提前拒绝, 把原因说清楚
            raise SystemExit("--router dec 只在阶段 B(train_jit)实现; 阶段 A 的 JiT 回归器请用 --router ec")
        refine = int(get("refine_head", 0) or 0)
        ridx = None
        if refine:
            layout = C.cond_layout(get("mode", C.DEFAULT_MODE))
            ridx = [layout.index(n) for n in C.STATIC_ORDER]
        return JiTRegressor(hw=hw, patch=int(get("patch", 32)), cond_ch=cin,
                            out_ch=cout, hidden=int(get("hidden", 384)),
                            depth=int(get("depth", 12)), num_heads=int(get("heads", 6)),
                            mlp_ratio=float(get("mlp_ratio", 4.0)),
                            bottleneck=int(get("bottleneck", 128)),
                            attn_drop=float(get("attn_dropout", 0.0)),
                            proj_drop=float(get("proj_dropout", 0.0)),
                            moe_config=jit_moe_config(cfg),
                            patch_margin=int(get("patch_margin", 0)),
                            refine_head=refine, refine_static_idx=ridx)
    raise ValueError(f"未知 arch: {arch!r} (可选: unet / corrdiff / jit)")


def target_selection(target):
    """--target 的口径: 'all' 为三目标联合, 输出通道按合同 TARGETS 顺序; 否则单目标。

    返回 (目标张量的通道切片, 模型输出通道数)。用切片而不是整数索引: 整数索引会丢掉通道维,
    (B,H,W) 对 (B,1,H,W) 在 masked_mse 里照样广播成功, 混用不报错, 只会算错。
    """
    if target == "all":
        return slice(None), len(C.TARGETS)
    it = C.TARGETS.index(target)
    return slice(it, it + 1), 1


def masked_mse_strict(pred, tgt, mask):
    """要求预测与目标形状完全一致的 masked_mse。

    (B,3,H,W) 的预测对上 (B,1,H,W) 的目标会被广播吞掉: loss 照常下降、作业正常结束,
    只是三个输出通道都在拟合同一个目标。形状不一致必须当场抛错, 不能交给广播。
    """
    if pred.shape != tgt.shape:
        raise RuntimeError(f"预测与目标形状不一致: {tuple(pred.shape)} vs {tuple(tgt.shape)}")
    return masked_mse(pred, tgt, mask)


def check_resume_args(ck, args, keys, world=None, path_keys=()):
    """核对续训命令行与 checkpoint 记录的关键参数, 不一致直接拒绝启动。

    这些字段改错不会改变任何张量形状, 权重照样装得回去 —— 只会静默换掉训练口径。
    """
    saved = ck.get("args", {}) or {}
    diffs = []
    for k in keys:
        a, b = saved.get(k), getattr(args, k, None)
        if k in path_keys:
            a = os.path.abspath(a) if a else a
            b = os.path.abspath(b) if b else b
        if _contract_value(a) != _contract_value(b):
            diffs.append(f"{k}: checkpoint={a!r}, current={b!r}")
    if world is not None and ck.get("world") not in (None, world):
        diffs.append(f"world: checkpoint={ck.get('world')}, current={world}")
    if diffs:
        raise SystemExit("续训参数与断点不一致:\n  - " + "\n  - ".join(diffs))


# ===========================================================================
# 4. 命令行
# ===========================================================================
def add_common_args(p):
    p.add_argument("--era5-dir", default=M.ERA5_DIR)
    p.add_argument("--daymet-dir", default=M.DAYMET_DIR)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default=C.TARGETS[0], choices=list(C.TARGETS) + ["all"],
                   help="单目标训练(三个目标各训一个模型), 或 all = 三目标联合"
                        "(输出通道按合同 TARGETS 顺序)")
    p.add_argument("--mode", default=C.DEFAULT_MODE, choices=sorted(C.MODES),
                   help="条件模式; history_51 与 history_control_21 共享同一帧集合")
    p.add_argument("--train-years", type=int, nargs="+", default=M.splits["train"])
    p.add_argument("--val-years", type=int, nargs="+", default=M.splits["val"])
    p.add_argument("--epochs", type=int, default=100, help="累计目标 epoch 数, 不是追加轮数")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--epoch-frames", type=int, default=4096,
                   help="每 epoch 全体 rank 合计看多少帧; steps 随数据并行度收缩")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="直接指定每 rank 步数; >0 时覆盖 --epoch-frames")
    p.add_argument("--val-steps", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-patience", type=int, default=4)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--improve-criterion", choices=["auto"] + sorted(IMPROVE_TOL_DEFAULT), default="auto",
                   help="val 算改善的判据, plateau 减半与早停共用: abs = val < best - tol; "
                        "rel = val < best * (1 - tol), 与损失量级无关。缺省 auto 按输入产品选: "
                        "真实 ERA5 -> abs, Daymet-oracle -> rel")
    p.add_argument("--improve-tol", type=float, default=None,
                   help="判据阈值; 缺省按判据取 abs 1e-4 / rel 1e-3")
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0,
                   help="权重初始化种子; 固定它才能对续训做逐位复现验证")
    p.add_argument("--resume-from", default="")
    p.add_argument("--crop", type=int, default=0,
                   help="只允许配 --smoke: 切窗训练与整幅学到的不是同一个函数")
    p.add_argument("--smoke", action="store_true", help="秒级自测: 少量帧与轮次")


def apply_smoke(args):
    args.train_years = [2019]
    args.val_years = [2020]
    args.epochs = 2
    args.epoch_frames = 0
    args.steps_per_epoch = 2
    args.val_steps = 2
    args.batch = 1
    args.workers = 0
    args.base = min(getattr(args, "base", 64), 16)
    if not args.crop:
        args.crop = 128


# ===========================================================================
# 5. 训练
# ===========================================================================
def fit(args, ddp_info):
    rank, world, local, device, is_dist = ddp_info
    is_main = (rank == 0)
    if args.crop and not args.smoke:
        raise SystemExit("--crop 只允许配 --smoke; 正式训练一律整幅(GroupNorm 依赖窗口大小)")
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    # 各 rank 同一初始权重: DDP 构造时会广播 rank0 的参数, 但固定种子让"不经 DDP 的
    # 单进程跑"也可复现, 续训的逐位对拍才有判据。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 改善判据落成显式值(写回 args 进 checkpoint 与续训契约), 之后全程只用这一对
    crit, tol = resolve_improve_args(args)
    if is_main:
        print(f"[判据] val 改善 = {crit} (tol={tol:g}); plateau 减半与早停共用", flush=True)

    stats = Stats(args.era5_dir, args.daymet_dir)
    need = C.pairing_history_days(args.mode)
    lags = C.history_lags(args.mode)
    avail = sorted(set(args.train_years) | set(args.val_years))
    tr_fi = FrameIndex(args.train_years, avail, need, lags, split="train")
    va_fi = FrameIndex(args.val_years, avail, need, lags, split="val")
    assert_same_frames_across_ranks(tr_fi, device, is_dist, "train")
    assert_same_frames_across_ranks(va_fi, device, is_dist, "val")

    # 需要加载哪些年由帧集合**实际引用**的年份决定, 不靠"val 年 + 训练最后一年"这类推算:
    # 验证集年初的历史落在上一个训练年, 少加载一年会在取数时才报 KeyError。
    def years_of(fi):
        return sorted({f[0] for f in fi.frames} | {h[0] for hs in fi.history for h in hs})

    tr_years, va_years = years_of(tr_fi), years_of(va_fi)
    # ERA5 按年常驻: 洗牌流的相邻样本来自随机年份, LRU 太小会每个样本重读一整年。
    tr_data = DownscaleData(args.era5_dir, args.daymet_dir, tr_years, stats,
                            mode=args.mode, era5_cache_years=len(tr_years))
    va_data = DownscaleData(args.era5_dir, args.daymet_dir, va_years, stats,
                            mode=args.mode, era5_cache_years=len(va_years))
    # ERA5 年数据在主进程一次性载满。DataLoader 的 worker 每个 epoch 重新 fork, 若靠
    # worker 惰性加载, 年缓存会随 worker 一同消亡, 每个 epoch 都从文件系统重读几十份
    # 年数据 —— 训练一切正常, 只是每步慢好几秒。在 fork 之前载满, 写时复制让所有
    # worker 零成本共享同一份只读缓存, 且逐 epoch 重掷 worker 的采样语义不受影响。
    # 各 rank 从不同年份起步轮转: 年文件各驻一个 OST, 全体同序读会挤在同一个上。
    t0 = time.time()
    off = rank % max(1, len(tr_years))
    for y in tr_years[off:] + tr_years[:off]:
        tr_data._era5_year(y)
    off = rank % max(1, len(va_years))
    for y in va_years[off:] + va_years[:off]:
        va_data._era5_year(y)
    if is_main:
        print(f"[预载] ERA5 train {len(tr_years)} 年 + val {len(va_years)} 年, "
              f"{time.time() - t0:.0f}s", flush=True)

    steps = (args.steps_per_epoch if args.steps_per_epoch > 0
             else max(1, math.ceil(args.epoch_frames / (world * args.batch))))
    tr_span = world * steps * args.batch
    val_steps = max(1, min(args.val_steps, math.ceil(len(va_fi) / max(1, world * args.batch))))
    val_len = val_steps * args.batch
    if is_main:
        print(f"[帧] train {len(tr_fi)} 帧(丢 {tr_fi.dropped}) | val {len(va_fi)} 帧"
              f"(丢 {va_fi.dropped}) | 配对 {tr_fi.signature()[:12]}", flush=True)
        print(f"[步] world={world} batch/rank={args.batch} steps/epoch={steps} "
              f"每 epoch {tr_span} 帧 | val {val_steps} 步/rank", flush=True)
        gb = len(C.ERA5_IN) * C.DAYS_PER_YEAR * C.LR_SHAPE[0] * C.LR_SHAPE[1] * 4 / 1e9
        print(f"[内存] ERA5 常驻 train {len(tr_years)} 年 + val {len(va_years)} 年 "
              f"≈ {(len(tr_years) + len(va_years)) * gb:.1f} GB/rank", flush=True)

    tr_ds = FrameDS(tr_data, tr_fi, steps * args.batch, seed=1234, stream=True,
                    index_offset=rank * steps * args.batch, epoch_span=tr_span, crop=args.crop)
    va_ds = FrameDS(va_data, va_fi, val_len, seed=987, deterministic=True,
                    index_offset=rank * val_len, crop=args.crop)
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, num_workers=args.workers,
                                     drop_last=True, worker_init_fn=ds_worker_init)
    va = torch.utils.data.DataLoader(va_ds, batch_size=args.batch,
                                     num_workers=max(0, args.workers // 2),
                                     drop_last=True, worker_init_fn=ds_worker_init)

    cin = C.cond_channels(args.mode)
    hw = (args.crop, args.crop) if args.crop else C.HR_SHAPE
    sel, cout = target_selection(args.target)
    model = build_regressor(cin, cout, args, hw=hw).to(device)
    # MoE 的专家不是每步全命中 -> 计算图逐步在变, 与 static_graph 不兼容; 必须改用
    # find_unused_parameters。稠密主体没有这个问题, 保留 static_graph(更快且更严格)。
    # broadcast_buffers 对 JiT 关掉: 它的 buffer 要么恒定(位置编码), 要么由全局 all-reduce
    # 保证一致(MoE 路由偏置); 逐前向广播只会用 rank0 的值盖掉不一致, 掩盖问题而非修复。
    use_moe = bool(getattr(args, "moe", False)) and getattr(args, "arch", "unet") == "jit"
    is_jit = getattr(args, "arch", "unet") == "jit"
    net = DDP(model, device_ids=([local] if device.type == "cuda" else None),
              static_graph=not use_moe, find_unused_parameters=use_moe,
              broadcast_buffers=not is_jit) if is_dist else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    contract = resume_contract(args, world, steps, tr_span,
                               tr_fi.metadata(), va_fi.metadata())
    # 与断点侧对称回填: 无论哪一侧缺后加入的口径键都按默认档解释, 比较才不被
    # "键存在与否"这种表示差异误伤; 显式改动开关的新 run 依旧会被逮住。
    apply_legacy_defaults(contract["values"])
    ckpt = os.path.join(args.out, "ckpt.pt")
    last = os.path.join(args.out, "last.pt")
    best, bad, plateau, cur_lr, start_epoch, history = float("inf"), 0, 0, args.lr, 0, []

    if args.resume_from:
        src = os.path.abspath(args.resume_from)
        if os.path.dirname(src) != os.path.abspath(args.out):
            raise RuntimeError("--resume-from 必须位于当前 --out 目录内")
        err, state = "", None
        try:
            state = torch.load(src, map_location=device, weights_only=False)
            if state.get("state_version") != _STATE_VERSION:
                raise RuntimeError(f"不支持的 checkpoint 版本 {state.get('state_version')!r}")
            filled = apply_legacy_defaults((state.get("resume_contract") or {}).get("values"))
            if filled and is_main:
                print(f"  旧断点缺口径键 {filled}, 按默认档解释", flush=True)
            diffs = contract_mismatches(state.get("resume_contract"), contract)
            if diffs:
                raise RuntimeError("续训配置与 checkpoint 不一致:\n  - " + "\n  - ".join(diffs))
            model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if is_dist:
            ok = torch.tensor([0 if err else 1], device=device, dtype=torch.int32)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if not int(ok.item()):
                raise RuntimeError(f"至少一个 rank 无法读取续训 checkpoint; 本 rank: {err or '正常'}")
        elif err:
            raise RuntimeError(err)
        start_epoch = int(state["epoch"]) + 1
        best, bad = float(state["best"]), int(state["bad"])
        plateau, cur_lr = int(state["plateau"]), float(state["cur_lr"])
        history = list(state.get("history", []))
        if bad >= args.patience:
            raise RuntimeError(f"checkpoint 已触发早停(bad={bad}), 不是墙钟截断, 不能原样续训")
        if args.epochs < start_epoch:
            raise RuntimeError(f"--epochs={args.epochs} 小于 checkpoint 的下一 epoch {start_epoch + 1}")
        if is_main:
            print(f"  ★续训 {src} -> 从 epoch {start_epoch + 1}; best={best:.4f} "
                  f"bad={bad}/{args.patience} lr={cur_lr:.2e}", flush=True)

    rand_offset = int(getattr(args, "rand_offset", 0) or 0)
    if rand_offset and getattr(args, "arch", "unet") != "jit":
        raise SystemExit("--rand-offset 只对 --arch jit 有意义: 其余主干的 forward 不收 offset")
    # 起点从 CPU 流抽: 只有两个整数, 与 GPU 型号无关, 换卡也抽到同一序列
    off_gen = torch.Generator() if rand_offset else None

    def offset_kw(ep, step):
        if not rand_offset:
            return {}
        off_gen.manual_seed(patch_offset_seed(args.seed, ep, step, rank))
        return {"offset": draw_patch_offset(args.patch, "cpu", off_gen)}

    for ep in range(start_epoch, args.epochs):
        tr_ds.epoch = ep
        for g in opt.param_groups:
            g["lr"] = cur_lr
        lr_used = cur_lr
        net.train(); t0 = time.time(); trt = trn = 0.0
        for si, (cond, tgt, m) in enumerate(tr):
            cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                loss = masked_mse_strict(net(cond, **offset_kw(ep, si)), tgt[:, sel], m)
            opt.zero_grad(); loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            trt += loss.item(); trn += 1
        model.eval(); vt = vn = 0.0
        with torch.no_grad():
            for vi, (cond, tgt, m) in enumerate(va):
                cond, tgt, m = cond.to(device), tgt.to(device), m.to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                    # ep=-1: 验证起点逐 batch 固定、跨 epoch 不变
                    vt += masked_mse_strict(model(cond, **offset_kw(-1, vi)),
                                            tgt[:, sel], m).item(); vn += 1
        if is_dist:
            tt = torch.tensor([vt, vn, trt, trn], device=device); dist.all_reduce(tt)
            vt, vn, trt, trn = (float(x) for x in tt)
        vloss, trloss = vt / max(vn, 1), trt / max(trn, 1)
        improved = is_improved(vloss, best, crit, tol)
        halved = False
        if improved:
            best, bad, plateau = vloss, 0, 0
        else:
            bad += 1; plateau += 1
            if plateau >= args.lr_patience and cur_lr > args.min_lr:
                cur_lr = max(cur_lr * args.lr_factor, args.min_lr)
                plateau = 0; halved = True
        err = ""
        if is_main:
            history.append({"epoch": ep + 1, "train": trloss, "val": vloss,
                            "best": best, "lr": lr_used})
            try:
                save_loss_history(args.out, history)
                if improved:
                    _atomic_torch_save({"model": model.state_dict(), "args": vars(args),
                                        "val_best": best, "epoch": ep,
                                        "resume_contract": contract}, ckpt)
                _atomic_torch_save({
                    "state_version": _STATE_VERSION, "model": model.state_dict(),
                    "opt": opt.state_dict(), "epoch": ep, "best": best, "bad": bad,
                    "plateau": plateau, "cur_lr": cur_lr, "history": history,
                    "resume_contract": contract, "args": vars(args),
                    "saved_job_id": os.environ.get("SLURM_JOB_ID"),
                    "stopped_early": bad >= args.patience}, last)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        if is_dist:
            ok = torch.tensor([0 if err else 1], device=device, dtype=torch.int32)
            dist.broadcast(ok, src=0)
            if not int(ok.item()):
                raise RuntimeError(f"rank0 无法保存 checkpoint: {err or '(见 rank0 日志)'}")
        elif err:
            raise RuntimeError(err)
        if is_main:
            h = f" ↓LR->{cur_lr:.2e}" if halved else ""
            print(f"  ep {ep + 1}/{args.epochs}  train={trloss:.4f}  val={vloss:.4f}  "
                  f"best={best:.4f}  bad={bad}/{args.patience}  lr={lr_used:.2e}{h}  "
                  f"{time.time() - t0:.0f}s  [last.pt]", flush=True)
        if bad >= args.patience:
            if is_main:
                print(f"  早停: val 连续 {args.patience} 轮未达改善判据 {crit} tol={tol:g} "
                      f"(best={best:.4f})", flush=True)
            break

    if is_main and history:
        save_loss_history(args.out, history)
    if is_dist:
        dist.barrier()
    return ckpt, model
