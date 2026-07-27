#!/usr/bin/env python
# Packaged implementation; code/train_vit.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
train_vit.py — Vision-Transformer (SETR 风格) 降尺度 (ERA5 -> Daymet, 6x)
============================================================================
独立训练 ViT。和 UNet 分开, 因为 ViT 慢、要单独调超参。DDP + 早停(默认 10 epoch)。
  train=1980-2017 / val=2018-2019(早停) / test=2020(载入 best ckpt 评测)

★可调超参(都在命令行):
  --vit-patch  patch 边长   (默认 2; 越小细节越多、token 越多越慢)
  --dim        embedding 维 (默认 384)
  --depth      Transformer block 数 (默认 8)
  --heads      注意力头数   (默认 6)
  --mlp        MLP 扩张倍数 (默认 4.0)
  --patch      训练裁剪块边长(=eval 分块边长, 默认 64 -> token 数 (64/2)^2=1024)

结构: Conv patch-embed -> +可学习位置编码 -> N×(LN+MHSA+MLP) -> ConvTranspose 还原
      -> ★卷积细化头(两层 conv)消除 "一格一格" 的 patch 拼接伪影。
评测: ViT 的位置编码绑定训练 patch 尺寸, 所以整帧按 --patch 大小分块、重叠平均
      (det_predict, eval_tile=--patch), 保证每块 token 数与训练一致。

用法:
  自测(合成数据, 需 torch):  python train_vit.py --smoke
  单卡:  python train_vit.py --stats-dir stats_train --out runs/vit \
             --vit-patch 2 --dim 384 --depth 8 --heads 6 --patch 64
  多节点: 把  srun -u python train_vit.py --stats-dir stats_train --out runs/vit
          放进 submit_ddp.sh
============================================================================
"""
import argparse
import os
import sys

from era5_daymet.training import train_downscale as TD

if TD.torch is not None:
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
        绑尺寸的可学习 pos (259200 裸参数是 ~1亿, 见 handoff §5.1)。
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


def main():
    p = argparse.ArgumentParser(description="独立 ViT 降尺度训练 (DDP + 早停)",
                                formatter_class=argparse.RawTextHelpFormatter)
    TD.add_common_args(p)
    # 默认对齐指南 V3_LARGE 的参数预算(~15.2M), 做与 U-Net U3_BASE192(14.2M)公平的同参数量对比。
    # V3 名义超参 = patch2/dim384/depth6/heads12/mlp4(=>15.182M in 指南实现)。但本实现的上采样头更轻
    # (PixelShuffle+卷积细化头仅~1M, 见 docs/reference/instruction.html 无此层规定), depth=6 只有 11.6M;
    # 故 depth 取 8 补回头部差额 -> 15.19M ≈ 指南 V3。其余(dim/heads/patch/mlp)严格取 V3 值。
    p.add_argument("--vit-patch", type=int, default=2, help="patch 边长(V3=2)")
    p.add_argument("--dim", type=int, default=384, help="embedding 维度(V3=384)")
    p.add_argument("--depth", type=int, default=8, help="Transformer block 数(V3名义=6; 取8对齐V3参数预算~15.2M)")
    p.add_argument("--heads", type=int, default=12, help="注意力头数(V3=12)")
    p.add_argument("--mlp", type=float, default=4.0, help="MLP 扩张倍数(V3=4)")
    p.add_argument("--dropout", type=float, default=0.0, help="注意力/MLP/pos dropout(抗过拟合; 默认 0=旧口径)")
    p.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth, 沿深度线性 0->此值")
    p.add_argument("--pos-type", choices=["sincos", "learned"], default="sincos",
                   help="位置编码: sincos(固定, 任意尺寸, 整幅必需) / learned(绑 crop, 旧口径)")
    p.add_argument("--head-up", choices=["pixelshuffle", "convt"], default="pixelshuffle",
                   help="上采样头: pixelshuffle(无棋盘) / convt(旧口径 ConvTranspose)")
    p.add_argument("--full-frame", action="store_true",
                   help="整幅 720x1440 训练+整幅评测(不切窗); 用 FullFrameDS, eval_tile=0")
    p.add_argument("--epoch-frames", type=int, default=0,
                   help="整幅: 每 epoch 见多少帧; >0 时有效 steps=ceil(N/(world*batch)), 加节点真省墙钟")
    p.add_argument("--seq-parallel", action="store_true",
                   help="序列并行: SP 组内多 GCD 协同算一帧的全局注意力, 单步~1/sp-size(整幅提步数用)")
    p.add_argument("--sp-size", type=int, default=8,
                   help="SP 组大小(节点内 GCD 数, 建议=每节点GCD数=8); world 须被其整除")
    p.add_argument("--smoke", action="store_true", help="合成数据秒级自测")
    # ViT 专属默认: crop 小一点(token 数可控) + 梯度裁剪(Transformer 需要)
    # patch=60 而非 64: 必须被 FACTOR=6 整除(见 TD.check_patch), 64 会在真数据上崩
    # lr=1e-4 + grad_clip=1.0: 3e-4 无裁剪时发散(runs/exp/20260712-vit-d384, val 随 lr 单调变差)
    # LR 策略: 无 warmup, plateau 减半(见 TD.fit_deterministic; --lr-patience/--lr-factor/--min-lr)
    p.set_defaults(patch=60, batch=8, lr=1e-4, grad_clip=1.0)
    args = p.parse_args()

    if TD.torch is None:
        sys.exit("需要 PyTorch(在 Frontier GPU 节点上跑)。本地只能 --smoke 且需装 torch。")
    args.model = "vit"
    if args.full_frame and args.pos_type != "sincos":
        sys.exit("--full-frame 需 --pos-type sincos(learned 位置编码绑 crop 尺寸, 无法整幅)。")
    # 整幅: 整幅一次前向评测(不切窗); crop: 按训练 patch 尺寸分块重叠评测
    args.eval_tile = 0 if args.full_frame else args.patch

    if args.smoke:
        TD.apply_smoke(args)
        args.patch = 24; args.vit_patch = 2; args.dim = 64; args.depth = 2; args.heads = 4
        args.eval_tile = 0 if args.full_frame else 24

    if not args.full_frame:                                # 整幅不切窗, 无 crop 整除约束(720/1440 天然合法)
        if args.patch % args.vit_patch:
            sys.exit(f"--patch({args.patch}) 必须能被 --vit-patch({args.vit_patch}) 整除")
        TD.check_patch(args)                               # --patch 还必须能被 FACTOR=6 整除

    rank, world, local, device, is_dist = TD.setup_ddp()
    if args.seq_parallel and not args.full_frame:
        sys.exit("--seq-parallel 仅用于 --full-frame(切分整幅 token 的全局注意力)。")
    # 2D mesh(SP × DP): SP 组=节点内连续 GCD 协同算一帧; DP 组跨节点(不同帧)
    sp_ctx = None; sp_group = None; sp_size = 1; sp_rank = 0
    if args.seq_parallel and is_dist:
        from era5_daymet.models import seq_parallel_attn as SP
        if world % args.sp_size:
            sys.exit(f"world({world}) 必须被 --sp-size({args.sp_size}) 整除")
        sp_group, dp_group, sp_rank, dp_rank, dp_size = SP.build_2d_mesh(args.sp_size)
        sp_size = args.sp_size
        sp_ctx = dict(sp_group=sp_group, dp_group=dp_group, sp_rank=sp_rank, dp_rank=dp_rank, dp_size=dp_size)
        if rank == 0:
            print(f"[SP] 序列并行: sp_size={sp_size} dp_size={dp_size} (world={world}); "
                  f"单步~1/{sp_size}, 全局 batch={dp_size*args.batch}", flush=True)
    stats = TD.Stats(args.stats_dir, args.in_vars, args.out_vars)
    Cin = TD.cond_channels(args.in_vars, args.out_vars, args.use_clim)   # 默认20通道; --use-clim=23
    Cout = len(args.out_vars)
    model = ViT(Cin, Cout, img=args.patch, patch=args.vit_patch,
                dim=args.dim, depth=args.depth, heads=args.heads, mlp=args.mlp,
                dropout=args.dropout, drop_path=args.drop_path,
                pos_type=args.pos_type, head_up=args.head_up, full_frame=args.full_frame,
                sp_group=sp_group, sp_size=sp_size, sp_rank=sp_rank).to(device)
    n_par = sum(x.numel() for x in model.parameters())
    if rank == 0:
        if args.full_frame:
            gh, gw = 720 // args.vit_patch, 1440 // args.vit_patch; toks = f"整幅 {gh}x{gw}={gh*gw}"
        else:
            toks = f"crop={args.patch} tokens={(args.patch//args.vit_patch)**2}"
        print(f"[ViT] patch={args.vit_patch} dim={args.dim} depth={args.depth} heads={args.heads} "
              f"pos={args.pos_type} head_up={args.head_up} {toks}  params={n_par/1e6:.1f}M  "
              f"world={world}  train={args.train_years[0]}-{args.train_years[-1]} test={args.test_year} "
              f"patience={args.patience} lr={args.lr:.1e} lr_patience={args.lr_patience} lr_factor={args.lr_factor}", flush=True)

    TD.fit_deterministic(model, stats, args, device, (rank, world, local, is_dist), sp_ctx=sp_ctx)
    if sp_ctx is not None:
        # 序列并行整幅评测: 保持 SP 开, 所有 rank 参与前向集合通信(~1/sp 快), 仅 rank0 算指标/写文件
        TD.evaluate(model, None, stats, args, device, is_main=(rank == 0))
    elif rank == 0:
        TD.evaluate(model, None, stats, args, device)     # 非 SP: rank0 单进程整幅/切窗评测

    if is_dist:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
