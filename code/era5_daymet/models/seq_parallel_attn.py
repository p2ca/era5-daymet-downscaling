#!/usr/bin/env python
# Packaged implementation; code/seq_parallel_attn.py remains a compatibility entry point.
# -*- coding: utf-8 -*-
"""
============================================================================
seq_parallel_attn.py — 序列并行(sequence-parallel)全局自注意力
============================================================================
目的: 把整幅 ViT 的 O(N²) 全局注意力(N=259200 token, 单卡 fwd+bwd ~136s)
      拆给 SP 组内 K 个 GCD 协同算, 单步耗时 ~1/K, 从而在 24h 墙钟内迈更多优化步。

★ 关键性质: 算的是**数学上精确的全局注意力**, 与单卡整幅注意力逐元素等价
  (仅差浮点归约顺序 ~1e-5)。不近似、不加窗、不砍感受野 -> **不损伤模型表现**。
  代价只有: 通信开销(all-gather K/V, 节点内 Infinity Fabric 很快) + 工程复杂度。

实现路线 = "all-gather-KV" 变体(非 true-ring 流式):
  - 每个 SP rank 只持有自己那段 token 的 Q/K/V(N/K 个)。
  - 注意力前, 在 SP 组内 **autograd-aware all_gather** K、V -> 每个 rank 拿到完整 K、V。
  - 本地 SDPA(Q_local[N/K], K_full[N], V_full[N]) -> 只算 N²/K, K 个 rank 并行 => 单步 ~1/K。
  - 全注意力(双向), 无 causal mask -> 比 causal ring 简单, 无负载均衡/掩码技巧。
  - K/V full 只 ~0.8GB(N·d), SDPA 走 flash 不 materialize N² -> 显存反而按 token 分片下降。
  为什么不用 true-ring: 此处显存不是瓶颈, all-gather 版可直接复用 SDPA + 标准 autograd all_gather,
  正确性风险低得多(true-ring 的 online-softmax 反向要自定义 autograd)。显存吃紧再换。

参数布局对齐 nn.MultiheadAttention(in_proj_weight/bias + out_proj) -> 可与之做等价性对拍,
也便于从已有 ckpt 迁移权重。
============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _sp_size(group):
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


class _AllGatherSeq(torch.autograd.Function):
    """沿序列维(dim=2)在 SP 组内 all_gather, 自定义正确反向。
    前向: out = concat_r(x_r) (每 rank 拿到完整序列)。
    反向: 每个 K/V shard 被组内所有 rank 的 SDPA 用到, 故对本地 shard 的梯度 =
          Σ_{组内所有 rank} grad_out[本 shard 段]。用 all_reduce(SUM) 后切本地段实现
          (数学等价 reduce_scatter, 但 gloo/nccl 兼容性更好)。"""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.world = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)
        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(ctx.world)]
        dist.all_gather(parts, x, group=group)
        return torch.cat(parts, dim=2)

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=ctx.group)   # Σ over ranks
        Nl = grad_out.shape[2] // ctx.world
        grad_x = grad_out[:, :, ctx.rank * Nl:(ctx.rank + 1) * Nl, :].contiguous()
        return grad_x, None


def all_gather_seq(t, group):
    """在 SP 组内沿序列维(dim=2)all_gather, 保留 autograd(反向见 _AllGatherSeq)。
    t: [B, heads, N_local, head_dim] -> [B, heads, N_full, head_dim]。"""
    return _AllGatherSeq.apply(t.contiguous(), group)


class SPSelfAttention(nn.Module):
    """序列并行全局自注意力。sp_group=None 或 size1 -> 退化为本地全注意力(与 MHA 等价)。
    forward 输入 x: [B, N_local, dim](已按 SP 组分片的 token);输出同形。"""
    def __init__(self, dim, heads, dropout=0.0, sp_group=None):
        super().__init__()
        assert dim % heads == 0, f"dim({dim}) 必须被 heads({heads}) 整除"
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.dropout = float(dropout)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))   # 对齐 MHA: [Wq;Wk;Wv]
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        self.sp_group = sp_group
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, t, B, N):
        return t.view(B, N, self.heads, self.hd).transpose(1, 2)       # [B, heads, N, hd]

    def forward(self, x):
        B, Nl, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)      # [B, Nl, 3*dim]
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q, B, Nl)                                # [B, heads, Nl, hd]
        k = self._split_heads(k, B, Nl)
        v = self._split_heads(v, B, Nl)
        if _sp_size(self.sp_group) > 1:                                # 组内聚齐完整 K/V(autograd)
            k = all_gather_seq(k, self.sp_group)                       # [B, heads, N_full, hd]
            v = all_gather_seq(v, self.sp_group)
        p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)     # [B, heads, Nl, hd], flash 后端
        out = out.transpose(1, 2).reshape(B, Nl, self.dim)            # [B, Nl, dim]
        return self.out_proj(out)


# ===========================================================================
# 自测: SP=1(单进程)时必须与 nn.MultiheadAttention 数值等价(前向+反向)
# ===========================================================================
def _selftest_single():
    torch.manual_seed(0)
    dim, heads, B, N = 64, 4, 2, 40
    mha = nn.MultiheadAttention(dim, heads, batch_first=True, bias=True)
    sp = SPSelfAttention(dim, heads, sp_group=None)
    # 拷贝 MHA 权重到 SP(参数布局一致)
    with torch.no_grad():
        sp.in_proj_weight.copy_(mha.in_proj_weight)
        sp.in_proj_bias.copy_(mha.in_proj_bias)
        sp.out_proj.weight.copy_(mha.out_proj.weight)
        sp.out_proj.bias.copy_(mha.out_proj.bias)
    x = torch.randn(B, N, dim, dtype=torch.float64)
    mha.double(); sp.double(); x = x.double()
    xr = x.clone().requires_grad_(True); xs = x.clone().requires_grad_(True)
    y_mha = mha(xr, xr, xr, need_weights=False)[0]
    y_sp = sp(xs)
    fwd_err = (y_mha - y_sp).abs().max().item()
    (y_mha.sum()).backward(); (y_sp.sum()).backward()
    grad_err = (xr.grad - xs.grad).abs().max().item()
    ok = fwd_err < 1e-9 and grad_err < 1e-9
    print(f"[selftest SP=1] 前向 max|Δ|={fwd_err:.2e}  输入梯度 max|Δ|={grad_err:.2e}  "
          f"-> {'PASS ✅ 与 MHA 数值等价' if ok else 'FAIL ❌'}")
    return ok


# ===========================================================================
# 2D mesh(SP × DP)进程组 + 梯度同步(供 SP-3 接入 ViT/fit 用)
# ===========================================================================
def build_2d_mesh(sp_size, world=None, rank=None):
    """把 flat world 划成 SP × DP 二维网格。**SP 组取节点内连续 rank**(如每节点 8 GCD,
    sp_size=8 -> SP 组=同节点 8 卡, all-gather 走 Infinity Fabric 快);DP 组跨节点。
    要求 world % sp_size == 0。返回 (sp_group, dp_group, sp_rank, dp_rank, dp_size)。"""
    if world is None: world = dist.get_world_size()
    if rank is None: rank = dist.get_rank()
    assert world % sp_size == 0, f"world({world}) 必须被 sp_size({sp_size}) 整除"
    dp_size = world // sp_size
    sp_group = dp_group = None
    # SP 组: [g*sp_size, g*sp_size+1, ..., g*sp_size+sp_size-1](节点内连续)
    for g in range(dp_size):
        ranks = list(range(g * sp_size, (g + 1) * sp_size))
        grp = dist.new_group(ranks)
        if rank in ranks: sp_group = grp
    # DP 组: 各 SP 组内同一位置 [j, j+sp_size, j+2*sp_size, ...]
    for j in range(sp_size):
        ranks = list(range(j, world, sp_size))
        grp = dist.new_group(ranks)
        if rank in ranks: dp_group = grp
    sp_rank = rank % sp_size
    dp_rank = rank // sp_size
    return sp_group, dp_group, sp_rank, dp_rank, dp_size


def shard_seq(t, sp_rank, sp_size):
    """把完整 token 序列 [B, N, dim] 沿 N 切成 sp_size 段, 取第 sp_rank 段(连续切分,
    与 all_gather_seq 的 concat 顺序一致)。纯本地切片(autograd 天然正确: 非本段梯度为 0)。"""
    N = t.shape[1]; assert N % sp_size == 0, f"token 数 {N} 必须被 sp_size {sp_size} 整除"
    Nl = N // sp_size
    return t[:, sp_rank * Nl:(sp_rank + 1) * Nl]


class _GatherSeqRedundant(torch.autograd.Function):
    """末端 token gather: 与 _AllGatherSeq 前向相同, 但**反向取 MEAN 而非 SUM**。
    因为 gather 之后每个 SP rank 做**相同的冗余计算**(整幅卷积头 -> 同一个 loss, 各 rank 算一遍),
    等于同一份 loss 被算了 K 遍; 梯度流回分片 token 时各 rank 的 grad_out 相同, 除以 K 才是
    "算一遍"的真实梯度。(对比 _AllGatherSeq: KV gather 之后各 rank 做不同 SDPA, 故 SUM。)"""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group; ctx.world = dist.get_world_size(group); ctx.rank = dist.get_rank(group)
        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(ctx.world)]
        dist.all_gather(parts, x, group=group)
        return torch.cat(parts, dim=2)

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=ctx.group)
        grad_out /= ctx.world                                          # MEAN: 冗余计算算了 K 遍
        Nl = grad_out.shape[2] // ctx.world
        return grad_out[:, :, ctx.rank * Nl:(ctx.rank + 1) * Nl, :].contiguous(), None


def gather_seq_tokens(t_local, sp_group):
    """把各 SP rank 的 [B, N_local, dim] 沿 N 聚回完整 [B, N, dim](autograd, 反向 MEAN 切本地段;
    因下游是冗余整幅卷积头, 见 _GatherSeqRedundant)。复用其 dim=2 操作, 先加 head 维再去掉。"""
    x = t_local.unsqueeze(1)                        # [B,1,N_local,dim]
    x = _GatherSeqRedundant.apply(x.contiguous(), sp_group)   # [B,1,N,dim]
    return x.squeeze(1)                             # [B,N,dim]


def _coalesced_allreduce(params, group, scale=1.0):
    """把一组参数的梯度 flatten 成一整块做**单次** all_reduce(SUM)(像 DDP 的 bucketing), 再散回。
    ★为什么: 逐参数上百个小集合通信在 512 rank 下又慢又易 deadlock(实测 dp all_reduce 超时)。
    ★None 补零: grad=None 的参数先补 0, 保证各 rank 参与的张量结构完全一致(否则 None 不一致 -> mismatch 挂死)。"""
    if not params:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    if scale != 1.0:
        flat.mul_(scale)
    for p, s in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
        p.grad.copy_(s)


def sp_sync_grads(sharded_params, redundant_params, sp_group, dp_group, dp_size):
    """SP 训练的梯度同步(不走 DDP, backward 后手动调用), 合并归约版:
    - 分片参数(embed/blocks/norm): 各 SP rank 只有自己那段贡献 -> SP 维 **SUM** 得全帧梯度(合并 1 次)。
    - 冗余参数(gather 后的 proj/head): 各 SP rank 已算完整一份、梯度相同 -> **不 SP-SUM**。
    - 全部参数再 DP 维 **平均**(跨节点不同帧, 合并 1 次)。等价 (1/dp)Σ_dp(全帧梯度)。
    每步只 2 次集合通信(sp-sum + dp-avg), 而非逐参数上百次。"""
    sp_size = dist.get_world_size(sp_group) if sp_group is not None else 1
    if sp_size > 1:                                   # sharded: SP 维 SUM
        _coalesced_allreduce(sharded_params, sp_group)
    if dp_size > 1:                                   # sharded+redundant: DP 维平均
        _coalesced_allreduce(list(sharded_params) + list(redundant_params), dp_group, scale=1.0 / dp_size)


if __name__ == "__main__":
    import sys
    ok = _selftest_single()
    sys.exit(0 if ok else 1)
