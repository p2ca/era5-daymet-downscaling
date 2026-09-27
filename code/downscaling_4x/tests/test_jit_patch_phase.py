#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JiT 的切块相位随机化(A)与重叠读取(B): 几何、随机流归属、以及"确实起作用"。

固定起点时同一像素永远落在块内同一位置, 而逐像素损失从不惩罚相邻块在边界上对不上,
接缝因此免费并在固定位置累积成网格。随机起点让同一像素时而在边界、时而在块内, 位置
特异的偏置在平均意义上要挨罚。重叠读取(B)则让每块看得见邻居的边缘, 但**只在读的一侧**
重叠 —— 写出范围不变, 所以块数、拼接与输出形状全不变, 不需要任何融合权重。

三件事必须钉死:
  * 几何: 任意起点下输出形状恒等于输入; 重叠不改变块数; 补边只用反射。
  * 随机流归属: 起点必须来自调用方给的 generator, 否则断点续训不再逐位复现,
    且验证集的起点会逐 epoch 漂移, val 数值失去可比性。
  * 有效性: 换起点必须真的改变计算 —— 否则随机化是空操作, 而这不会有任何报错。

用法:
    python -m downscaling_4x.tests.test_jit_patch_phase
"""
import torch

from downscaling_4x.models.jit_backbone import JiT, draw_patch_offset

FAILS = []
H, W, P, MARGIN, CIN = 120, 240, 32, 8, 5


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def _net(margin=MARGIN):
    torch.manual_seed(0)
    return JiT(hw=(H, W), patch=P, cond_ch=CIN, out_ch=1, hidden=32, depth=2,
               num_heads=2, bottleneck=8, patch_margin=margin).eval()


def _inputs(seed=1):
    torch.manual_seed(seed)
    return torch.randn(1, 1, H, W), torch.randn(1, CIN, H, W), torch.rand(1)


def test_geometry():
    """补边网格要能容下任意起点, 且形状恒定(位置编码与 RoPE 按固定网格预生成)。"""
    net = _net()
    gh, gw = net.x_embedder.gh, net.x_embedder.gw
    check("网格补到可容纳任意起点",
          net.grid_hw == (gh * P, gw * P) and gh * P >= H + P - 1 and gw * P >= W + P - 1,
          f"{(H, W)} -> {net.grid_hw} = {gh}x{gw}")
    z, cond, t = _inputs()
    for off in [(0, 0), (7, 3), (P - 1, P - 1)]:
        with torch.no_grad():
            out = net(z, t, cond, offset=off)
        check(f"起点{off} 输出形状还原", tuple(out.shape) == (1, 1, H, W), str(tuple(out.shape)))


def test_overlap_keeps_block_count():
    """B 的重叠只在读的一侧: 块数、输出形状都不变, 只有 kernel 与参数变大。"""
    n0, n8 = _net(margin=0), _net(margin=MARGIN)
    check("重叠不改变块数",
          (n0.x_embedder.gh, n0.x_embedder.gw) == (n8.x_embedder.gh, n8.x_embedder.gw),
          f"{n0.x_embedder.gh}x{n0.x_embedder.gw}")
    k0 = n0.x_embedder.proj1.weight.shape[-1]
    k8 = n8.x_embedder.proj1.weight.shape[-1]
    check("kernel = patch + 2*margin", k0 == P and k8 == P + 2 * MARGIN, f"{k0} -> {k8}")
    check("填充用反射而非补零", n8.x_embedder.proj1.padding_mode == "reflect")
    p0 = sum(q.numel() for q in n0.x_embedder.parameters())
    p8 = sum(q.numel() for q in n8.x_embedder.parameters())
    check("重叠的代价是参数, 不是块数", p8 > p0, f"{p0:,} -> {p8:,}")


def test_offset_changes_computation():
    """换起点必须真的改变切块结果。断言落在 patch 嵌入的输出上, 不在网络输出上 ——
    输出层是零初始化的, 未训练网络的输出恒为 0, 拿它验证无论如何都会"通过"。"""
    net = _net()
    z, cond, t = _inputs()
    seen = {}
    h = net.x_embedder.register_forward_hook(lambda m, i, o: seen.__setitem__("v", o.detach().clone()))
    got = {}
    try:
        for off in [(0, 0), (7, 3), (0, 0)]:
            with torch.no_grad():
                net(z, t, cond, offset=off)
            got.setdefault(off, []).append(seen["v"])
    finally:
        h.remove()
    a, b = got[(0, 0)][0], got[(7, 3)][0]
    d, mag = float((a - b).abs().max()), float(a.abs().max())
    check("换起点改变切块结果", d > 0.1 * mag, f"max|Δ|={d:.3e} (量级 {mag:.3e})")
    check("同起点重复调用逐位相同", torch.equal(got[(0, 0)][0], got[(0, 0)][1]))


def test_offset_comes_from_the_given_stream():
    """起点必须由调用方的 generator 决定: 同种子同序列, 不同种子不同序列。
    否则断点续训不再逐位复现, 验证集起点也会逐 epoch 漂移。"""
    def seq(seed, n=8):
        g = torch.Generator(); g.manual_seed(seed)
        return [draw_patch_offset(P, torch.device("cpu"), g) for _ in range(n)]
    check("同种子给出同一串起点", seq(7) == seq(7), str(seq(7)[:3]))
    check("不同种子给出不同串", seq(7) != seq(8))
    offs = seq(11, 200)
    check("起点落在 [0,patch) 内", all(0 <= a < P and 0 <= b < P for a, b in offs))
    check("起点确实在变(不是常数)", len(set(offs)) > 20, f"200 次抽到 {len(set(offs))} 个不同值")
    check("patch<=1 时退化为固定起点", draw_patch_offset(1, torch.device("cpu")) == (0, 0))


def test_ensemble_members_get_different_phases():
    """采样时每个集合成员带不同 generator, 应落在不同起点 —— 网格伪影才会在成员间错开。"""
    from downscaling_4x.models import jit_sampler as JS
    net = _net()
    cond = torch.randn(1, CIN, H, W)
    used = []
    orig = JS._velocity
    JS._velocity = lambda n, z, t, c, te, l, off=(0, 0), cache=None: (used.append(off), orig(n, z, t, c, te, l, off, cache))[1]
    try:
        for member in range(6):
            g = torch.Generator(); g.manual_seed(1000 + member)
            JS.generate(net, cond, steps=2, method="euler", generator=g)
    finally:
        JS._velocity = orig
    per_member = [used[i] for i in range(0, len(used), len(used) // 6)]
    check("不同成员落在不同起点", len(set(per_member)) > 1, f"{per_member}")
    check("同一成员整条轨迹用同一起点", len(set(used[:len(used) // 6])) == 1)


def main():
    for t in (test_geometry, test_overlap_keeps_block_count, test_offset_changes_computation,
              test_offset_comes_from_the_given_stream, test_ensemble_members_get_different_phases):
        print(t.__doc__.splitlines()[0])
        t()
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}", flush=True)
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
