#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
input_identity.py — 条件输入产品的身份与一致性守卫
============================================================================
真实 ERA5 与 "Daymet-oracle" 输入目录同名、同形状、同布局: 后者把 ERA5 年度归档里的三个
目标成员换成同日 Daymet 的陆地 4x4 块平均, 其余成员逐字节相同。两者在运行时不可区分 ——
用 oracle 训出的 checkpoint 喂真实输入、或反过来, 前向照跑, 指标照出, 只是全部错标。

因此每个消费 checkpoint / 系数 / μ 缓存的入口都核对一次: 产物记录的输入目录与本次给的
是否同一个。不一致直接拒绝, 只有显式放行标志才能越过(留给有意的交叉测试)。

`describe()` 给出可写进 meta 的身份: 目录、产品名, 以及 oracle 目录自带 manifest 里的
算法名与配置哈希, 使 "这份产物基于哪个输入" 可追溯而不依赖目录名。
============================================================================
"""
import json
import os

ORACLE_MANIFEST = "oracle_dataset_manifest.json"
PRODUCT_ERA5 = "era5"
PRODUCT_ORACLE = "daymet-oracle-4x"
ALLOW_FLAG = "--allow-input-mismatch"
ALLOW_DEST = "allow_input_mismatch"


def normalize(path):
    """记录用的规范路径: 绝对路径, 不解析符号链接(保持人可读)。"""
    return os.path.abspath(str(path)) if path else None


def _same(a, b):
    """比较用: 解析符号链接后再比, 免得同一目录的两种写法被判成不同输入。"""
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def describe(era5_dir):
    """{era5_dir, input_product[, oracle]} —— 由目录内容而非目录名判定。"""
    d = normalize(era5_dir)
    out = {"era5_dir": d, "input_product": PRODUCT_ERA5}
    mp = os.path.join(d, ORACLE_MANIFEST) if d else None
    if mp and os.path.isfile(mp):
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        out["input_product"] = PRODUCT_ORACLE
        out["oracle"] = {k: m.get(k) for k in ("algorithm", "config_sha256", "created_utc",
                                              "replaced_variables", "scientific_scope")}
    return out


def check(recorded, given, what, allow=False, warn=print):
    """核对产物记录的输入目录与本次目录。

    一致返回 True; 不一致抛 SystemExit, allow=True 时只警告并返回 False。
    recorded 为 None 表示旧产物没有记录 —— 无法核对, 只警告, 由使用者自行确认。
    """
    b = normalize(given)
    if not recorded:
        warn(f"[input] ⚠ {what} 未记录输入目录, 无法核对是否与 {b} 一致")
        return False
    a = normalize(recorded)
    if _same(a, b):
        return True
    msg = (f"{what} 基于输入目录\n    {a}\n  本次给的是\n    {b}\n"
           "  两者布局相同, 混用不会报错, 只会得到错标的结果。")
    if allow:
        warn(f"[input] ⚠ 已按 {ALLOW_FLAG} 放行: " + msg)
        return False
    raise SystemExit("[input] 拒绝: " + msg + f"  有意的交叉测试请加 {ALLOW_FLAG}。")


def add_arg(parser):
    """给入口挂放行标志; 缺省关闭。"""
    parser.add_argument(ALLOW_FLAG, dest=ALLOW_DEST, action="store_true",
                        help="产物记录的输入目录与 --era5-dir 不一致时仍继续(有意的交叉测试)")
    return parser
