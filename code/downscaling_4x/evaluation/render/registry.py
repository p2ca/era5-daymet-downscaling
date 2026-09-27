# -*- coding: utf-8 -*-
"""图/统计模块注册表。

每个 figures/ 下的模块用 @figure(name, needs=(...)) 登记一个渲染函数, 声明它依赖哪些场;
resolve 按 --only/--skip 选出本次要跑的模块。热插拔即: 加模块=登记新名, 关某图=--skip 它。
"""
from dataclasses import dataclass, field
from typing import Callable, Tuple


@dataclass
class Figure:
    name: str
    fn: Callable
    needs: Tuple[str, ...] = ()
    doc: str = ""
    full_year: bool = False


REGISTRY: "dict[str, Figure]" = {}


def figure(name, needs=(), full_year=False):
    """把渲染函数登记为一张图/一项统计。函数签名 fn(ctx) -> 产出路径(单个或列表)。

    needs 列出所依赖的场名(如 'crps'); 驱动会在场缺失时跳过该模块。
    full_year=True 表示该图的含义建立在"整年覆盖"之上(年均、逐月、按月排名等) —— 落场
    只有若干天时它照样画得出来, 结果却被标成年均或逐月, 图面看不出任何异常; 驱动因此在
    覆盖不全时跳过这类模块。只描述单日的图不设此标志。"""
    def deco(fn):
        if name in REGISTRY:
            raise ValueError(f"figure 名冲突: {name}")
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        REGISTRY[name] = Figure(name, fn, tuple(needs), doc, bool(full_year))
        return fn
    return deco


def resolve(only, skip):
    """按登记顺序返回启用的模块名; only 非空则只留其中的, skip 再剔除。"""
    names = list(REGISTRY)
    if only:
        s = set(only)
        names = [n for n in names if n in s]
    if skip:
        s = set(skip)
        names = [n for n in names if n not in s]
    return names
