# -*- coding: utf-8 -*-
"""图模块自动发现。

本目录下每个非下划线开头的 .py 一经 import 即触发其 @figure 登记。新增一张图 = 丢一个
模块文件进来, 无需改动别处; 用 --skip <名> 可临时关掉某张。
"""
import importlib
import pkgutil
from pathlib import Path

for _m in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if not _m.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_m.name}")
