#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mpl_style.py — 各出图入口共用的 matplotlib 全局设置

目前只有一项: 注册能显示中文的字体。图注、图例与坐标标签里混有中文, 而 matplotlib 的
默认 DejaVu Sans 不含 CJK 字形, 缺字时只在 stderr 出一条 UserWarning, 图照样存盘, 缺的
字变成空心方框 —— 出图脚本一般不看 stderr, 所以这类退化要靠共用入口一次设好, 而不是
每个脚本各写一份。
"""
import os

import matplotlib.pyplot as plt

# 字体名关键字与文件名关键字: 两轮各查一遍, 覆盖"已注册"与"装在用户目录尚未注册"两种情形
_NAME_KEYS = ("Noto Sans CJK", "Noto Sans SC", "WenQuanYi", "Source Han")
_FILE_KEYS = ("notosanssc", "notosanscjk", "cjk", "wqy", "sourcehans")


def use_cjk():
    """注册一个能显示中文的字体并返回其名字; 找不到就返回 None(标签缺字, 但不至于崩)。"""
    import matplotlib.font_manager as fm

    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in _NAME_KEYS):
            return _apply(f.name)
    for path in fm.findSystemFonts(fontext="otf") + fm.findSystemFonts(fontext="ttf"):
        if any(k in os.path.basename(path).lower() for k in _FILE_KEYS):
            fm.fontManager.addfont(path)
            return _apply(fm.FontProperties(fname=path).get_name())
    print("[mpl_style] 未找到中文字体, 中文标签会缺字")
    return None


def _apply(name):
    plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 该字体的减号字形与默认负号不同, 关掉替换
    return name
