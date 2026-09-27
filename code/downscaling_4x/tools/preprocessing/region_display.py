# -*- coding: utf-8 -*-
"""区域展示编号(地理顺序)。

展示编号 = 下列顺序中的位置(1..19), 与 regions_v1 的内部 region_id 解耦(内部 id 不变)。
编号自西向东、组内按地形分带排布, 因此从编号即可读出西/东与大致地带:

  1-2   太平洋海岸      PacificNW PacificSW
  3-8   西部山区/内陆    Cascades SierraNevada NRockies GreatBasin SRockies Southwest
  9-12  大平原(近中部)  NPlains CPlains SPlains Mezquital
  ---- 西/东分界: 编号 <= 12 为西部, >= 13 为东部 ----
  13-14 中部低地        Prairie GreatLakes
  15-16 东南低地        DeepSouth Southeast
  17    阿巴拉契亚      Appalachia
  18-19 大西洋海岸      MidAtlantic NorthAtlantic
"""

REGION_DISPLAY_ORDER = [
    "PacificNW", "PacificSW",
    "Cascades", "SierraNevada", "NRockies", "GreatBasin", "SRockies", "Southwest",
    "NPlains", "CPlains", "SPlains", "Mezquital",
    "Prairie", "GreatLakes",
    "DeepSouth", "Southeast",
    "Appalachia",
    "MidAtlantic", "NorthAtlantic",
]

WEST_EAST_DIVIDER = 12  # 展示编号 <= 12 为西部, >= 13 为东部


def display_id_map(region_names):
    """region_name -> 展示编号(1..); 不在展示顺序内的名字按尾部兜底(保持稳定)。"""
    pos = {n: i + 1 for i, n in enumerate(REGION_DISPLAY_ORDER)}
    nxt = len(REGION_DISPLAY_ORDER)
    out = {}
    for n in region_names:
        if n in pos:
            out[n] = pos[n]
        else:
            nxt += 1
            out[n] = nxt
    return out
