#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
ablation_groups.py — 条件通道的机理分组(通道敏感性实验的唯一定义处)
============================================================================
按机理成组置换而不是逐通道置换, 是因为**冗余通道会互相掩护**: t2m 与 tmax 高度相关,
只动 tmax 时 t2m 会把信息补回来, 结果是"tmax 不重要"的假结论。凡是彼此可替代的通道
必须同组同时置换。

八个基本组覆盖全部 20 个条件通道且互不重叠, 另有三个复合项。每个目标都有一组是它的
**直接预测子**(C1/C2 -> SFC-T, C3 -> SFC-W), 该组的效应必须显著 —— 它是管线的标定项,
若它不显著说明置换根本没生效, 其余结果一概不可信。

置换值按通道的归一化方式分别选取, 不是一律填 0:

  * 动态通道走 (x - mean) / std, 而 mean/std 是**标量**(每个变量全域全年一个数)。
    因此归一化空间填 0 等于把整个通道换成一个空间均匀的常数场, 同时抹掉天气异常、
    季节循环与南北梯度, 且这种场在训练里从未出现(分布外)。mode="doy" 改为取另一年
    同一 day-of-year 的实测场, 只抹掉"今天的天气"而保留季节与空间结构, 留在分布内。
  * Δz 只除以 std、不减均值, 填 0 恰好等于物理上 Δz=0, 即"高分辨率地形等于粗网格
    地形" —— 语义干净, 固定用 zero。
  * land_sea_mask 是未归一化的 0/1, 填 0 等于宣告"全域皆海", 在陆地上是强分布外输入;
    固定填 1(全陆地), 即"抹掉海陆对比"。
============================================================================
"""
from era5_daymet.training import train_downscale as TD

# 静态通道的符号名(位于动态通道之后, 顺序与 DownscaleData.get_patch 拼接顺序一致)
DZ = "dz"
LANDCOVER = "landcover"
LSM = "land_sea_mask"
STATIC_ORDER = (DZ, LANDCOVER, LSM)

# 每个通道的置换方式: zero=归一化空间填 0; one=填 1; doy=可跨年同日历日重采样
FILL_ZERO, FILL_ONE, FILL_DOY = "zero", "one", "doy"
STATIC_FILL = {DZ: FILL_ZERO, LANDCOVER: FILL_ZERO, LSM: FILL_ONE}

GROUPS = {
    "SFC-T": {
        "channels": ["2m_temperature", "2m_temperature_max", "2m_temperature_min"],
        "mechanism": "近地面温度: 目标变量本身的大尺度值",
        "direct_predictor_for": ["2m_temperature_max", "2m_temperature_min"],
    },
    "SFC-W": {
        "channels": ["total_precipitation_24hr", "volumetric_soil_water_layer_1"],
        "mechanism": "近地面水分: 潜热与蒸散、Bowen ratio; 降水与土壤湿度互为代理, 必须同组",
        "direct_predictor_for": ["total_precipitation_24hr"],
    },
    "SFC-V": {
        "channels": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "mechanism": "近地面风: 平流、通风与近地层混合",
    },
    "UPR-T": {
        "channels": ["temperature_500", "temperature_850"],
        "mechanism": "高空热力: 气团属性、递减率与稳定度",
    },
    "UPR-Q": {
        "channels": ["specific_humidity_500", "specific_humidity_850"],
        "mechanism": "高空湿度: 水汽供给与云量",
    },
    "UPR-D": {
        "channels": ["geopotential_500", "geopotential_850",
                     "u_component_of_wind_500", "u_component_of_wind_850",
                     "v_component_of_wind_500", "v_component_of_wind_850"],
        "mechanism": "高空环流: 天气型与水汽输送; 地转关系下位势梯度即风, 分开置换会互相掩护",
    },
    "STA-Z": {
        "channels": [DZ],
        "mechanism": "亚网格地形 Δz = HR 高程 − 上采样 LR 高程, 递减率订正的唯一来源",
    },
    "STA-S": {
        "channels": [LANDCOVER, LSM],
        "mechanism": "地表属性: 反照率、粗糙度与海陆对比",
    },
}

# 复合项。ALL 是**上界标定**: 输入全部失效时输出应塌向气候态, 用
# ΔCRPS(组) / ΔCRPS(ALL) 把每组表达成"信息份额", 才能跨区、跨目标比较。
COMPOSITES = {
    "ALL-UPR": ["UPR-T", "UPR-Q", "UPR-D"],
    "ALL-STA": ["STA-Z", "STA-S"],
    "ALL": list(GROUPS),
}

# 二轮拆分: 只对一轮里显著的那一组做, 用来分辨组内是谁在起作用
SPLITS = {
    "SFC-T": [["2m_temperature_max"], ["2m_temperature", "2m_temperature_min"]],
    "SFC-W": [["total_precipitation_24hr"], ["volumetric_soil_water_layer_1"]],
    "UPR-D": [["geopotential_500", "geopotential_850"],
              ["u_component_of_wind_500", "u_component_of_wind_850",
               "v_component_of_wind_500", "v_component_of_wind_850"]],
    "STA-S": [[LANDCOVER], [LSM]],
}

NONE = "none"          # 对照: 走完全相同的代码路径但不置换任何通道


def resolve(name):
    """组名/复合名/逗号分隔的通道名 -> 通道名列表; NONE 返回空列表。"""
    if name in (None, "", NONE):
        return []
    if name in COMPOSITES:
        out = []
        for g in COMPOSITES[name]:
            out.extend(GROUPS[g]["channels"])
        return out
    if name in GROUPS:
        return list(GROUPS[name]["channels"])
    chans = [s.strip() for s in name.split(",") if s.strip()]
    known = set(TD.DEFAULT_IN) | set(STATIC_ORDER)
    bad = [c for c in chans if c not in known]
    if bad:
        raise ValueError(f"未知通道 {bad}; 可用组 {sorted(GROUPS)} / 复合 {sorted(COMPOSITES)}")
    return chans


def channel_slots(chans, in_vars=None):
    """通道名列表 -> [(条件张量的通道下标, 置换方式), ...]。

    条件张量的通道顺序 = 动态通道(in_vars 顺序) + dz + landcover + land_sea_mask。
    """
    in_vars = list(in_vars or TD.DEFAULT_IN)
    out = []
    for c in chans:
        if c in in_vars:
            out.append((in_vars.index(c), FILL_DOY))          # 动态通道可选 zero/doy
        elif c in STATIC_ORDER:
            out.append((len(in_vars) + STATIC_ORDER.index(c), STATIC_FILL[c]))
        else:
            raise ValueError(f"未知通道 {c!r}")
    return out


def describe(name):
    """给 meta.json 用的分组说明。"""
    chans = resolve(name)
    return {"group": name, "channels": chans, "n_channels": len(chans),
            "mechanism": (GROUPS[name]["mechanism"] if name in GROUPS else
                          f"复合: {'+'.join(COMPOSITES[name])}" if name in COMPOSITES else
                          "自定义通道列表")}
