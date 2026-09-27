#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
ablation_groups.py — 条件通道的机理分组(通道敏感性实验的唯一定义处)
============================================================================
敏感性实验的缺省协议是**逐通道消融**: 合同里的每个条件通道单独置换(见 per_channel),
给出"其余通道都在时, 这一个还额外提供什么"的边际贡献全图。机理分组是按需的补充分析:
**冗余通道会互相掩护**(t2m 与 tmax 高度相关, 只动 tmax 时 t2m 会把信息补回来,
逐通道读数接近 0 不等于该通道不重要), 把彼此可替代的通道成组同时置换, 才能回答
"这套机理总共贡献多少"。两套结果不可换算。

基本组覆盖合同的全部条件通道且互不重叠, 另有复合项与 ALL 上界标定。每个目标都有
一组是它的**直接预测子**(tmax/tmin -> SFC-T, precip -> SFC-W), 该组的效应必须显著 ——
它是管线的标定项, 若它不显著说明置换根本没生效, 其余结果(含逐通道)一概不可信。

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
from era5_daymet import contract as C
# 静态通道的符号名与顺序直接取自数据合同, 不在此另写一份: 合同增删静态通道时, 本地副本
# 不会报错, 只会让下标整体错位, 把某一组的置换悄悄打到相邻通道上。
DZ = C.DZ
ELEVATION = C.ELEVATION
LANDCOVER = C.LANDCOVER
LSM = C.LAND_SEA_MASK
STATIC_ORDER = C.STATIC_ORDER

# 每个通道的置换方式: zero=归一化空间填 0; one=填 1; doy=可跨年同日历日重采样
FILL_ZERO, FILL_ONE, FILL_DOY = "zero", "one", "doy"
STATIC_FILL = {DZ: FILL_ZERO, ELEVATION: FILL_ZERO, LANDCOVER: FILL_ZERO, LSM: FILL_ONE,
               C.DOY_SIN: FILL_ZERO, C.DOY_COS: FILL_ZERO}

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
        "channels": [DZ, ELEVATION],
        "mechanism": ("地形高度: Δz(亚网格起伏, 递减率订正的唯一来源)与绝对高程(递减率与气压高度"
                      "的绝对参考)。两者由 Δz = 高程 − 上采样 LR 高程 线性相关, 分开置换会互相掩护, "
                      "故同组; 组内谁在起作用交给 SPLITS 判"),
    },
    "STA-S": {
        "channels": [LANDCOVER, LSM],
        "mechanism": "地表属性: 反照率、粗糙度与海陆对比",
    },
    "TIM-D": {
        "channels": [C.DOY_SIN, C.DOY_COS],
        "mechanism": ("季节相位: 年内位置本身携带的气候信息(白昼长度、太阳高度角), 与当日天气无关。"
                      "sin/cos 成对才构成相位, 单置换其一只是把相位旋到别处而非抹掉"),
    },
}

# 复合项。ALL 是**上界标定**: 输入全部失效时输出应塌向气候态, 用
# ΔCRPS(组) / ΔCRPS(ALL) 把每组表达成"信息份额", 才能跨区、跨目标比较。
COMPOSITES = {
    "ALL-SFC": ["SFC-T", "SFC-W"],
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
    "STA-Z": [[DZ], [ELEVATION]],
}

NONE = "none"          # 对照: 走完全相同的代码路径但不置换任何通道


def per_channel(in_vars=None):
    """逐通道消融的"组"名列表 = 合同里的每个通道自成一组。

    与分组消融回答的不是同一个问题: 分组问"这套机理贡献多少", 逐通道问"在其余通道都在的
    前提下, 这一个还额外提供什么"。彼此可替代的通道(t2m/tmax/tmin、位势与风、Δz 与绝对
    高程、doy 的 sin 与 cos)在逐通道下会互相掩护, 各自的 Δ 都接近 0 —— 那是边际贡献的
    真实读数, 不等于该通道不重要。两套结果不可换算: 组的 Δ 不等于组内各通道 Δ 之和。
    """
    return list(C.cond_layout(list(in_vars or C.DEFAULT_IN)))


def group_of(channel):
    """通道名 -> 它所属的分组名; 不属于任何组时返回 None。"""
    for g, d in GROUPS.items():
        if channel in d["channels"]:
            return g
    return None


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
    known = set(C.cond_layout(C.DEFAULT_IN))      # 动态 + 静态 + 时间, 即合同的全部通道
    bad = [c for c in chans if c not in known]
    if bad:
        raise ValueError(f"未知通道 {bad}; 可用组 {sorted(GROUPS)} / 复合 {sorted(COMPOSITES)}")
    return chans


def channel_slots(chans, in_vars=None):
    """通道名列表 -> [(条件张量的通道下标, 置换方式), ...]。

    通道顺序即 contract.cond_layout(in_vars); 下标一律由它给出, 不按段偏移自行推算。
    """
    in_vars = list(in_vars or C.DEFAULT_IN)
    layout = C.cond_layout(in_vars)
    out = []
    for c in chans:
        if c not in layout:
            raise ValueError(f"通道 {c!r} 不在现行条件合同里; 可用通道 {layout}")
        i = layout.index(c)
        out.append((i, FILL_DOY if c in in_vars else STATIC_FILL.get(c, FILL_ZERO)))
    return out


def describe(name):
    """给 meta.json 用的分组说明。"""
    chans = resolve(name)
    if name in GROUPS:
        mech = GROUPS[name]["mechanism"]
    elif name in COMPOSITES:
        mech = f"复合: {'+'.join(COMPOSITES[name])}"
    elif len(chans) == 1:
        g = group_of(chans[0])
        mech = f"单通道(边际贡献)" + (f"; 分组消融里属于 {g}" if g else "")
    else:
        mech = "自定义通道列表"
    return {"group": name, "channels": chans, "n_channels": len(chans), "mechanism": mech}
