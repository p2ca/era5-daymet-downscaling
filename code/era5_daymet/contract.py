#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
contract.py — 数据合同: 通道顺序、目标变量、空间倍率与降水单位空间
============================================================================
本模块是这些口径的唯一定义处, 训练与评测都从这里取。改动其中任何一项都会让既有实验
失去可比性(条件通道换序会静默改变模型输入, 降水钳制上界会改变反变换结果), 因此它是
两侧共管的冻结区: 修改前须同时确认训练侧与评测侧, 并在实验记录中注明口径变更。

`runs/STATUS.md` 的数据合同段由脚本从本模块的常量派生, 不是手写, 因此这里改了那边会
跟着变, 而已有实验的记录不会 —— 这正是不可随手改动的原因。

降水存在两种单位空间: `log1p(mm)` 与 `m/day`。两者的 RMSE 之间没有换算关系, 排名甚至
相反; 任何跨方法比较前先确认单位一致。
============================================================================
"""
import numpy as np

# 空间倍率: ERA5 双线性上采样到 Daymet 网格的倍数
FACTOR = 6

# 预测目标(单目标模型逐个训练, 多目标模型一次输出全部)
TARGETS = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
PRECIP = "total_precipitation_24hr"

# 输入合同: 15 个 ERA5 动态变量, 必须严格按此顺序读取。
# 加 4 个 Daymet 静态通道与 2 个时间通道 => 条件通道 = 15+4+2 = 21(默认)。
DEFAULT_IN = ["2m_temperature", "2m_temperature_max", "2m_temperature_min",
              "total_precipitation_24hr",
              "volumetric_soil_water_layer_1", "geopotential_500", "geopotential_850",
              "specific_humidity_500", "specific_humidity_850", "temperature_500", "temperature_850",
              "u_component_of_wind_500", "u_component_of_wind_850",
              "v_component_of_wind_500", "v_component_of_wind_850"]

# 静态与时间通道的符号名。这里的顺序**就是**条件张量里静态段与时间段的拼接顺序: 取数
# (DownscaleData.get_patch)与通道敏感性分组都从这里派生, 不在别处再写一份。通道错位不会
# 报错, 只会让指标悄悄变差, 因此布局只能有一个定义处。
DZ = "dz"                       # HR 高程 − 上采样 LR 高程: 亚网格地形, 递减率订正的来源
ELEVATION = "elevation"         # HR 绝对高程: 递减率与气压高度的绝对参考
LANDCOVER = "landcover"
LAND_SEA_MASK = "land_sea_mask"
STATIC_ORDER = (DZ, ELEVATION, LANDCOVER, LAND_SEA_MASK)

DOY_SIN = "doy_sin"
DOY_COS = "doy_cos"
TIME_ORDER = (DOY_SIN, DOY_COS)

# 日历每年恒 365 天(闰年丢 12/31), 所以"年内第几天"直接就是季节相位, 与逐日气候态的
# slot 同口径, 跨年对齐无漂移。
DAYS_PER_YEAR = 365

# log1p(mm) 的物理上界: 世界日降水纪录约 1825 mm -> log1p ≈ 7.51。取 8.0 (≈2980 mm) 已极宽松。
PRECIP_LOG_MAX = 8.0


def doy_sincos(day_index):
    """年内第 day_index 天(0 起) -> (sin, cos), 用作空间上为常数的时间条件通道。

    (sin, cos) 对 day index 是双射, 年内相位无损, 且 12/31 -> 1/1 处不跳变; 只取 sin 会让
    春秋各有一天落到同一个值上, 网络分不出来。二次谐波 sin(4πd/D) = 2·sin(2πd/D)·cos(2πd/D)
    只是这两个通道的乘积, 带逐点非线性的网络自己就能合成, 所以不另设通道。
    值域已在 [-1,1] 且全年均值为 0, 不再做 z-score。
    """
    a = 2.0 * np.pi * (float(day_index) % DAYS_PER_YEAR) / DAYS_PER_YEAR
    return float(np.sin(a)), float(np.cos(a))


def cond_layout(in_vars):
    """条件张量的通道名列表, 顺序即 DownscaleData.get_patch 的拼接顺序:
    ERA5 动态 -> 静态(STATIC_ORDER) -> 时间(TIME_ORDER)。"""
    return list(in_vars) + list(STATIC_ORDER) + list(TIME_ORDER)


def cond_channels(in_vars):
    """条件输入通道数 = len(cond_layout(in_vars)); 15 动态 + 4 静态 + 2 时间 = 21。

    训练/评测/构模所有地方都用它, 保证与 DownscaleData.get_patch 拼出的 cond 通道数一致。"""
    return len(cond_layout(in_vars))


def precip_fwd(p, clip, scale):
    """降水建模空间: ×scale 变毫米 -> drizzle clip -> log1p(与 compute_norm_stats 一致)。"""
    p = np.maximum(p, 0.0) * scale
    p = np.where(p < clip, 0.0, p)
    return np.log1p(p)


def precip_inv(x, scale, max_log=PRECIP_LOG_MAX, return_clipped=False):
    """log1p(mm) -> m/day。

    必须钳制: 这是 expm1, 是指数函数。生成式模型(扩散)的样本有重尾, 只要有一个离群像素
    (比如 log 空间值 50), expm1(50)≈5e21 就会把整幅图的 RMSE 炸成 inf。确定性方法从不
    触发这个 —— L2 损失把它们的输出压得很平, 永远到不了那个量级 —— 只有生成式模型才会
    走到需要钳制的区间。

    钳到 8.0 对任何真实降水都是无操作(远超世界纪录), 因此不改变任何已记录的确定性结果;
    它只在生成式模型吐出物理上不可能的值时兜底。return_clipped=True 会一并返回被钳的
    像素比例 —— 这个数字本身是个诊断: 训练良好的模型应该≈0, 若显著>0 就是红旗。
    """
    n_clip = float(np.mean(x > max_log)) if return_clipped else 0.0
    p = np.maximum(np.expm1(np.minimum(x, max_log)), 0.0) / scale
    return (p, n_clip) if return_clipped else p
