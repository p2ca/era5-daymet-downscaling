#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
contract.py — 4x 跨产品降尺度的数据合同
============================================================================
ERA5(0.25 度, 120x240) -> Daymet(3.75 arcmin, 480x960), 空间倍率 4。

条件通道分四段, 顺序即本模块 `cond_layout()` 的返回顺序, 也是取数拼接的唯一依据:

    ERA5 当天(15)  ->  静态(4)  ->  年内相位(2)  ->  因果历史(30)

前三段共 21 个通道, 与上一代合同逐通道一致; 本合同相对它的唯一增量就是最后那段历史。
三档条件模式的通道数为 21 / 21 / 51, 其中两档 21 通道的差别不在通道而在帧集合。

降水存在 `log1p(mm)` 与 `m/day` 两种单位空间, RMSE 之间没有换算关系, 排名甚至相反;
任何跨方法比较前先确认单位一致。
============================================================================
"""
import numpy as np

# 空间倍率与网格。HR/LR 必须恰为 FACTOR 倍, 由 check_shapes() 兜底。
FACTOR = 4
LR_SHAPE = (120, 240)
HR_SHAPE = (480, 960)

# 预测目标(单目标模型逐个训练)
TARGETS = ["2m_temperature_max", "2m_temperature_min", "total_precipitation_24hr"]
PRECIP = "total_precipitation_24hr"

# ERA5 动态输入, 必须严格按此顺序读取。数据集里还有 10m 纬向/经向风, 本合同不取:
# 近地面风与高空风信息重叠, 且不进条件张量可让当天段与上一代口径逐通道一致。
ERA5_IN = ["2m_temperature", "2m_temperature_max", "2m_temperature_min",
           "total_precipitation_24hr",
           "volumetric_soil_water_layer_1",
           "geopotential_500", "geopotential_850",
           "specific_humidity_500", "specific_humidity_850",
           "temperature_500", "temperature_850",
           "u_component_of_wind_500", "u_component_of_wind_850",
           "v_component_of_wind_500", "v_component_of_wind_850"]

# 静态通道。第一个是**亚网格地形 Δz**, 不是绝对高程:
#   Δz = Daymet 高分辨率高程 − 双线性上采样的 ERA5 粗网格高程
# 它是 4x 降尺度里唯一能产生亚网格温度结构的输入; 数据源是两侧 static.npz 的 orography,
# 但入通道的是二者之差, 源文件名与通道语义不是一回事。
DZ = "dz"
ELEVATION = "elevation"         # 高分辨率绝对高程: 递减率与气压高度的绝对参考
LANDCOVER = "landcover"
LAND_SEA_MASK = "land_sea_mask"
STATIC_ORDER = (DZ, ELEVATION, LANDCOVER, LAND_SEA_MASK)

# 每个静态通道的归一化方式与所用统计量的键名:
#   scale  只除标准差、不减均值 —— Δz 天然零中心, 填 0 恰好等于"高分辨率地形 = 粗网格地形",
#          语义干净; 若再减一个非零均值, 填 0 就不再对应这个物理含义
#   zscore 减均值再除标准差 —— 绝对高程均值约 800 m, 不减均值会给整幅压一个常数偏置
#   raw    原值(0/1)
# Δz 与绝对高程共用同一份 orography 统计量, 只是用法不同; 这个耦合写在这里, 不散在取数里。
STATIC_NORM = {DZ: ("scale", "orography"),
               ELEVATION: ("zscore", "orography"),
               LANDCOVER: ("zscore", "landcover"),
               LAND_SEA_MASK: ("raw", None)}

# 静态通道在有效域外一律填 0: 那里没有可用的地形与地表信息, 留原值等于把无意义的数值
# 喂给卷积, 而感受野会把它带进域内边缘的预测。
STATIC_OCEAN_FILL = 0.0

# **有效域 = 目标有真值 且 输入有数据**, 即 Daymet 陆地掩膜与 ERA5 valid_mask 的交集。
# 缺一不可: 只有目标没有输入的地方, 模型只能靠边界外推去猜, 那里的得分衡量的是填充策略
# 而不是模型; 只有输入没有目标的地方本来就无从算 loss。
#
# ERA5 的 valid_mask 是上一代 Daymet 产品的覆盖范围按 FACTOR 粗化而来, 因此这个交集在数据上
# 就等于上一代合同的 CONUS 范围 —— 它是从两侧数据推出来的, 不是手画的经纬框。
# 现行 4x Daymet 产品的陆地掩膜比它大得多(把加拿大与墨西哥也算作陆地), 直接拿它当有效域会
# 让三分之一的域没有输入; 那种情况不会报错, 只会让指标里混进一段与模型无关的分数。
EFFECTIVE_DOMAIN = "daymet_land AND era5_valid"


# 时间通道: 年内相位, 运行时生成, 不进归一化档案也不进年度 npz。
# 不设 x/y 位置平面: 主干网络自带位置编码(UNet 的 pos_grid、扩散主干的位置网格),
# 再从输入端注入一份会与之重复。
DOY_SIN = "doy_sin"
DOY_COS = "doy_cos"
TIME_ORDER = (DOY_SIN, DOY_COS)

# 因果历史: 前两天的同一套 ERA5 变量, 顺序固定旧->新。
HISTORY_LAGS = (2, 1)
HISTORY_PREFIX = "t_minus_"

# 日历每年恒 365 天(闰年丢 12/31), 因此 day_index 直接就是年内相位。
# 闰年缺 12/31 会让次年 1 月 1-2 日取不到完整历史, 那两帧按规则丢弃, 不复制也不 wrap。
DAYS_PER_YEAR = 365
LEAP_DROP = "dec31"

# 降水: m/day -> mm -> 小于 clip 置零 -> log1p。反变换 expm1 前钳到此上界。
# 世界日降水纪录约 1825 mm -> log1p 约 7.51; 取 8.0(约 2980 mm)已极宽松。
PRECIP_LOG_MAX = 8.0
PRECIP_SCALE = 1000.0
PRECIP_CLIP_MM = 0.1

# 域的地理范围(度)。lat/lon 数组给的是网格单元下边界, ERA5 与 Daymet 在此域上严格套合:
# 一个 ERA5 单元恰好覆盖 FACTOR x FACTOR 个 Daymet 单元。绘图 extent 与任何按经纬选区
# 都用它, 并由 data.grid.check_domain() 对着实际 lat/lon 文件核对。
DOMAIN_LAT = (24.0, 54.0)
DOMAIN_LON = (-125.0, -65.0)

# 条件模式 -> (历史 lag, 配对所需历史天数)。
# baseline 的 21 个通道与上一代合同逐通道一致, 因此新旧两条线的差别只剩空间倍率与目标网格。
# control 的通道与 baseline 完全相同, 只是要求同样的历史窗口, 于是帧集合与 history 档逐帧
# 相同 —— 衡量历史通道的增量必须拿 history 去比 control, 比 baseline 会把"训练集变小"
# 混进来, 而两者通道数一样, 不会有任何东西报错。
MODES = {
    "baseline_21":        ((),           0),
    "history_control_21": ((),           2),
    "history_51":         (HISTORY_LAGS, 2),
}
DEFAULT_MODE = "history_51"


def history_name(lag, var):
    """历史通道名, 与 lag 和变量名一一对应。"""
    return f"{HISTORY_PREFIX}{lag}:{var}"


def cond_layout(mode=DEFAULT_MODE):
    """条件张量的通道名列表; 顺序即拼接顺序, 是全包唯一的布局定义处。"""
    if mode not in MODES:
        raise ValueError(f"未知条件模式 {mode!r}; 可用 {sorted(MODES)}")
    lags, _ = MODES[mode]
    names = list(ERA5_IN) + list(STATIC_ORDER) + list(TIME_ORDER)
    for lag in lags:
        names += [history_name(lag, v) for v in ERA5_IN]
    return names


def cond_channels(mode=DEFAULT_MODE):
    """条件输入通道数 = len(cond_layout(mode))。"""
    return len(cond_layout(mode))


def history_lags(mode=DEFAULT_MODE):
    return MODES[mode][0]


def pairing_history_days(mode=DEFAULT_MODE):
    """建帧集合时要求前几天必须存在; 大于 0 的模式共享同一个帧集合。"""
    return MODES[mode][1]


def doy_sincos(day_index):
    """年内第 day_index 天(0 起) -> (sin, cos), 用作空间上为常数的时间条件通道。

    (sin, cos) 对 day index 是双射, 年内相位无损, 且 12/31 -> 1/1 处不跳变; 只取 sin 会让
    春秋各有一天落到同一个值上。值域已在 [-1,1] 且全年均值为 0, 不再做 z-score。
    """
    a = 2.0 * np.pi * (float(day_index) % DAYS_PER_YEAR) / DAYS_PER_YEAR
    return float(np.sin(a)), float(np.cos(a))


def precip_fwd(x, clip_mm=PRECIP_CLIP_MM, scale=PRECIP_SCALE):
    """降水正变换: m/day -> mm -> 小于 clip_mm 的毛毛雨置零 -> log1p。"""
    mm = np.maximum(np.asarray(x, np.float32), 0.0) * scale
    return np.log1p(np.where(mm < clip_mm, 0.0, mm))


def precip_inv(x, scale=PRECIP_SCALE, log_max=PRECIP_LOG_MAX):
    """降水反变换: log1p(mm) -> m/day; 先钳上界再 expm1, 免得离群值放大成天文数字。"""
    return np.expm1(np.clip(np.asarray(x, np.float32), 0.0, log_max)) / scale


def check_shapes(lr_shape, hr_shape):
    """LR/HR 形状必须恰为 FACTOR 倍且与合同一致; 不符当场抛错。"""
    lr, hr = tuple(lr_shape[-2:]), tuple(hr_shape[-2:])
    if lr != LR_SHAPE or hr != HR_SHAPE:
        raise ValueError(f"网格与合同不符: LR {lr} 期望 {LR_SHAPE}, HR {hr} 期望 {HR_SHAPE}")
    if tuple(h // l for h, l in zip(hr, lr)) != (FACTOR, FACTOR) or \
       tuple(h % l for h, l in zip(hr, lr)) != (0, 0):
        raise ValueError(f"HR {hr} 不是 LR {lr} 的 {FACTOR} 倍")
