# ERA5–Daymet Downscaling Code Guide

本仓库用于代码检查；当前维护的实现集中在 `code/era5_daymet/`。代码主流程为：

`数据匹配与统计 → 模型训练 → 统一评估 → 专项诊断与汇报`

## 训练与模型

| 功能 | 代码 | 主要职责 |
|---|---|---|
| 共享训练核心 | `training/train_downscale.py` | 定义数据集与 DataLoader、归一化、UNet 主体、分布式训练、切片推理和公共评估流程 |
| UNet | `training/train_unet.py` | UNet baseline 的训练入口 |
| ViT crop / global | `training/train_vit.py` | 定义 ViT、Transformer block、位置编码和上采样结构，并负责 crop 与 full-frame/global 两种训练方式 |
| CorrDiff | `training/train_corrdiff.py` | 两阶段 CorrDiff：确定性均值模型、残差扩散训练和集合采样 |
| SCD | `training/train_scd.py` | Scale-Consistent Decomposition 扩散模型训练 |
| 序列并行注意力 | `models/seq_parallel_attn.py` | global ViT 使用的 sequence-parallel attention、切分/聚合与梯度同步 |

## 数据与统计基线

| 功能 | 代码 | 主要职责 |
|---|---|---|
| ERA5–Daymet 数据匹配 | `data/match_era5_daymet.py` | 查找并按日期配对 ERA5 与 Daymet 文件，定义数据年份划分和匹配流程 |
| 训练集统计量 | `data/compute_norm_stats.py` | 仅使用训练年份计算均值、标准差和气候态统计量 |
| 公共数据工具 | `data/downscale_baseline.py` | 提供文件读取、插值、掩膜、单位转换和基础指标等底层函数 |
| 统计降尺度 baseline | `baselines/train_statistical.py` | 训练并评估插值和 BCSD 等统计方法 |
| BCSD 系数拟合 | `baselines/fit_bcsd_coefs.py` | 拟合并保存逐网格 BCSD 参数 |

## 评估

| 功能 | 代码 | 主要职责 |
|---|---|---|
| 统一评估工具 | `evaluation/eval_common.py` | 汇总 RMSE、MAE、bias、correlation、CRPS、SSIM、频谱和空间图等公共评估逻辑 |
| 多模型统一评估 | `evaluation/eval_all_methods.py` | 在相同日期、单位和指标口径下比较统计方法、UNet、ViT 与 CorrDiff |
| BCSD 双空间评估 | `evaluation/eval_bcsd_both_spaces.py` | 同时在降水物理空间与 `log1p` 空间评估 BCSD |

## 数据预处理

| 功能 | 代码 |
|---|---|
| 将 ERA5 有效区域与 Daymet 网格对齐 | `tools/preprocessing/align_era5_to_daymet.py` |
| 提取固定样例帧供模型对比 | `tools/preprocessing/extract_golden_frames.py` |
| 补齐测试集中缺失的 ERA5 最后一天 | `tools/preprocessing/fill_era5_lastday.py` |

## ViT global 诊断

| 功能 | 代码 |
|---|---|
| 计算并绘制 ViT crop/global 的网格相位 bias | `tools/diagnostics/diagnose_vit_gridphase_bias.py` |
| 检查 ViT 网格结构与降水网格结构 | `tools/diagnostics/_eda_vit_grid.py`、`tools/diagnostics/_eda_vit_grid_precip.py` |
| 检查边界接缝和阶梯状误差 | `tools/diagnostics/_r0_seam_zoom.py`、`tools/diagnostics/_r0_staircase_check.py` |
| 探测 full-frame ViT 的数据与前向行为 | `tools/diagnostics/_probe_fullframe_vit.py` |
| 生成全年模型图并补充 ViT global | `tools/diagnostics/_annual_individual_maps.py`、`tools/diagnostics/_annual_vit_global.py` |

## 其他分析、绘图与汇报工具

| 功能 | 代码 |
|---|---|
| 数据局部性、误差分解和模型机制分析 | `tools/diagnostics/_eda_task_locality.py`、`_eda_error_decomp.py`、`_eda_bcsd_why.py`、`_eda_corrdiff.py` |
| 模型上限、频谱和统一诊断 | `tools/diagnostics/_eda_ceiling.py`、`_eda_unified.py` |
| 全幅降水对比图 | `tools/diagnostics/_full_precip_map.py`、`_full_precip_maps_all_vit.py` |
| 日常数据与模型结果绘图 | `tools/plotting/` |
| 汇总评估结果并更新 HTML 指标表 | `tools/reporting/` |

## 分布式测试

`tests/distributed/` 用于验证 sequence parallel 的通信、网格划分、收敛性和 ViT 端到端行为。

## 路径入口

`paths.py` 统一提供代码目录和项目目录定位，供包内工具使用。
