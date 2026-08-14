# ERA5–Daymet Downscaling

把 ERA5 再分析（0.25°，120×240）降尺度到 Daymet 观测网格（2.5 arcmin，720×1440）的 CONUS
实验代码，空间倍率 6×。三条基线并行比较：统计方法（插值 / BCSD）、确定性神经网络
（UNet / ViT）、生成式模型（NVIDIA CorrDiff 的两阶段残差扩散）。

## 任务设置

| 项 | 设置 |
|---|---|
| 条件输入 | 20 通道 = 17 个 ERA5 动态变量（双线性上采样 6×）+ 3 个 Daymet 静态场（Δz / landcover / land_sea_mask） |
| 预测目标 | tmax、tmin、24 小时降水 |
| 数据划分 | train 1980–2017 / val 2018–2019 / test 2020，365 天历 |
| 降水处理 | ×1000 → mm → <0.1 mm 置零 → log1p → z-score；评测时逆变换 |
| 有效域 | 只在 Daymet 陆地掩膜上计算 loss 与指标 |

降水在 `log1p(mm)` 与 `m/day` 两种空间都评测过，两者的 RMSE 之间没有换算关系，排名甚至相反；
跨方法比较前必须确认单位空间一致。

## 代码布局

维护中的实现全部在 `code/era5_daymet/` 包内；`code/` 下的零散 `.py` 只是旧命令的兼容入口，
不放新实现。

```text
era5_daymet/
├── contract.py    数据合同: 通道顺序、目标变量、空间倍率、降水单位空间
├── data/          数据发现、读写、取数与归一化统计
├── models/        SongUNet、EDM preconditioning、patching、JiT 主干、分块推理等模型组件
├── training/      确定性、ViT、UNet、SCD、CorrDiff 与 JiT 的训练入口
├── baselines/     统计基线与 BCSD
├── evaluation/    打分原语、多方法统一评估与采样落盘
├── tools/
│   ├── preprocessing/
│   ├── diagnostics/
│   ├── plotting/
│   └── reporting/
└── tests/
    └── distributed/
```

### 训练与模型

| 功能 | 代码 | 主要职责 |
|---|---|---|
| 共享训练核心 | `training/train_downscale.py` | 分布式初始化、训练与验证循环、断点续训契约、命令行参数；数据合同、取数、打分原语与分块推理已各自独立成模块，此处按旧名转发以兼容既有调用 |
| 数据合同 | `contract.py` | 17 个 ERA5 动态通道的强制顺序、目标变量、空间倍率、降水 log1p 变换与钳制上界 |
| UNet | `training/train_unet.py` | UNet baseline 训练入口 |
| ViT crop / global | `training/train_vit.py` | ViT、Transformer block、位置编码与上采样头；crop 与 full-frame 两种训练方式 |
| CorrDiff | `training/train_corrdiff.py`、`training/train_stage_b.py` | 两阶段 CorrDiff：确定性均值模型、残差扩散训练与集合采样 |
| JiT / JiTMoE | `training/train_jit.py`、`models/jit_backbone.py`、`models/moe_ffn.py`、`models/jit_sampler.py` | 整幅像素空间条件扩散（x-prediction + v 空间损失），可选 DeepSeek 风格稀疏 FFN |
| SCD | `training/train_scd.py` | Scale-Consistent Decomposition 扩散模型 |
| 序列并行注意力 | `models/seq_parallel_attn.py` | global ViT 的 sequence-parallel attention、切分/聚合与梯度同步 |

`models/` 下的 `song_unet.py`、`preconditioning.py`、`patching.py`、`stochastic_sampler.py`
移植自 NVIDIA PhysicsNeMo（Apache-2.0），`tests/test_vendored_equivalence.py` 以「同权重下
输出逐比特相同」作为移植可信度的依据。

### 数据与统计基线

| 功能 | 代码 | 主要职责 |
|---|---|---|
| ERA5–Daymet 数据匹配 | `data/match_era5_daymet.py` | 按日期配对 ERA5 与 Daymet 文件，定义年份划分 |
| 训练集统计量 | `data/compute_norm_stats.py` | 仅用训练年份计算均值、标准差与气候态统计 |
| 取数与归一化统计 | `data/dataset.py` | `Stats` 加载训练集统计与气候态；`DownscaleData` 按年持有场并切出 (cond, target, mask, 原值真值)，训练与评测共用同一实现 |
| 公共数据工具 | `data/downscale_baseline.py` | 文件读取、插值、掩膜、单位转换与基础指标 |
| 统计降尺度 | `baselines/train_statistical.py` | 训练并评估插值与 BCSD |
| BCSD 系数拟合 | `baselines/fit_bcsd_coefs.py` | 拟合并保存逐网格 BCSD 参数 |

### 评估与诊断

| 功能 | 代码 | 主要职责 |
|---|---|---|
| 打分原语 | `evaluation/metrics.py` | 集合 CRPS（含逐像素场）、名次直方图、径向功率谱、分析窗口选取；纯 numpy |
| 分块推理 | `models/tiled_inference.py` | 确定性模型的整帧/分块预测与羽化加权融合，训练验证与评测共用 |
| 统一评估工具 | `evaluation/eval_common.py` | RMSE、MAE、bias、correlation、CRPS、SSIM、频谱与空间图 |
| 多模型统一评估 | `evaluation/eval_all_methods.py` | 同日期、同单位、同指标口径下比较各方法 |
| BCSD 双空间评估 | `evaluation/eval_bcsd_both_spaces.py` | 同时在物理空间与 `log1p` 空间评估 BCSD |
| CorrDiff 阶段 B 检验 | `tools/diagnostics/stage_b_big_check.py` | rank histogram、功率谱、逐月 CRPS/CRPSS、spread-skill、逐像素 CRPS |
| 分层与区域诊断 | `tools/diagnostics/stratified_eval.py`、`regional_seasonal_eval.py` | 按高程/起伏/离海距离与命名区域×逐月的多方法对比 |
| 绘图与汇报 | `tools/plotting/`、`tools/reporting/` | 结果图与指标汇总 |

## 代码所有权

多个协作 session 并行工作时按下表划分改动权限。分档依据是"改动的影响范围"，不是目录名：

| 档 | 范围 | 规则 |
|---|---|---|
| 训练侧 | `training/**`、`code/submit/**` | 训练 session 自主改动 |
| 评测侧 | `evaluation/**`、`tools/plotting/**`、`tools/reporting/**`、`tools/diagnostics/**` | 评测 session 自主改动 |
| 共管 | `contract.py`、`data/**`、`models/**`、`paths.py` | 改动前需两侧确认 |
| 只读引入 | `models/song_unet.py`、`preconditioning.py`、`patching.py`、`stochastic_sampler.py` | 移植自上游，不改动 |

共管区之所以单列：`runs/STATUS.md` 的数据合同段由脚本从 `contract.py` 的常量派生，改动会让
该文件自动跟着变，而已记录实验的元数据不会——两者一旦分叉，跨实验比较就在无声地失效。
同理，`data/dataset.py` 决定条件通道的拼接顺序与降水变换，任一侧另起炉灶都会让验证集与
测试集的输入口径分叉。

方法特有的输入组装不在共管区：JiT 把加噪目标与条件拼成 21 通道在 `models/jit_backbone.py`，
CorrDiff 阶段 B 叠加 μ、全域插值与位置网格在 `models/patching.py` 与 `song_unet.py`，两者互不影响。

## 运行

```bash
python -m era5_daymet.training.train_vit --help          # 从 code/ 目录
python -m era5_daymet.evaluation.eval_all_methods --help
python code/train_vit.py --help                          # 旧路径仍可用
```

包内新增 import 一律用绝对路径：

```python
from era5_daymet.training.train_downscale import FullFrameDS
```

## 测试

```bash
python -m era5_daymet.tests.test_spec_contract       # 固定数据合同（通道/顺序/划分/降水管线）
python -m era5_daymet.tests.test_worker_sampling     # DataLoader 采样唯一性与分片
python -m era5_daymet.tests.test_crps_per_pixel      # CRPS 逐像素分解与分层可加性
python -m era5_daymet.tests.test_vendored_equivalence  # 移植代码与官方实现等价
python -m era5_daymet.tests.test_jit_moe             # DSMoE 稀疏 FFN 对拍与路由不变量
python -m era5_daymet.tests.test_jit_backbone        # JiT 主干几何与初始化契约
python -m era5_daymet.tests.test_jit_sampler         # ODE 采样解析检验与分布回收
python -m era5_daymet.tests.test_jit_resume          # train_jit 断点接力逐位复现与取帧分片
torchrun --nproc_per_node=2 -m era5_daymet.tests.test_jit_ddp_sync  # train_jit DDP 同步正反对照
```

`tests/distributed/` 用于验证 sequence parallel 的通信、网格划分与收敛性。

## 依赖

Python 3.11 + PyTorch（ROCm 或 CUDA）、numpy、scipy、matplotlib、xarray、netCDF4。
训练使用 AMP bfloat16。
