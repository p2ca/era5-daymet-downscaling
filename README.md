# ERA5–Daymet Downscaling

把 ERA5 再分析（0.25°，120×240）降尺度到 Daymet 观测网格（2.5 arcmin，720×1440）的 CONUS
实验代码，空间倍率 6×。三条基线并行比较：统计方法（插值 / BCSD）、确定性神经网络
（UNet / ViT）、生成式模型（NVIDIA CorrDiff 的两阶段残差扩散）。

## 任务设置

| 项 | 设置 |
|---|---|
| 条件输入 | 21 通道 = 15 个 ERA5 动态变量（双线性上采样 6×）+ 4 个 Daymet 静态场（Δz / 绝对高程 / landcover / land_sea_mask）+ 2 个 day-of-year 时间通道（sin / cos） |
| 预测目标 | tmax、tmin、24 小时降水 |
| 数据划分 | train 1980–2017 / val 2018–2019 / test 2020，365 天历 |
| 降水处理 | ×1000 → mm → <0.1 mm 置零 → log1p → z-score；评测时逆变换 |
| 有效域 | 只在 Daymet 陆地掩膜上计算 loss 与指标 |

降水在 `log1p(mm)` 与 `m/day` 两种空间都评测过，两者的 RMSE 之间没有换算关系，排名甚至相反；
跨方法比较前必须确认单位空间一致。

## 模型命名

单目标模型统一用「家族前缀 + 目标序号」命名，序号 1/2/3 = tmax/tmin/precip：

| 家族 | tmax | tmin | precip |
|---|---|---|---|
| CorrDiff 阶段A (μ) | CA1 | CA2 | CA3 |
| CorrDiff 阶段B (残差扩散) | CB1 | CB2 | CB3 |
| JiT dense | JD1 | JD2 | JD3 |
| JiT MoE | JM1 | JM2 | JM3 |

UNet U1–U3 与 ViT-V3 的数字是规模/版本档，语义与上表不同。早期实验目录名与文档中的
C1/C2/C3 旧称按家族对应上表；实验目录名是历史 id，不随命名调整改名。规范名由
`tools/reporting/model_names.py` 依 meta.json 的 (method, target) 推导，指标汇总自动使用。

## 两条并行的实验线

仓库同时有两个互不依赖的包，各自持有一份完整的数据合同：

| 包 | 合同 | 状态 |
|---|---|---|
| `code/era5_daymet/` | 6×，ERA5 120×240 → Daymet 720×1440，21 条件通道 | 冻结，只读既有结果 |
| `code/downscaling_4x/` | 4×，ERA5 120×240 → Daymet 480×960，条件通道 = 上一代那 21 个 + 可选 t−2/t−1 因果历史（共 51） | 在建 |

后者的当天段（15 个 ERA5 动态 + 4 个静态 + 2 个年内相位）与上一代逐通道一致，唯一的增量是
前两天的 ERA5 历史帧；不注入 x/y 位置平面，因为主干网络自带位置编码。

**有效域 = 目标有真值 且 输入有数据**，即 Daymet 陆地掩膜与 ERA5 `valid_mask` 的交集，
在 480×960 网格上是 219,069 格。ERA5 的 `valid_mask` 是上一代产品覆盖范围粗化而来，所以这个
交集就是上一代合同的 CONUS 范围——它由两侧数据推出，不是手画的经纬框。4× Daymet 产品的
陆地掩膜本身大得多（把加拿大与墨西哥也算陆地），但那三分之一的区域没有 ERA5 输入，直接
拿来打分等于在衡量缺测填充策略而不是模型。

它把任务定性为**跨产品映射**而非同产品超分辨率：ERA5 与 Daymet 出自不同的生成系统，两者
对同一天的雨区位置、湿区频率与强度本就不一致，因此目标不是唯一还原当天的细尺度场，而是
利用预测时真实可用的信息取得可验证的改进。

两条线的指标仍然不可比：空间倍率与目标网格不同（6× 的 720×1440 vs 4× 的 480×960），
只有评测域是一致的。

`code/downscaling_4x/` 由 `code/era5_daymet/` 复制后改动得来。
`python -m downscaling_4x.tools.port_manifest` 随时算出两者的逐文件对照：哪些逐字节相同
（即与合同无关）、哪些改过（附实质 diff 行数）、哪些是新写的。

## 代码布局

上一代维护中的实现全部在 `code/era5_daymet/` 包内；`code/` 下的零散 `.py` 只是旧命令的
兼容入口，不放新实现。

```text
era5_daymet/
├── contract.py    数据合同: 通道顺序、目标变量、空间倍率、降水单位空间
├── data/          数据发现、读写、取数与归一化统计
├── models/        SongUNet、EDM preconditioning、patching、JiT 主干、分块推理等模型组件
├── training/      确定性、ViT、UNet、SCD、CorrDiff 与 JiT 的训练入口
├── baselines/     统计基线与 BCSD
├── evaluation/    打分原语、逐日场落盘、统一指标与渲染管线
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
| 数据合同 | `contract.py` | 15 个 ERA5 动态通道的强制顺序、静态与时间通道的拼接布局、目标变量、空间倍率、降水 log1p 变换与钳制上界 |
| UNet | `training/train_unet.py` | UNet baseline 训练入口 |
| ViT crop / global | `training/train_vit.py` | ViT、Transformer block、位置编码与上采样头；crop 与 full-frame 两种训练方式 |
| CorrDiff | `training/train_unet.py --arch corrdiff`（阶段 A）、`training/train_stage_b.py`（阶段 B） | 两阶段 CorrDiff：确定性均值模型 μ，再由移植自官方的 `ResidualLoss` + EDM 预条件训练残差扩散 |
| EDM 扩散与分块采样 | `models/edm_diffusion.py` | EDM 损失、Algorithm-2 二阶采样与羽化融合的分块集合采样；与条件通道口径无关 |
| JiT / JiTMoE | `training/train_jit.py`、`models/jit_backbone.py`、`models/moe_ffn.py`、`models/jit_sampler.py` | 整幅像素空间条件扩散（x-prediction + v 空间损失），可选 DeepSeek 风格稀疏 FFN |
| SCD | `training/train_scd.py` | Scale-Consistent Decomposition 扩散模型 |
| 序列并行注意力 | `models/seq_parallel_attn.py` | global ViT 的 sequence-parallel attention、切分/聚合与梯度同步 |

`models/` 下的 `song_unet.py`、`preconditioning.py`、`patching.py`、`stochastic_sampler.py`
移植自 NVIDIA PhysicsNeMo（Apache-2.0）。移植当时以「同权重下输出逐比特相同」验收，该验收
固化在 `tests/test_vendored_equivalence.py` 里。**这些文件可以修改**——上游本身存在缺陷，
逐比特跟随上游没有意义；但每一处有意偏离都必须登记在该测试的偏离表中，使「我们与上游差在
哪里」是一份可执行、被审查的清单，而不是口口相传。

上游快照存放在 `reference_corrdiff_official/`（与上游逐字节一致），供对照与查阅；
它是只读参照，不要修改，否则失去参照价值。

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
| 打分原语 | `evaluation/metrics.py` | 集合 CRPS（含逐像素场）、名次直方图、径向功率谱、分析窗口选取；SSIM 三件套 `ssim_masked` 标量 / `ssim_field` 整幅图 / `ssim_components` 的 l·c·s 三分量共用一份口径，标量即图在腐蚀掩膜上的均值；纯 numpy |
| 分块推理 | `models/tiled_inference.py` | 确定性模型的整帧/分块预测与羽化加权融合，训练验证与评测共用 |
| 统一评估工具 | `evaluation/eval_common.py` | RMSE、MAE、bias、correlation、CRPS、SSIM 的累加与汇总 |
| 确定性方法落场 | `evaluation/det_dump.py` | UNet / ViT / CorrDiff 阶段A μ / BCSD / 插值的整年逐日场，布局与生成式采样落盘一致 |
| 逐日场 → 标量指标 | `evaluation/dump_metrics.py` | 从落场目录算 MAE/CRPS、RMSE、CORR、SSIM；降水另给 `log1p(mm)` 一套 |
| 场 → 图 | `evaluation/render/` | 年均场/年均 bias、单日场/单日 bias 与单日真值（`--days` 显式指定）、区域与月份错误贡献排名、失败区逐月图；同组共色标并落 `scales.json`。`--display-smooth` 与 `--interp` 只改画出来的样子：前者在有效域内做归一化卷积高斯（域外 NaN 不参与也不被填充），平滑量自动写进图标题，两者都随 `manifest.json` 落盘；**指标一律取自未平滑的场** |
| 整年前向 | `evaluation/fields.py` | 从 checkpoint 现算逐日物理场的唯一实现，落场与诊断共用 |
| 各方法预测构造 | `evaluation/predictors.py` | checkpoint / 预拟合系数 / 插值算子 → 统一的当日预测可调用体 |
| BCSD 双空间评估 | `evaluation/eval_bcsd_both_spaces.py` | 同时在物理空间与 `log1p` 空间评估 BCSD |
| CorrDiff 阶段 B 检验 | `tools/diagnostics/stage_b_big_check.py` | rank histogram、功率谱、逐月 CRPS/CRPSS、spread-skill、逐像素 CRPS |
| 分层与区域诊断 | `tools/diagnostics/stratified_eval.py`、`regional_seasonal_eval.py` | 按高程/起伏/离海距离与命名区域×逐月的多方法对比 |
| 损失曲线 | `evaluation/plot_loss.py` | 训练损失曲线的唯一出图入口；横轴由 `loss_history.json` 的键自动决定（epoch 或累计样本数），两种横轴不允许画进同一张图 |
| 单元/像素子集打分 | `tools/diagnostics/stage_b_worst_cells.py`、`stage_b_worst_pixels.py` | 按相对劣势或自身分数挑出最差的一批 (区域,日) 或 (像素,日)，在该集合与其余集上池化 MAE/RMSE/bias/corr/CRPS/SSIM；全集池化必须复现落场 `metrics.json`，不一致即退出 |
| SSIM 排序的子集 | `tools/diagnostics/stage_b_ssim_sets.py` | 按逐像素 SSIM 取最差/最优比例的两个独立二分；逐像素数组缓存带身份校验（模型、年份、天数、掩膜校验和、落场内容指纹），拿错缓存会被拒绝 |
| 尺度分带（多模型） | `tools/diagnostics/scale_band_multi.py` | 任意多个模型的逐波长带 MAE/误差方差/相关/振幅比；波长带与带通口径引用 `scale_band_compare`，两者在同样日期上逐位一致 |
| 专家路由出图 | `tools/plotting/plot_routing.py`、`plot_expert_region_month.py`、`plot_expert_by_ssim_set.py` | 前者出地图与全部路由量；后两者只出 (区域×月)，用按 offset 缓存的 token→区域权重直接求和，不把 token 铺成像素 |
| 路由容量与地形 | `tools/plotting/plot_capacity_vs_terrain.py` | 逐像素平均激活专家数与地形粗糙度的关系：晕渲底图叠 k 等值线、k 填色叠高程等值线、粗糙度-k 的分箱曲线、扣掉粗糙度后的残差图 |
| 子集上的技能与增益 | `tools/plotting/plot_gain_by_ssim_set.py`、`plot_cross_set_gain.py` | 前者出 (区域×月) 相对阶段A μ 的 CRPSS；后者出交叉集合下的 CRPS 相对差，并给两个方向的中点以抵消“去掉了谁的哪一段”造成的偏移 |
| 绘图与汇报 | `tools/plotting/`、`tools/reporting/` | 结果图与指标汇总 |

## 代码所有权

多个协作 session 并行工作时按下表划分改动权限。分档依据是"改动的影响范围"，不是目录名：

| 档 | 范围 | 规则 |
|---|---|---|
| 训练侧 | `training/**`、`code/submit/**` | 训练 session 自主改动 |
| 评测侧 | `evaluation/**`、`tools/plotting/**`、`tools/reporting/**`、`tools/diagnostics/**` | 评测 session 自主改动 |
| 共管 | `contract.py`、`data/**`、`models/**`、`paths.py` | 改动前需两侧确认 |

共管区之所以单列：`runs/STATUS.md` 的数据合同段由脚本从 `contract.py` 的常量派生，改动会让
该文件自动跟着变，而已记录实验的元数据不会——两者一旦分叉，跨实验比较就在无声地失效。
同理，`data/dataset.py` 决定条件通道的拼接顺序与降水变换，任一侧另起炉灶都会让验证集与
测试集的输入口径分叉。

方法特有的输入组装不在共管区：JiT 把加噪目标与条件拼成 22 通道在 `models/jit_backbone.py`，
CorrDiff 阶段 B 叠加 μ、全域插值与位置网格在 `models/patching.py` 与 `song_unet.py`，两者互不影响。

## 运行

```bash
python -m era5_daymet.training.train_vit --help          # 从 code/ 目录
python -m era5_daymet.evaluation.det_dump --help
python code/train_vit.py --help                          # 旧路径仍可用
```

包内新增 import 一律用绝对路径：

```python
from era5_daymet.training.train_downscale import FullFrameDS
```

## 测试

```bash
python -m era5_daymet.tests.test_spec_contract       # 固定数据合同（通道/顺序/划分/降水管线）
python -m era5_daymet.tests.test_cond_channels       # 条件张量内容与合同布局逐通道对拍（需真实数据）
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

### 4× 线（`downscaling_4x`）

全部可在 login node 跑完，不需要 GPU，也不占队列：

```bash
python -m downscaling_4x.tests.test_spec_contract    # 合同: 通道布局/三档模式/倍率/降水管线/闰年历
python -m downscaling_4x.tests.test_causal_pairing   # 因果配对与"年内减索引"负对照
python -m downscaling_4x.tests.test_cond_channels    # 真实数据上逐槽位对拍（需真实数据）
python -m downscaling_4x.tests.test_train_resume     # 断点续训逐位复现、契约守卫、取帧无放回
python -m downscaling_4x.tests.test_joint_target     # --target all 三目标联合: 输出通道/逐通道梯度/形状守卫
python -m downscaling_4x.tests.test_arch_defaults    # 按 --arch 回填的口径默认值逐项钉死; 显式传参优先
python -m downscaling_4x.tests.test_input_guard      # 输入身份守卫: checkpoint/系数/μ缓存记录的输入目录与本次不一致即拒绝
python -m downscaling_4x.tests.test_improve_criterion # val 改善判据开关(abs/rel): 缺省与既有 run 一致、rel 与量级无关、续训契约钉住
python -m downscaling_4x.tests.test_precip_log_space # 降水 log1p(mm) 空间唯一定义: 统计基线与落场两条路径同场同数
python -m downscaling_4x.tests.test_oracle_inputs    # Daymet-oracle 输入目录与合同取数逐项吻合（需真实数据）
python -m downscaling_4x.tests.test_dec_router       # D-EC 路由: 全关=ec、域外永不入选、跨帧池、先验只进选择、难度头 stop-grad、网格同源、契约钉住
python -m downscaling_4x.tests.test_routing_dump     # MoE 路由截获: 只读补丁、计数不变量、D-EC 直方图对拍、专家输出范数钩子
python -m downscaling_4x.tests.test_jit_two_stream   # 两条流开关契约: 缺省=单流逐位相同、开关矩阵、按名拆流、公里制与风推移、粗token掩膜、缓存等价、域外舍弃、两级分诊、键零初始化、续训契约
torchrun --nproc_per_node=2 -m downscaling_4x.tests.test_ddp_grad_sync  # DDP 包装体梯度同步
torchrun --nproc_per_node=2 -m downscaling_4x.tests.test_train_ddp      # 真实训练入口的跨 rank 一致性
torchrun --nproc_per_node=2 -m downscaling_4x.tests.test_jit_ddp_sync   # train_jit 接线: dense / moe / dec / 两条流全开 四档的梯度同步与首步自检
```

后两条用 gloo 后端跑在 CPU 上。`DDP_SYNC_TEST_BYPASS=1` 会人为制造"前向绕过 DDP 包装体"
的缺陷，用于验证测试自身有分辨力（预期失败）。

训练入口：

```bash
python -m downscaling_4x.training.train_unet --out runs/_smoke/u1 --smoke   # 秒级自测
python -m downscaling_4x.training.train_unet --out runs/exp/<id> --mode history_51 --target all --amp
```

`--mode` 三档：`baseline_21` / `history_control_21` / `history_51`。**衡量历史通道的增量必须拿
`history_51` 比 `history_control_21`**——两者通道数相同，差别只在帧集合；比 `baseline_21` 会把
"训练集变小"算进"加了历史通道"里，而不会有任何东西报错。

`train_unet` 这条入口（unet / corrdiff / 阶段 A 的 jit 回归）的 plateau 减半与早停共用一个
"val 算改善"的判据 `--improve-criterion`：`abs` 是真实 ERA5 输入的口径，val 比 best 低至少
`--improve-tol`（缺省 1e-4）才算改善；`rel` 按比例判（缺省 1e-3），是 Daymet-oracle 输入的
口径。缺省 `auto` 按输入目录的产品自动选，真实 ERA5 用 abs、oracle 用 rel，显式给定时以给定为准。
oracle 输入的损失量级只有真实输入的几十分之一，绝对阈值会把每轮都在创新低的缓降判成停滞，
学习率提前塌缩、早停提前触发，而作业正常结束。固定预算口径（jit 回归）下该判据只决定 ckpt.pt
何时保存。解析后的判据与阈值进 checkpoint 与续训契约，续训段改判据会被拒。`train_jit` 与
`train_stage_b` 按固定 duration 训练，没有早停，也没有这个开关。

CorrDiff 两阶段（阶段 A 的均值网 μ → μ 缓存 → 阶段 B 的残差扩散）：

```bash
python -m downscaling_4x.training.train_unet --out runs/exp/<A> --arch corrdiff --mode history_51
python -m downscaling_4x.tools.build_mu_cache --ckpt <target>=runs/exp/<A>/ckpt.pt \
    --out runs/mu_cache/<id> --mode history_51
python -m downscaling_4x.training.train_stage_b --cache runs/mu_cache/<id> \
    --stage-a-ckpt runs/exp/<A>/ckpt.pt --target <target> --out runs/exp/<B> \
    --mode history_51 --sigma-data <残差标准差>
```

阶段 B 内置两道启动自检：patch 位置确实在重掷、以及首步后各 rank 参数逐元素一致。后者的
分辨力可用 `STAGEB_DDP_BYPASS=1` 验证（人为绕过 DDP 包装体，预期当场失败）。

JiT 两阶段（JDA/JMA → μ 缓存 → σ_r → JDB/JMB）：

```bash
# 阶段 A: 确定性 JiT (--moe 即 JMA)
python -m downscaling_4x.training.train_unet --out runs/exp/<JDA> --arch jit --mode history_51
python -m downscaling_4x.tools.build_mu_cache --ckpt <target>=runs/exp/<JDA>/ckpt.pt \
    --out runs/mu_cache/<id> --mode history_51
python -m downscaling_4x.tools.residual_scale --cache runs/mu_cache/<id> \
    --target <target> --out runs/mu_cache/<id>/residual_scale.json
# 阶段 B: 给了 --mu-cache 即进入残差模式; 不给则是单阶段 JiT
python -m downscaling_4x.training.train_jit --out runs/exp/<JDB> --mode history_51 \
    --mu-cache runs/mu_cache/<id> --stage-a-ckpt runs/exp/<JDA>/ckpt.pt \
    --residual-scale runs/mu_cache/<id>/residual_scale.json
```

阶段 A 与阶段 B 共用同一份 MoE 配置（`train_downscale.jit_moe_config`），四档 JDA/JMA/JDB/JMB
没有任何一处为 MoE 单独写的代码。残差模式必须给 `--residual-scale`：残差方差远小于 1，不归一
会让噪声调度整体偏掉，而且不会报错。

路由有三档：`--router tc`（每 token 挑专家）、`ec`（每专家帧内挑 token，FLOPs 与 tc 对齐）、
`dec`（D-EC，难度感知的池化专家选择，只在阶段 B 实现）。D-EC 在 ec 的选择方向上叠三个可独立
开关的成分，缺省全开：`--dec-pool` 专家在本 rank 本步的全部帧上挑 token；`--dec-drop` 有效域外
token 不参与路由、容量按域内 token 数计；`--dec-prior` 难度头从早层隐状态预测该 token 本步的误差
幅度，以本步逐 token 训练损失为监督（只更新头），系数按 `--dec-prior-ramp` 比例的预算从 0 升到
`--dec-prior-max`，只进选择分不进门控。三个成分全关等价于 ec，会被拒绝；推理期按帧内 top-C 乘
`jit_dump --dec-capacity` 决策，与 batch 组成无关。语义见 `models/moe_ffn.py` 模块开头。

```bash
python -m downscaling_4x.training.train_jit --out runs/exp/<JMB> --mu-cache ... --moe --router dec
python -m downscaling_4x.training.train_jit ... --router dec --dec-pool 0     # 成分消融: 关批级池
```

两条流结构（`train_jit`，单阶段与残差模式都可用）：每个部件一个开关，缺省全关即单流模型，旧
checkpoint 不给新参数逐位复现；依赖关系由 `train_downscale.jit_stream_config` 单点执法，违反即拒绝
启动。`--two-stream 1` 把条件按通道名拆成细流（噪声目标 + 静态 + 年内相位）与粗流（ERA5 动态通道
+ 年内相位），两条流同一切块网格、同一随机起点、各自一套 patch 嵌入；粗流自处理为 token 网格上的
3×3 卷积（`--coarse-conv`）加若干"自注意力 + SwiGLU"块（`--coarse-blocks`），不接 t，采样时一条
轨迹只算一次。`--cross-attn 1` 在每个细流块里加一支交叉问询：Q 来自细 token，K/V 来自粗 token，
ERA5 无数据的粗 token 不参与（`--cross-mask-outside`）。`--rope-units km` 把位置章改成以一个南北
token 间距为单位、东西向乘 cos(纬度)；在此之上 `--lagrangian 1` 让 K 的位置按该粗 token 的日均风
推移 τ 小时，逐头由 `--tau-spec` 指定（缺省 `0,0,850:6,850:12,500:12,500:24`），风统计量与 ERA5
有效掩膜随 checkpoint 保存。`--expert-two-card` / `--dense-two-card` 让路由专家 / 稠密 FFN 读
"细 token 投影 ‖ 正上方粗 token 投影"（`--two-card-dims`，缺省 192,192，专家大小不变）。
`--wards N` 把 tc 路由改成两级分诊（先选 1 个科室，`--ward-topk`，再在科室内取 top-K），
`--ward-key 1` 把正上方粗 token 的投影经零初始化矩阵加进科室分，`--terrain-key 1` 把该块静态场
单独嵌成的地形键经零初始化矩阵加进专家分（`--terrain-key-window` 缺省 20 像素）；
`--drop-outside 1` 让域外 token 只走共享专家。推理期可做两种记进 meta 的干预：
`jit_dump --tau-scale 0` 把全部头的推移置零，`--zero-keys` 把两个键的投影置零。

评测（复刻上一代三段式管线，产物布局/图样式一致；统计随数据目录走，无 `--stats-dir`）。
降水的 `log1p(mm)` 空间在评测侧只有一个定义：合同的 `precip_fwd`，即 <0.1 mm 置零后 log1p，
预测与真值同一条变换，落场的 `crps_log`、指标、SSIM 与 log 空间的图全部经
`evaluation/metrics.py` 的 `precip_log_mm`，与统计基线（`eval_common`）和训练目标同口径。
直接写 `log1p(max(x, 0))` 会保留预测在干区的毛毛雨纹理，在 log 空间里变成结构差，与置零口径
的数字放在一张表里并不可比，而不会有任何东西报错；`tests/test_precip_log_space.py` 用同一份
场走两条路径断言一致。已有的确定性落场可用 `tools/rebuild_det_crps_log.py` 按此口径重建
`crps_log` 场，无需重跑前向。

```bash
# 落场 -> 指标 -> 图 一条命令（sbatch 或 login node 直接 bash 均可）
METHOD=jda RUN_ID=<exp-id> sbatch code/submit/submit_4x_det_eval.sh
RUN_ID=<阶段B exp-id> AMP=1 sbatch -N 32 code/submit/submit_4x_stageb_dump.sh   # 阶段 B 集合采样落盘 -> 再走 dump_metrics / render
#   一段装不下就同命令 --dependency=afterany 再排一段: 已落盘的天跳过, 段内按 MAXSEC 预算不开装不下的新天
python -m downscaling_4x.evaluation.det_dump --help       # 逐日场落盘(方法↔结构硬校验)
python -m downscaling_4x.evaluation.stageb_dump --help    # CorrDiff 阶段 B patched 采样落盘(布局同 jit_dump)
python -m downscaling_4x.tools.diagnostics.seam_ratio_check --fields runs/exp/<id>-eval2020 --figs   # patch 拼接缝验收 seam ratio ≤ 1.10
python -m downscaling_4x.evaluation.dump_metrics --help   # 场 -> RMSE/MAE/bias/corr/SSIM/CRPS
python -m downscaling_4x.evaluation.render.cli --list     # 图模块(共享 scales.json 色标)
python -m downscaling_4x.evaluation.plot_loss --help      # 损失曲线唯一出图入口
python -m downscaling_4x.tools.plotting.plot_spectrum --help  # 径向功率谱(按需显式出图)
python -m downscaling_4x.tools.plotting.plot_routing --help   # 从路由落盘出专家图(容量/地盘/参与专家数/随 t 变化), 不重放采样
```

JiT-MoE 的采样落盘可以顺手保存路由决策（`jit_dump --routing-dump member0|all`，缺省不落、场文件
逐字节不变）：每成员每天一个 `routing/<年>_d<日>_m<成员>.npz`，记每层每个 token 被各专家接住的次数
（总计与按噪声水平 t 分档）、专家数、门控权重、亲和分、专家输出占 token 隐状态的比例与 D-EC 的难度
先验，连同该轨迹的切块起点。`--routing-days` 只在指定日期落路由，配合 `all` 做 case study；
`--routing-only` 只采样落路由成员、不写场，用于给已有落场目录补录。截获是只读补丁；落盘时
自检前向次数、TC 的每 token 选中次数与 D-EC 直方图和模型自身统计的一致性。之后
`tools/plotting/plot_routing` 直接从落盘出图，任意成员与日期子集都不必再采样。

逐通道敏感性（缺省 = none 基线 + 全部条件通道逐个消融；逐像素逐月落盘，任选区域的
聚合都无需重跑实验；none 组与正式落场逐位一致是内置自检）：

```bash
METHOD=jda RUN_ID=<exp-id> sbatch code/submit/submit_4x_ablation.sh
python -m downscaling_4x.tools.plotting.plot_ablation --ablation runs/exp/<id>/ablation2020 \
    --regions runs/exp/20260902-regions-4x-v1/regions_v1.npz --region <展示编号> --out <dir>
```

命名区域（Bukovsky 19 区落到 480×960，陆地=有效域）由
`downscaling_4x.tools.preprocessing.build_regions` 生成，现行产物
`runs/exp/20260902-regions-4x-v1/regions_v1.npz`，render 与敏感性出图共用。

#### 输入产品替换（同产品信息上限诊断）

所有训练、μ 缓存与评测入口都收 `--era5-dir`。把它指向 Daymet-oracle 目录——ERA5 年度归档中
tmax / tmin / precip 三个成员被替换为同日 Daymet 的陆地 4×4 块平均，其余成员、目标、掩膜、
静态通道与帧集合都不变——即得到"同产品"条件下的得分，用来估计跨产品差距的量级。它含同日
目标信息，不是可部署的输入，结果不进基线表，只与同一方法在真实输入上的结果并排。

真实 ERA5 与 oracle 目录同名、同形状、同布局，拿错不会报错。因此 checkpoint、BCSD 系数、
μ 缓存与落场 meta 都记录产生它们的输入目录，消费方核对，不一致即拒绝，只有显式
`--allow-input-mismatch` 才放行。训练走 `submit/submit_4x_unet.sh unet-oracle`（工作区代码，
产物进 `runs/exp/`，启动前按 `submit/MANIFEST-data-oracle-4x.sha256` 核对数据文件）；
评测给 `submit_4x_det_eval.sh` 传 `ERA5_DIR`，`SCALES` 可指向真实输入 run 的 `scales.json`
以共用色标。

## 依赖

Python 3.11 + PyTorch（ROCm 或 CUDA）、numpy、scipy、matplotlib、xarray、netCDF4。
训练使用 AMP bfloat16。
