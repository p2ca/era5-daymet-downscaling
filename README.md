# Diffusion Weather Downscaling (ERA5 → observation product)

跨产品（cross-product）气象降尺度扩散模型项目。以 **CorrDiff** 为可复现基线，目标是 ERA5 再分析 → 独立观测产品的大分布偏移降尺度。

## 三端架构（Dropbox / ORNL / 本地）

| 端 | 角色 | 说明 |
|---|---|---|
| **Dropbox** | 冷存储（原始数据唯一真相源） | ~50GB、720×1440 每日气象帧；检查点备份 |
| **本地（这台机器）** | 大脑：写代码/配置、调试、结果分析 | Claude Code 在此协助；无法直连 Dropbox/ORNL |
| **ORNL（OLCF Frontier）** | 肌肉：预处理 + 训练 + 集合推理 | AMD MI250X；SLURM；数据落 Lustre |

**连接方式**：代码走 Git/GitHub；数据走 rclone（DTN）+ Globus/bbcp；三端不互为副本，各司其职。

- 环境事实与路径 → [`docs/reference/ornl-environment.md`](docs/reference/ornl-environment.md)
- 基线论文 → `oripaper/s43247-025-02042-5.pdf`（CorrDiff）
- 文献综述 → `overview/Diffusion_Downscaling_Literature_Review.docx`

## 单次训练时间估算（基于 CorrDiff）
锚点：NVIDIA CONUS CorrDiff ≈ 5,000 A100-GPU-小时。本项目 720×1440 面积约为参考 448×448 的 5.2×
→ 完整运行 **~15,000–30,000 A100-GPU-小时**（128–256 GPU 上数天）。冒烟测试先用 ~10–50 GPU-小时的 mini 配置。

## 当前状态 / 下一步
- [x] 环境侦察（项目 `atm112`、MI250X/ROCm、SLURM、登录节点+DTN 可上网）
- [x] 发现学长（patrickfan）已在 world-shared 备好全部数据（100+TB）+ 完整代码
- [x] 数据来源改为读 Lustre（不再从 Dropbox 下载）；有可复用 ROCm PyTorch 环境 `wea_env`
- [x] 工作区确认 = `/lustre/orion/atm112/scratch/hjsong/downscaling`
- [x] 在 Frontier 登录节点架起 tmux 里的 Claude Code（集群内动手）
- [ ] **阶段 1：跑通 UNet + ViT baseline**（复用学长 `train_unet.py`/`train_vit.py`，cp 到工作区）
- [ ] **阶段 2：CorrDiff 两阶段**（cnn 均值 + 扩散残差）

> 协作规则见 [AGENTS.md](AGENTS.md)，重大历史见 [docs/HISTORY.md](docs/HISTORY.md)，方法背景见 [docs/reference/baseline-pipeline.md](docs/reference/baseline-pipeline.md)。
