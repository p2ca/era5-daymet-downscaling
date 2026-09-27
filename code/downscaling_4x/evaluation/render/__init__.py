# -*- coding: utf-8 -*-
"""阶段 B 场渲染管线。

以 stageb_dump 落盘的逐日逐像素场(ens_mean/crps/spread/rank/crps_log)为输入, 热插拔地
出单图与统计: 每张图是 figures/ 下一个 @figure 注册的模块, 用 --only/--skip 开关。渲染全在
CPU/login node 完成, 秒~分钟级, 可反复重跑; 共享聚合缓存在 <fields>/agg/。
"""
