#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锁定降水 log1p(mm) 空间的唯一定义: 评测侧两条路径(统计基线的 MultiMethodEval 与落场的
dump_metrics 口径)在同一份场上必须给出同一个 log 空间指标与 SSIM。

两条路径各自都"正确"却口径不同, 不会有任何东西报错, 只会让同一张表里的数字不可比。
本测试: (1) precip_log_mm 与合同 precip_fwd 逐点相同; (2) 含毛毛雨的合成场上两条路径的
log 空间 RMSE/MAE/bias/corr/SSIM 一致; (3) 负对照: 不置零的 log1p 在同一场上给出不同数字,
证明测试有分辨力; (4) BCSD 的两处预测构造引用同一个函数。纯 CPU, 不需要数据。

运行: python -m downscaling_4x.tests.test_precip_log_space
"""
import numpy as np

from downscaling_4x import contract as C
from downscaling_4x.data import downscale_baseline as DB
from downscaling_4x.evaluation import det_dump
from downscaling_4x.evaluation import metrics as MT
from downscaling_4x.evaluation.eval_common import MultiMethodEval
from downscaling_4x.baselines import eval_baselines as EB
from scipy.ndimage import binary_erosion

H, W = 96, 128


def _fields(seed=0):
    """合成一天: 真值有干区(精确 0)、毛毛雨区(<0.1 mm)与雨区; 预测在干区带零点几毫米噪声。"""
    rng = np.random.default_rng(seed)
    land = np.ones((H, W), bool); land[:, :8] = False; land[-8:, :] = False
    truth = np.zeros((H, W), np.float32)
    truth[20:60, 30:90] = rng.gamma(2.0, 3.0, (40, 60)).astype(np.float32)   # 雨区, mm
    truth[70:80, 100:120] = 0.05                                             # 毛毛雨 <0.1 mm
    pred = truth * rng.uniform(0.7, 1.3, (H, W)).astype(np.float32)
    pred += rng.uniform(0.0, 0.3, (H, W)).astype(np.float32)                 # 全域零点几毫米噪声
    pred[5:10, 40:50] = -0.2                                                 # 少量负值
    return land, truth, pred


def test_helper_matches_contract():
    x = np.array([-1.0, 0.0, 0.05, 0.0999, 0.1, 0.5, 20.0, 3000.0], np.float32)
    got = MT.precip_log_mm(x)
    want = C.precip_fwd(x / 1000.0)                    # 合同以 m/day 为入参
    assert np.allclose(got, want, rtol=1e-6, atol=1e-7), (got, want)
    assert got[0] == 0 and got[2] == 0 and got[3] == 0 and got[4] > 0, "0.1 mm 以下必须置零"


def _dump_metrics_path(land, truth_mm, pred_mm):
    er = binary_erosion(land, iterations=MT.SSIM_ERODE)
    lg = lambda x: MT.precip_log_mm(x)
    acc = DB.Acc(); acc.add(lg(pred_mm[land]), lg(truth_mm[land]))
    ssim = MT.ssim_masked(np.where(land, lg(pred_mm), 0.0), lg(truth_mm), land, er)
    return {**acc.result(), "ssim": ssim}


def _eval_common_path(land, truth_mm, pred_mm):
    ev = MultiMethodEval(["m"], list(C.TARGETS), H, W, land, precip_scale=C.PRECIP_SCALE, precip_log=True)
    raw = np.zeros((3, H, W), np.float32); raw[2] = truth_mm / C.PRECIP_SCALE    # m/day
    pr = np.zeros((1, 3, H, W), np.float32); pr[0, 2] = pred_mm / C.PRECIP_SCALE
    ev.add_day(raw, land[None].astype(np.float32), {"m": pr})
    r = ev.acc_log["m"].result(); r["ssim"] = ev.ssim_log["m"][0] / ev.ssim_log["m"][1]
    return r


def test_two_paths_agree():
    land, truth, pred = _fields()
    a, b = _dump_metrics_path(land, truth, pred), _eval_common_path(land, truth, pred)
    for k in ("rmse", "mae", "bias", "corr", "ssim"):
        assert abs(a[k] - b[k]) < 1e-5, f"{k}: dump_metrics 口径 {a[k]} vs eval_common 口径 {b[k]}"


def test_uncensored_log_would_differ():
    """负对照: 若落场路径仍用 log1p(max(x,0)), 同一场上的 SSIM 与 RMSE 必须与置零口径不同。"""
    land, truth, pred = _fields()
    er = binary_erosion(land, iterations=MT.SSIM_ERODE)
    old = lambda x: np.log1p(np.maximum(x, 0.0))
    s_old = MT.ssim_masked(np.where(land, old(pred), 0.0), old(truth), land, er)
    s_new = MT.ssim_masked(np.where(land, MT.precip_log_mm(pred), 0.0), MT.precip_log_mm(truth), land, er)
    assert abs(s_old - s_new) > 1e-3, "毛毛雨场上两种口径 SSIM 竟相同, 本测试无分辨力"
    acc_old = DB.Acc(); acc_old.add(old(pred[land]), old(truth[land]))
    acc_new = DB.Acc(); acc_new.add(MT.precip_log_mm(pred[land]), MT.precip_log_mm(truth[land]))
    assert abs(acc_old.result()["rmse"] - acc_new.result()["rmse"]) > 1e-4


def test_bcsd_single_formula():
    """det_dump 与 eval_baselines 的 BCSD 预测必须是同一个函数; 降水先上采样再变换。"""
    assert det_dump.bcsd_predict is EB.bcsd_predict
    x = np.array([[0.0, 0.00005, 0.002]], np.float32)          # m/day: 干 / 0.05 mm / 2 mm
    a, b = np.full_like(x, 0.8), np.full_like(x, 0.1)
    got = EB.bcsd_predict(x, a, b, True)
    want = C.precip_inv(a * C.precip_fwd(x) + b)
    assert np.allclose(got, want)
    assert got[0, 0] == got[0, 1], "0.05 mm 的输入必须先被置零, 与干格给出同一预测"
    assert np.allclose(EB.bcsd_predict(x, a, b, False), a * x + b)


TESTS = [test_helper_matches_contract, test_two_paths_agree, test_uncensored_log_would_differ,
         test_bcsd_single_formula]


def main():
    for fn in TESTS:
        fn()
        print(f"[PASS] {fn.__name__}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
