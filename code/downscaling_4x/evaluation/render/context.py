# -*- coding: utf-8 -*-
"""渲染上下文: 一次性加载分区/日历/场访问器与绘图助手, 传给每个图模块。

单图约定(同组共色标, 落 scales.json; 图由使用者自行拼接)贯穿所有助手方法。month_of_day
取自数据层 calendar_365, 与逐日场的日索引一致, 闰年按项目 365 日历约定处理。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from downscaling_4x.data import grid as G
from downscaling_4x.data import match_era5_daymet as M
from downscaling_4x.evaluation.metrics import precip_log_mm
from downscaling_4x.data.mu_cache import MuCache
from downscaling_4x import contract as C
from downscaling_4x.data.dataset import DownscaleData, Stats
from downscaling_4x.tools.preprocessing.region_display import display_id_map
from downscaling_4x.evaluation.render.aggregate import Aggregator

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

KIND_LABEL = {"stageb_sample_dump": "CorrDiff-B",
              "jit_sample_dump": "JiT",
              "deterministic_field_dump": None}      # 用 meta["method"] 更具体


def model_identity(meta):
    """落场 meta -> (文件名用的短标识, 图注用的标签, 成员数)。

    图一旦离开自己的目录就再没有别的地方说明它出自哪个模型, 因此标识必须进图注与文件名。
    三个生产者(stageb_dump / jit_dump / det_dump)记录 checkpoint 的字段名不同, 在此归一;
    取产生该落场的 run 目录名作标识, 与工作区的实验命名一致。
    """
    src = ((meta.get("diffusion_ckpt") or {}).get("path")
           or ((meta.get("model_sources") or [{}])[0] or {}).get("path")
           or meta.get("run") or "")
    src = str(src)
    tag = Path(src).parent.name if src.endswith(".pt") else Path(src).name
    kind = KIND_LABEL.get(meta.get("kind"), None)
    label = kind or str(meta.get("method") or meta.get("kind") or "model")
    members = int(meta.get("members", 1) or 1)
    return (tag or "model"), (f"{label} · {tag}" if tag else label), members


def _smooth_masked(a, sigma):
    """有效域内的高斯平滑; 域外 NaN 不参与也不被填充。

    直接对含 NaN 的场做高斯会把 NaN 蔓延到整幅, 所以按归一化卷积做: 分子是填零后的场,
    分母是同样平滑过的有效掩膜, 相除即只用域内像素的加权平均。纯显示用途。
    """
    from scipy.ndimage import gaussian_filter
    m = np.isfinite(a)
    if not m.any():
        return a
    num = gaussian_filter(np.where(m, a, 0.0), sigma, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(m, num / np.maximum(den, 1e-12), np.nan)
    return out


class RenderContext:
    def __init__(self, fields_dir, regions_npz, target, out_dir, years=(2020,),
                 mu_cache=None, failure_regions=(), days=(),
                 scales_path=None, model_tag=None, boxes=(), interp="nearest",
                 display_smooth=0.0):
        self.fields = Path(fields_dir)
        # 纯显示参数: 只影响画出来的样子, 不进任何指标。平滑会写进图标题, 免得一张
        # 磨平过的图被当成原生场读
        self.interp = str(interp)
        self.display_smooth = float(display_smooth or 0.0)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.ti = C.TARGETS.index(target)
        self.is_precip = (target == C.PRECIP)
        self.unit = "mm/day" if self.is_precip else "K"
        self.years = list(years)

        z = np.load(regions_npz, allow_pickle=False)
        self.region_id = z["region_id"].astype(np.int64)
        self.compound_id = z["compound_id"].astype(np.int64)
        self.land = z["land"].astype(bool)
        self.region_names = [str(s) for s in z["region_names"]]
        self.compound_names = [str(s) for s in z["compound_names"]]
        self.H, self.W = self.region_id.shape

        # day_idx -> 月份, 与数据层 calendar_365 同一约定(闰年删一天)
        self.month_of_day = {y: np.array([d.month for d in M.daymet_dates(y)], np.int64)
                             for y in self.years}
        self.MONTHS = MONTHS
        self.mu_cache = mu_cache
        mp = self.fields / "meta.json"
        self.meta = json.load(open(mp)) if mp.exists() else {}
        auto_tag, self.model_label, self.members = model_identity(self.meta)
        self.model_tag = str(model_tag or auto_tag)
        # 单成员方法没有"集合", 沿用 ens_mean 的叫法会把确定性预测说成集合均值
        self.pred_name = "ens_mean" if self.members > 1 else "prediction"
        self.days = tuple(int(d) for d in (days or ()))       # 单日图的日索引, 空=不出单日图
        # 地图上的标注框 (lon0, lon1, lat0, lat1), 经纬度; 与 imshow 的 extent 同一坐标系,
        # 因此同一组框画在任意模型的图上都落在同一片地面, 可跨图对照。
        self.boxes = tuple(tuple(float(v) for v in b) for b in (boxes or ()))
        # 载入已有色标并累积: 重渲子集不清空其它组; 同组跨图/跨重跑复用同一色标。
        # scales_path 指向共享文件时, 多个模型落在同一把尺子上 —— 否则各算各的百分位,
        # 两张图的颜色不可比, 而图面看不出这件事。
        self._scales_path = Path(scales_path) if scales_path else (self.out / "scales.json")
        self._scales_path.parent.mkdir(parents=True, exist_ok=True)
        sp = self._scales_path
        self._scales = json.load(open(sp)) if sp.exists() else {}
        self.agg = Aggregator(self)
        # 展示编号(地理: 西→东, 组内地形分带); 区域级图用它代替正式名, 对照见 regions_v1_id_map
        self._disp = display_id_map(self.region_names)
        self.region_ids = [str(self._disp[n]) for n in self.region_names]
        self.failure_regions = [self._resolve_region(r) for r in (failure_regions or [])]
        self._dd = None
        self._mu = None
        self._stats = None
        self._oro = None

    def _resolve_region(self, r):
        """把区域条目(展示编号或正式名)解析为规范区名。"""
        r = str(r).strip()
        if r.isdigit():
            inv = {v: k for k, v in self._disp.items()}
            if int(r) in inv:
                return inv[int(r)]
        return r

    def region_id_of(self, name):
        """规范区名 -> 内部 region_id(1..19, 用于栅格掩膜); 未知返回 None。"""
        return self.region_names.index(name) + 1 if name in self.region_names else None

    def display_id(self, name):
        """规范区名 -> 展示编号(地理); 未知返回 None。"""
        return self._disp.get(name)

    # ---- 阶段A(μ)与真值访问; 供 bias 图与 A-vs-B 分类离线复算(不重采样) ----
    def _ensure_data(self):
        """初始化真值数据层(统计随数据目录走, 无需额外参数); μ 缓存另按需初始化。"""
        if self._dd is not None:
            return
        self._stats = Stats()
        # 只读真值与掩膜, 用不到条件张量, 取无历史的模式免去多载一年
        self._dd = DownscaleData(M.ERA5_DIR, M.DAYMET_DIR, self.years, self._stats,
                                 mode="baseline_21")

    def _ensure_mu(self):
        if self._mu is not None:
            return
        if not self.mu_cache:
            raise RuntimeError("需要 --mu-cache 才能取阶段A μ")
        self._mu = MuCache(self.mu_cache, [self.target])

    def truth(self, y, day):
        """物理单位真值场 (H,W); 降水为 mm/day(x precip_scale)。有效域外 NaN。"""
        self._ensure_data()
        _, hr = self._dd.target(y, day)
        tr = hr[self.ti].astype(np.float64)
        if self.is_precip:
            tr = tr * self._stats.precip_scale
        return np.where(self._dd.mask, tr, np.nan)

    def mu_phys(self, y, day):
        """阶段A 均值 μ 的物理场 (H,W); 与 stageb_dump 的反变换口径一致。"""
        self._ensure_data()
        self._ensure_mu()
        s = self._stats
        mu = self._mu.get(self.target, y, day) * s.d_std[self.ti] + s.d_mean[self.ti]
        if self.is_precip and s.precip_log:
            mu = C.precip_inv(mu, s.precip_scale) * s.precip_scale
            mu = np.where(mu < s.precip_clip, 0.0, mu)
        return mu

    def ensure_mae_a(self):
        """把阶段A 逐日 |μ-truth| 落成一等场 <fields>/mae_a/(确定性 CRPS=MAE, 陆外 NaN);
        幂等, 已存即跳过。之后按普通场用(mass_matrix/annual_mean)。"""
        dd = self.fields / "mae_a"
        dd.mkdir(exist_ok=True)
        for y, t in self.available_days("crps"):
            p = dd / f"{y}_d{t}.npy"
            if p.exists():
                continue
            tr = self.truth(y, t)
            mae = np.abs(self.mu_phys(y, t) - tr)          # 陆外 truth 为 NaN -> mae NaN
            np.save(p, mae.astype(np.float32))

    def annual_truth(self):
        """陆地年均真值场 (H,W), 缓存到 agg/; 供 bias 图。"""
        c = self.agg._load("annual_truth")
        if c is not None:
            return c["mean"]
        s = None
        n = 0
        for y, t in self.available_days("ens_mean"):
            a = self.truth(y, t)
            m = np.isfinite(a)
            if s is None:
                s = np.zeros_like(a)
                cnt = np.zeros_like(a)
            s[m] += a[m]
            cnt[m] += 1
            n += 1
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        self.agg._save("annual_truth", mean=mean.astype(np.float32))
        return mean

    def annual_truth_log(self):
        """陆地年均 log1p 真值场 (H,W), 缓存到 agg/; 与 Aggregator.annual_mean_log 同口径。"""
        c = self.agg._load("annual_truth_log")
        if c is not None:
            return c["mean"]
        s = cnt = None
        for y, t in self.available_days("ens_mean"):
            a = precip_log_mm(self.truth(y, t))
            m = np.isfinite(a)
            if s is None:
                s = np.zeros_like(a)
                cnt = np.zeros_like(a)
            s[m] += a[m]
            cnt[m] += 1
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        self.agg._save("annual_truth_log", mean=mean.astype(np.float32))
        return mean

    def elevation(self):
        """陆地高程场 (H,W) [m], 缓存; 供地形晕渲底图(静态量, 与年份无关)。"""
        if self._oro is None:
            self._oro = M.load_static_2d(M.DAYMET_DIR, "orography").astype(np.float64)
        return self._oro

    # ---- 场访问 ----
    def field_path(self, field, y, day):
        return self.fields / field / f"{y}_d{day}.npy"

    def has_field(self, field):
        return (self.fields / field).is_dir()

    def load_field(self, field, y, day):
        return np.load(self.field_path(field, y, day))

    def coverage(self, field="ens_mean"):
        """(实际落盘天数, 整年应有天数)。

        分母取日历年而**不是**落场自己的 n_days_expected: 只采了几天的落场会把那个字段
        记成几天, 拿它当分母的话部分覆盖会自称完整, 于是一天的场被画成"年均"。
        """
        return len(self.available_days(field)), C.DAYS_PER_YEAR * len(self.years)

    def is_full_year(self, field="ens_mean"):
        """是否够得上"整年"。落场若明确声明只采了部分日期, 直接判否。"""
        if self.meta.get("all_days") is False:
            return False
        have, want = self.coverage(field)
        return have >= want

    def available_days(self, field):
        """该场实际落盘的 (year, day) 列表(与 self.years 取交)。"""
        out = []
        for y in self.years:
            d = self.fields / field
            for t in range(365):
                if (d / f"{y}_d{t}.npy").exists():
                    out.append((y, t))
        return out

    def partition(self, level):
        """level='region'(19) | 'compound'(8) -> (id_grid, names)。"""
        if level == "compound":
            return self.compound_id, self.compound_names
        return self.region_id, self.region_names

    # ---- 绘图助手(单图) ----
    def savefig(self, fig, name):
        p = self.out / f"{name}__{self.model_tag}.png"     # 身份进文件名, 图拷走也认得出
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return p

    def scale(self, group, data=None, cmap="viridis", label="", vmin=None, vmax=None):
        """解析色标组 [vmin,vmax,cmap,label] 并登记, 返回 (vmin,vmax,cmap)。

        显式 vmin/vmax 优先; 否则复用已存组(跨图/跨重跑稳定, 实现同组共色标);
        再否则由 data 的有限值计算。要强制重算某组: 删 scales.json 或传显式 vmin/vmax。
        """
        g = self._scales.get(group)
        if g is not None:
            vmin = g["vmin"] if vmin is None else vmin
            vmax = g["vmax"] if vmax is None else vmax
            cmap = g.get("cmap", cmap)
            label = g.get("label", label) or label
        elif data is not None:
            fin = np.asarray(data, float)
            fin = fin[np.isfinite(fin)]
            if vmin is None:
                vmin = float(fin.min()) if fin.size else 0.0
            if vmax is None:
                vmax = float(fin.max()) if fin.size else 1.0
        vmin = 0.0 if vmin is None else float(vmin)
        vmax = 1.0 if vmax is None else float(vmax)
        self._scales[group] = {"vmin": vmin, "vmax": vmax, "cmap": cmap, "label": label}
        return vmin, vmax, cmap

    def write_scales(self):
        json.dump(self._scales, open(self._scales_path, "w"),
                  indent=1, ensure_ascii=False)

    def heatmap(self, mat, row_labels, col_labels, name, cbar="", scale_group=None,
                cmap="viridis", vmin=None, vmax=None, title=None, annotate=False):
        mat = np.asarray(mat, float)
        vmin, vmax, cmap = self.scale(scale_group or name, data=mat, cmap=cmap,
                                      label=cbar, vmin=vmin, vmax=vmax)
        h = max(3.0, 0.30 * len(row_labels) + 1.2)
        w = max(5.0, 0.52 * len(col_labels) + 2.2)
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label=cbar)
        if annotate:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="0.9")
        return self.savefig(fig, name)

    def bar(self, values, labels, name, ylabel="", title=None, color="#4878a8", ref=None,
            annotations=None):
        """竖直条形(保持给定顺序); annotations(与 labels 同序)非空时在条顶标注文本。"""
        values = np.asarray(values, float)
        fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(labels) + 1.0), 4.0),
                               constrained_layout=True)
        x = range(len(labels))
        ax.bar(x, values, color=color)
        if annotations is not None:
            pad = 0.02 * max(values.max(), 1e-12)
            for i, (v, t) in enumerate(zip(values, annotations)):
                ax.text(i, v + pad, t, ha="center", va="bottom", fontsize=7)
            ax.set_ylim(0, values.max() * 1.18)
        ax.set_xticks(list(x))
        rot = 45 if max((len(str(l)) for l in labels), default=0) > 3 else 0
        ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=8)
        ax.set_ylabel(ylabel)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=1)
        if title:
            ax.set_title(title, fontsize=10)
        return self.savefig(fig, name)

    def barh(self, values, labels, name, xlabel="", title=None, color="#4878a8",
             sort=True, annotations=None):
        """降序水平条形; annotations(与 labels 同序)非空时在条端标注文本。"""
        values = np.asarray(values, float)
        order = np.argsort(values)[::-1] if sort else np.arange(len(values))
        v = values[order]
        lab = [labels[i] for i in order]
        ann = [annotations[i] for i in order] if annotations is not None else None
        fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.34 * len(labels) + 1.0)),
                               constrained_layout=True)
        y = range(len(lab))
        ax.barh(list(y), v, color=color)
        if ann is not None:
            xmax = float(np.nanmax(v)) if len(v) else 1.0
            for i, s in enumerate(ann):
                ax.text(v[i] + 0.01 * xmax, i, s, va="center", fontsize=7)
            ax.set_xlim(0, xmax * 1.18)
        ax.invert_yaxis()
        ax.set_yticks(list(y))
        ax.set_yticklabels(lab, fontsize=8)
        ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title, fontsize=10)
        return self.savefig(fig, name)

    def map(self, field2d, name, cmap="turbo", vmin=None, vmax=None, scale_group=None,
            cbar="", title=None, diverging=False):
        """陆地掩膜下的地理场地图(单图); diverging=True 时围绕 0 对称配色。"""
        a = np.asarray(field2d, float)
        if self.display_smooth > 0:
            a = _smooth_masked(a, self.display_smooth)
            # 标注用英文: 出图字体没有 CJK 字形, 中文会渲染成方框
            title = (title or "") + f"  [display-smoothed σ={self.display_smooth:g} px]"
        if diverging and vmax is None and self._scales.get(scale_group or name) is None:
            vmax = float(np.nanpercentile(np.abs(a[np.isfinite(a)]), 99)) if np.isfinite(a).any() else 1.0
            vmin = -vmax
        vmin, vmax, cmap = self.scale(scale_group or name, data=a, cmap=cmap,
                                      label=cbar, vmin=vmin, vmax=vmax)
        ext = G.extent()
        aspect = G.aspect()
        fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                       extent=ext, aspect=aspect, interpolation=self.interp)
        ax.set_facecolor("0.85")
        for lo0, lo1, la0, la1 in self.boxes:
            ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False,
                                       edgecolor="#101418", lw=1.8, zorder=5))
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label=cbar)
        return self.savefig(fig, name)
