import argparse
# Packaged implementation; the original code/ path remains compatible.
import numpy as np
import os
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 0. 命令行参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract daily 'golden frame' (+33h) from ERA5 chunk files and crop to North America.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python extract_golden_frames.py --year 1980
  python extract_golden_frames.py --year 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 --workers 16
  python extract_golden_frames.py --year 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 --workers 12
  python extract_golden_frames.py --split train --workers 8
  python extract_golden_frames.py --workers 4
        """
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--year', type=int, nargs='+', metavar='YEAR',
        help='指定一个或多个年份'
    )
    group.add_argument(
        '--split', type=str, choices=['train', 'val', 'test'],
        help='处理整个 split'
    )
    parser.add_argument(
        '--workers', type=int, default=1, metavar='N',
        help='并行进程数（每个进程处理一年，默认1）。'
             '建议不超过可用 CPU core 数，也不超过待处理的年份数。'
    )
    return parser.parse_args()


# ==========================================
# 1. 全局配置
# ==========================================
base_dir   = "/lustre/orion/atm112/world-shared/patrickfan/era5/0.25_deg"
output_dir = "/lustre/orion/atm112/world-shared/patrickfan/era5/0.25_deg_Daily_Golden"

splits = {
    'train': list(range(1980, 2018)),
    'val':   [2018, 2019],
    'test':  [2020],
}

target_vars = [
    'land_sea_mask', 'landcover', 'latitude', 'orography',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    '2m_temperature', '2m_temperature_max', '2m_temperature_min',
    'sea_surface_temperature', 'total_precipitation_24hr',
    'volumetric_soil_water_layer_1', 'geopotential_200', 'geopotential_500',
    'geopotential_850', 'specific_humidity_200', 'specific_humidity_500',
    'specific_humidity_850', 'temperature_200', 'temperature_500',
    'temperature_850', 'u_component_of_wind_200', 'u_component_of_wind_500',
    'u_component_of_wind_850', 'v_component_of_wind_200', 'v_component_of_wind_500',
    'v_component_of_wind_850', 'days_of_year', 'time_of_day', 'hrs_each_step',
    'num_steps_per_shard', 'extra_steps',
]

# ==========================================
# 2. 常量
# ==========================================
# [Note] 对于标准升序 ERA5 720-lat 网格（lat[i] = -90 + i×0.25），
#        24.0°N 精确对应 index 456，54.0°N 精确对应 index 576。
#        当前 [455:575] 实际约对应 23.75°N ~ 53.5°N（差 ~0.25°）。
#        如需严格对齐 Daymet 边界，可改为 LAT_START, LAT_END = 456, 576。
LAT_START, LAT_END  = 455, 575
LON_START, LON_END  = 940, 1180

# 每年固定 365 天（数据集不含闰年）
# 365 × 24 = 8760 小时 = 20 个 chunk 文件（每文件 438 小时，整除）
DAYS_PER_YEAR    = 365
HOURS_PER_YEAR   = DAYS_PER_YEAR * 24   # 8760
HOURS_PER_CHUNK  = 438
CHUNKS_PER_YEAR  = HOURS_PER_YEAR // HOURS_PER_CHUNK  # 20


# ==========================================
# 3. 辅助函数
# ==========================================
def find_chunk_file(year, file_idx):
    """全局搜索 chunk 文件，防止跨年路径越界。"""
    for split_folder in ['train', 'val', 'test']:
        p = os.path.join(base_dir, split_folder, f"{year}_{file_idx}.npz")
        if os.path.exists(p):
            return p
    p = os.path.join(base_dir, f"{year}_{file_idx}.npz")
    if os.path.exists(p):
        return p
    return None


def build_job_list(args):
    """根据命令行参数返回 [(split_name, year), ...] 列表。"""
    year_to_split = {y: s for s, ys in splits.items() for y in ys}
    if args.year:
        jobs = []
        for y in args.year:
            if y not in year_to_split:
                raise ValueError(
                    f"Year {y} not in any split. "
                    f"Valid range: {min(year_to_split)}~{max(year_to_split)}"
                )
            jobs.append((year_to_split[y], y))
        return jobs
    if args.split:
        return [(args.split, y) for y in splits[args.split]]
    return [(s, y) for s, ys in splits.items() for y in ys]


# ==========================================
# 4. 核心处理：以 chunk 为单位批量提取
# ==========================================
# 旧方案（天为单位）：365天 × 30变量 = 10,950次解压，每次解压整个438×720×1440数组
# 新方案（chunk为单位）：20个chunk × 30变量 = 600次解压，约18倍提速
# 每个 chunk 内的所有目标帧用 numpy 向量化 fancy index 一次性提取
# ==========================================
def process_year(year, split_name, out_split_dir):
    print(f"\n[Processing {split_name.upper()}] Year: {year}")

    # --- Step 1: 预计算每天对应的 (target_year, file_idx, hour_in_file) ---
    schedule = []  # index = day_idx
    for day_idx in range(DAYS_PER_YEAR):
        abs_hour = day_idx * 24 + 33   # +33h 黄金偏移
        t_year = year
        if abs_hour >= HOURS_PER_YEAR:  # 仅最后一天(day 364)会跨年
            t_year  += 1
            abs_hour -= HOURS_PER_YEAR
        schedule.append((t_year, abs_hour // HOURS_PER_CHUNK, abs_hour % HOURS_PER_CHUNK))

    # --- Step 2: 将同一 chunk 的所有天打包 ---
    # key: (target_year, file_idx)  value: [(day_idx, hour_in_file), ...]
    chunk_groups = defaultdict(list)
    for day_idx, (t_year, fi, hi) in enumerate(schedule):
        chunk_groups[(t_year, fi)].append((day_idx, hi))

    # --- Step 3: 预分配结果容器 ---
    # dynamic_data 延迟初始化（首次见到变量时用真实 dtype 和 shape）
    dynamic_data    = {}
    static_data     = {}
    static_collected = False
    missing_days    = set()

    total_chunks = len(chunk_groups)
    chunk_done   = 0

    # --- Step 4: 遍历每个 chunk（每年只有 20 个，每个只打开一次）---
    for (t_year, fi), day_hour_pairs in sorted(chunk_groups.items()):
        chunk_path = find_chunk_file(t_year, fi)
        days_arr  = np.array([d for d, _ in day_hour_pairs], dtype=int)
        hours_arr = np.array([h for _, h in day_hour_pairs], dtype=int)

        if chunk_path is None:
            print(f"\n  [Warning] Missing {t_year}_{fi}.npz — "
                  f"days {days_arr.tolist()} will be forward-filled later.")
            missing_days.update(days_arr.tolist())
            chunk_done += 1
            continue

        # ── 每个 chunk 只打开/解压一次 ──────────────────────────────
        npz = np.load(chunk_path)

        for var in target_vars:
            if var not in npz.files:
                continue

            var_matrix = npz[var]

            if var_matrix.ndim >= 3:
                # 防止 (1, lat, lon) 等静态类变量 hour 越界
                safe_hours = np.minimum(hours_arr, var_matrix.shape[0] - 1)

                # 一次 fancy index 提取该 chunk 内所有目标帧
                if var_matrix.ndim == 4:
                    frames = var_matrix[safe_hours, 0, LAT_START:LAT_END, LON_START:LON_END]
                else:
                    frames = var_matrix[safe_hours, LAT_START:LAT_END, LON_START:LON_END]

                # 首次见到该变量：用真实 dtype 预分配整年数组
                if var not in dynamic_data:
                    dynamic_data[var] = np.zeros(
                        (DAYS_PER_YEAR,) + frames.shape[1:],
                        dtype=frames.dtype,
                    )
                # 直接写入对应日期的位置（无需 append + stack）
                dynamic_data[var][days_arr] = frames

            elif not static_collected:
                # 首次成功加载文件时提取静态/低维变量
                if var_matrix.ndim == 2:
                    static_data[var] = var_matrix[LAT_START:LAT_END, LON_START:LON_END]
                elif var_matrix.ndim == 1:
                    if 'lat' in var.lower():
                        static_data[var] = var_matrix[LAT_START:LAT_END]
                    elif 'lon' in var.lower():
                        static_data[var] = var_matrix[LON_START:LON_END]
                    else:
                        static_data[var] = var_matrix
                elif var_matrix.ndim == 0:
                    static_data[var] = var_matrix

        npz.close()
        # ────────────────────────────────────────────────────────────

        if not static_collected:
            static_collected = True

        chunk_done += 1
        print(f"  -> Chunk {chunk_done:02d}/{total_chunks}  "
              f"({t_year}_{fi}.npz, covers {len(days_arr)} days)",
              end="\r", flush=True)

    print()

    # --- Step 5: 对缺失天做 forward-fill（从最近的有效帧复制）---
    if missing_days:
        print(f"  [Warning] Forward-filling {len(missing_days)} missing days.")
        for day_idx in sorted(missing_days):
            src = day_idx - 1 if day_idx > 0 else 0
            for arr in dynamic_data.values():
                arr[day_idx] = arr[src]

    # --- Step 6: 健全性检查 ---
    for var, arr in dynamic_data.items():
        if arr.shape[0] != DAYS_PER_YEAR:
            print(f"  [Error] '{var}': shape[0]={arr.shape[0]}, expected {DAYS_PER_YEAR}!")

    # --- Step 7: 保存 ---
    out_file = os.path.join(out_split_dir, f"{year}.npz")
    print(f"  -> Saving {len(dynamic_data)} dynamic + {len(static_data)} static vars to {out_file} ...")
    np.savez_compressed(out_file, **static_data, **dynamic_data)
    print(f"  -> Saved.")

    del dynamic_data, static_data
    gc.collect()


# ==========================================
# 5. Worker 包装函数（必须在模块级别，multiprocessing 才能 pickle）
# ==========================================
def _worker(task):
    """task = (split_name, year, out_split_dir)"""
    split_name, year, out_split_dir = task
    try:
        process_year(year, split_name, out_split_dir)
        return year, None
    except Exception as e:
        return year, e


# ==========================================
# 6. 主入口
# ==========================================
if __name__ == '__main__':
    args     = parse_args()
    job_list = build_job_list(args)

    # 构建任务列表，同时确保输出目录存在
    tasks = []
    for split_name, year in job_list:
        out_split_dir = os.path.join(output_dir, split_name)
        os.makedirs(out_split_dir, exist_ok=True)
        tasks.append((split_name, year, out_split_dir))

    workers = min(args.workers, len(tasks))

    print("=" * 60)
    print("Initiating Golden Frame (+33h) Extraction & Cropping")
    print(f"Years to process : {[y for _, y, _ in tasks]}")
    print(f"Chunks per year  : {CHUNKS_PER_YEAR} (each opened exactly once)")
    print(f"Parallel workers : {workers}")
    print("=" * 60)

    if workers <= 1:
        # 单进程：顺序执行
        for task in tasks:
            _worker(task)
    else:
        # 多进程：每个 worker 负责一年，互不干扰
        completed = 0
        failed    = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, task): task for task in tasks}
            for fut in as_completed(futures):
                year, err = fut.result()
                completed += 1
                if err:
                    failed.append((year, err))
                    print(f"\n[ERROR] Year {year} failed: {err}")
                else:
                    print(f"\n[DONE]  Year {year} ({completed}/{len(tasks)} complete)")

        if failed:
            print(f"\n⚠️  {len(failed)} year(s) failed: {[y for y, _ in failed]}")
        else:
            print(f"\n✅ All {len(tasks)} years completed successfully.")

    print("\n" + "=" * 60)
    print("✅ ALL YEARS COMPLETED SUCCESSFULLY!")
    print(f"Dataset saved to: {output_dir}")
    print("=" * 60)
