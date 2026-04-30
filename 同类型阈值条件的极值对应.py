# -*- coding: utf-8 -*-
"""
双端行波故障定位算法 v2.2
高斯滤波 + 波峰/波谷极值对齐法（严格模式 + 幅度阈值）

v2.2 新增：
    - 极值点判定增加幅度条件：|极值| > 1.05 * |基线中值|
      该条件用于滤除幅度不足的噪声极值，提高检测可靠性。

v2.1 已有特性：
    - STRICT_ALIGN 参数：要求两端首波极值类型相同（均为 peak 或均为 valley）
    - 高斯滤波、导数变号+导数阈值检测、置信度基于基线中值比
"""

import os
import re
import glob
import math
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# scipy 用于高斯滤波（可选，若无则回退手动高斯核）
try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy 未安装，将使用手动高斯卷积核，性能较低。建议安装 scipy。")

# =========================================================
# 0. 参数配置（用户可修改）
# =========================================================

A_FOLDER = r"D:\work\Qinghua19\故障定位\A"
B_FOLDER = r"D:\work\Qinghua19\故障定位\B"
OUTPUT_DIR = r"D:\work\Qinghua19\故障定位\output_v2_strict_amp"

# 线路参数
LINE_LENGTH_M = 100000.0      # 线路总长度（m）
WAVE_SPEED = 2.8e8            # 行波传播速度（m/s）
FS = 4.21e6                   # 采样频率（Hz）

# 预处理参数
BASE_RATIO = 0.2             # 前 5% 采样点作为基线区间
GAUSS_SIGMA = 5.0             # 高斯滤波标准差（采样点数），<=0 则跳过滤波
DERIV_THR = 0.00001               # 极值检测导数阈值（V/点），量纲需与电压/采样点一致

# 双端对齐模式
STRICT_ALIGN = True           # True: 严格模式，要求两端极值类型相同（peak-peak 或 valley-valley）
                              # False: 宽松模式，任意极值类型均可用（与 v2.0 行为一致）

# 极值幅度阈值（相对于基线中值）
EXTREMA_AMP_RATIO = 1.00000005      # 极值的绝对值必须 > 基线中值绝对值 * 该系数

# 可视化与输出
ENABLE_VIS = True
SAVE_PNG = True
SAVE_HTML_REPORT = True
PLOT_DIR_NAME = "plots"
MAX_PLOT_PAIRS = 30
PLOT_DPI = 130

# 结果排序策略
SORT_BEST_BY = ("confidence",)   # 置信度降序


# =========================================================
# 1. 基础工具函数（沿用旧代码，未变）
# =========================================================

def setup_matplotlib_fonts() -> bool:
    """配置中文字体。"""
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "PingFang SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


HAS_CJK_FONT = setup_matplotlib_fonts()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(name))


def to_1d_array(x) -> np.ndarray:
    """转为一维 float 数组，并删除 NaN/Inf。"""
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def odd_int(n: int, minimum: int = 3) -> int:
    """返回不小于 minimum 的奇数。"""
    n = int(max(n, minimum))
    if n % 2 == 0:
        n += 1
    return n


def gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """手动生成一维高斯卷积核。"""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    """
    对一维信号进行高斯滤波。
    若 sigma <= 0 或 SCIPY 不可用，回退到手动画卷积。
    """
    x = to_1d_array(signal)
    if len(x) == 0:
        return x
    if sigma <= 0:
        return x.copy()

    if SCIPY_AVAILABLE:
        # scipy 的 mode='reflect' 避免边界突变
        smoothed = gaussian_filter1d(x, sigma=sigma, mode='reflect')
    else:
        kernel = gaussian_kernel_1d(sigma)
        # 使用反射边界进行卷积（等同 mode='reflect'）
        pad_width = len(kernel) // 2
        padded = np.pad(x, pad_width, mode='reflect')
        smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.astype(float)


def extract_timestamp_key(name: str) -> str:
    """从文件名提取配对键。"""
    base = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", base)
    return m.group(0) if m else base


def list_csv_files(folder: str) -> List[str]:
    files = glob.glob(os.path.join(folder, "*.csv"))
    return sorted(files, key=lambda f: os.path.basename(f))


def pair_files_by_key(a_files: List[str], b_files: List[str]) -> List[Tuple[str, str, str]]:
    a_groups = {}
    b_groups = {}
    for f in a_files:
        k = extract_timestamp_key(f)
        a_groups.setdefault(k, []).append(f)
    for f in b_files:
        k = extract_timestamp_key(f)
        b_groups.setdefault(k, []).append(f)
    common_keys = sorted(set(a_groups.keys()) & set(b_groups.keys()))
    pairs = []
    for k in common_keys:
        a_list = sorted(a_groups[k], key=lambda f: os.path.basename(f))
        b_list = sorted(b_groups[k], key=lambda f: os.path.basename(f))
        n = min(len(a_list), len(b_list))
        for i in range(n):
            a = a_list[i]
            b = b_list[i]
            pair_name = f"{os.path.splitext(os.path.basename(a))[0]}__{os.path.splitext(os.path.basename(b))[0]}"
            pairs.append((a, b, pair_name))
    if len(pairs) == 0:
        n = min(len(a_files), len(b_files))
        for i in range(n):
            a = a_files[i]
            b = b_files[i]
            pair_name = f"{os.path.splitext(os.path.basename(a))[0]}__{os.path.splitext(os.path.basename(b))[0]}"
            pairs.append((a, b, pair_name))
    return pairs


# =========================================================
# 2. 数据读取（沿用旧代码）
# =========================================================

def read_voltage_csv(filepath: str) -> np.ndarray:
    """
    读取CSV并返回电压列（默认第2列）。
    兼容无表头/多列。
    """
    df = pd.read_csv(filepath, header=None)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]
    if len(valid_cols) == 0:
        raise ValueError(f"文件中没有可用数值列: {filepath}")
    # 优先取第2列（索引1），否则取最后一个有效列
    if 1 in numeric_df.columns and numeric_df[1].notna().sum() > 0:
        col = 1
    else:
        col = valid_cols[-1]
    signal = numeric_df[col].dropna().to_numpy(dtype=float)
    signal = to_1d_array(signal)
    if len(signal) < 20:
        raise ValueError(f"有效波形点太少: {filepath}")
    return signal


# =========================================================
# 3. 极值点检测（v2.2 增加幅度条件）
# =========================================================

def detect_extrema(smooth_signal: np.ndarray,
                   baseline_n: int,
                   deriv_thr: float,
                   baseline_median: float,
                   amp_ratio: float = EXTREMA_AMP_RATIO) -> List[Dict[str, Any]]:
    """
    在平滑信号上检测所有满足条件的极值点（波峰/波谷）。
    条件：
        1) 导数变号 (d_left * d_right < 0)
        2) 左右导数绝对值均 > deriv_thr
        3) |极值| > amp_ratio * |baseline_median|   （新增）
    
    参数
    ----
    smooth_signal : 平滑后的信号
    baseline_n : 基线结束索引（极值检测从此之后开始）
    deriv_thr : 导数阈值
    baseline_median : 基线中值（用于幅度判断）
    amp_ratio : 幅度比例阈值，默认为 1.05

    返回值
    ----
    extrema : 字典列表，每个字典包含 index, type, value, left_deriv, right_deriv
    """
    x = to_1d_array(smooth_signal)
    n = len(x)
    if n < 3:
        return []

    # 计算左右导数
    d_left = np.zeros(n)
    d_right = np.zeros(n)
    d_right[0] = x[1] - x[0] if n > 1 else 0.0
    for i in range(1, n-1):
        d_left[i] = x[i] - x[i-1]
        d_right[i] = x[i+1] - x[i]
    if n > 1:
        d_left[n-1] = x[n-1] - x[n-2]
        d_right[n-1] = 0.0

    # 幅度阈值（绝对值）
    amp_threshold = amp_ratio * abs(baseline_median)
    # 避免 baseline_median 过小导致条件过于宽松，增加一个最小值阈值（例如1e-6）
    amp_threshold = max(amp_threshold, 1e-6)

    extrema = []
    start_idx = max(baseline_n, 1)
    for i in range(start_idx, n-1):
        # 条件1：导数变号
        if d_left[i] * d_right[i] >= 0:
            continue
        # 条件2：左右导数幅度均超过阈值
        if abs(d_left[i]) <= deriv_thr or abs(d_right[i]) <= deriv_thr:
            continue
        # 条件3：极值的绝对值超过幅度阈值（新增）
        if abs(x[i]) <= amp_threshold:
            continue

        # 判断类型
        if d_left[i] > 0 and d_right[i] < 0:
            etype = "peak"
        elif d_left[i] < 0 and d_right[i] > 0:
            etype = "valley"
        else:
            continue

        extrema.append({
            "index": i,
            "type": etype,
            "value": float(x[i]),
            "left_deriv": float(d_left[i]),
            "right_deriv": float(d_right[i]),
        })
    return extrema


def find_first_extremum(extrema_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """返回极值点列表中的第一个（索引最小）。"""
    if not extrema_list:
        return None
    return extrema_list[0]


def estimate_baseline_median(signal: np.ndarray, base_ratio: float) -> float:
    """计算前 base_ratio 部分的信号中位数。"""
    x = to_1d_array(signal)
    n = len(x)
    baseline_n = max(5, int(math.floor(n * base_ratio)))
    baseline_n = min(baseline_n, n)
    base = x[:baseline_n]
    median = float(np.median(base)) if len(base) > 0 else 0.0
    return median


# =========================================================
# 4. 单对 A/B 故障检测与定位（v2.2 增加幅度条件参数）
# =========================================================

def fault_location_single_v2(va: np.ndarray,
                             vb: np.ndarray,
                             fs: float,
                             line_length: float,
                             wave_speed: float = WAVE_SPEED,
                             base_ratio: float = BASE_RATIO,
                             gauss_sigma: float = GAUSS_SIGMA,
                             deriv_thr: float = DERIV_THR,
                             strict_align: bool = STRICT_ALIGN,
                             amp_ratio: float = EXTREMA_AMP_RATIO) -> Dict[str, Any]:
    """
    对一对 A/B 波形执行 v2.2 检测与定位。
    增加幅度条件：极值绝对值 > amp_ratio * |baseline_median|
    """
    va = to_1d_array(va)
    vb = to_1d_array(vb)
    n = min(len(va), len(vb))
    va = va[:n]
    vb = vb[:n]

    # 1. 高斯滤波
    va_smooth = gaussian_smooth(va, gauss_sigma)
    vb_smooth = gaussian_smooth(vb, gauss_sigma)

    # 2. 计算基线长度和中值（基于平滑信号）
    baseline_n = max(5, int(math.floor(n * base_ratio)))
    baseline_n = min(baseline_n, n)
    baseline_median_a = estimate_baseline_median(va_smooth, base_ratio)
    baseline_median_b = estimate_baseline_median(vb_smooth, base_ratio)

    # 3. 检测极值点（传入基线中值和幅度比例）
    extrema_a = detect_extrema(va_smooth, baseline_n, deriv_thr, baseline_median_a, amp_ratio)
    extrema_b = detect_extrema(vb_smooth, baseline_n, deriv_thr, baseline_median_b, amp_ratio)

    # 4. 取第一个有效极值点
    first_a = find_first_extremum(extrema_a)
    first_b = find_first_extremum(extrema_b)

    # 基础检测是否两端都有极值点
    both_detected = (first_a is not None) and (first_b is not None)

    # 严格模式检查：如果要求同类型对齐，则检查类型是否相同
    type_match = False
    if both_detected:
        type_match = (first_a["type"] == first_b["type"])

    if strict_align:
        success = both_detected and type_match
    else:
        success = both_detected

    # 构建结果字典（基础字段）
    result: Dict[str, Any] = {
        "success": success,
        "method_used": "gauss_peak_valley",
        "A_len": len(va),
        "B_len": len(vb),
        "baseline_n": baseline_n,
        "gauss_sigma": gauss_sigma,
        "deriv_thr": deriv_thr,
        "amp_ratio": amp_ratio,
        "baseline_median_a": baseline_median_a,
        "baseline_median_b": baseline_median_b,
        "extrema_count_a": len(extrema_a),
        "extrema_count_b": len(extrema_b),
        "strict_align": strict_align,
    }

    # 处理失败情况
    if not success:
        reason_parts = []
        if first_a is None:
            reason_parts.append("A端未检测到满足条件的极值点")
        if first_b is None:
            reason_parts.append("B端未检测到满足条件的极值点")
        if both_detected and strict_align and not type_match:
            reason_parts.append(f"严格模式下两端首波类型不一致 (A:{first_a['type']} vs B:{first_b['type']})")
        result["reason"] = "/".join(reason_parts) if reason_parts else "未知原因"
        result["confidence"] = 0.0
        result["distance_m"] = None
        result["distance_raw_m"] = None
        result["tA_s"] = None
        result["tB_s"] = None
        result["delta_t_s"] = None
        result["onset_a"] = first_a["index"] if first_a else None
        result["onset_b"] = first_b["index"] if first_b else None
        result["extreme_type_a"] = first_a["type"] if first_a else None
        result["extreme_type_b"] = first_b["type"] if first_b else None
        result["extreme_value_a"] = first_a["value"] if first_a else None
        result["extreme_value_b"] = first_b["value"] if first_b else None
        result["score_quality_a"] = 0.0
        result["score_quality_b"] = 0.0
        # 保存平滑数据用于可视化
        result["plot_data"] = {
            "va_raw": va,
            "vb_raw": vb,
            "va_smooth": va_smooth,
            "vb_smooth": vb_smooth,
            "baseline_n": baseline_n,
            "extrema_a": extrema_a,
            "extrema_b": extrema_b,
            "baseline_median_a": baseline_median_a,
            "baseline_median_b": baseline_median_b,
        }
        return result

    # 成功检测到极值点（且严格模式下类型相同）
    onset_a = first_a["index"]
    onset_b = first_b["index"]
    type_a = first_a["type"]
    type_b = first_b["type"]
    value_a = first_a["value"]
    value_b = first_b["value"]

    # 计算时间差与距离
    delta_samples = onset_a - onset_b
    delta_s = delta_samples / fs
    distance_raw_m = (line_length + wave_speed * delta_s) / 2.0
    distance_m = float(np.clip(distance_raw_m, 0.0, line_length))

    # 计算置信度：信号质量分（单端）
    eps = 1e-6
    ratio_a = abs(value_a) / (abs(baseline_median_a) + eps)
    ratio_b = abs(value_b) / (abs(baseline_median_b) + eps)
    q_a = np.clip((ratio_a - 1.0) / ratio_a, 0.0, 1.0) if ratio_a > 1e-12 else 0.0
    q_b = np.clip((ratio_b - 1.0) / ratio_b, 0.0, 1.0) if ratio_b > 1e-12 else 0.0
    signal_quality = 0.5 * q_a + 0.5 * q_b

    # 物理边界惩罚
    boundary_penalty = 1.0
    if abs(distance_raw_m - distance_m) > 0.05 * line_length:
        boundary_penalty = 0.85

    confidence = float(np.clip(signal_quality * boundary_penalty, 0.0, 1.0))

    # 填充结果
    result.update({
        "distance_m": distance_m,
        "distance_raw_m": distance_raw_m,
        "tA_s": float(onset_a / fs),
        "tB_s": float(onset_b / fs),
        "delta_t_s": delta_s,
        "confidence": confidence,
        "onset_a": onset_a,
        "onset_b": onset_b,
        "extreme_type_a": type_a,
        "extreme_type_b": type_b,
        "extreme_value_a": value_a,
        "extreme_value_b": value_b,
        "score_quality_a": float(q_a),
        "score_quality_b": float(q_b),
        "reason": "",
    })

    # 保存绘图所需数据
    result["plot_data"] = {
        "va_raw": va,
        "vb_raw": vb,
        "va_smooth": va_smooth,
        "vb_smooth": vb_smooth,
        "baseline_n": baseline_n,
        "first_a": first_a,
        "first_b": first_b,
        "extrema_a": extrema_a,
        "extrema_b": extrema_b,
        "baseline_median_a": baseline_median_a,
        "baseline_median_b": baseline_median_b,
    }

    return result


# =========================================================
# 5. 可视化（与 v2.1 相同，无变化）
# =========================================================

def plot_voltage_extrema(plot_data: Dict[str, Any],
                         title: str,
                         save_path: str,
                         dpi: int = PLOT_DPI,
                         meta: Optional[Dict[str, Any]] = None) -> bool:
    """
    绘制 A/B 端的原始电压（灰细线）、高斯滤波后电压（蓝粗线）、
    基线结束竖线、首波极值点红色星号、基线中值水平虚线。
    """
    if not plot_data:
        return False

    va_raw = np.asarray(plot_data.get("va_raw", []), dtype=float)
    vb_raw = np.asarray(plot_data.get("vb_raw", []), dtype=float)
    va_smooth = np.asarray(plot_data.get("va_smooth", []), dtype=float)
    vb_smooth = np.asarray(plot_data.get("vb_smooth", []), dtype=float)
    if len(va_raw) == 0 or len(vb_raw) == 0:
        return False

    baseline_n = int(plot_data.get("baseline_n", 0))
    baseline_median_a = float(plot_data.get("baseline_median_a", np.nan))
    baseline_median_b = float(plot_data.get("baseline_median_b", np.nan))
    first_a = plot_data.get("first_a")
    first_b = plot_data.get("first_b")
    extrema_a = plot_data.get("extrema_a", [])
    extrema_b = plot_data.get("extrema_b", [])

    meta = meta or {}
    pair_tag = meta.get("pair_name", title)
    success = meta.get("success", False)
    status = "成功" if success else "失败"
    if not HAS_CJK_FONT:
        status = "success" if success else "failed"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(
        f"{pair_tag} | 高斯滤波+极值点检测 | 状态={status}"
        if HAS_CJK_FONT else
        f"{pair_tag} | Gaussian filter + extremum detection | status={status}",
        fontsize=11
    )

    # A 端子图
    ax = axes[0]
    ax.plot(va_raw, color="gray", lw=0.8, alpha=0.7, label="Raw voltage")
    ax.plot(va_smooth, color="blue", lw=1.5, label="Smoothed (Gaussian)")
    ax.axvline(baseline_n - 1, color="gray", ls="--", lw=1.0, label=f"Baseline end (n={baseline_n})")
    if not np.isnan(baseline_median_a):
        ax.axhline(baseline_median_a, color="orange", ls=":", lw=1.2, label=f"Baseline median = {baseline_median_a:.3f}")
    # 标记所有极值点（用小圆圈）
    for e in extrema_a:
        color = "red" if e["type"] == "peak" else "green"
        ax.scatter(e["index"], e["value"], color=color, s=20, alpha=0.6, zorder=3)
    # 标记第一个极值点（大红星）
    if first_a is not None:
        ax.scatter(first_a["index"], first_a["value"], color="red", marker="*", s=120, zorder=5,
                   edgecolors="k", linewidths=0.5)
        ax.annotate(f"{first_a['type']}\nidx={first_a['index']}\nval={first_a['value']:.2f}",
                    xy=(first_a["index"], first_a["value"]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    ax.set_title("A端：原始电压 vs 平滑电压（蓝色）" if HAS_CJK_FONT else "A-side: Raw vs Smoothed (blue)")
    ax.set_ylabel("电压 (V)" if HAS_CJK_FONT else "Voltage (V)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    # B 端子图
    ax = axes[1]
    ax.plot(vb_raw, color="gray", lw=0.8, alpha=0.7, label="Raw voltage")
    ax.plot(vb_smooth, color="blue", lw=1.5, label="Smoothed (Gaussian)")
    ax.axvline(baseline_n - 1, color="gray", ls="--", lw=1.0, label=f"Baseline end (n={baseline_n})")
    if not np.isnan(baseline_median_b):
        ax.axhline(baseline_median_b, color="orange", ls=":", lw=1.2, label=f"Baseline median = {baseline_median_b:.3f}")
    for e in extrema_b:
        color = "red" if e["type"] == "peak" else "green"
        ax.scatter(e["index"], e["value"], color=color, s=20, alpha=0.6, zorder=3)
    if first_b is not None:
        ax.scatter(first_b["index"], first_b["value"], color="red", marker="*", s=120, zorder=5,
                   edgecolors="k", linewidths=0.5)
        ax.annotate(f"{first_b['type']}\nidx={first_b['index']}\nval={first_b['value']:.2f}",
                    xy=(first_b["index"], first_b["value"]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    ax.set_title("B端：原始电压 vs 平滑电压（蓝色）" if HAS_CJK_FONT else "B-side: Raw vs Smoothed (blue)")
    ax.set_xlabel("采样点" if HAS_CJK_FONT else "Sample index")
    ax.set_ylabel("电压 (V)" if HAS_CJK_FONT else "Voltage (V)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_summary(result_df: pd.DataFrame, save_path: str, dpi: int = PLOT_DPI) -> bool:
    """总览图：置信度与距离（与旧版兼容）"""
    if result_df is None or len(result_df) == 0:
        return False
    df = result_df.copy()
    if "confidence" not in df.columns:
        return False
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df = df.sort_values(by="confidence", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))

    total_pairs = len(df)
    success_pairs = int(df["success"].sum()) if "success" in df.columns else 0
    mean_conf = float(df["confidence"].mean()) if total_pairs > 0 else 0.0
    success_rate = (success_pairs / total_pairs * 100.0) if total_pairs > 0 else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    title = (
        f"全局总览 | 成功率={success_rate:.1f}% | 平均置信度={mean_conf:.3f}"
        if HAS_CJK_FONT else
        f"Overview | success_rate={success_rate:.1f}% | mean_conf={mean_conf:.3f}"
    )
    fig.suptitle(title, fontsize=11)

    axes[0].bar(x, df["confidence"].values, width=0.8)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("置信度" if HAS_CJK_FONT else "Confidence")
    axes[0].set_title("置信度（降序）" if HAS_CJK_FONT else "Confidence (descending)")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].plot(x, df["distance_m"].values, lw=1.0, marker="o", ms=3)
    axes[1].set_ylabel("距离(m)" if HAS_CJK_FONT else "Distance (m)")
    axes[1].set_xlabel("按置信度排序后的样本序号" if HAS_CJK_FONT else "Sample rank by confidence")
    axes[1].set_title("距离分布" if HAS_CJK_FONT else "Distance distribution")
    axes[1].grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return True


def build_auto_insights(result_df: pd.DataFrame, params_snapshot: Dict[str, Any]) -> List[str]:
    insights = []
    if result_df is None or len(result_df) == 0:
        return ["未检测到结果数据，请先检查输入文件与配对。"]
    total_pairs = len(result_df)
    success_pairs = int(result_df["success"].sum()) if "success" in result_df.columns else 0
    success_rate = success_pairs / max(1, total_pairs)
    mean_conf = float(pd.to_numeric(result_df["confidence"], errors="coerce").fillna(0.0).mean())
    insights.append(f"总配对数={total_pairs}，成功数={success_pairs}，成功率={success_rate*100:.1f}%，平均置信度={mean_conf:.3f}。")

    if "reason" in result_df.columns:
        fail_reasons = result_df.loc[result_df["success"] == False, "reason"].dropna()
        if len(fail_reasons) > 0:
            top_reason = fail_reasons.value_counts().index[0]
            top_count = int(fail_reasons.value_counts().iloc[0])
            insights.append(f"主要失败原因：{top_reason}（{top_count}次）。")

    # 检查是否因极值检测阈值过高导致失败率过高
    if success_rate < 0.3:
        insights.append("成功率偏低，尝试降低 `DERIV_THR` 或增大 `GAUSS_SIGMA` 以增强平滑，或降低 `EXTREMA_AMP_RATIO`。")
    elif success_rate > 0.9 and mean_conf < 0.4:
        insights.append("成功率高但置信度偏低，可适当提高 `DERIV_THR` 或 `EXTREMA_AMP_RATIO` 滤除弱极值。")

    # 如果严格模式开启且失败原因中包含类型不一致，给出提示
    if params_snapshot.get("STRICT_ALIGN", False) and success_rate < 0.5:
        insights.append("当前启用了严格模式（要求两端极值类型相同），若实际故障行波极性可能相反，可尝试设置 STRICT_ALIGN=False 获得更高成功率。")

    return insights


def generate_html_report(output_path: str,
                         result_df: pd.DataFrame,
                         best_summary: Dict[str, Any],
                         pair_plot_records: List[Dict[str, Any]],
                         summary_plot_relpath: str,
                         params_snapshot: Dict[str, Any]) -> None:
    insights = build_auto_insights(result_df, params_snapshot)
    params_lines = "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in params_snapshot.items()])
    insights_lines = "".join([f"<li>{x}</li>" for x in insights])

    rows_html = []
    for rec in pair_plot_records:
        img = f"<img src='{rec.get('plot_relpath', '')}' width='950'>" if rec.get("plot_relpath") else "<em>无</em>"
        rows_html.append(
            f"""
            <h3>Pair {rec.get('pair_index')} | {rec.get('pair_name')}</h3>
            <p>success={rec.get('success')} | distance_m={rec.get('distance_m')} | confidence={rec.get('confidence')} | reason={rec.get('reason','')}</p>
            <p>
              A onset={rec.get('onset_a')} ({rec.get('extreme_type_a')}) |
              B onset={rec.get('onset_b')} ({rec.get('extreme_type_b')}) |
              Δt={rec.get('delta_t_s')}
            </p>
            <div><b>图：原始电压 + 平滑曲线 + 极值点标记</b><br>{img}</div>
            <hr>
            """
        )

    summary_img_html = f"<img src='{summary_plot_relpath}' width='950'>" if summary_plot_relpath else "<em>无</em>"
    total_pairs = len(result_df) if result_df is not None else 0
    success_pairs = int(result_df["success"].sum()) if (result_df is not None and "success" in result_df.columns) else 0

    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>双端行波故障定位报告 v2.2（高斯滤波+极值检测+严格模式+幅度阈值）</title>
</head>
<body>
  <h1>双端行波故障定位报告 v2.2</h1>
  <p>算法：高斯滤波 + 波峰/波谷极值对齐法</p>
  <p>严格模式：{'启用' if params_snapshot.get('STRICT_ALIGN', False) else '禁用'}</p>
  <p>极值幅度阈值：|极值| > {params_snapshot.get('EXTREMA_AMP_RATIO', 1.05)} * |基线中值|</p>
  <h2>运行概览</h2>
  <p>总配对数: {total_pairs} | 成功配对数: {success_pairs}</p>
  <p>最佳结果: {best_summary}</p>

  <h2>参数快照</h2>
  <ul>{params_lines}</ul>

  <h2>自动结论与建议</h2>
  <ul>{insights_lines}</ul>

  <h2>全局总览图</h2>
  {summary_img_html}

  <h2>逐对图像</h2>
  {''.join(rows_html)}
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# =========================================================
# 6. 主流程
# =========================================================

def run_fault_location(a_folder: str = A_FOLDER,
                       b_folder: str = B_FOLDER,
                       output_dir: str = OUTPUT_DIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ensure_dir(output_dir)
    plot_dir = os.path.join(output_dir, PLOT_DIR_NAME)
    if ENABLE_VIS and SAVE_PNG:
        ensure_dir(plot_dir)

    a_files = list_csv_files(a_folder)
    b_files = list_csv_files(b_folder)
    if len(a_files) == 0:
        raise FileNotFoundError(f"A文件夹没有找到CSV文件: {a_folder}")
    if len(b_files) == 0:
        raise FileNotFoundError(f"B文件夹没有找到CSV文件: {b_folder}")

    pairs = pair_files_by_key(a_files, b_files)

    results = []
    pair_plot_records = []

    print("=" * 110)
    print(f"A文件数: {len(a_files)} | B文件数: {len(b_files)} | 配对数: {len(pairs)}")
    print(f"严格模式: {'启用 (要求两端极值类型相同)' if STRICT_ALIGN else '禁用 (任意极值类型均可)'}")
    print(f"极值幅度阈值: |极值| > {EXTREMA_AMP_RATIO} * |基线中值|")
    print("=" * 110)

    plotted_pairs = 0

    for idx, (a_path, b_path, pair_name) in enumerate(pairs, start=1):
        try:
            va = read_voltage_csv(a_path)
            vb = read_voltage_csv(b_path)

            res = fault_location_single_v2(
                va=va,
                vb=vb,
                fs=FS,
                line_length=LINE_LENGTH_M,
                wave_speed=WAVE_SPEED,
                base_ratio=BASE_RATIO,
                gauss_sigma=GAUSS_SIGMA,
                deriv_thr=DERIV_THR,
                strict_align=STRICT_ALIGN,
                amp_ratio=EXTREMA_AMP_RATIO,
            )

            row = {
                "pair_index": idx,
                "pair_name": pair_name,
                "A_file": os.path.basename(a_path),
                "B_file": os.path.basename(b_path),
                "success": res.get("success", False),
                "method_used": res.get("method_used", ""),
                "distance_m": res.get("distance_m", None),
                "distance_raw_m": res.get("distance_raw_m", None),
                "confidence": res.get("confidence", 0.0),
                "reason": res.get("reason", ""),
                "A_len": res.get("A_len", None),
                "B_len": res.get("B_len", None),
                "onset_a": res.get("onset_a", None),
                "onset_b": res.get("onset_b", None),
                "tA_s": res.get("tA_s", None),
                "tB_s": res.get("tB_s", None),
                "delta_t_s": res.get("delta_t_s", None),
                "extreme_type_a": res.get("extreme_type_a", None),
                "extreme_type_b": res.get("extreme_type_b", None),
                "extreme_value_a": res.get("extreme_value_a", None),
                "extreme_value_b": res.get("extreme_value_b", None),
                "baseline_median_a": res.get("baseline_median_a", None),
                "baseline_median_b": res.get("baseline_median_b", None),
                "score_quality_a": res.get("score_quality_a", 0.0),
                "score_quality_b": res.get("score_quality_b", 0.0),
                "extrema_count_a": res.get("extrema_count_a", 0),
                "extrema_count_b": res.get("extrema_count_b", 0),
                "gauss_sigma": res.get("gauss_sigma", GAUSS_SIGMA),
                "deriv_thr": res.get("deriv_thr", DERIV_THR),
                "amp_ratio": res.get("amp_ratio", EXTREMA_AMP_RATIO),
                "strict_align": STRICT_ALIGN,
            }
            results.append(row)

            if ENABLE_VIS and SAVE_PNG and plotted_pairs < int(MAX_PLOT_PAIRS):
                safe_pair = safe_filename(pair_name)
                prefix = f"{idx:04d}_{safe_pair}"
                plot_name = f"{prefix}_gauss_extrema.png"
                plot_path = os.path.join(plot_dir, plot_name)

                meta = {"pair_name": pair_name, "success": row["success"]}
                plot_ok = False
                try:
                    plot_ok = plot_voltage_extrema(
                        plot_data=res.get("plot_data", {}),
                        title=prefix,
                        save_path=plot_path,
                        dpi=PLOT_DPI,
                        meta=meta,
                    )
                except Exception as viz_err:
                    print(f"[WARN] 绘图失败 {pair_name}: {viz_err}")

                pair_plot_records.append({
                    "pair_index": idx,
                    "pair_name": pair_name,
                    "success": row["success"],
                    "distance_m": row["distance_m"],
                    "confidence": row["confidence"],
                    "reason": row["reason"],
                    "onset_a": row["onset_a"],
                    "onset_b": row["onset_b"],
                    "extreme_type_a": row["extreme_type_a"],
                    "extreme_type_b": row["extreme_type_b"],
                    "delta_t_s": row["delta_t_s"],
                    "plot_relpath": os.path.join(PLOT_DIR_NAME, plot_name).replace("\\", "/") if plot_ok else "",
                })
                plotted_pairs += 1

            print(
                f"[{idx}/{len(pairs)}] {pair_name} | "
                f"success={row['success']} | "
                f"d={row['distance_m']} m | "
                f"conf={row['confidence']:.4f} | "
                f"A_onset={row['onset_a']}({row['extreme_type_a']}) "
                f"B_onset={row['onset_b']}({row['extreme_type_b']}) | "
                f"reason={row['reason']}"
            )

        except Exception as e:
            row = {
                "pair_index": idx,
                "pair_name": pair_name,
                "A_file": os.path.basename(a_path),
                "B_file": os.path.basename(b_path),
                "success": False,
                "method_used": "gauss_peak_valley",
                "distance_m": None,
                "distance_raw_m": None,
                "confidence": 0.0,
                "reason": str(e),
                "A_len": None,
                "B_len": None,
                "onset_a": None,
                "onset_b": None,
                "tA_s": None,
                "tB_s": None,
                "delta_t_s": None,
                "extreme_type_a": None,
                "extreme_type_b": None,
                "extreme_value_a": None,
                "extreme_value_b": None,
                "baseline_median_a": None,
                "baseline_median_b": None,
                "score_quality_a": 0.0,
                "score_quality_b": 0.0,
                "extrema_count_a": 0,
                "extrema_count_b": 0,
                "gauss_sigma": GAUSS_SIGMA,
                "deriv_thr": DERIV_THR,
                "amp_ratio": EXTREMA_AMP_RATIO,
                "strict_align": STRICT_ALIGN,
            }
            results.append(row)
            print(f"[{idx}/{len(pairs)}] {pair_name} | ERROR: {e}")

    result_df = pd.DataFrame(results)
    result_csv = os.path.join(output_dir, "fault_location_results.csv")
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    # 选取最佳结果
    success_df = result_df[(result_df["success"] == True) & (result_df["distance_m"].notna())].copy()
    if len(success_df) > 0:
        sort_cols = [c for c in SORT_BEST_BY if c in success_df.columns]
        if not sort_cols:
            sort_cols = ["confidence"]
        success_df = success_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        best_row = success_df.iloc[0]
        best_summary = {
            "best_pair_name": best_row["pair_name"],
            "best_A_file": best_row["A_file"],
            "best_B_file": best_row["B_file"],
            "best_method_used": best_row["method_used"],
            "best_distance_m": float(best_row["distance_m"]),
            "best_confidence": float(best_row["confidence"]),
            "best_onset_a": int(best_row["onset_a"]) if pd.notna(best_row["onset_a"]) else None,
            "best_onset_b": int(best_row["onset_b"]) if pd.notna(best_row["onset_b"]) else None,
            "best_extreme_type_a": best_row["extreme_type_a"],
            "best_extreme_type_b": best_row["extreme_type_b"],
        }
    else:
        best_summary = {
            "best_pair_name": None,
            "best_A_file": None,
            "best_B_file": None,
            "best_method_used": None,
            "best_distance_m": None,
            "best_confidence": 0.0,
            "best_onset_a": None,
            "best_onset_b": None,
            "best_extreme_type_a": None,
            "best_extreme_type_b": None,
        }

    summary_path = os.path.join(output_dir, "best_fault_location_summary.csv")
    pd.DataFrame([best_summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 汇总图
    summary_plot_relpath = ""
    if ENABLE_VIS and SAVE_PNG and len(result_df) > 0:
        summary_plot_path = os.path.join(plot_dir, "summary_overview.png")
        try:
            ok = plot_summary(result_df, summary_plot_path, dpi=PLOT_DPI)
            if ok:
                summary_plot_relpath = os.path.join(PLOT_DIR_NAME, "summary_overview.png").replace("\\", "/")
        except Exception as viz_err:
            print(f"[WARN] 汇总图绘制失败: {viz_err}")

    # HTML 报告
    report_path = os.path.join(output_dir, "fault_location_report.html")
    if ENABLE_VIS and SAVE_HTML_REPORT:
        params_snapshot = {
            "LINE_LENGTH_M": LINE_LENGTH_M,
            "WAVE_SPEED": WAVE_SPEED,
            "FS": FS,
            "BASE_RATIO": BASE_RATIO,
            "GAUSS_SIGMA": GAUSS_SIGMA,
            "DERIV_THR": DERIV_THR,
            "EXTREMA_AMP_RATIO": EXTREMA_AMP_RATIO,
            "STRICT_ALIGN": STRICT_ALIGN,
            "MAX_PLOT_PAIRS": MAX_PLOT_PAIRS,
            "PLOT_DPI": PLOT_DPI,
        }
        generate_html_report(
            output_path=report_path,
            result_df=result_df,
            best_summary=best_summary,
            pair_plot_records=pair_plot_records,
            summary_plot_relpath=summary_plot_relpath,
            params_snapshot=params_snapshot
        )

    print("\n" + "=" * 110)
    print("全部配对结果已保存：", result_csv)
    print("最佳定位结果已保存：", summary_path)
    if ENABLE_VIS and SAVE_PNG:
        print("可视化图片目录：", plot_dir)
    if ENABLE_VIS and SAVE_HTML_REPORT:
        print("可视化报告已保存：", report_path)
    print("最佳结果：")
    print(best_summary)
    print("=" * 110)

    return result_df, best_summary


if __name__ == "__main__":
    run_fault_location()