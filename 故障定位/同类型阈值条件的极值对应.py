# -*- coding: utf-8 -*-
"""
双端行波故障定位算法 v3.0
基于专利 CN120275771A 的小波模极大值法

v3.0 核心算法：
    - 4层小波分解 (db4小波基)
    - 硬阈值去噪
    - 逐层重构得到降噪信号 y(n)
    - 一阶差分绝对值 |Δy| 提取
    - 自适应阈值过滤
    - 三条件第一极大点提取 + 极性判断

v2.x 特性保留：
    - STRICT_ALIGN 参数：要求两端首波极值类型相同
    - 置信度基于基线中值比
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
import pywt

# =========================================================
# 0. 参数配置（用户可修改）
# =========================================================

# 获取脚本所在目录的绝对路径，用于构建相对路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

A_FOLDER = os.path.join(_SCRIPT_DIR, "A")
B_FOLDER = os.path.join(_SCRIPT_DIR, "B")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output_v3_wavelet")

# 线路参数
LINE_LENGTH_M = 100000.0      # 线路总长度（m）
WAVE_SPEED = 2.8e8            # 行波传播速度（m/s）
FS = 4.21e6                   # 采样频率（Hz）

# 预处理参数
BASE_RATIO = 0.2             # 前 20% 采样点作为基线区间

# 小波方法参数
WAVELET_LEVELS = 4            # 小波分解层数
WAVELET_DY_TH = 0.2           # 差分阈值倍数（全局最大差分的比例）
WAVELET_M = 300               # 前M个点用于阈值计算
WAVELET_THRESH_RATIOS = [1.0, 0.3, 0.2, 0.3]  # 各层阈值比例

# 双端对齐模式
STRICT_ALIGN = True           # True: 严格模式，要求两端极值类型相同
                              # False: 宽松模式，任意极值类型均可用

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
# 2. 数据读取（保留原有逻辑）
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
# 3. 小波模极大值法核心算法（专利 CN120275771A）
# =========================================================

def wavelet_decompose(signal: np.ndarray, n_levels: int = 4) -> Dict:
    """使用 PyWavelets 进行4层小波分解，返回 wavedec 系数列表。"""
    return pywt.wavedec(signal, 'db4', level=n_levels)


def wavelet_threshold(coeffs: List[np.ndarray]) -> List[np.ndarray]:
    """硬阈值处理细节系数，d_i_p = d_i if |d_i| > d_i_set else 0。
    
    系数结构: [cA_n, cD_n, cD_n-1, ..., cD_1]
    阈值比例: d1_set = max(|d1|)×1.0, d2_set = max(|d2|)×0.3, ...
    """
    ratios = WAVELET_THRESH_RATIOS  # [1.0, 0.3, 0.2, 0.3]
    thresholded = [coeffs[0].copy()]  # 近似系数不变
    
    # 细节系数从最深层到最浅层（cD_n, cD_n-1, ..., cD_1）
    for i, d_i in enumerate(coeffs[1:]):
        # 反转索引：i=0 对应最深层（第4层），i=3 对应第1层
        level_from_top = len(coeffs) - 1 - i  # 4,3,2,1
        ratio = ratios[level_from_top - 1]  # ratios[3],ratios[2],ratios[1],ratios[0]
        
        d_i_set = np.max(np.abs(d_i)) * ratio
        thresholded.append(np.where(np.abs(d_i) > d_i_set, d_i, 0.0))
    
    return thresholded


def wavelet_reconstruct(thresholded: List[np.ndarray], signal_len: int) -> np.ndarray:
    """使用 PyWavelets 从阈值后的系数重构信号。"""
    y = pywt.waverec(thresholded, 'db4')
    # 确保输出长度与原始信号一致
    if len(y) > signal_len:
        y = y[:signal_len]
    elif len(y) < signal_len:
        y = np.pad(y, (0, signal_len - len(y)), mode='edge')
    return y


def wavelet_denoise(signal: np.ndarray, n_levels: int = 4) -> np.ndarray:
    """完整小波降噪：分解 → 阈值 → 重构。"""
    coeffs = wavelet_decompose(signal, n_levels)
    thresholded = wavelet_threshold(coeffs)
    return wavelet_reconstruct(thresholded, len(signal))


def compute_adaptive_threshold(delta_y: np.ndarray, 
                                M: int = WAVELET_M, 
                                dy_th: float = WAVELET_DY_TH) -> float:
    """自适应过滤阈值：max(max(|Δy|)×0.2, max(|Δy(1:M)|))。"""
    max_th = np.max(delta_y) * dy_th
    max_M = np.max(delta_y[:M]) if len(delta_y) >= M else np.max(delta_y)
    return max(max_th, max_M)


def detect_extrema_wavelet(y: np.ndarray,
                           baseline_n: int,
                           threshold: float) -> List[Dict[str, Any]]:
    """
    找到所有满足三条件的极大点：
      1. |Δy(s)| > |Δy(s-1)|
      2. |Δy(s)| > |Δy(s+1)|
      3. |Δy(s)| > threshold
    极性: y(s) > 0 → peak, y(s) ≤ 0 → valley
    """
    delta_y = np.abs(np.diff(y))
    extrema = []
    start_idx = max(baseline_n, 1)
    
    for s in range(start_idx, len(delta_y) - 1):
        if (delta_y[s] > delta_y[s - 1] and
            delta_y[s] > delta_y[s + 1] and
            delta_y[s] > threshold):
            
            etype = "peak" if y[s] > 0 else "valley"
            
            extrema.append({
                "index": int(s),
                "type": etype,
                "value": float(y[s]),
                "left_deriv": float(delta_y[s - 1]),
                "right_deriv": float(delta_y[s + 1]),
            })
    
    return extrema


def find_first_extremum(extrema_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """返回极值点列表中的第一个（索引最小）。"""
    if not extrema_list:
        return None
    return extrema_list[0]


def process_single_terminal_wavelet(signal: np.ndarray,
                                     base_ratio: float = BASE_RATIO,
                                     n_levels: int = WAVELET_LEVELS) -> Dict[str, Any]:
    """
    对单端信号执行完整的小波处理流程。
    返回: {'y_smooth', 'baseline_n', 'baseline_median', 'delta_y', 'threshold', 'extrema'}
    """
    n = len(signal)
    
    # 1. 小波降噪
    y = wavelet_denoise(signal, n_levels)
    
    if len(y) > n:
        y = y[:n]
    elif len(y) < n:
        y = np.pad(y, (0, n - len(y)), mode='edge')
    
    # 2. 基线计算
    baseline_n = max(5, int(math.floor(n * base_ratio)))
    baseline_n = min(baseline_n, n)
    baseline_median = float(np.median(y[:baseline_n]))
    
    # 3. 一阶差分绝对值
    delta_y = np.abs(np.diff(y))
    
    # 4. 自适应阈值
    threshold = compute_adaptive_threshold(delta_y)
    
    # 5. 极值检测
    extrema = detect_extrema_wavelet(y, baseline_n, threshold)
    
    return {
        'y_smooth': y,
        'baseline_n': baseline_n,
        'baseline_median': baseline_median,
        'delta_y': delta_y,
        'threshold': threshold,
        'extrema': extrema,
    }


# =========================================================
# 4. 单对 A/B 故障检测与定位（专利算法 v3.0）
# =========================================================

def fault_location_single_v3(va: np.ndarray,
                              vb: np.ndarray,
                              fs: float,
                              line_length: float,
                              wave_speed: float = WAVE_SPEED,
                              base_ratio: float = BASE_RATIO,
                              strict_align: bool = STRICT_ALIGN,
                              n_levels: int = WAVELET_LEVELS) -> Dict[str, Any]:
    """
    专利算法：小波模极大值法故障定位
    
    流程:
      1. A/B端分别小波降噪 → y_a, y_b
      2. 计算自适应阈值 → threshold_a, threshold_b
      3. 提取极值点 → extrema_a, extrema_b
      4. 取第一个极值点 → first_a, first_b
      5. 严格模式对齐检查 → 类型是否相同
      6. 计算时间差 → 距离 → 置信度
    """
    va = to_1d_array(va)
    vb = to_1d_array(vb)
    n = min(len(va), len(vb))
    va = va[:n]
    vb = vb[:n]
    
    # 1. 单端处理
    result_a = process_single_terminal_wavelet(va, base_ratio, n_levels)
    result_b = process_single_terminal_wavelet(vb, base_ratio, n_levels)
    
    # 2. 取第一个极值点
    extrema_a = result_a['extrema']
    extrema_b = result_b['extrema']
    first_a = find_first_extremum(extrema_a)
    first_b = find_first_extremum(extrema_b)
    
    # 3. 检测判断
    both_detected = (first_a is not None) and (first_b is not None)
    type_match = (first_a["type"] == first_b["type"]) if both_detected else False
    
    success = both_detected and (not strict_align or type_match)
    
    # 4. 构建结果
    result: Dict[str, Any] = {
        "success": success,
        "method_used": "wavelet_modulus_maxima",
        "A_len": len(va),
        "B_len": len(vb),
        "baseline_n": result_a['baseline_n'],
        "n_levels": n_levels,
        "threshold_a": result_a['threshold'],
        "threshold_b": result_b['threshold'],
        "baseline_median_a": result_a['baseline_median'],
        "baseline_median_b": result_b['baseline_median'],
        "extrema_count_a": len(extrema_a),
        "extrema_count_b": len(extrema_b),
        "strict_align": strict_align,
    }
    
    # 5. 失败处理
    if not success:
        reason_parts = []
        if first_a is None:
            reason_parts.append("A端未检测到满足条件的极值点")
        if first_b is None:
            reason_parts.append("B端未检测到满足条件的极值点")
        if both_detected and strict_align and not type_match:
            reason_parts.append(f"严格模式下两端首波类型不一致 (A:{first_a['type']} vs B:{first_b['type']})")
        
        result.update({
            "reason": "/".join(reason_parts) if reason_parts else "未知原因",
            "confidence": 0.0,
            "distance_m": None,
            "distance_raw_m": None,
            "tA_s": None,
            "tB_s": None,
            "delta_t_s": None,
            "onset_a": first_a["index"] if first_a else None,
            "onset_b": first_b["index"] if first_b else None,
            "extreme_type_a": first_a["type"] if first_a else None,
            "extreme_type_b": first_b["type"] if first_b else None,
            "extreme_value_a": first_a["value"] if first_a else None,
            "extreme_value_b": first_b["value"] if first_b else None,
            "score_quality_a": 0.0,
            "score_quality_b": 0.0,
        })
        result["plot_data"] = {
            "va_raw": va, "vb_raw": vb,
            "va_smooth": result_a['y_smooth'],
            "vb_smooth": result_b['y_smooth'],
            "delta_y_a": result_a['delta_y'],
            "delta_y_b": result_b['delta_y'],
            "threshold_a": result_a['threshold'],
            "threshold_b": result_b['threshold'],
            "baseline_n": result_a['baseline_n'],
            "extrema_a": extrema_a,
            "extrema_b": extrema_b,
            "baseline_median_a": result_a['baseline_median'],
            "baseline_median_b": result_b['baseline_median'],
        }
        return result
    
    # 6. 成功 - 计算距离
    onset_a = first_a["index"]
    onset_b = first_b["index"]
    
    delta_samples = onset_a - onset_b
    delta_s = delta_samples / fs
    distance_raw_m = (line_length + wave_speed * delta_s) / 2.0
    distance_m = float(np.clip(distance_raw_m, 0.0, line_length))
    
    # 7. 置信度计算
    eps = 1e-6
    ratio_a = abs(first_a["value"]) / (abs(result_a['baseline_median']) + eps)
    ratio_b = abs(first_b["value"]) / (abs(result_b['baseline_median']) + eps)
    q_a = np.clip((ratio_a - 1.0) / ratio_a, 0.0, 1.0) if ratio_a > 1e-12 else 0.0
    q_b = np.clip((ratio_b - 1.0) / ratio_b, 0.0, 1.0) if ratio_b > 1e-12 else 0.0
    signal_quality = 0.5 * q_a + 0.5 * q_b
    
    boundary_penalty = 1.0
    if abs(distance_raw_m - distance_m) > 0.05 * line_length:
        boundary_penalty = 0.85
    
    confidence = float(np.clip(signal_quality * boundary_penalty, 0.0, 1.0))
    
    result.update({
        "distance_m": distance_m,
        "distance_raw_m": distance_raw_m,
        "tA_s": float(onset_a / fs),
        "tB_s": float(onset_b / fs),
        "delta_t_s": delta_s,
        "confidence": confidence,
        "onset_a": onset_a,
        "onset_b": onset_b,
        "extreme_type_a": first_a["type"],
        "extreme_type_b": first_b["type"],
        "extreme_value_a": first_a["value"],
        "extreme_value_b": first_b["value"],
        "score_quality_a": float(q_a),
        "score_quality_b": float(q_b),
        "reason": "",
    })
    
    result["plot_data"] = {
        "va_raw": va, "vb_raw": vb,
        "va_smooth": result_a['y_smooth'],
        "vb_smooth": result_b['y_smooth'],
        "delta_y_a": result_a['delta_y'],
        "delta_y_b": result_b['delta_y'],
        "threshold_a": result_a['threshold'],
        "threshold_b": result_b['threshold'],
        "baseline_n": result_a['baseline_n'],
        "first_a": first_a,
        "first_b": first_b,
        "extrema_a": extrema_a,
        "extrema_b": extrema_b,
        "baseline_median_a": result_a['baseline_median'],
        "baseline_median_b": result_b['baseline_median'],
    }
    
    return result


# =========================================================
# 5. 可视化（适配小波方法）
# =========================================================

def plot_voltage_wavelet(plot_data: Dict[str, Any],
                         title: str,
                         save_path: str,
                         dpi: int = PLOT_DPI,
                         meta: Optional[Dict[str, Any]] = None) -> bool:
    """
    绘制 A/B 端的原始电压（灰细线）、小波重构信号（蓝粗线）、
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
        f"{pair_tag} | 小波模极大值法 | 状态={status}"
        if HAS_CJK_FONT else
        f"{pair_tag} | Wavelet modulus maxima | status={status}",
        fontsize=11
    )

    # A 端子图
    ax = axes[0]
    ax.plot(va_raw, color="gray", lw=0.8, alpha=0.7, label="Raw voltage")
    ax.plot(va_smooth, color="blue", lw=1.5, label="Denoised (Wavelet)")
    ax.axvline(baseline_n - 1, color="gray", ls="--", lw=1.0, label=f"Baseline end (n={baseline_n})")
    if not np.isnan(baseline_median_a):
        ax.axhline(baseline_median_a, color="orange", ls=":", lw=1.2, label=f"Baseline median = {baseline_median_a:.3f}")
    for e in extrema_a:
        color = "red" if e["type"] == "peak" else "green"
        ax.scatter(e["index"], e["value"], color=color, s=20, alpha=0.6, zorder=3)
    if first_a is not None:
        ax.scatter(first_a["index"], first_a["value"], color="red", marker="*", s=120, zorder=5,
                   edgecolors="k", linewidths=0.5)
        ax.annotate(f"{first_a['type']}\nidx={first_a['index']}\nval={first_a['value']:.2f}",
                    xy=(first_a["index"], first_a["value"]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    ax.set_title("A端：原始电压 vs 小波降噪信号（蓝色）" if HAS_CJK_FONT else "A-side: Raw vs Denoised (blue)")
    ax.set_ylabel("电压 (V)" if HAS_CJK_FONT else "Voltage (V)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    # B 端子图
    ax = axes[1]
    ax.plot(vb_raw, color="gray", lw=0.8, alpha=0.7, label="Raw voltage")
    ax.plot(vb_smooth, color="blue", lw=1.5, label="Denoised (Wavelet)")
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
    ax.set_title("B端：原始电压 vs 小波降噪信号（蓝色）" if HAS_CJK_FONT else "B-side: Raw vs Denoised (blue)")
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
        insights.append("成功率偏低，尝试降低 `WAVELET_DY_TH` 或调整 `WAVELET_THRESH_RATIOS` 以增强信号提取。")
    elif success_rate > 0.9 and mean_conf < 0.4:
        insights.append("成功率高但置信度偏低，可适当提高 `WAVELET_DY_TH` 滤除弱极值。")

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
  <title>双端行波故障定位报告 v3.0（小波模极大值法）</title>
</head>
<body>
  <h1>双端行波故障定位报告 v3.0</h1>
  <p>算法：小波模极大值法（专利 CN120275771A）</p>
  <p>严格模式：{'启用' if params_snapshot.get('STRICT_ALIGN', False) else '禁用'}</p>
  <p>小波层数: {params_snapshot.get('WAVELET_LEVELS', 4)} | 差分阈值倍数: {params_snapshot.get('WAVELET_DY_TH', 0.2)}</p>
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
    print(f"小波层数: {WAVELET_LEVELS} | 差分阈值倍数: {WAVELET_DY_TH} | M={WAVELET_M}")
    print("=" * 110)

    plotted_pairs = 0

    for idx, (a_path, b_path, pair_name) in enumerate(pairs, start=1):
        try:
            va = read_voltage_csv(a_path)
            vb = read_voltage_csv(b_path)

            res = fault_location_single_v3(
                va=va,
                vb=vb,
                fs=FS,
                line_length=LINE_LENGTH_M,
                wave_speed=WAVE_SPEED,
                base_ratio=BASE_RATIO,
                strict_align=STRICT_ALIGN,
                n_levels=WAVELET_LEVELS,
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
                "n_levels": res.get("n_levels", WAVELET_LEVELS),
                "threshold_a": res.get("threshold_a", None),
                "threshold_b": res.get("threshold_b", None),
                "strict_align": STRICT_ALIGN,
            }
            results.append(row)

            if ENABLE_VIS and SAVE_PNG and plotted_pairs < int(MAX_PLOT_PAIRS):
                safe_pair = safe_filename(pair_name)
                prefix = f"{idx:04d}_{safe_pair}"
                plot_name = f"{prefix}_wavelet_extrema.png"
                plot_path = os.path.join(plot_dir, plot_name)

                meta = {"pair_name": pair_name, "success": row["success"]}
                plot_ok = False
                try:
                    plot_ok = plot_voltage_wavelet(
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
                "method_used": "wavelet_modulus_maxima",
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
                "n_levels": WAVELET_LEVELS,
                "threshold_a": None,
                "threshold_b": None,
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
            "WAVELET_LEVELS": WAVELET_LEVELS,
            "WAVELET_DY_TH": WAVELET_DY_TH,
            "WAVELET_M": WAVELET_M,
            "WAVELET_THRESH_RATIOS": WAVELET_THRESH_RATIOS,
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