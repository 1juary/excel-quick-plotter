import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from typing import Optional

from numeric_coercion import (
    coerce_numeric_series,
    detect_header_row as _detect_header_row,
    is_blank_cell as _is_blank_cell,
)

def render_paired_trend_chart(
    ax: Axes,
    df: pd.DataFrame,
    sheet_name: str = "Data",
    excel_start_row: Optional[int] = None,
) -> None:
    # 1. 提取表头与纯数据 (将完全空白的单元格处理为空字符串，以在 X 轴留白)
    has_header = _detect_header_row(df)
    if has_header:
        header_values = df.iloc[0, :].tolist()
        x_labels = ["" if _is_blank_cell(v) else str(v).strip() for v in header_values]
        data_df = df.iloc[1:, :].copy()
        actual_start_row = (excel_start_row or 1) + 1 
    else:
        x_labels = [str(col) for col in df.columns]
        data_df = df.copy()
        actual_start_row = excel_start_row or 1

    # 2. 强制转换数值，Excel 中的空列会变成全 NaN
    for col in data_df.columns:
        data_df[col] = coerce_numeric_series(data_df[col])

    num_stages = len(x_labels)
    x_positions = np.arange(num_stages)
    
    # 3. UI 审美设定：多巴胺调色
    color_up = "#EE884C"    
    color_down = "#3DC2EC"  
    color_flat = "#BDC3C7"  
    
    ax.set_facecolor("#FFFFFF")
    fig = ax.figure
    
    base_width = max(6.0, num_stages * 1.5)
    try:
        fig.set_size_inches(base_width, 5.0, forward=True)
    except Exception:
        pass

    all_values = []
    scatter_artists = []

    # 4. 逐行绘制
    for idx, row_series in data_df.iterrows():
        y_values = row_series.to_numpy(dtype=float)
        valid_mask = ~np.isnan(y_values)
        
        # 如果整行全空，直接跳过
        if not np.any(valid_mask):
            continue
            
        all_values.extend(y_values[valid_mask])
        
        # 计算整体趋势 (基于该行首尾第一个和最后一个非空值)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) >= 2:
            first_val = y_values[valid_indices[0]]
            last_val = y_values[valid_indices[-1]]
            delta = last_val - first_val
            threshold = abs(first_val * 0.01) if first_val != 0 else 0.01
            if delta > threshold:
                line_color = color_up
            elif delta < -threshold:
                line_color = color_down
            else:
                line_color = color_flat
        else:
            line_color = color_flat

        # 【核心逻辑 1】绘制折线：直接传入包含 NaN 的完整数组，Matplotlib 会在此处自动断开连线
        ax.plot(
            x_positions, 
            y_values, 
            color=line_color, 
            linewidth=1.5, 
            alpha=0.6,
            zorder=2
        )
        
        # 【核心逻辑 2】绘制散点：仅传入非空坐标，防止悬浮提示选到空气
        scatter = ax.scatter(
            x_positions[valid_mask], 
            y_values[valid_mask], 
            color="white", 
            edgecolors=line_color,
            s=45, 
            linewidths=1.5,
            zorder=3
        )
        
        # 绑定元数据供交互使用 (仅保留非空阶段的数据)
        excel_row_num = actual_start_row + int(idx) - (1 if has_header else 0)
        try:
            setattr(
                scatter,
                "_eqp_meta",
                {
                    "label": f"Sample Row {excel_row_num}",
                    "rows": [excel_row_num] * np.sum(valid_mask),
                    "values": y_values[valid_mask],
                    "x_labels": [x_labels[i] for i, valid in enumerate(valid_mask) if valid],
                    "is_outlier": [False] * np.sum(valid_mask) 
                },
            )
        except Exception:
            pass
        scatter_artists.append(scatter)

    # 5. 坐标轴与背景网格设置
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold', color="#2C3E50")
    ax.set_ylabel("Measurement Value", fontsize=11, color="#2C3E50")
    
    ax.yaxis.grid(True, linestyle="--", which="major", color="#E9ECEF", alpha=0.8)
    ax.xaxis.grid(False)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color("#BDC3C7")
        ax.spines[spine].set_linewidth(1.5)

    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_title(f"Stage Trend Analysis_{sheet_name}", fontsize=14, fontweight='bold', color="#2C3E50", pad=15)

    legend_elements = [
        Line2D([0], [0], color=color_up, lw=2, marker='o', label="Upward Trend"),
        Line2D([0], [0], color=color_down, lw=2, marker='o', label="Downward Trend"),
        Line2D([0], [0], color=color_flat, lw=2, marker='o', label="Stable"),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=False)

    # 6. 绑定悬浮提示交互
    try:
        import mplcursors
        prev_cursor = getattr(ax.figure, "_eqp_mplcursors_cursor", None)
        if prev_cursor is not None:
            try:
                prev_cursor.remove()
            except Exception:
                pass

        if scatter_artists:
            cursor = mplcursors.cursor(scatter_artists, hover=True)
            @cursor.connect("add")
            def _on_add(sel):
                meta = getattr(sel.artist, "_eqp_meta", None)
                if not meta:
                    return
                idx = int(sel.index)
                rows = meta.get("rows")
                values = meta.get("values")
                x_labs = meta.get("x_labels")
                
                excel_row = rows[idx] if rows is not None else "Unknown"
                val = values[idx]
                stage = x_labs[idx] if x_labs and idx < len(x_labs) else f"Col {idx}"
                
                delta_str = ""
                # values 数组已经剔除了空值，所以 values[0] 就是该行的初始数据基准
                if idx > 0:
                    delta_t0 = val - values[0]
                    sign = "+" if delta_t0 > 0 else ""
                    delta_str = f"\nΔ初始: {sign}{delta_t0:.4g}"

                sel.annotation.set_text(
                    f"Row: {excel_row}\n"
                    f"Stage: {stage}\n"
                    f"Value: {val:.4g}{delta_str}"
                )
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, ec=sel.artist.get_edgecolors()[0])
            setattr(ax.figure, "_eqp_mplcursors_cursor", cursor)
    except Exception:
        pass