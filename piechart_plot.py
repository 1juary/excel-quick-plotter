import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from typing import Optional

try:
    from numeric_coercion import coerce_numeric_series
except ImportError:
    # 兼容性 Fallback：若 numeric_coercion 模块未处于 Python 环境变量下
    def coerce_numeric_series(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors='coerce')


def _is_blank_cell(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _is_numeric_type_cell(value) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.number))


def _detect_header_row(df: pd.DataFrame) -> bool:
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        return False

    first_row = df.iloc[0, :].tolist()
    for cell in first_row:
        if _is_blank_cell(cell):
            continue
        if not _is_numeric_type_cell(cell):
            return True
    return False


def _render_empty_placeholder(fig: Figure, message: str) -> None:
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="gray")
    ax.axis('off')


def render_pie_and_radial_chart(
    fig: Figure,
    df: pd.DataFrame,
    sheet_name: str = "Data",
    excel_start_row: Optional[int] = None,
) -> None:
    """
    在传入的 Matplotlib Figure 上绘制南丁格尔玫瑰图与径向柱状图双图。
    """
    fig.clear()

    # 1. 检测与清洗数据
    has_header = _detect_header_row(df)
    if has_header:
        data_df = df.iloc[1:, :]
    else:
        data_df = df

    if data_df.shape[0] == 0 or data_df.shape[1] == 0:
        _render_empty_placeholder(fig, "No data available in selection.")
        return

    # 2. 提取类别标签与数据值
    # 若大于等于2列，默认第1列为标签，第2列为数值；若只有1列，自动生成虚拟标签
    if data_df.shape[1] >= 2:
        raw_labels = data_df.iloc[:, 0]
        raw_values = data_df.iloc[:, 1]
    else:
        raw_values = data_df.iloc[:, 0]
        raw_labels = [f"Item {i+1}" for i in range(len(raw_values))]

    # 滤除缺失值并强转为数值，由于是饼图类需要滤除 <= 0 的非正数数据
    numeric_values = coerce_numeric_series(raw_values)
    valid_mask = numeric_values.notna() & (numeric_values > 0)

    filtered_values = numeric_values[valid_mask].astype(float).tolist()
    if isinstance(raw_labels, pd.Series):
        filtered_labels = raw_labels[valid_mask].tolist()
    else:
        filtered_labels = [raw_labels[i] for i, valid in enumerate(valid_mask) if valid]

    if not filtered_values:
        _render_empty_placeholder(fig, "No positive numeric data found to plot.")
        return

    # 格式化清洗后的标签
    clean_labels = []
    for l in filtered_labels:
        if _is_blank_cell(l):
            clean_labels.append("N/A")
        else:
            clean_labels.append(str(l).strip())

    total = sum(filtered_values)
    percentages = [v / total * 100 for v in filtered_values]

    # 3. 动态配置画布比例与间距（宽幅设计以容纳双图）
    try:
        fig.set_size_inches(13, 6.5, forward=True)
    except Exception:
        pass

    try:
        fig.subplots_adjust(wspace=0.35, bottom=0.22, top=0.85)
    except Exception:
        pass

    # 4. 配色与图例配置
    color_palette = ['#5B85B0', '#C8B04A', '#D4894A', '#5DA85A', '#7ABFBD', '#D47076', '#E5A93C', '#9467BD', '#8C564B', '#BCBD22']
    colors = [color_palette[i % len(color_palette)] for i in range(len(clean_labels))]

    # 校验数据本身是否本来就是百分比形式
    is_percentage_already = (95.0 <= total <= 105.0)
    legend_patches = []
    for c, l, v, p in zip(colors, clean_labels, filtered_values, percentages):
        if is_percentage_already:
            label_text = f"{l}  {v:.1f}%"
        else:
            label_text = f"{l}  {v:.1f} ({p:.1f}%)"
        legend_patches.append(mpatches.Patch(color=c, label=label_text))

    # ---------------------------------------------------
    # 图1：南丁格尔玫瑰图
    # ---------------------------------------------------
    ax1 = fig.add_subplot(121, projection='polar')
    widths = [v / total * 2 * np.pi for v in filtered_values]

    angles = []
    start = 0
    for w in widths:
        angles.append(start + w / 2)
        start += w

    max_val = max(filtered_values)
    # 高度自适应缩放：最大柱体高度设为 10
    bar_heights = [v / max_val * 10 for v in filtered_values]
    bottom_offset = 3

    bars1 = ax1.bar(
        angles, bar_heights, width=widths, color=colors,
        edgecolor='white', linewidth=1.5, bottom=bottom_offset
    )

    ax1.set_axis_off()
    ax1.set_title(
        f"Nightingale Rose Chart\n(南丁格尔玫瑰图)\n_{sheet_name}", 
        pad=20, fontsize=12, fontweight='bold'
    )

    ax1.legend(
        handles=legend_patches,
        loc='upper center', bbox_to_anchor=(0.5, -0.08),
        ncol=min(3, len(clean_labels)), fontsize=8, framealpha=0.9, edgecolor='#cccccc',
        title="Shares" if not is_percentage_already else "Values", title_fontsize=8,
        borderaxespad=0, handletextpad=0.8, labelspacing=0.4
    )

    # 注入元数据供鼠标 Hover 交互展示
    for j, patch in enumerate(bars1.patches):
        try:
            setattr(
                patch,
                "_eqp_meta",
                {
                    "label": clean_labels[j],
                    "value": filtered_values[j],
                    "percentage": percentages[j],
                    "type": "Rose Chart"
                },
            )
        except Exception:
            pass

    # ---------------------------------------------------
    # 图2：径向柱状图
    # ---------------------------------------------------
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_theta_zero_location('S')
    ax2.set_theta_direction(1)

    ring_width = 2
    # 环状间距根据元素个数自适应堆叠
    bottoms = np.arange(1, len(clean_labels) + 1) * 3 + 6
    thetas = [w / 2 for w in widths]

    bars2 = ax2.bar(
        thetas, height=ring_width, width=widths, bottom=bottoms,
        color=colors, edgecolor='white', linewidth=0.5
    )

    ax2.set_axis_off()
    ax2.set_title(
        f"Radial Bar Chart\n(径向柱状图)\n_{sheet_name}", 
        pad=20, fontsize=12, fontweight='bold'
    )

    ax2.legend(
        handles=legend_patches,
        loc='upper center', bbox_to_anchor=(0.5, -0.08),
        ncol=min(3, len(clean_labels)), fontsize=8, framealpha=0.9, edgecolor='#cccccc',
        title="Shares" if not is_percentage_already else "Values", title_fontsize=8,
        borderaxespad=0, handletextpad=0.8, labelspacing=0.4
    )

    for j, patch in enumerate(bars2.patches):
        try:
            setattr(
                patch,
                "_eqp_meta",
                {
                    "label": clean_labels[j],
                    "value": filtered_values[j],
                    "percentage": percentages[j],
                    "type": "Radial Chart"
                },
            )
        except Exception:
            pass

    # ---------------------------------------------------
    # 5. 基础交互：绑定鼠标悬浮查看扇区数据
    # ---------------------------------------------------
    try:
        import mplcursors

        prev_cursor = getattr(fig, "_eqp_mplcursors_cursor", None)
        if prev_cursor is not None:
            try:
                prev_cursor.remove()
            except Exception:
                pass

        artists = list(bars1.patches) + list(bars2.patches)
        if artists:
            cursor = mplcursors.cursor(artists, hover=True)

            try:
                setattr(fig, "_eqp_mplcursors_annotations", [])
            except Exception:
                pass

            @cursor.connect("add")
            def _on_add(sel):
                meta = getattr(sel.artist, "_eqp_meta", None)
                if not meta:
                    return

                try:
                    ann_list = getattr(fig, "_eqp_mplcursors_annotations", None)
                    if ann_list is None:
                        ann_list = []
                        setattr(fig, "_eqp_mplcursors_annotations", ann_list)
                    ann_list.append(sel.annotation)
                except Exception:
                    pass

                label = meta.get("label", "N/A")
                val = meta.get("value", 0.0)
                pct = meta.get("percentage", 0.0)
                chart_type = meta.get("type", "Chart")

                sel.annotation.set_text(
                    f"{chart_type}\n"
                    f"Label: {label}\n"
                    f"Value: {val:.6g}\n"
                    f"Share: {pct:.1f}%"
                )

            setattr(fig, "_eqp_mplcursors_cursor", cursor)
    except Exception:
        # 若运行环境未配置 mplcursors，则静默跳过交互逻辑
        pass