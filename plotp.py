import matplotlib.pyplot as plt
import numpy as np

# =========================
# 数据区（来自你发的结果）
# =========================

# --- HF overall ---
hf_models = ["baseline", "medusa1", "medusa2"]
hf_tps = np.array([65.66, 151.07, 195.95], dtype=float)

# --- HF category (baseline + medusa2) ---
hf_base_cat = {
    "coding":65.84, "extraction":64.88, "humanities":65.69, "math":65.79,
    "reasoning":65.41, "roleplay":65.75, "stem":65.72, "writing":65.74
}
hf_medusa2_cat = {
    "coding":225.64, "extraction":216.46, "humanities":184.09, "math":212.98,
    "reasoning":179.32, "roleplay":177.21, "stem":192.67, "writing":184.00
}

# --- Jittor overall ---
jt_models = ["baseline", "medusa1"]
jt_tps = np.array([30.41, 67.37], dtype=float)

# --- Jittor category (baseline + medusa1) ---
jt_base_cat = {
    "writing":29.79, "roleplay":29.88, "reasoning":29.76, "math":29.95,
    "coding":30.14, "extraction":31.57, "stem":32.18, "humanities":29.98
}
jt_medusa1_cat = {
    "writing":62.46, "roleplay":65.28, "reasoning":62.82, "math":74.34,
    "coding":76.09, "extraction":67.92, "stem":65.38, "humanities":62.05
}

# =========================
# 绘图工具函数
# =========================

def annotate_top(ax, bars, labels, dy_ratio=0.02, fontsize=11):
    """在柱顶上方标注文本"""
    y0, y1 = ax.get_ylim()
    dy = (y1 - y0) * dy_ratio
    for b, lab in zip(bars, labels):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + dy,
            lab,
            ha="center", va="bottom",
            fontsize=fontsize
        )

def plot_overall_tps_only_speedup(models, tps, title, out_png, colors):
    """
    总体速度柱状图：
    - 柱高 = TPS
    - 柱顶只标 speedup（不标TPS）
    """
    baseline = tps[0]
    speedup = tps / baseline

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    x = np.arange(len(models))

    bars = ax.bar(
        x, tps,
        width=0.55,
        color=colors,
        edgecolor="black",
        linewidth=0.7
    )

    ax.set_xticks(x, models)
    ax.set_ylabel("Tokens per second (TPS)")
    ax.set_title(title)
    ax.set_ylim(0, max(tps) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    labels = [f"{s:.2f}×" for s in speedup]
    annotate_top(ax, bars, labels, dy_ratio=0.015, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_category_speedup_sorted(base_cat, medusa_cat, title, out_png):
    """
    分类柱状图：
    - 排序依据 = speedup 从低到高
    - 柱高 = speedup（保证单调递增）
    - 每根柱颜色不同
    - 柱顶标 speedup
    """
    cats = list(medusa_cat.keys())
    speedup = {c: medusa_cat[c] / base_cat[c] for c in cats}

    sorted_cats = sorted(cats, key=lambda c: speedup[c])
    sp = np.array([speedup[c] for c in sorted_cats], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=220)
    x = np.arange(len(sorted_cats))

    # 每个柱子不同颜色（用 colormap）
    cmap = plt.get_cmap("tab10")
    bar_colors = [cmap(i % 10) for i in range(len(sorted_cats))]

    bars = ax.bar(
        x, sp,
        width=0.62,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7
    )

    ax.set_xticks(x, sorted_cats, rotation=20, ha="right")
    ax.set_ylabel("Speedup (× over baseline)")
    ax.set_title(title)
    ax.set_ylim(0, max(sp) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    labels = [f"{v:.2f}×" for v in sp]
    annotate_top(ax, bars, labels, dy_ratio=0.02, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# =========================
# 生成四张图
# =========================

if __name__ == "__main__":
    # 配色：baseline 蓝、medusa1 绿、medusa2 橙
    hf_colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    jt_colors = ["#4C78A8", "#59A14F"]

    # 1) HF overall（只标倍数）
    plot_overall_tps_only_speedup(
        models=hf_models,
        tps=hf_tps,
        title="HF group: Overall decoding speed",
        out_png="hf_overall_speedup_label.png",
        colors=hf_colors
    )

    # 2) HF category：用 Medusa2，画 speedup（单调递增 + 彩色柱）
    plot_category_speedup_sorted(
        base_cat=hf_base_cat,
        medusa_cat=hf_medusa2_cat,
        title="HF group: Medusa2 category speedup (sorted)",
        out_png="hf_medusa2_category_speedup_sorted.png"
    )

    # 3) Jittor overall（只标倍数）
    plot_overall_tps_only_speedup(
        models=jt_models,
        tps=jt_tps,
        title="Jittor group: Overall decoding speed",
        out_png="jittor_overall_speedup_label.png",
        colors=jt_colors
    )

    # 4) Jittor category：Medusa1 speedup（单调递增 + 彩色柱）
    plot_category_speedup_sorted(
        base_cat=jt_base_cat,
        medusa_cat=jt_medusa1_cat,
        title="Jittor group: Medusa1 category speedup (sorted)",
        out_png="jittor_medusa1_category_speedup_sorted.png"
    )

    print("Done! Generated PNGs:")
    print(" - hf_overall_speedup_label.png")
    print(" - hf_medusa2_category_speedup_sorted.png")
    print(" - jittor_overall_speedup_label.png")
    print(" - jittor_medusa1_category_speedup_sorted.png")
