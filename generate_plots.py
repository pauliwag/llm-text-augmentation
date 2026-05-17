#!/usr/bin/env python3
"""
Plot Generation
================
Creates all figures for the final report from saved results.

Generates:
  fig1_accuracy_f1_vs_k.png      - Accuracy and Macro F1 across methods and k (combined)
  fig2_accuracy_bars.png         - Grouped bar chart at each k
  fig3_per_class_f1.png          - Per-class F1 heatmap
  fig4_improvement_over_base.png - Relative improvement over no-augmentation baseline
  fig5_augmentation_volume.png   - Accuracy vs. training set size

Usage:
    python generate_plots.py [--results results/results_aggregated.json]
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({"font.size": 11})


# ============================================================================
# CONFIGURATION
# ============================================================================


DEFAULT_RESULTS_FILE = "results/results_aggregated.json"
OUTPUT_DIR = "results"
FIGURES_DIR = "results/figures"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
K_VALUES = [50, 100, 200]

METHOD_DISPLAY = {
    "none": "No Augmentation",
    "eda": "EDA (Synonym)",
    "backtranslation": "Back-Translation",
    "llm_zero_shot": "LLM Zero-Shot",
    "llm_paraphrase": "LLM Paraphrase",
    "llm_few_shot": "LLM Few-Shot",
}

METHOD_COLORS = {
    "none": "#95a5a6",
    "eda": "#e74c3c",
    "backtranslation": "#f39c12",
    "llm_zero_shot": "#2ecc71",
    "llm_paraphrase": "#3498db",
    "llm_few_shot": "#9b59b6",
}

METHOD_MARKERS = {
    "none": "o",
    "eda": "s",
    "backtranslation": "D",
    "llm_zero_shot": "^",
    "llm_paraphrase": "v",
    "llm_few_shot": "P",
}


def load_results(path):
    """Load aggregated results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def _get_display(method):
    return METHOD_DISPLAY.get(method, method)


def _get_color(method):
    return METHOD_COLORS.get(method, "#333333")


def _get_marker(method):
    return METHOD_MARKERS.get(method, "o")


# ============================================================================
# PLOT 1: ACCURACY AND F1 VS K (COMBINED TWO-PANEL)
# ============================================================================


def plot_accuracy_f1_vs_k(agg):
    """Combined line plot: accuracy (left) and macro F1 (right) vs. k."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    methods = list(agg.keys())
    n_methods = len(methods)

    # Horizontal jitter so overlapping points and error bars are readable
    jitter_width = 13  # Total spread in k-axis units
    offsets = np.linspace(-jitter_width / 2, jitter_width / 2, n_methods)

    for idx, method in enumerate(methods):
        ks = sorted([int(k) for k in agg[method]])
        label = _get_display(method)
        color = _get_color(method)
        marker = _get_marker(method)
        jittered_ks = [k + offsets[idx] for k in ks]

        # Accuracy
        acc_means = [agg[method][str(k)]["accuracy_mean"] for k in ks]
        acc_stds = [agg[method][str(k)]["accuracy_std"] for k in ks]
        ax1.errorbar(
            jittered_ks,
            acc_means,
            yerr=acc_stds,
            marker=marker,
            color=color,
            linewidth=2,
            capsize=4,
            markersize=8,
            label=label,
        )

        # Macro F1
        f1_means = [agg[method][str(k)]["macro_f1_mean"] for k in ks]
        f1_stds = [agg[method][str(k)]["macro_f1_std"] for k in ks]
        ax2.errorbar(
            jittered_ks,
            f1_means,
            yerr=f1_stds,
            marker=marker,
            color=color,
            linewidth=2,
            capsize=4,
            markersize=8,
            label=label,
        )

    for ax, ylabel, title in [
        (ax1, "Accuracy", "Classification Accuracy"),
        (ax2, "Macro F1", "Macro F1 Score"),
    ]:
        ax.set_xlabel("k (examples per class)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0.84, 0.895)
        ax.grid(True, alpha=0.3)

    # Single legend below
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(n_methods, 3),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.1),
    )

    fig.suptitle(
        "Performance vs. Training Set Size by Augmentation Method\n"
        "(mean ± std across 3 seeds)",
        fontsize=14,
        y=1,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig1_accuracy_f1_vs_k.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig1_accuracy_f1_vs_k.png')}")
    plt.close()


# ============================================================================
# PLOT 2: GROUPED BAR CHART
# ============================================================================


def plot_grouped_bars(agg):
    """Grouped bar chart: accuracy for each method at each k."""
    methods = list(agg.keys())
    n_methods = len(methods)
    n_k = len(K_VALUES)

    fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 6), sharey=True)
    if n_k == 1:
        axes = [axes]

    for ax, k in zip(axes, K_VALUES):
        k_str = str(k)
        means = [agg[m].get(k_str, {}).get("accuracy_mean", 0) for m in methods]
        stds = [agg[m].get(k_str, {}).get("accuracy_std", 0) for m in methods]
        colors = [_get_color(m) for m in methods]
        labels = [_get_display(m) for m in methods]

        bars = ax.bar(
            range(n_methods),
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"k = {k}", fontsize=13)
        ax.set_ylabel("Accuracy" if k == K_VALUES[0] else "", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, m_val, s_val in zip(bars, means, stds):
            if m_val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s_val + 0.01,
                    f"{m_val:.3f}",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    fig.suptitle(
        "Classification Accuracy by Method and Training Size", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig2_accuracy_bars.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig2_accuracy_bars.png')}")
    plt.close()


# ============================================================================
# PLOT 3: PER-CLASS F1 HEATMAP
# ============================================================================


def plot_per_class_heatmap(agg, k_to_show=None):
    """Heatmap: per-class F1 for each method at a given k."""
    import seaborn as sns

    # Default to first available k if not specified
    if k_to_show is None:
        available_ks = set()
        for m in agg:
            available_ks.update(int(k) for k in agg[m])
        k_to_show = sorted(available_ks)[len(available_ks) // 2]  # Pick middle k

    k_str = str(k_to_show)
    methods = [m for m in agg if k_str in agg[m]]
    if not methods:
        print(f"No data for k={k_to_show}; skipping heatmap.")
        return

    labels_display = [_get_display(m) for m in methods]
    matrix = np.array([agg[m][k_str]["per_class_f1_mean"] for m in methods])

    fig, ax = plt.subplots(figsize=(8, max(4, len(methods) * 0.6 + 1)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=LABEL_NAMES,
        yticklabels=labels_display,
        ax=ax,
        cbar_kws={"label": "F1 Score"},
        vmin=max(0.4, matrix.min() - 0.05),
        vmax=min(1.0, matrix.max() + 0.05),
    )
    ax.set_title(
        f"Per-Class F1 Score by Augmentation Method (k={k_to_show})", fontsize=13
    )
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Method", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig3_per_class_f1.png"), dpi=150, bbox_inches="tight"
    )
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig3_per_class_f1.png')}")
    plt.close()


# ============================================================================
# PLOT 4: IMPROVEMENT OVER BASELINE
# ============================================================================


def plot_improvement(agg):
    """Bar chart: relative improvement in accuracy over no-augmentation baseline."""
    if "none" not in agg:
        print("Skipping improvement plot (no baseline).")
        return

    methods = [m for m in agg if m != "none"]
    if not methods:
        return

    fig, axes = plt.subplots(
        1, len(K_VALUES), figsize=(5 * len(K_VALUES), 6), sharey=True
    )
    if len(K_VALUES) == 1:
        axes = [axes]

    for ax, k in zip(axes, K_VALUES):
        k_str = str(k)
        baseline_acc = agg["none"].get(k_str, {}).get("accuracy_mean", 0)
        if baseline_acc == 0:
            continue

        improvements, colors, labels = [], [], []
        for m in methods:
            if k_str in agg[m]:
                imp = (agg[m][k_str]["accuracy_mean"] - baseline_acc) * 100
                improvements.append(imp)
                colors.append(_get_color(m))
                labels.append(_get_display(m))

        bars = ax.bar(
            range(len(improvements)),
            improvements,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"k = {k}", fontsize=13)
        ax.set_ylabel(
            "Accuracy Improvement (pp)" if k == K_VALUES[0] else "", fontsize=11
        )
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, improvements):
            y_pos = bar.get_height() - 0.09 if val >= 0 else bar.get_height() + 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:+.2f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle(
        "Accuracy Improvement Over No-Augmentation Baseline (percentage points)",
        fontsize=13,
        y=1,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig4_improvement_over_base.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig4_improvement_over_base.png')}")
    plt.close()


# ============================================================================
# PLOT 5: ACCURACY VS TRAINING SET SIZE
# ============================================================================


def plot_accuracy_vs_train_size(agg):
    """Scatter: accuracy vs. total training examples used."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in agg:
        sizes, accs, stds = [], [], []
        for k_str in sorted(agg[method].keys(), key=int):
            sizes.append(agg[method][k_str]["n_train"])
            accs.append(agg[method][k_str]["accuracy_mean"])
            stds.append(agg[method][k_str]["accuracy_std"])

        ax.errorbar(
            sizes,
            accs,
            yerr=stds,
            marker=_get_marker(method),
            color=_get_color(method),
            linewidth=2,
            capsize=4,
            markersize=10,
            label=_get_display(method),
        )

    ax.set_xlabel("Total Training Examples (real + augmented)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        "Accuracy vs. Effective Training Set Size",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "fig5_augmentation_volume.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig5_augmentation_volume.png')}")
    plt.close()


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================


def generate_latex_table(agg):
    """Generate a LaTeX-formatted results table for the report."""
    lines = []
    lines.append("% Auto-generated results table")
    lines.append("\\begin{table}[t]")
    lines.append(
        "\\caption{Classification accuracy and macro F1 by augmentation "
        "method and training size (mean $\\pm$ std across 3 seeds).}"
    )
    lines.append("\\label{tab:main_results}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "rr" * len(K_VALUES) + "}")
    lines.append("\\toprule")

    header = "\\textbf{Method}"
    for k in K_VALUES:
        header += f" & \\textbf{{Acc (k={k})}} & \\textbf{{F1 (k={k})}}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for method in agg:
        display = _get_display(method)
        row = display
        for k in K_VALUES:
            k_str = str(k)
            if k_str in agg[method]:
                am = agg[method][k_str]["accuracy_mean"]
                a_s = agg[method][k_str]["accuracy_std"]
                fm = agg[method][k_str]["macro_f1_mean"]
                fs = agg[method][k_str]["macro_f1_std"]
                row += f" & {am:.3f}$\\pm${a_s:.3f} & {fm:.3f}$\\pm${fs:.3f}"
            else:
                row += " & --- & ---"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "results_table.tex"), "w") as f:
        f.write(table_str)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'results_table.tex')}")
    return table_str


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate plots from results")
    parser.add_argument(
        "--results",
        type=str,
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to aggregated results JSON (default: {DEFAULT_RESULTS_FILE})",
    )
    args = parser.parse_args()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating Plots from Experiment Results")
    print("=" * 60)

    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found. Run run_experiments.py first.")
        return

    agg = load_results(args.results)
    methods = list(agg.keys())
    print(f"Methods found: {methods}")

    # Fig 1: Combined accuracy + F1 vs k
    plot_accuracy_f1_vs_k(agg)

    # Fig 2: Grouped bar chart
    plot_grouped_bars(agg)

    # Fig 3: Per-class F1 heatmap
    try:
        plot_per_class_heatmap(agg, k_to_show=100)
    except ImportError:
        print("seaborn not installed; skipping heatmap.")

    # Fig 4: Improvement over baseline
    plot_improvement(agg)

    # Fig 5: Accuracy vs training size
    plot_accuracy_vs_train_size(agg)

    # LaTeX table
    latex = generate_latex_table(agg)
    print("\nLaTeX table:")
    print(latex)

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
