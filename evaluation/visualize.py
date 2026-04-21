"""
evaluation/visualize.py
论文级可视化 —— 生成可直接用于课程论文的图表。
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


FIGURE_DIR = "evaluation/figures"


def _ensure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def _setup_style():
    """统一论文配图风格。"""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    if HAS_SNS:
        sns.set_palette("Set2")


# ═══════════════════════════════════════════════════════════
#  Fig 1: Schema Compliance Per Node (Bar)
# ═══════════════════════════════════════════════════════════

def plot_compliance_by_node(consistency_results: List[dict], save_name: str = "fig1_schema_compliance.png"):
    """各节点 JSON 可解析率 & Pydantic 通过率柱状图。"""
    if not HAS_MPL:
        print("⚠️  matplotlib not installed, skipping plot.")
        return
    _ensure_dir()
    _setup_style()

    nodes_list = ["visual_agent", "semantic_agent", "anomaly_agent"]
    json_rates = {n: [] for n in nodes_list}
    fallback_rates = {n: [] for n in nodes_list}

    for run in consistency_results:
        sc = run.get("schema_compliance", {})
        fb = run.get("fallback_triggered", {})
        for n in nodes_list:
            json_rates[n].append(1 if sc.get(n, {}).get("json_parseable", False) else 0)
            fallback_rates[n].append(1 if fb.get(n, False) else 0)

    x = np.arange(len(nodes_list))
    width = 0.35

    json_means = [np.mean(json_rates[n]) * 100 for n in nodes_list]
    no_fallback = [(1 - np.mean(fallback_rates[n])) * 100 for n in nodes_list]

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, json_means, width, label="JSON Parseable (%)", color="#4C72B0")
    ax.bar(x + width / 2, no_fallback, width, label="No Fallback (%)", color="#55A868")

    ax.set_ylabel("Rate (%)")
    ax.set_title("RQ1: Schema Compliance by Agent Node")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_agent", "").title() for n in nodes_list])
    ax.set_ylim(0, 110)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Fig 2: Field Completeness Radar
# ═══════════════════════════════════════════════════════════

def plot_completeness_radar(consistency_results: List[dict], save_name: str = "fig2_field_completeness.png"):
    """各节点字段完整度雷达图。"""
    if not HAS_MPL:
        return
    _ensure_dir()
    _setup_style()

    nodes_check = ["visual_agent", "semantic_agent"]
    rates = {n: [] for n in nodes_check}
    for run in consistency_results:
        fc = run.get("field_completeness", {})
        for n in nodes_check:
            rates[n].append(fc.get(n, {}).get("completeness_rate", 0))

    labels = [n.replace("_agent", "").title() for n in nodes_check]
    means = [np.mean(rates[n]) for n in nodes_check]
    stds = [np.std(rates[n]) for n in nodes_check]

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Completeness Rate")
    ax.set_title("RQ1: Field Completeness by Agent (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{m:.1%}", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Fig 3: Cross-run Consistency Box Plot
# ═══════════════════════════════════════════════════════════

def plot_consistency_boxplot(consistency_results: List[dict], save_name: str = "fig3_consistency.png"):
    """跨运行一致性：各节点输出丰富度的箱线图。"""
    if not HAS_MPL:
        return
    _ensure_dir()
    _setup_style()

    richness_data = {}
    for run in consistency_results:
        for node, rich in run.get("output_richness", {}).items():
            if node not in richness_data:
                richness_data[node] = []
            richness_data[node].append(rich.get("total_text_length", 0))

    if not richness_data:
        print("⚠️  No output_richness data, skipping.")
        return

    labels = list(richness_data.keys())
    data = [richness_data[l] for l in labels]
    display_labels = [l.replace("_agent", "").title() for l in labels]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, labels=display_labels, patch_artist=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Total Text Length (chars)")
    ax.set_title("RQ2: Output Richness Consistency Across Runs")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Fig 4: Temperature Sensitivity Line Plot
# ═══════════════════════════════════════════════════════════

def plot_sensitivity_line(
    sweep_results: List[dict],
    param_name: str = "temperature",
    save_name: str = None,
):
    """
    参数敏感性折线图：横轴为参数值，纵轴为多个指标。
    """
    if not HAS_MPL:
        return
    _ensure_dir()
    _setup_style()

    if save_name is None:
        save_name = f"fig4_sensitivity_{param_name}.png"

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in sweep_results:
        grouped[r["sweep_value"]].append(r)

    param_vals = sorted(grouped.keys())

    compliance_means, compliance_stds = [], []
    time_means, time_stds = [], []
    richness_means = []

    for v in param_vals:
        runs = grouped[v]

        comp = []
        for r in runs:
            sc = r.get("schema_compliance", {})
            n_ok = sum(1 for s in sc.values() if s.get("json_parseable", False))
            comp.append(n_ok / max(len(sc), 1))
        compliance_means.append(np.mean(comp) * 100)
        compliance_stds.append(np.std(comp) * 100)

        times = [r.get("total_time", 0) for r in runs]
        time_means.append(np.mean(times))
        time_stds.append(np.std(times))

        rich = []
        for r in runs:
            tl = sum(
                v2.get("total_text_length", 0)
                for v2 in r.get("output_richness", {}).values()
                if isinstance(v2, dict)
            )
            rich.append(tl)
        richness_means.append(np.mean(rich))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].errorbar(param_vals, compliance_means, yerr=compliance_stds,
                     marker="o", capsize=4, color="#4C72B0")
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("JSON Compliance (%)")
    axes[0].set_title("Schema Compliance")
    axes[0].set_ylim(-5, 110)

    axes[1].errorbar(param_vals, time_means, yerr=time_stds,
                     marker="s", capsize=4, color="#DD8452")
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("Total Time (s)")
    axes[1].set_title("Pipeline Latency")

    axes[2].plot(param_vals, richness_means, marker="^", color="#55A868")
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel("Total Text Length")
    axes[2].set_title("Output Richness")

    fig.suptitle(f"RQ3: Sensitivity to {param_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Fig 5: Image Robustness Comparison (Grouped Bar)
# ═══════════════════════════════════════════════════════════

def plot_robustness_comparison(perturbation_results: List[dict], save_name: str = "fig5_robustness.png"):
    """图片扰动鲁棒性对比图。"""
    if not HAS_MPL:
        return
    _ensure_dir()
    _setup_style()

    perturbations = [r["perturbation"] for r in perturbation_results]

    compliance = []
    coherence = []
    for r in perturbation_results:
        sc = r.get("schema_compliance", {})
        n_ok = sum(1 for s in sc.values() if s.get("json_parseable", False))
        compliance.append(n_ok / max(len(sc), 1) * 100)
        coherence.append(r.get("coherence", {}).get("overall_coherence", 0) * 100)

    x = np.arange(len(perturbations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, compliance, width, label="Schema Compliance (%)", color="#4C72B0")
    ax.bar(x + width / 2, coherence, width, label="Pipeline Coherence (%)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=35, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("RQ4: Robustness Under Image Perturbations")
    ax.set_ylim(0, 110)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Fig 6: Latency Breakdown (Stacked Bar)
# ═══════════════════════════════════════════════════════════

def plot_latency_breakdown(consistency_results: List[dict], save_name: str = "fig6_latency.png"):
    """各节点延迟占比堆叠柱状图。"""
    if not HAS_MPL:
        return
    _ensure_dir()
    _setup_style()

    node_order = ["visual_agent", "semantic_agent", "anomaly_agent", "report_agent"]
    runs_label = [f"Run {r.get('run_index', i)+1}" for i, r in enumerate(consistency_results)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(consistency_results))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for node, color in zip(node_order, colors):
        vals = [r.get("node_times", {}).get(node, 0) for r in consistency_results]
        ax.bar(runs_label, vals, bottom=bottom, label=node.replace("_", " ").title(), color=color)
        bottom += np.array(vals)

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Latency Breakdown per Run")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Fig 7: Coherence Heatmap
# ═══════════════════════════════════════════════════════════

def plot_coherence_heatmap(consistency_results: List[dict], save_name: str = "fig7_coherence.png"):
    """流水线一致性指标热力图。"""
    if not HAS_MPL:
        return
    _ensure_dir()
    _setup_style()

    metrics_keys = [
        "visual_semantic_theme",
        "visual_in_report",
        "insights_in_report_mean",
        "anomalies_in_report_mean",
        "title_in_report",
        "overall_coherence",
    ]
    display_labels = [
        "Visual↔Semantic\nTheme",
        "Visual→Report",
        "Insights→Report",
        "Anomalies→Report",
        "Title→Report",
        "Overall",
    ]

    matrix = []
    for run in consistency_results:
        coh = run.get("coherence", {})
        row = [coh.get(k, 0) for k in metrics_keys]
        matrix.append(row)

    matrix = np.array(matrix)
    run_labels = [f"Run {i+1}" for i in range(len(consistency_results))]

    fig, ax = plt.subplots(figsize=(9, max(4, len(consistency_results) * 0.6 + 1)))

    if HAS_SNS:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                    xticklabels=display_labels, yticklabels=run_labels,
                    vmin=0, vmax=1, ax=ax)
    else:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(display_labels)))
        ax.set_xticklabels(display_labels, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(run_labels)))
        ax.set_yticklabels(run_labels)
        plt.colorbar(im, ax=ax)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("RQ5: Inter-Agent Coherence Scores")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  Fig 8: Prompt Variant Comparison
# ═══════════════════════════════════════════════════════════

def plot_prompt_variants(prompt_results: List[dict], save_name: str = "fig8_prompt_variants.png"):
    """Prompt 变体对比图。"""
    if not HAS_MPL:
        return
    _ensure_dir()
    _setup_style()

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in prompt_results:
        grouped[r["variant"]].append(r)

    variants = list(grouped.keys())
    json_rates = []
    pydantic_rates = []
    avg_times = []

    for v in variants:
        runs = grouped[v]
        json_rates.append(np.mean([1 if r["json_parseable"] else 0 for r in runs]) * 100)
        pydantic_rates.append(np.mean([1 if r["pydantic_valid"] else 0 for r in runs]) * 100)
        avg_times.append(np.mean([r["duration_s"] for r in runs]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(variants))
    width = 0.35
    ax1.bar(x - width / 2, json_rates, width, label="JSON Parseable", color="#4C72B0")
    ax1.bar(x + width / 2, pydantic_rates, width, label="Pydantic Valid", color="#55A868")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=20, ha="right")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Compliance by Prompt Variant")
    ax1.set_ylim(0, 110)
    ax1.legend()

    ax2.bar(variants, avg_times, color="#DD8452")
    ax2.set_ylabel("Avg Time (s)")
    ax2.set_title("Latency by Prompt Variant")
    plt.xticks(rotation=20, ha="right")

    fig.suptitle("RQ3: Prompt Engineering Sensitivity", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved → {path}")


# ═══════════════════════════════════════════════════════════
#  生成全部图表（从已保存的 JSON 结果）
# ═══════════════════════════════════════════════════════════

def generate_all_figures(results_dir: str = "evaluation/results"):
    """从已保存的结果文件一次性生成全部论文图表。"""
    import json

    def _load(name):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    consistency = _load("consistency_results.json")
    temp_sweep = _load("temperature_sweep.json")
    top_p_sweep = _load("top_p_sweep.json")
    token_sweep = _load("max_tokens_sweep.json")
    robustness = _load("robustness_results.json")
    prompts = _load("prompt_variant_results.json")

    if consistency:
        plot_compliance_by_node(consistency)
        plot_completeness_radar(consistency)
        plot_consistency_boxplot(consistency)
        plot_latency_breakdown(consistency)
        plot_coherence_heatmap(consistency)

    if temp_sweep:
        plot_sensitivity_line(temp_sweep, "temperature", "fig4_sensitivity_temperature.png")
    if top_p_sweep:
        plot_sensitivity_line(top_p_sweep, "top_p", "fig4_sensitivity_top_p.png")
    if token_sweep:
        plot_sensitivity_line(token_sweep, "max_new_tokens", "fig4_sensitivity_max_tokens.png")
    if robustness:
        plot_robustness_comparison(robustness)
    if prompts:
        plot_prompt_variants(prompts)

    print(f"\n✅ All figures saved to {FIGURE_DIR}/")
