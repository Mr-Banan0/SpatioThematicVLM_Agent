"""
evaluation/run_all.py
评测框架总入口 —— 执行全部或指定实验，生成结果 + 图表。

Usage:
    # 快速模式（3 次一致性 + 3 温度值，适合调试）
    python -m evaluation.run_all --image test.png --mode quick

    # 标准模式（5 次一致性 + 全参数扫描）
    python -m evaluation.run_all --image test.png --mode standard

    # 完整模式（10 次一致性 + 全参数扫描 + 图片扰动 + Prompt 变体）
    python -m evaluation.run_all --image test.png --mode full

    # 仅从已有结果生成图表
    python -m evaluation.run_all --figures-only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.runner import run_consistency_test, save_results, load_results
from evaluation.sensitivity import (
    sweep_temperature,
    sweep_top_p,
    sweep_max_tokens,
    sweep_image_perturbations,
    sweep_prompt_variants,
)
from evaluation.metrics import consistency_analysis, latency_stats
from evaluation.visualize import generate_all_figures


MODE_CONFIG = {
    "quick": {
        "n_consistency": 3,
        "temp_values": [0.1, 0.6, 1.0],
        "top_p_values": [0.7, 0.95],
        "token_values": [256, 512],
        "temp_repeats": 1,
        "do_robustness": False,
        "do_prompt_variants": False,
        "prompt_repeats": 1,
    },
    "standard": {
        "n_consistency": 5,
        "temp_values": [0.1, 0.3, 0.6, 0.8, 1.0],
        "top_p_values": [0.5, 0.7, 0.85, 0.95],
        "token_values": [128, 256, 512, 768, 1024],
        "temp_repeats": 2,
        "do_robustness": True,
        "do_prompt_variants": True,
        "prompt_repeats": 2,
    },
    "full": {
        "n_consistency": 10,
        "temp_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "top_p_values": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        "token_values": [128, 256, 384, 512, 768, 1024],
        "temp_repeats": 3,
        "do_robustness": True,
        "do_prompt_variants": True,
        "prompt_repeats": 3,
    },
}

RESULTS_DIR = "evaluation/results"


def _print_header(title: str):
    w = 60
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def _print_summary(all_results: dict):
    """打印可用于论文的汇总数据。"""
    _print_header("EVALUATION SUMMARY")

    # RQ1 + RQ2: Consistency results
    cr = all_results.get("consistency")
    if cr:
        n = len(cr)
        json_rates = []
        fallback_rates = []
        coherence_scores = []
        total_times = []

        for run in cr:
            sc = run.get("schema_compliance", {})
            fb = run.get("fallback_triggered", {})
            n_nodes = max(len(sc), 1)
            json_rates.append(sum(1 for s in sc.values() if s.get("json_parseable", False)) / n_nodes)
            fallback_rates.append(sum(1 for f in fb.values() if f) / max(len(fb), 1))
            coherence_scores.append(run.get("coherence", {}).get("overall_coherence", 0))
            total_times.append(run.get("total_time", 0))

        print(f"\n📋 RQ1 & RQ2: Consistency Test ({n} runs)")
        print(f"   JSON Compliance:      {sum(json_rates)/n*100:.1f}% (mean)")
        print(f"   Fallback Rate:        {sum(fallback_rates)/n*100:.1f}% (mean)")
        print(f"   Pipeline Coherence:   {sum(coherence_scores)/n:.3f} (mean)")
        print(f"   Total Time:           {sum(total_times)/n:.1f}s ± {max(total_times)-min(total_times):.1f}s")

        # 跨运行一致性
        visual_outputs = [r.get("node_outputs", {}).get("visual_agent", {}) for r in cr]
        ca = consistency_analysis(visual_outputs, ["theme", "map_title", "overall_summary"])
        print(f"\n   Visual Agent Consistency:")
        for field, stats in ca.items():
            if "mean_jaccard" in stats:
                print(f"     {field}: Jaccard={stats['mean_jaccard']:.3f} ± {stats['std_jaccard']:.3f}")

    # RQ3: Parameter sensitivity
    for param_name in ["temperature", "top_p", "max_tokens"]:
        key = f"{param_name}_sweep"
        data = all_results.get(key)
        if data:
            print(f"\n📋 RQ3: {param_name} Sensitivity ({len(data)} runs)")
            from collections import defaultdict
            grouped = defaultdict(list)
            for r in data:
                grouped[r["sweep_value"]].append(r)
            for v in sorted(grouped.keys()):
                runs = grouped[v]
                times = [r.get("total_time", 0) for r in runs]
                sc = []
                for r in runs:
                    s = r.get("schema_compliance", {})
                    sc.append(sum(1 for x in s.values() if x.get("json_parseable", False)) / max(len(s), 1))
                print(f"     {param_name}={v}: compliance={sum(sc)/len(sc)*100:.0f}%, time={sum(times)/len(times):.1f}s")

    # RQ4: Robustness
    rob = all_results.get("robustness")
    if rob:
        print(f"\n📋 RQ4: Image Robustness ({len(rob)} perturbations)")
        for r in rob:
            sc = r.get("schema_compliance", {})
            n_ok = sum(1 for s in sc.values() if s.get("json_parseable", False))
            coh = r.get("coherence", {}).get("overall_coherence", 0)
            print(f"     {r['perturbation']:20s}: compliance={n_ok}/{len(sc)}, coherence={coh:.3f}")

    print(f"\n{'═' * 60}")
    print(f"  Results saved in: {RESULTS_DIR}/")
    print(f"  Figures saved in: evaluation/figures/")
    print(f"{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="SpatioThematicVLM Evaluation Framework")
    parser.add_argument("--image", type=str, default="test.png", help="测试图片路径")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "standard", "full"])
    parser.add_argument("--figures-only", action="store_true", help="仅从已有结果生成图表")
    parser.add_argument("--skip-sensitivity", action="store_true", help="跳过参数敏感性实验")
    parser.add_argument("--skip-robustness", action="store_true", help="跳过图片鲁棒性实验")
    args = parser.parse_args()

    if args.figures_only:
        print("📊 Generating figures from saved results...")
        generate_all_figures(RESULTS_DIR)
        return

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    cfg = MODE_CONFIG[args.mode]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    total_start = time.time()

    # ── Experiment config ──
    exp_config = {
        "image": args.image,
        "mode": args.mode,
        "config": cfg,
        "start_time": datetime.now().isoformat(),
    }
    save_results(exp_config, os.path.join(RESULTS_DIR, "experiment_config.json"))

    # ═══════════════════════════════════════════════════
    #  Experiment 1: Consistency Test (RQ1 + RQ2 + RQ5)
    # ═══════════════════════════════════════════════════
    _print_header(f"Experiment 1: Consistency Test (n={cfg['n_consistency']})")
    consistency = run_consistency_test(args.image, n_runs=cfg["n_consistency"])
    all_results["consistency"] = consistency
    save_results(consistency, os.path.join(RESULTS_DIR, "consistency_results.json"))

    # ═══════════════════════════════════════════════════
    #  Experiment 2: Parameter Sensitivity (RQ3)
    # ═══════════════════════════════════════════════════
    if not args.skip_sensitivity:
        _print_header("Experiment 2a: Temperature Sweep")
        temp_res = sweep_temperature(args.image, cfg["temp_values"], cfg["temp_repeats"])
        all_results["temperature_sweep"] = temp_res
        save_results(temp_res, os.path.join(RESULTS_DIR, "temperature_sweep.json"))

        _print_header("Experiment 2b: top_p Sweep")
        tp_res = sweep_top_p(args.image, cfg["top_p_values"], cfg["temp_repeats"])
        all_results["top_p_sweep"] = tp_res
        save_results(tp_res, os.path.join(RESULTS_DIR, "top_p_sweep.json"))

        _print_header("Experiment 2c: max_new_tokens Sweep")
        mt_res = sweep_max_tokens(args.image, cfg["token_values"], cfg["temp_repeats"])
        all_results["max_tokens_sweep"] = mt_res
        save_results(mt_res, os.path.join(RESULTS_DIR, "max_tokens_sweep.json"))

    # ═══════════════════════════════════════════════════
    #  Experiment 3: Image Robustness (RQ4)
    # ═══════════════════════════════════════════════════
    if cfg["do_robustness"] and not args.skip_robustness:
        _print_header("Experiment 3: Image Perturbation Robustness")
        rob_res = sweep_image_perturbations(args.image)
        all_results["robustness"] = rob_res
        save_results(rob_res, os.path.join(RESULTS_DIR, "robustness_results.json"))

    # ═══════════════════════════════════════════════════
    #  Experiment 4: Prompt Variants (RQ3 补充)
    # ═══════════════════════════════════════════════════
    if cfg["do_prompt_variants"]:
        _print_header("Experiment 4: Prompt Variant Sensitivity")
        pr_res = sweep_prompt_variants(args.image, n_repeats=cfg["prompt_repeats"])
        all_results["prompt_variants"] = pr_res
        save_results(pr_res, os.path.join(RESULTS_DIR, "prompt_variant_results.json"))

    # ═══════════════════════════════════════════════════
    #  Generate Figures
    # ═══════════════════════════════════════════════════
    _print_header("Generating Figures")
    generate_all_figures(RESULTS_DIR)

    # ═══════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════
    total_elapsed = time.time() - total_start
    exp_config["end_time"] = datetime.now().isoformat()
    exp_config["total_elapsed_s"] = round(total_elapsed, 1)
    save_results(exp_config, os.path.join(RESULTS_DIR, "experiment_config.json"))

    _print_summary(all_results)
    print(f"⏱️  Total evaluation time: {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
