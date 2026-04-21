"""
evaluation/runner.py
流水线执行引擎 —— 逐节点运行 pipeline 并收集完整指标。
"""

import copy
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from evaluation.metrics import (
    check_json_parseable,
    detect_fallback,
    field_completeness,
    inter_agent_coherence,
    output_richness,
)


def _import_nodes():
    """延迟导入 app.nodes，避免在无 GPU 环境下的 import 崩溃。"""
    import app.nodes as nodes
    return nodes


def run_pipeline_with_metrics(
    image_path: str,
    shp_zip_path: Optional[str] = None,
    **vlm_kwargs,
) -> tuple[dict, dict]:
    """
    手动逐节点执行 pipeline，收集每一步的时间和输出。
    返回 (final_state, metrics_dict)。
    """
    nodes = _import_nodes()

    original_config = nodes.VLM_GENERATION_CONFIG.copy()
    if vlm_kwargs:
        nodes.VLM_GENERATION_CONFIG.update(vlm_kwargs)

    nodes.clear_vlm_call_log()

    state: Dict[str, Any] = {
        "map_image": image_path,
        "shp_zip_path": shp_zip_path,
        "gis_metadata": {},
        "visual_features": {},
        "semantic_themes": {},
        "anomalies": None,
        "analysis_steps": [],
        "report_markdown": "",
        "pdf_path": None,
        "next": "",
    }

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "image": image_path,
        "vlm_config": {**nodes.VLM_GENERATION_CONFIG},
        "node_times": {},
        "node_outputs": {},
        "schema_compliance": {},
        "field_completeness": {},
        "output_richness": {},
        "fallback_triggered": {},
    }

    node_sequence = [
        ("visual_agent", nodes.visual_agent_node),
        ("semantic_agent", nodes.semantic_agent_node),
        ("anomaly_agent", nodes.anomaly_agent_node),
        ("report_agent", nodes.report_agent_node),
    ]

    pydantic_classes = {
        "visual_agent": nodes.VisualFeatures,
        "semantic_agent": nodes.SemanticThemes,
    }

    state_keys = {
        "visual_agent": "visual_features",
        "semantic_agent": "semantic_themes",
        "anomaly_agent": "anomalies",
        "report_agent": "report_markdown",
    }

    try:
        for node_name, node_func in node_sequence:
            log_before = len(nodes.get_vlm_call_log())
            t0 = time.time()
            result = node_func(state)
            elapsed = time.time() - t0

            state.update(result)
            metrics["node_times"][node_name] = round(elapsed, 2)

            sk = state_keys[node_name]
            output_val = state.get(sk, {})
            metrics["node_outputs"][node_name] = output_val

            # 从 VLM call log 拿原始输出做 schema compliance 检查
            log_after = nodes.get_vlm_call_log()
            if len(log_after) > log_before:
                raw = log_after[-1]["output"]
                parseable, _ = check_json_parseable(raw)
                metrics["schema_compliance"][node_name] = {
                    "json_parseable": parseable,
                    "raw_output_length": len(raw),
                }

            if isinstance(output_val, dict):
                metrics["fallback_triggered"][node_name] = detect_fallback(output_val)
                metrics["output_richness"][node_name] = output_richness(output_val)
                if node_name in pydantic_classes:
                    metrics["field_completeness"][node_name] = field_completeness(
                        output_val, pydantic_classes[node_name]
                    )

    finally:
        nodes.VLM_GENERATION_CONFIG.update(original_config)

    # 流水线一致性
    metrics["coherence"] = inter_agent_coherence(
        state.get("visual_features", {}),
        state.get("semantic_themes", {}),
        state.get("anomalies", []),
        state.get("report_markdown", ""),
    )

    metrics["total_time"] = round(sum(metrics["node_times"].values()), 2)
    return state, metrics


def run_consistency_test(
    image_path: str,
    n_runs: int = 5,
    **vlm_kwargs,
) -> List[dict]:
    """同一输入运行 n 次，收集全部指标，用于一致性分析。"""
    all_metrics = []
    for i in range(n_runs):
        print(f"\n{'='*60}")
        print(f"  Consistency Run {i+1}/{n_runs}")
        print(f"{'='*60}")
        _, m = run_pipeline_with_metrics(image_path, **vlm_kwargs)
        m["run_index"] = i
        all_metrics.append(m)
    return all_metrics


def save_results(data: Any, path: str):
    """将评测结果保存为 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _default(obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)
    print(f"✅ Results saved → {path}")


def load_results(path: str) -> Any:
    """从 JSON 加载已有评测结果。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
