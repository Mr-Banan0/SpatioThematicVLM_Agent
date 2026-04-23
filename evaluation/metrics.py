"""
evaluation/metrics.py
自动化评测指标 —— 全部基于输出结构/文本分析，不依赖 Ground Truth。
"""

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Type

import numpy as np
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════
#  RQ1: Schema Compliance — 结构化输出有效性
# ═══════════════════════════════════════════════════════════

def check_json_parseable(raw_text: str) -> tuple[bool, Optional[dict]]:
    """检查 VLM 原始输出是否为合法 JSON。"""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    try:
        data = json.loads(cleaned)
        return True, data
    except (json.JSONDecodeError, ValueError):
        return False, None


def check_pydantic_valid(raw_text: str, model_class: Type[BaseModel]) -> tuple[bool, Optional[dict]]:
    """检查 VLM 原始输出是否通过 Pydantic 模型验证。"""
    parseable, data = check_json_parseable(raw_text)
    if not parseable or data is None:
        return False, None
    try:
        obj = model_class.model_validate(data)
        return True, obj.model_dump()
    except Exception:
        return False, None


def detect_fallback(parsed_dict: dict) -> bool:
    """检测解析结果是否触发了 fallback（包含失败标记）。"""
    fallback_markers = ["解析失败", "模型输出不符合Schema", "analysis failed"]
    text = json.dumps(parsed_dict, ensure_ascii=False)
    return any(marker in text for marker in fallback_markers)


def schema_compliance_summary(raw_outputs: List[str], model_class: Type[BaseModel]) -> dict:
    """对一组 VLM 原始输出计算 schema 合规率。"""
    n = len(raw_outputs)
    json_ok = sum(1 for r in raw_outputs if check_json_parseable(r)[0])
    pydantic_ok = sum(1 for r in raw_outputs if check_pydantic_valid(r, model_class)[0])
    return {
        "total": n,
        "json_parseable": json_ok,
        "json_rate": json_ok / n if n else 0,
        "pydantic_valid": pydantic_ok,
        "pydantic_rate": pydantic_ok / n if n else 0,
    }


# ═══════════════════════════════════════════════════════════
#  RQ1 补充: Field Completeness — 字段完整度
# ═══════════════════════════════════════════════════════════

def field_completeness(data_dict: dict, model_class: Type[BaseModel]) -> dict:
    """衡量已填充字段占总字段的比例。"""
    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    total = len(properties)
    filled = 0
    details = {}

    for name in properties:
        value = data_dict.get(name)
        is_filled = value is not None and value != "" and value != [] and value != {}
        details[name] = {"filled": is_filled, "required": name in required}
        if is_filled:
            filled += 1

    return {
        "total_fields": total,
        "filled_fields": filled,
        "completeness_rate": filled / total if total else 0,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════
#  RQ1 补充: Output Richness — 输出丰富度
# ═══════════════════════════════════════════════════════════

def output_richness(data_dict: dict) -> dict:
    """衡量输出内容的丰富程度（文本长度、列表元素数等）。"""
    total_text_len = 0
    total_list_items = 0
    field_stats = {}

    for key, value in data_dict.items():
        if isinstance(value, str):
            field_stats[f"{key}_len"] = len(value)
            total_text_len += len(value)
        elif isinstance(value, list):
            field_stats[f"{key}_count"] = len(value)
            total_list_items += len(value)
        elif isinstance(value, dict):
            field_stats[f"{key}_keys"] = len(value)

    field_stats["total_text_length"] = total_text_len
    field_stats["total_list_items"] = total_list_items
    return field_stats


# ═══════════════════════════════════════════════════════════
#  RQ2: Cross-run Consistency — 跨运行一致性
# ═══════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """中英文混合分词：中文按字，英文按词。"""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    """两段文本的 Jaccard 相似度。"""
    s1, s2 = set(_tokenize(text1)), set(_tokenize(text2))
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _lcs_length(a: list, b: list) -> int:
    """最长公共子序列长度。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l_f1(text1: str, text2: str) -> float:
    """ROUGE-L F1 分数。"""
    t1, t2 = _tokenize(text1), _tokenize(text2)
    if not t1 or not t2:
        return 0.0
    lcs = _lcs_length(t1, t2)
    p, r = lcs / len(t1), lcs / len(t2)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def consistency_analysis(runs: List[dict], key_fields: List[str]) -> dict:
    """对多次运行结果的指定字段做一致性分析。"""
    result = {}
    for field in key_fields:
        values = [r.get(field, "") for r in runs]
        if not values:
            continue

        if all(isinstance(v, str) for v in values):
            sims = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    sims.append(jaccard_similarity(values[i], values[j]))
            result[field] = {
                "type": "text",
                "mean_jaccard": float(np.mean(sims)) if sims else 0,
                "std_jaccard": float(np.std(sims)) if sims else 0,
                "unique_values": len(set(values)),
                "n_runs": len(values),
            }

        elif all(isinstance(v, (int, float)) for v in values):
            result[field] = {
                "type": "numeric",
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "cv": float(np.std(values) / abs(np.mean(values))) if np.mean(values) != 0 else 0,
            }

        elif all(isinstance(v, list) for v in values):
            counts = [len(v) for v in values]
            result[field] = {
                "type": "list",
                "mean_count": float(np.mean(counts)),
                "std_count": float(np.std(counts)),
                "range": [int(np.min(counts)), int(np.max(counts))],
            }

    return result


# ═══════════════════════════════════════════════════════════
#  RQ5: Inter-Agent Coherence — 流水线一致性
# ═══════════════════════════════════════════════════════════

def inter_agent_coherence(
    visual: dict,
    semantic: dict,
    anomalies: list,
    report: str,
) -> dict:
    """衡量多智能体输出之间的内在一致性。"""
    scores: Dict[str, float] = {}

    # Visual.theme ↔ Semantic.main_theme
    v_theme = visual.get("theme", "")
    s_theme = semantic.get("main_theme", "")
    scores["visual_semantic_theme"] = jaccard_similarity(v_theme, s_theme)

    # Visual.overall_summary → Report 引用
    v_summary = visual.get("overall_summary", "")
    scores["visual_in_report"] = rouge_l_f1(v_summary, report) if report else 0

    # Semantic.key_insights → Report 引用
    insights = semantic.get("key_insights", [])
    if insights and report:
        ins_scores = [rouge_l_f1(ins, report) for ins in insights]
        scores["insights_in_report_mean"] = float(np.mean(ins_scores))
    else:
        scores["insights_in_report_mean"] = 0.0

    # Anomalies → Report 引用
    if anomalies and report:
        a_scores = [rouge_l_f1(a.get("description", ""), report) for a in anomalies]
        scores["anomalies_in_report_mean"] = float(np.mean(a_scores))
    else:
        scores["anomalies_in_report_mean"] = 1.0 if not anomalies else 0.0

    # Map title → Report 引用
    title = visual.get("map_title", "")
    if title and report:
        scores["title_in_report"] = 1.0 if title.lower() in report.lower() else jaccard_similarity(title, report)
    else:
        scores["title_in_report"] = 0.0

    weights = {
        "visual_semantic_theme": 0.25,
        "visual_in_report": 0.15,
        "insights_in_report_mean": 0.20,
        "anomalies_in_report_mean": 0.15,
        "title_in_report": 0.25,
    }
    scores["overall_coherence"] = sum(scores.get(k, 0) * w for k, w in weights.items())
    return scores


# ═══════════════════════════════════════════════════════════
#  辅助: Latency Statistics
# ═══════════════════════════════════════════════════════════

def latency_stats(node_times: Dict[str, List[float]]) -> dict:
    """按节点计算延迟统计量。"""
    stats = {}
    for node, times in node_times.items():
        if not times:
            continue
        arr = np.array(times)
        stats[node] = {
            "mean_s": round(float(arr.mean()), 2),
            "std_s": round(float(arr.std()), 2),
            "min_s": round(float(arr.min()), 2),
            "max_s": round(float(arr.max()), 2),
            "median_s": round(float(np.median(arr)), 2),
        }
    return stats
