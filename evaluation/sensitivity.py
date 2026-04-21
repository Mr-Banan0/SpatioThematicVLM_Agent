"""
evaluation/sensitivity.py
敏感性分析实验 —— 参数扫描、图片扰动、Prompt 变体。
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from evaluation.runner import run_pipeline_with_metrics


# ═══════════════════════════════════════════════════════════
#  RQ3: 生成参数敏感性
# ═══════════════════════════════════════════════════════════

def sweep_temperature(
    image_path: str,
    values: List[float] = None,
    n_repeats: int = 2,
) -> List[dict]:
    """温度参数扫描。"""
    if values is None:
        values = [0.1, 0.3, 0.6, 0.8, 1.0]
    results = []
    for temp in values:
        for rep in range(n_repeats):
            print(f"\n--- Temperature={temp}, repeat={rep+1}/{n_repeats} ---")
            _, m = run_pipeline_with_metrics(image_path, temperature=temp)
            m["sweep_param"] = "temperature"
            m["sweep_value"] = temp
            m["repeat"] = rep
            results.append(m)
    return results


def sweep_top_p(
    image_path: str,
    values: List[float] = None,
    n_repeats: int = 2,
) -> List[dict]:
    """top_p 参数扫描。"""
    if values is None:
        values = [0.5, 0.7, 0.85, 0.95]
    results = []
    for tp in values:
        for rep in range(n_repeats):
            print(f"\n--- top_p={tp}, repeat={rep+1}/{n_repeats} ---")
            _, m = run_pipeline_with_metrics(image_path, top_p=tp)
            m["sweep_param"] = "top_p"
            m["sweep_value"] = tp
            m["repeat"] = rep
            results.append(m)
    return results


def sweep_max_tokens(
    image_path: str,
    values: List[int] = None,
    n_repeats: int = 2,
) -> List[dict]:
    """max_new_tokens 参数扫描。"""
    if values is None:
        values = [128, 256, 512, 768, 1024]
    results = []
    for mt in values:
        for rep in range(n_repeats):
            print(f"\n--- max_new_tokens={mt}, repeat={rep+1}/{n_repeats} ---")
            _, m = run_pipeline_with_metrics(image_path, max_new_tokens=mt)
            m["sweep_param"] = "max_new_tokens"
            m["sweep_value"] = mt
            m["repeat"] = rep
            results.append(m)
    return results


# ═══════════════════════════════════════════════════════════
#  RQ4: 输入鲁棒性 — 图片扰动
# ═══════════════════════════════════════════════════════════

def _apply_perturbation(image_path: str, name: str, output_dir: str) -> str:
    """对图片施加一种扰动并保存，返回新路径。"""
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_{name}.png")

    if name == "original":
        img.save(out_path)

    elif name == "resize_50":
        w, h = img.size
        img.resize((w // 2, h // 2), Image.LANCZOS).save(out_path)

    elif name == "resize_200":
        w, h = img.size
        img.resize((w * 2, h * 2), Image.LANCZOS).save(out_path)

    elif name == "jpeg_q20":
        img.save(out_path.replace(".png", ".jpg"), "JPEG", quality=20)
        out_path = out_path.replace(".png", ".jpg")

    elif name == "jpeg_q50":
        img.save(out_path.replace(".png", ".jpg"), "JPEG", quality=50)
        out_path = out_path.replace(".png", ".jpg")

    elif name == "blur":
        img.filter(ImageFilter.GaussianBlur(radius=3)).save(out_path)

    elif name == "noise":
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(out_path)

    elif name == "brightness_low":
        ImageEnhance.Brightness(img).enhance(0.5).save(out_path)

    elif name == "brightness_high":
        ImageEnhance.Brightness(img).enhance(1.5).save(out_path)

    elif name == "crop_center_80":
        w, h = img.size
        dw, dh = int(w * 0.1), int(h * 0.1)
        img.crop((dw, dh, w - dw, h - dh)).save(out_path)

    else:
        img.save(out_path)

    return out_path


def sweep_image_perturbations(
    image_path: str,
    perturbation_names: List[str] = None,
    output_dir: str = "evaluation/perturbed_images",
) -> List[dict]:
    """图片扰动鲁棒性测试。"""
    if perturbation_names is None:
        perturbation_names = [
            "original",
            "resize_50", "resize_200",
            "jpeg_q20", "jpeg_q50",
            "blur", "noise",
            "brightness_low", "brightness_high",
            "crop_center_80",
        ]

    results = []
    for pname in perturbation_names:
        print(f"\n--- Image perturbation: {pname} ---")
        perturbed_path = _apply_perturbation(image_path, pname, output_dir)
        _, m = run_pipeline_with_metrics(perturbed_path)
        m["perturbation"] = pname
        m["perturbed_image"] = perturbed_path
        results.append(m)
    return results


# ═══════════════════════════════════════════════════════════
#  RQ3 补充: Prompt 变体敏感性
# ═══════════════════════════════════════════════════════════

PROMPT_VARIANTS = {
    "baseline": {
        "description": "当前默认 prompt（含 CRITICAL RULES）",
        "modify": None,
    },
    "minimal": {
        "description": "仅保留 schema，去掉所有规则说明",
        "rules_section": "",
    },
    "few_shot": {
        "description": "在 prompt 中加入一个 JSON 输出示例",
        "extra_section": (
            '\n\nExample output:\n'
            '{"map_title":"Sample Map","theme":"land use",'
            '"legend":[],"color_scale":null,'
            '"main_elements":[],"overall_summary":"A land use map.",'
            '"analysis_complete":true}\n'
        ),
    },
    "chinese": {
        "description": "将系统 prompt 翻译为中文",
        "system_prompt_override": (
            "你是一名专业的地图学与视觉分析专家。\n"
            "请分析这张专题地图图像，提取所有视觉元素。\n"
            "按照以下 JSON Schema 返回结果：\n"
            "{schema}\n\n"
            "重要规则：\n"
            "1. 只返回合法 JSON，不要有任何额外文字或 markdown。\n"
            "2. 所有描述字段简洁（不超过15个词）。\n"
            "3. 不要添加 schema 中没有的字段。\n"
            "4. 输出必须能被 Pydantic 直接解析。"
        ),
    },
}


def sweep_prompt_variants(
    image_path: str,
    variant_names: List[str] = None,
    n_repeats: int = 2,
) -> List[dict]:
    """
    Prompt 变体敏感性测试（仅测试 Visual Agent 节点）。
    通过直接调用 call_vlm 实现，不经过完整 pipeline。
    """
    import app.nodes as nodes

    if variant_names is None:
        variant_names = list(PROMPT_VARIANTS.keys())

    schema_str = json.dumps(
        nodes.VisualFeatures.model_json_schema(), ensure_ascii=False, indent=2
    )

    base_system = (
        "You are a professional cartographer and visual analysis expert.\n"
        "Analyze the thematic map image and the provided GIS metadata...\n"
        f"{schema_str}\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY valid JSON, nothing else — no explanations, no markdown, no ```json, no extra text.\n"
        "2. Keep ALL description fields short and concise (max 15 words each).\n"
        "3. Do not add any extra fields.\n"
        "4. Output must be parseable by Pydantic immediately."
    )
    user_prompt = "Please analyze the visual elements of this thematic map."

    results = []
    for vname in variant_names:
        variant = PROMPT_VARIANTS.get(vname, {})
        for rep in range(n_repeats):
            print(f"\n--- Prompt variant: {vname}, repeat={rep+1}/{n_repeats} ---")

            if vname == "baseline":
                sys_p = base_system
            elif vname == "minimal":
                sys_p = f"Return a JSON object following this schema:\n{schema_str}"
            elif vname == "few_shot":
                sys_p = base_system + variant.get("extra_section", "")
            elif vname == "chinese":
                sys_p = variant["system_prompt_override"].replace("{schema}", schema_str)
            else:
                sys_p = base_system

            t0 = time.time()
            raw_output = nodes.call_vlm(sys_p, user_prompt, image_path)
            elapsed = time.time() - t0

            json_ok, parsed = nodes._extract_and_validate_json(raw_output, nodes.VisualFeatures), None
            parseable, _ = (True, None)
            try:
                cleaned = raw_output.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                parsed = json.loads(cleaned)
                parseable = True
            except Exception:
                parseable = False

            pydantic_ok = False
            if parsed is not None:
                try:
                    nodes.VisualFeatures.model_validate(parsed)
                    pydantic_ok = True
                except Exception:
                    pass

            results.append({
                "variant": vname,
                "variant_desc": variant.get("description", ""),
                "repeat": rep,
                "duration_s": round(elapsed, 2),
                "json_parseable": parseable,
                "pydantic_valid": pydantic_ok,
                "output_length": len(raw_output),
                "raw_output_preview": raw_output[:500],
            })

    return results
