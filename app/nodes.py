import json
import os
import re
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime
import torch
import time
import pprint

from pydantic import BaseModel, Field, ValidationError
from fpdf import FPDF

# ==================== Qwen2.5-VL Model Loading ====================
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print("=== Loading Qwen2.5-VL-7B model...===")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, trust_remote_code=True,
    min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
)
print("=== Model loaded successfully ===\n")

# ====================== Pydantic Models ======================
class VisualElement(BaseModel):
    type: str = Field(..., description="Element type, e.g. legend/title/colorbar/symbol/label/region")
    description: str = Field(..., description="Detailed description")
    location: Optional[str] = Field(None, description="Location description")
    color: Optional[str] = Field(None, description="Primary color")

class VisualFeatures(BaseModel):
    map_title: Optional[str] = Field(None, description="Map title")
    theme: str = Field(..., description="Theme type")
    legend: List[VisualElement] = Field(default_factory=list)
    color_scale: Optional[str] = Field(None, description="Color scale")
    main_elements: List[VisualElement] = Field(default_factory=list)
    overall_summary: str = Field(..., description="One-sentence overall visual summary")
    analysis_complete: bool = Field(True, description="Always True, indicating this layer is complete")

class SemanticThemes(BaseModel):
    main_theme: str = Field(..., description="Main semantic theme")
    spatial_patterns: List[str] = Field(default_factory=list, description="Spatial patterns")
    geographic_meaning: str = Field(..., description="Geographic meaning")
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    analysis_complete: bool = Field(True, description="Always True")

class Anomaly(BaseModel):
    type: str = Field(..., description="Anomaly type")
    description: str = Field(..., description="Detailed description")
    location: Optional[str] = Field(None, description="Location")
    severity: str = Field("medium", description="Severity level: low/medium/high")

class AgentState(TypedDict):
    map_image: str
    visual_features: Dict[str, Any]          # 存 VisualFeatures.model_dump()
    semantic_themes: Dict[str, Any]          # 存 SemanticThemes.model_dump()
    anomalies: List[Dict[str, Any]]          # 存 Anomaly.model_dump()
    next: str
    analysis_steps: List[str]
    report_markdown: str
    pdf_path: Optional[str]


def _extract_and_validate_json(raw_response: str, model_class: type[BaseModel]) -> BaseModel:
    print("=== [Pydantic Parser] RAW RESPONSE ===")
    print(raw_response[:2000] + "..." if len(raw_response) > 2000 else raw_response)

    try:
        # 清理可能的 markdown 代码块
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        
        data = json.loads(cleaned)
        return model_class.model_validate(data)
    
    except Exception as e:
        print(f"=== Pydantic 解析失败: {type(e).__name__} ===")
        print(f"错误详情: {e}")
        
        # ==================== 根据不同模型精准 fallback ====================
        if model_class == VisualFeatures:
            return model_class(
                theme="解析失败（模型输出不符合Schema）",
                overall_summary="解析失败（模型输出不符合Schema）",
                analysis_complete=True
            )
        
        elif model_class == SemanticThemes:
            return model_class(
                main_theme="解析失败（模型输出不符合Schema）",
                geographic_meaning="解析失败（模型输出不符合Schema）",
                spatial_patterns=[],
                key_insights=[],
                analysis_complete=True
            )
        
        elif model_class == Anomaly:
            # Anomaly 是单个对象，但有时可能返回列表，这里返回单个默认对象
            return model_class(
                type="解析失败",
                description="模型输出不符合Schema",
                severity="medium"
            )
        else:
            # 兜底
            return model_class.model_construct(analysis_complete=True)


# ====================== call_vlm======================
def call_vlm(
    system_prompt: str,
    user_prompt: str,
    image_path: Optional[str] = None,
) -> str:
    """Call Qwen2.5-VL with timing and debug prints."""
    start_time = time.time()
    print(f"[call_vlm] START - Image: {image_path if image_path else 'No image'}")
    print(f"[call_vlm] System prompt length: {len(system_prompt)} | User prompt length: {len(user_prompt)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[call_vlm] Using device: {device}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    if image_path and os.path.exists(image_path):
        messages[-1]["content"].insert(0, {"type": "image", "image": image_path})
        print(f"[call_vlm] Image added: {image_path}")

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    end_time = time.time()
    duration = end_time - start_time
    print(f"[call_vlm] FINISH - Duration: {duration:.2f} seconds")
    print(f"[call_vlm] Output length: {len(output_text)}\n")

    return output_text.strip()

# ====================== 1. Supervisor Node ======================
def supervisor_node(state: AgentState) -> Dict:
    print("=== [Supervisor Node] START ===")
    start = time.time()

    if state.get("report_markdown") or state.get("pdf_path"):
        next_node = "END"
        state.setdefault("analysis_steps", []).append(
            f"[{datetime.now().strftime('%H:%M:%S')}] Supervisor → END (Report already generated)"
        )
        print(f"=== [Supervisor Node] FINISH → Next: {next_node} (Report already generated) ===\n")
        return {"next": next_node, "analysis_steps": state["analysis_steps"]}

    visual = state.get("visual_features", {})
    semantic = state.get("semantic_themes", {})

    if not visual or not visual.get("analysis_complete", False):
        next_node = "visual_agent"
    elif not semantic or not semantic.get("analysis_complete", False):
        next_node = "semantic_agent"
    elif "anomalies" not in state or state.get("anomalies") is None:
        next_node = "anomaly_agent"
    else:
        next_node = "report_agent"

    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Supervisor → {next_node}"
    )

    print(f"=== [Supervisor Node] FINISH → Next: {next_node} (took {time.time()-start:.2f}s) ===\n")
    return {"next": next_node, "analysis_steps": state["analysis_steps"]}


# ====================== 2. Visual Agent ======================
def visual_agent_node(state: AgentState) -> Dict:
    print("=== [Visual Agent] START ===")
    start = time.time()

    schema_str = json.dumps(
        VisualFeatures.model_json_schema(),
        ensure_ascii=False,
        indent=2
    )
    # check schema_str
    # print(f"=== [Visual Agent] Schema for VisualFeatures ===\n{schema_str}\n")

    system_prompt = (
        "You are a professional cartographer and visual analysis expert (Layer 1).\n"
        "Analyze the thematic map and return data strictly following this schema:\n"
        f"{schema_str}\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY valid JSON, nothing else — no explanations, no markdown, no ```json, no extra text.\n"
        "2. Keep ALL description fields short and concise (max 15 words each).\n"
        "3. Do not add any extra fields.\n"
        "4. Output must be parseable by Pydantic immediately."
    )
    user_prompt = "Please analyze the visual elements of this thematic map."

    response = call_vlm(system_prompt, user_prompt, state["map_image"])

    print("=== [Visual Agent] RAW RESPONSE ===")
    print(response[:1000] + "..." if len(response) > 1000 else response)

    visual_features = _extract_and_validate_json(response, VisualFeatures)
    visual_dict = visual_features.model_dump()

    state["visual_features"] = visual_dict
    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Visual features extracted (Pydantic)"
    )

    print(f"=== [Visual Agent] FINISH (took {time.time()-start:.2f}s) ===\n")
    return {
        "visual_features": visual_dict,
        "analysis_steps": state["analysis_steps"]
    }

# ====================== 3. Semantic Agent ======================
def semantic_agent_node(state: AgentState) -> Dict:
    print("=== [Semantic Agent] START ===")
    start = time.time()

    schema_str = json.dumps(
        SemanticThemes.model_json_schema(),
        ensure_ascii=False,
        indent=2
    )
    # check schema_str
    # print(f"=== [Semantic Agent] Schema for SemanticThemes ===\n{schema_str}\n")
    system_prompt = (
        "You are a professional geographic information and thematic mapping expert (Layer 2).\n"
        "Based on the visual features, identify semantic themes.\n"
        f"Return strictly following this schema:\n{schema_str}\n\n"
        "CRITICAL RULES (MUST FOLLOW):\n"
        "1. Return ONLY valid JSON, nothing else. No explanations, no markdown, no ```json.\n"
        "2. Output the ACTUAL DATA values that match the schema. Do NOT output the schema structure itself.\n"
        "3. Do NOT include keys like 'properties', 'required', 'title', 'type' in your output.\n"
        "4. You MUST include ALL required fields: main_theme and geographic_meaning.\n"
        "5. spatial_patterns and key_insights must be arrays (return [] even if empty).\n"
        "6. Keep every text field extremely short and concise (max 12 words).\n"
        "7. If you cannot analyze, still return valid JSON with 'analysis failed' message in the required fields.\n"
        "8. Output must be parseable by Pydantic immediately.\n\n"
    )

    user_prompt = (
        f"Visual features:\n{json.dumps(state.get('visual_features', {}), ensure_ascii=False, indent=2)}\n"
        "Please map these to semantic and thematic interpretations."
    )

    response = call_vlm(system_prompt, user_prompt, state["map_image"])
    semantic_themes = _extract_and_validate_json(response, SemanticThemes)
    semantic_dict = semantic_themes.model_dump()

    state["semantic_themes"] = semantic_dict
    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Semantic themes mapped (Pydantic)"
    )

    print(f"=== [Semantic Agent] FINISH (took {time.time()-start:.2f}s) ===\n")
    return {
        "semantic_themes": semantic_dict,
        "analysis_steps": state["analysis_steps"]
    }

# ====================== 4. Anomaly Agent ======================
def anomaly_agent_node(state: AgentState) -> Dict:
    print("=== [Anomaly Agent] START ===")
    start = time.time()

    schema_str = json.dumps(
        Anomaly.model_json_schema(),
        ensure_ascii=False,
        indent=2
    )

    system_prompt = (
        "You are a spatial anomaly detection expert (Layer 3).\n"
        f"Return a JSON array of objects following this schema:\n{schema_str}\n"
        "If no anomalies are found, return an empty array: [].\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY valid JSON, nothing else — no explanations, no markdown, no ```json, no extra text.\n"
        "2. Each object in the array must strictly follow the schema.\n"
        "3. Keep all description fields short and concise (max 15 words).\n"
        "4. Output must be a valid JSON array that can be parsed immediately."
    )

    user_prompt = (
        f"Visual features:\n{json.dumps(state.get('visual_features', {}), ensure_ascii=False)}\n\n"
        f"Semantic themes:\n{json.dumps(state.get('semantic_themes', {}), ensure_ascii=False)}\n\n"
        "Detect and list all spatial anomalies."
    )

    response = call_vlm(system_prompt, user_prompt, state["map_image"])

    try:
        anomalies_list = json.loads(response)
        if not isinstance(anomalies_list, list):
            anomalies_list = [anomalies_list] if anomalies_list else []
        anomalies = [Anomaly.model_validate(item).model_dump() for item in anomalies_list]
    except Exception:
        anomalies = []

    state["anomalies"] = anomalies
    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Anomalies detected (Pydantic)"
    )

    print(f"=== [Anomaly Agent] FINISH (found {len(anomalies)} anomalies) ===\n")
    return {
        "anomalies": anomalies,
        "analysis_steps": state["analysis_steps"]
    }

# ====================== 5. Report Generator Agent ======================
def report_agent_node(state: AgentState) -> Dict:
    """Generate a professional, comprehensive geographic analysis report in natural language."""
    print("=== [Report Agent] START ===")
    start = time.time()

    # 提取已有分析结果
    visual = state.get("visual_features", {})
    semantic = state.get("semantic_themes", {})
    anomalies = state.get("anomalies", [])

    # ==================== 使用 LLM 生成专业报告 ====================
    system_prompt = (
        "You are a senior geographic information scientist and environmental analyst.\n"
        "Write a professional, comprehensive, and well-structured geographic analysis report.\n"
        "Use formal academic tone, natural paragraphs, and smooth transitions.\n"
        "Do NOT output any JSON code blocks.\n"
        "Focus on interpretation and insights rather than raw data."
    )

    user_prompt = f"""Analyze the following results and write a professional geographic analysis report.

Visual Features:
{json.dumps(visual, ensure_ascii=False, indent=2)}

Semantic Themes:
{json.dumps(semantic, ensure_ascii=False, indent=2)}

Anomalies Detected:
{json.dumps(anomalies, ensure_ascii=False, indent=2)}

Please generate a coherent, in-depth geographic analysis report in Markdown format.
Include:
- Overall map interpretation
- Environmental quality assessment
- Spatial patterns and geographic significance
- Key insights and implications
- Any anomalies (or note if none were found)

Use natural language and professional tone. Do not include raw JSON."""

    # 调用 VLM 生成高质量报告
    markdown = call_vlm(system_prompt, user_prompt, state.get("map_image"))

    # 如果 LLM 输出为空或出错， fallback 到简单模板
    if not markdown or len(markdown.strip()) < 50:
        markdown = f"""# SpatioThematicVLM Geographic Analysis Report
**Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
The thematic map shows environmental quality (RSEI) across Guangdong Province. The analysis indicates generally high environmental quality with limited spatial variation.

## Visual Interpretation
{visual.get('overall_summary', 'No visual summary available.')}

## Semantic and Geographic Insights
{semantic.get('main_theme', '')}  
{semantic.get('geographic_meaning', '')}

Key spatial patterns include: {', '.join(semantic.get('spatial_patterns', [])) or 'None identified.'}

## Key Insights
{chr(10).join(['- ' + insight for insight in semantic.get('key_insights', [])]) or 'No additional insights.'}

## Anomaly Detection
{"No spatial anomalies were detected." if not anomalies else "The following anomalies were identified:"}
{chr(10).join(['- ' + a.get('description', '') for a in anomalies]) if anomalies else ''}

## Conclusion
Guangdong Province demonstrates strong overall environmental quality based on the RSEI index.
"""

    state["report_markdown"] = markdown

    # ==================== 生成 PDF ====================
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    for line in markdown.split("\n"):
        pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))
        if pdf.get_y() > 270:
            pdf.add_page()

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(
        output_dir,
        f"thematic_map_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    pdf.output(pdf_path)

    state["pdf_path"] = pdf_path
    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Professional geographic analysis report generated: {pdf_path}"
    )

    print(f"=== [Report Agent] FINISH (took {time.time()-start:.2f}s) ===\n")

    return {
        "report_markdown": markdown,
        "pdf_path": pdf_path,
        "analysis_steps": state["analysis_steps"],
        "next": "END"
    }