import json
import os
import re
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime
import torch
import time
import pprint
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import requests

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Qwen2.5-VL Model Loading
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
_model = None
_processor = None

# model and processor Lazy Loading
def get_model_and_processor():
    global _model, _processor
    if _model is None:
        logger.info("=== First loading Qwen2.5-VL-7B-Instruct ===")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
            min_pixels=256*28*28, max_pixels=1280*28*28,
        )
        logger.info("=== model loaded successfully ===")
    return _model, _processor

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
    shp_zip_path: Optional[str]   
    gis_metadata: Dict[str, Any]     
    visual_features: Dict[str, Any]
    semantic_themes: Dict[str, Any]
    tool_results: Optional[Dict[str, Any]]
    geographic_insights: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    next: str
    analysis_steps: List[str]
    report_markdown: str
    pdf_path: Optional[str]

def _extract_and_validate_json(raw_response: str, model_class: type[BaseModel]) -> BaseModel:
    print("=== [Pydantic Parser] RAW RESPONSE ===")
    print(raw_response[:2000] + "..." if len(raw_response) > 2000 else raw_response)

    try:
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
    max_new_tokens: int = 512,   # tokens default 512
) -> str:
    """Call Qwen2.5-VL with timing and debug prints."""
    start_time = time.time()
    print(f"[call_vlm] START - Image: {image_path if image_path else 'No image'}")
    print(f"[call_vlm] System prompt length: {len(system_prompt)} | User prompt length: {len(user_prompt)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[call_vlm] Using device: {device}")

    model, processor = get_model_and_processor()
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
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        do_sample=True,
        top_p=0.85,
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

    visual = state.get("visual_features") or {}
    semantic = state.get("semantic_themes") or {}

    if state.get("shp_zip_path") and state.get("gis_metadata") is None:
        next_node = "gis_preprocessor"
    elif state.get("visual_features") is None or not visual.get("analysis_complete", False):
        next_node = "visual_agent"
    elif state.get("semantic_themes") is None or not semantic.get("analysis_complete", False):
        next_node = "semantic_agent"
    elif state.get("gis_metadata") is not None and state.get("tool_results") is None:  
        next_node = "gis_tool_agent"
    elif state.get("anomalies") is None:
        next_node = "anomaly_agent"
    elif state.get("report_markdown") is None or not state.get("report_markdown"):
        next_node = "report_agent"
    else:
        next_node = "END"

    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Supervisor → {next_node}"
    )

    print(f"=== [Supervisor Node] FINISH → Next: {next_node} (took {time.time()-start:.2f}s) ===\n")
    return {"next": next_node, "analysis_steps": state["analysis_steps"]}

def gis_preprocessor_node(state: AgentState) -> Dict:
    """提取 shp zip 的 GIS 元数据（不再生成 PNG）"""
    print("=== [GIS Preprocessor Node] START ===")
    start = time.time()

    raw_path = state.get("shp_zip_path")
    if not raw_path:
        raise Exception("No shp_zip_path found in state")

    print(f"正在提取元数据: {raw_path}")

    try:
        with open(raw_path, "rb") as file_obj:
            resp = requests.post(
                "http://localhost:8002/preprocess",
                files={"file": file_obj},
                timeout=60
            )
        
        print(f"预处理服务返回状态码: {resp.status_code}")

        if resp.status_code != 200:
            raise Exception(f"预处理服务错误: {resp.text}")

        data = resp.json()
        metadata = data.get("gis_metadata", {})

        # 更新 state
        state["gis_metadata"] = metadata

        state.setdefault("analysis_steps", []).append(
            f"[{datetime.now().strftime('%H:%M:%S')}] GIS Metadata extracted (vector, {metadata.get('feature_count', 0)} features)"
        )

        print(f"=== [GIS Preprocessor] FINISH (took {time.time()-start:.2f}s) ===\n")
        return {
            "gis_metadata": metadata,
            "analysis_steps": state["analysis_steps"]
        }

    except requests.exceptions.RequestException as e:
        # ArcPy 服务是可选组件；未启动时降级为仅图片分析，而不是让整个 /analyze 失败。
        warning_message = (
            "GIS Preprocessor service unavailable on localhost:8002; "
            "continuing without shapefile metadata. "
            "Start `python gis_preprocessor_service.py` in an ArcGIS Pro Python environment "
            "to enable GIS preprocessing."
        )
        print(f"GIS Preprocessor failed: {e}")
        state.setdefault("analysis_steps", []).append(
            f"[{datetime.now().strftime('%H:%M:%S')}] GIS preprocessing skipped: service unavailable"
        )
        return {
            "shp_zip_path": None,
            "gis_metadata": None,
            "analysis_steps": state["analysis_steps"],
            "tool_results": {
                "warning": warning_message,
                "detail": str(e)
            }
        }

    except Exception as e:
        print(f"GIS Preprocessor failed: {e}")
        raise

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
    shp_info = ""
    if state.get("gis_metadata"):
        shp_info = f"\nAdditional GIS Data from Shapefile:\n{json.dumps(state['gis_metadata'], ensure_ascii=False, indent=2)}"
    system_prompt = (
        "You are a professional cartographer and visual analysis expert.\n"
        "Analyze the thematic map image and the provided GIS metadata...\n"
        f"{schema_str}\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY valid JSON, nothing else — no explanations, no markdown, no ```json, no extra text.\n"
        "2. Keep ALL description fields short and concise (max 15 words each).\n"
        "3. Do not add any extra fields.\n"
        "4. Output must be parseable by Pydantic immediately."
    )
    user_prompt = f"Please analyze the visual elements of this thematic map.{shp_info}"

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
    shp_info = ""
    if state.get("gis_metadata"):
        shp_info = f"\nAdditional GIS Data from Shapefile:\n{json.dumps(state['gis_metadata'], ensure_ascii=False, indent=2)}"
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
        f"{shp_info}"
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

# ====================== 4. GIS Tool Agent ======================
def gis_tool_agent_node(state: AgentState) -> Dict:
    """GIS Tool Agent: 使用 LLM 智能判断字段，并调用 ArcPy 工具（含 Reverse Geocoding）"""
    print("=== [GIS Tool Agent] START ===")
    start = time.time()

    if not state.get("shp_zip_path") and not state.get("gis_metadata"):
        print("No vector data available, skipping GIS tools")
        tool_results = {"error": "No vector data provided"}
    else:
        feature_class = state.get("gis_metadata", {}).get("raw_path") or state.get("shp_zip_path")
        fields = state.get("gis_metadata", {}).get("fields", [])

        # LLM 智能选择字段
        visual = state.get("visual_features", {})
        semantic = state.get("semantic_themes", {})

        system_prompt = (
            "You are an expert GIS analyst.\n"
            "Given the map theme and list of fields, return ONLY the name of the SINGLE most suitable numeric field "
            "for spatial autocorrelation analysis (Moran's I).\n"
            "Do not return any explanation, only the field name."
        )

        user_prompt = f"""Map Information:
            - Title: {visual.get('map_title', 'Unknown')}
            - Theme: {visual.get('theme', 'Unknown')}
            - Semantic Theme: {semantic.get('main_theme', 'Unknown')}
            Available Fields: {fields}
            Return ONLY the most relevant field name for spatial analysis."""

        try:
            llm_response = call_vlm(system_prompt, user_prompt).strip()
            value_field = llm_response.strip().strip('"').strip("'")
            print(f"✅ LLM 推荐字段: {value_field}")
        except Exception as e:
            print(f"LLM field selection failed: {e}")
            value_field = next((f for f in fields if f.lower() not in ["fid", "shape", "id", "objectid", "oid"]), None)
            if not value_field and fields:
                value_field = fields[0]

        tool_results = {}

        # Global Moran's I
        try:
            resp_global = requests.post(
                "http://localhost:8002/gis/global_morans_i",
                json={"feature_class": feature_class, "value_field": value_field},
                timeout=60
            )
            if resp_global.status_code == 200:
                tool_results["global_morans_i"] = resp_global.json()
                print("[DEBUG] global_morans_i 返回内容：", json.dumps(resp_global.json(), ensure_ascii=False, indent=2))
            else:
                tool_results["global_morans_i"] = {"error": resp_global.text}
        except Exception as e:
            tool_results["global_morans_i"] = {"error": str(e)}

        # Local Moran's I
        try:
            resp_local = requests.post(
                "http://localhost:8002/gis/local_morans_i",
                json={"feature_class": feature_class, "value_field": value_field},
                timeout=60
            )
            if resp_local.status_code == 200:
                tool_results["local_morans_i"] = resp_local.json()
                print("[DEBUG] local_morans_i 返回内容：", json.dumps(resp_local.json(), ensure_ascii=False, indent=2))
            else:
                tool_results["local_morans_i"] = {"error": resp_local.text}
        except Exception as e:
            tool_results["local_morans_i"] = {"error": str(e)}

        # ==================== 新增：调用 Reverse Geocoding 服务 ====================
        geocoded_info = {"status": "failed", "locations": {}, "message": "未执行"}
        if tool_results.get("local_morans_i") and "output_feature_class" in tool_results["local_morans_i"]:
            try:
                output_fc = tool_results["local_morans_i"]["output_feature_class"]
                resp_geo = requests.post(
                    "http://localhost:8002/gis/reverse_geocode",
                    json={"output_feature_class": output_fc},
                    timeout=60
                )
                if resp_geo.status_code == 200:
                    geocoded_info = resp_geo.json()
                    print(f"[Reverse Geocoding] 服务调用成功 → {geocoded_info.get('locations', {})}")
                else:
                    geocoded_info["message"] = resp_geo.text
            except Exception as e:
                geocoded_info["message"] = str(e)
                print(f"[Reverse Geocoding] 服务调用失败: {e}")

        tool_results["geocoded_locations"] = geocoded_info

    # 统一更新 analysis_steps
    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] GIS Tool Agent completed (Global + Local Moran + Reverse Geocoding)"
    )

    print(f"=== [GIS Tool Agent] FINISH (took {time.time()-start:.2f}s) ===\n")
    return {
        "tool_results": tool_results,
        "analysis_steps": state["analysis_steps"]
    }

# ====================== 5. Anomaly Agent ======================
def anomaly_agent_node(state: AgentState) -> Dict:
    print("=== [Anomaly Agent] START ===")
    start = time.time()

    tool_results = state.get("tool_results", {})
    anomalies = []

    # Global Moran's I
    if tool_results.get("global_morans_i"):
        g = tool_results["global_morans_i"]
        z_score = g.get("z_score", 0)
        try:
            z_score = float(g.get("z_score", 0))
        except (ValueError, TypeError):
            z_score = 0.0
        if abs(z_score) > 1.96:
            anomalies.append({
                "type": "Global Clustering",
                "description": f"Significant global spatial autocorrelation detected (Moran's I Z-score = {z_score:.3f}, p-value ≈ {g.get('p_value', 0)})",
                "location": "Entire map",
                "severity": "high" if abs(z_score) > 2.58 else "medium"
            })

    # Local Moran's I + Geocoded locations（为 report 提供更丰富的信息）
    if tool_results.get("local_morans_i"):
        geocoded = tool_results.get("geocoded_locations", {}).get("locations", {})
        desc = "Local Moran's I identified statistically significant hot spots (HH) and cold spots (LL)"
        if geocoded.get("HH") or geocoded.get("LL"):
            desc += f". Notable clusters were detected near: {', '.join(geocoded.get('HH', [])[:2] + geocoded.get('LL', [])[:2])}"
        
        anomalies.append({
            "type": "Local Clusters",
            "description": desc,
            "location": "Multiple local clusters",
            "severity": "medium"
        })

    # Visual fallback（保持不变）
    if not anomalies and state.get("visual_features"):
        anomalies.append({
            "type": "Visual Anomaly",
            "description": "Potential visual anomalies detected in color distribution",
            "location": "Unknown",
            "severity": "low"
        })

    state["anomalies"] = anomalies
    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Anomaly Agent used GIS Tool results"
    )

    print(f"=== [Anomaly Agent] FINISH (found {len(anomalies)} anomalies) ===\n")
    return {
        "anomalies": anomalies,
        "analysis_steps": state["analysis_steps"]
    }

# ====================== 6. Report Generator Agent ======================
def report_agent_node(state: AgentState) -> Dict:
    """Generate a professional, comprehensive geographic analysis report in Markdown format."""
    print("=== [Report Agent] START ===")
    start = time.time()

    visual = state.get("visual_features", {})
    semantic = state.get("semantic_themes", {})
    anomalies = state.get("anomalies", [])
    tool_results = state.get("tool_results", {})
    geocoded = tool_results.get("geocoded_locations", {}).get("locations", {})

    shp_info = ""
    if state.get("gis_metadata"):
        shp_info = f"\nAdditional GIS Data from Shapefile:\n{json.dumps(state['gis_metadata'], ensure_ascii=False, indent=2)}"

    # ==================== System Prompt ====================
    system_prompt = (
        "You are a senior geographic information scientist and environmental acoustics expert. "
        "Write a highly professional, academically rigorous, and well-structured geographic analysis report in Markdown. "
        "Requirements:\n"
        "1. MUST output the COMPLETE report without any truncation.\n"
        "2. Use formal academic tone with precise geospatial terminology and smooth transitions.\n"
        "3. Strictly follow the exact section structure requested.\n"
        "4. In the Anomaly Analysis section, first briefly describe global clustering with Z-score, "
        "then focus on local clusters and seamlessly integrate the real geocoded location names into the analysis.\n"
        "5. Do NOT create separate subsections for Real Location References — weave them naturally into the Local Clusters discussion.\n"
        "6. Do NOT output JSON, code, or explanations — only the pure Markdown report."
    )

    # ==================== User Prompt ====================
    user_prompt = f"""Analyze the following analysis results and generate a COMPLETE, professional geographic analysis report.

Visual Features:
{json.dumps(visual, ensure_ascii=False, indent=2)}

Semantic Themes:
{json.dumps(semantic, ensure_ascii=False, indent=2)}

Anomalies (from Anomaly Agent):
{json.dumps(anomalies, ensure_ascii=False, indent=2)}

Geocoded Real Locations (请自然融入 Anomaly Analysis 的 Local Clusters 部分，进行详细空间解读):
{json.dumps(geocoded, ensure_ascii=False, indent=2)}

{shp_info}

Please generate a well-structured Markdown report with these exact sections:

- Main Title (using #)
- Executive Summary
- Visual Interpretation
- Spatial Patterns and Geographic Significance
- Key Insights and Implications
- Anomaly Analysis （先简要说明 Global Clustering 并量化 Z-score，然后重点分析 Local Clusters，并自然融入真实地名进行详细解读）
- Conclusion and Recommendations

Use formal academic language. Emphasize geographic significance, environmental implications, and policy relevance. Make the report concise yet in-depth."""

    # 调用 VLM
    markdown = call_vlm(system_prompt, user_prompt, state.get("map_image"), max_new_tokens=2048)

    state["report_markdown"] = markdown
    state.setdefault("analysis_steps", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Markdown report generated successfully (Anomaly Analysis optimized)"
    )

    print(f"=== [Report Agent] FINISH (took {time.time()-start:.2f}s) ===\n")
    return {
        "report_markdown": markdown,
        "pdf_path": None,
        "analysis_steps": state["analysis_steps"],
        "next": "END"
    }
