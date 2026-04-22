from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime
from dotenv import load_dotenv
import os
import time

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import requests

from app.nodes import AgentState, supervisor_node, gis_tool_agent_node, visual_agent_node, semantic_agent_node, anomaly_agent_node, report_agent_node, gis_preprocessor_node

# build app
load_dotenv()
app = FastAPI(title="SpatioThematicVLM Enterprise", version="0.1.0")

# define workflow
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("gis_preprocessor", gis_preprocessor_node)
workflow.add_node("visual_agent", visual_agent_node)
workflow.add_node("semantic_agent", semantic_agent_node)
workflow.add_node("anomaly_agent", anomaly_agent_node)
workflow.add_node("report_agent", report_agent_node)
workflow.add_node("gis_tool_agent", gis_tool_agent_node)
workflow.add_edge(START, "supervisor")

# True conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next"], 
    {
        "gis_preprocessor": "gis_preprocessor",
        "visual_agent": "visual_agent",
        "semantic_agent": "semantic_agent",
        "anomaly_agent": "anomaly_agent",
        "report_agent": "report_agent",
        "gis_tool_agent": "gis_tool_agent",
        "END": END,
    }
)

for node in ["gis_preprocessor", "visual_agent", "semantic_agent", "anomaly_agent", "report_agent","gis_tool_agent"]:
    workflow.add_edge(node, "supervisor")

graph = workflow.compile(checkpointer=MemorySaver())

@app.post("/analyze")
async def analyze_map(
    image_file: UploadFile = File(..., description="必填:已做好的专题地图图片 (PNG/JPG)"),
    shp_zip: UploadFile = File(None, description="可选:shp 的 zip 压缩包")
):
    ext = image_file.filename.lower()
    if not ext.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(400, detail="image_file must be PNG or JPG")

    os.makedirs("uploads", exist_ok=True)
    image_path = f"uploads/{image_file.filename}"
    with open(image_path, "wb") as f:
        f.write(await image_file.read())

    shp_path = None
    if shp_zip:
        shp_path = f"uploads/{shp_zip.filename}"
        with open(shp_path, "wb") as f:
            f.write(await shp_zip.read())

    print(f"✅ Received image: {image_file.filename} | shp_zip: {'Yes' if shp_zip else 'No'}")

    total_start_time = time.time()
    thread_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

    initial_state = {
        "map_image": image_path,
        "shp_zip_path": shp_path,
        "gis_metadata": None,
        "tool_results": None,
        "visual_features": None,
        "semantic_themes": None,
        "anomalies": None,
        "report_markdown": None
    }

    final_state = await graph.ainvoke(
        initial_state,
        {"configurable": {"thread_id": thread_id}}
    )

    total_duration = time.time() - total_start_time

    return {
        "status": "success",
        "report_markdown": final_state["report_markdown"],
        "total_time_seconds": round(total_duration, 2),
        "analysis_steps": final_state.get("analysis_steps", []),
        "thread_id": thread_id,
        "filename": image_file.filename,
        "gis_metadata": final_state.get("gis_metadata", {}),
        "message": "report generated successfully"
    }

# start app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)