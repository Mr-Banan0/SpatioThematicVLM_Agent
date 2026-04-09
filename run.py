from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from nodes import AgentState, supervisor_node, visual_agent_node, semantic_agent_node, anomaly_agent_node, report_agent_node
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("visual_agent", visual_agent_node)
workflow.add_node("semantic_agent", semantic_agent_node)
workflow.add_node("anomaly_agent", anomaly_agent_node)
workflow.add_node("report_agent", report_agent_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],
    {
        "visual_agent": "visual_agent",
        "semantic_agent": "semantic_agent",
        "anomaly_agent": "anomaly_agent",
        "report_agent": "report_agent",
        "END": END,
    }
)

for node in ["visual_agent", "semantic_agent", "anomaly_agent", "report_agent"]:
    workflow.add_edge(node, "supervisor")

graph = workflow.compile(checkpointer=MemorySaver())

# ==================== 一键运行 ====================
if __name__ == "__main__":
    image_path = "D:\\LLM\\GeoAnalyst\\SpatioThematicVLM\\test.png" 
    
    final_state = graph.invoke(
        {"map_image": image_path},
        {"configurable": {"thread_id": "test_run"}}
    )
    
    print("🎉 分析完成！")
    print("📄 PDF 报告路径：", final_state.get("pdf_path"))