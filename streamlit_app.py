import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="SpatioThematicVLM GIS", layout="wide")

st.title("🗺️ SpatioThematicVLM Enterprise")
st.markdown("**必填：专题地图图片 + 可选：shp zip**")
st.caption("PNG/JPG（必填） + SHP/ZIP（可选）")

# Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "report" not in st.session_state:
    st.session_state.report = None
if "gis_metadata" not in st.session_state:
    st.session_state.gis_metadata = None

col1, col2 = st.columns([5, 7])

with col1:
    st.subheader("📤 1. 上传专题地图图片（必填）")
    image_file = st.file_uploader(
        "PNG / JPG / JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    st.subheader("📤 2. 上传 shp zip（可选）")
    shp_zip = st.file_uploader(
        "shp 的完整压缩包（.zip）",
        type=["zip"],
        label_visibility="collapsed"
    )

    if image_file:
        try:
            image = Image.open(image_file)
            st.image(image, caption="地图图片预览", use_container_width=True)
        except:
            st.warning("无法预览图片")

    if shp_zip:
        st.info(f"✅ 已选择 shp zip: {shp_zip.name} ({shp_zip.size/1024:.1f} KB)")

    if st.button("🚀 开始 GIS 分析", type="primary", use_container_width=True):
        if not image_file:
            st.error("请先上传专题地图图片（PNG/JPG）")
        else:
            with st.spinner("正在分析..."):
                files = {}
                # 必填：图片
                files["image_file"] = (image_file.name, image_file.getvalue(), image_file.type)
                
                # 可选：shp zip
                if shp_zip:
                    files["shp_zip"] = (shp_zip.name, shp_zip.getvalue(), shp_zip.type)

                response = requests.post(
                    "http://localhost:8000/analyze",
                    files=files,
                    timeout=180
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.thread_id = data["thread_id"]
                    st.session_state.report = data["report_markdown"]
                    st.session_state.gis_metadata = data.get("gis_metadata", {})
                    st.success(f"✅ 分析完成！Thread ID: {data['thread_id']}")
                else:
                    st.error(f"分析失败: {response.text}")

with col2:
    if st.session_state.report:
        st.subheader("📊 分析报告")
        st.markdown(st.session_state.report)

        if st.session_state.gis_metadata:
            st.subheader("📍 GIS 元数据（来自 shp）")
            st.json(st.session_state.gis_metadata)

    st.divider()

    st.subheader("💬 与 GIS Agent 对话")
    if not st.session_state.thread_id:
        st.info("请先完成左侧分析")
    else:
        st.caption(f"当前线程: `{st.session_state.thread_id}`")

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])

        if prompt := st.chat_input("例如：分析这个地图的空间自相关性..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.spinner("Agent 思考中..."):
                payload = {"message": prompt, "thread_id": st.session_state.thread_id}
                resp = requests.post("http://localhost:8000/chat", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.chat_history.append({"role": "assistant", "content": data["response"]})
                    st.chat_message("assistant").write(data["response"])

with st.sidebar:
    st.success("✅ ArcPy 预处理服务已运行")
    st.success("✅ 主后端已连接")
    if st.button("🔄 清空当前会话"):
        st.session_state.clear()
        st.rerun()