import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


st.set_page_config(
    page_title="SpatioThematicVLM GIS Demo",
    page_icon="🗺️",
    layout="wide",
)

BACKEND_BASE_URL = os.getenv("SPATIOTHEMATIC_BACKEND_URL", "http://localhost:8000").rstrip("/")
ANALYZE_ENDPOINT = f"{BACKEND_BASE_URL}/analyze"
CHAT_ENDPOINT = f"{BACKEND_BASE_URL}/chat"
ANALYZE_TIMEOUT_SECONDS = 180
CHAT_TIMEOUT_SECONDS = 120
APP_STATE_DEFAULTS = {
    "analysis_status": "idle",
    "chat_status": "idle",
    "thread_id": None,
    "report_markdown": None,
    "gis_metadata": {},
    "chat_messages": [],
    "uploaded_image_name": None,
    "uploaded_shp_name": None,
    "analysis_error_message": None,
    "chat_error_message": None,
    "last_analyze_status": "Not run yet",
    "last_chat_status": "Not run yet",
    "last_analyzed_signature": None,
    "uploader_version": 0,
}

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #edf2ef;
            --panel: rgba(248, 252, 249, 0.94);
            --panel-strong: #fdfefd;
            --border: rgba(52, 87, 73, 0.16);
            --text: #14201b;
            --muted: #5d7168;
            --accent: #1f6b5c;
            --accent-soft: rgba(31, 107, 92, 0.12);
            --accent-2: #87612b;
            --grid: rgba(31, 107, 92, 0.06);
            --shadow: 0 18px 40px rgba(29, 60, 50, 0.08);
        }

        .stApp {
            background:
                linear-gradient(var(--grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--grid) 1px, transparent 1px),
                radial-gradient(circle at top left, rgba(31, 107, 92, 0.10), transparent 26%),
                radial-gradient(circle at top right, rgba(135, 97, 43, 0.10), transparent 22%),
                linear-gradient(180deg, #f4f8f4 0%, var(--app-bg) 100%);
            background-size: 32px 32px, 32px 32px, auto, auto, auto;
            color: var(--text);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(139, 191, 166, 0.12), transparent 30%),
                linear-gradient(180deg, #102722 0%, #162f2a 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f5efe7;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(24, 82, 71, 0.96), rgba(18, 51, 60, 0.95)),
                linear-gradient(120deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 28px;
            padding: 1.75rem 1.85rem;
            box-shadow: 0 22px 50px rgba(11, 44, 57, 0.22);
            color: white;
            margin-bottom: 1.2rem;
            overflow: hidden;
            position: relative;
        }

        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(rgba(255,255,255,0.07) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.07) 1px, transparent 1px);
            background-size: 26px 26px;
            mask-image: linear-gradient(180deg, rgba(0,0,0,0.38), transparent 92%);
            pointer-events: none;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -40px -54px auto;
            width: 260px;
            height: 260px;
            border-radius: 999px;
            background:
                radial-gradient(circle, rgba(255,255,255,0.15), transparent 64%);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.74rem;
            opacity: 0.82;
            margin-bottom: 0.6rem;
            font-weight: 700;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            line-height: 1.05;
            color: white;
        }

        .hero p {
            margin: 0.8rem 0 0 0;
            max-width: 760px;
            font-size: 1.02rem;
            line-height: 1.7;
            color: rgba(255,255,255,0.90);
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 1.5rem 0;
        }

        .workflow-card,
        .metric-card,
        .info-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        .workflow-card {
            padding: 1rem 1rem 1.05rem 1rem;
            min-height: 132px;
            position: relative;
            overflow: hidden;
        }

        .workflow-card::after {
            content: "";
            position: absolute;
            inset: auto 14px 14px auto;
            width: 58px;
            height: 58px;
            border-radius: 999px;
            border: 1px dashed rgba(31, 107, 92, 0.18);
            opacity: 0.8;
        }

        .workflow-step {
            width: 2rem;
            height: 2rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 0.9rem;
        }

        .workflow-card h4,
        .metric-card h4,
        .info-card h4 {
            margin: 0;
            font-size: 1rem;
            color: var(--text);
        }

        .workflow-card p,
        .metric-card p,
        .info-card p {
            margin: 0.5rem 0 0 0;
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.95rem;
        }

        .section-shell {
            background: rgba(250, 253, 251, 0.78);
            border: 1px solid rgba(52, 87, 73, 0.12);
            border-radius: 26px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            margin: 1rem 0 1.25rem 0;
            box-shadow: 0 14px 36px rgba(29, 60, 50, 0.06);
            position: relative;
            overflow: hidden;
        }

        .section-shell::after {
            content: "";
            position: absolute;
            inset: auto -10px -18px auto;
            width: 140px;
            height: 140px;
            background:
                radial-gradient(circle, rgba(31, 107, 92, 0.08), transparent 67%);
            pointer-events: none;
        }

        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--accent-2);
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            line-height: 1.65;
            margin-bottom: 0.4rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.8rem 0 0.35rem 0;
        }

        .metric-card {
            padding: 1rem 1rem 0.95rem 1rem;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, #1f6b5c 0%, #6f8c51 100%);
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text);
            margin-top: 0.25rem;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 0.25rem;
        }

        .panel-copy {
            color: var(--muted);
            line-height: 1.6;
            margin-bottom: 0.85rem;
        }

        .info-card {
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 0.9rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 700;
            font-size: 0.85rem;
            margin-bottom: 0.55rem;
        }

        .report-shell {
            background: var(--panel-strong);
            border: 1px solid rgba(52, 87, 73, 0.12);
            border-radius: 24px;
            padding: 1.3rem 1.35rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }

        .report-shell::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(var(--grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--grid) 1px, transparent 1px);
            background-size: 24px 24px;
            mask-image: linear-gradient(180deg, rgba(0,0,0,0.18), transparent 100%);
            pointer-events: none;
        }

        .report-shell h1,
        .report-shell h2,
        .report-shell h3 {
            color: #17443d;
        }

        .report-shell p,
        .report-shell li {
            line-height: 1.8;
        }

        .chat-shell {
            background: rgba(250, 253, 251, 0.9);
            border: 1px solid rgba(52, 87, 73, 0.12);
            border-radius: 24px;
            padding: 1rem 1rem 0.5rem 1rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }

        .chat-shell::before {
            content: "";
            position: absolute;
            inset: auto auto -40px -40px;
            width: 140px;
            height: 140px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(31, 107, 92, 0.10), transparent 68%);
            pointer-events: none;
        }

        .helper-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.55rem 0 0.85rem 0;
        }

        .helper-chip {
            display: inline-flex;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(31, 107, 92, 0.08);
            border: 1px solid rgba(31, 107, 92, 0.12);
            color: #1d5a50;
            font-size: 0.88rem;
        }

        .map-frame {
            position: relative;
            background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(241, 247, 243, 0.95));
            border: 1px solid rgba(52, 87, 73, 0.14);
            border-radius: 24px;
            padding: 1rem;
            box-shadow: var(--shadow);
            overflow: hidden;
        }

        .map-frame::before {
            content: "";
            position: absolute;
            inset: 12px;
            border: 1px solid rgba(31, 107, 92, 0.12);
            border-radius: 18px;
            pointer-events: none;
        }

        .map-frame::after {
            content: "MAP VIEW";
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.2rem 0.45rem;
            border-radius: 999px;
            background: rgba(20, 32, 27, 0.74);
            color: rgba(255,255,255,0.92);
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            font-weight: 700;
        }

        .legend-card {
            background: linear-gradient(180deg, rgba(248, 252, 249, 0.92), rgba(241, 247, 243, 0.95));
            border: 1px solid rgba(52, 87, 73, 0.12);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: var(--shadow);
            margin-bottom: 0.9rem;
        }

        .legend-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-top: 0.55rem;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .legend-swatch {
            width: 14px;
            height: 14px;
            border-radius: 4px;
            flex: 0 0 auto;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(15, 118, 110, 0.18);
            min-height: 2.9rem;
            font-weight: 700;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e 0%, #155e75 100%);
            color: white;
            border: none;
            box-shadow: 0 12px 24px rgba(21, 94, 117, 0.24);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stFileUploader,
        .stChatInput {
            border-radius: 16px !important;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 252, 247, 0.7);
            border: 1px solid rgba(125, 96, 67, 0.10);
            border-radius: 18px;
            padding: 0.25rem 0.4rem;
            margin-bottom: 0.65rem;
        }

        @media (max-width: 980px) {
            .workflow-grid,
            .metric-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 1.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    for key, value in APP_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    st.session_state.analysis_status = "idle"
    st.session_state.chat_status = "idle"
    st.session_state.thread_id = None
    st.session_state.report_markdown = None
    st.session_state.gis_metadata = {}
    st.session_state.chat_messages = []
    st.session_state.analysis_error_message = None
    st.session_state.chat_error_message = None


def reset_app_state() -> None:
    uploader_version = st.session_state.uploader_version + 1
    for key, value in APP_STATE_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.uploader_version = uploader_version


def get_file_signature(image_file: Any, shp_zip: Any) -> tuple[Any, ...]:
    image_signature = None
    shp_signature = None

    if image_file is not None:
        image_signature = (image_file.name, image_file.size)
    if shp_zip is not None:
        shp_signature = (shp_zip.name, shp_zip.size)

    return image_signature, shp_signature


def summarize_metadata(metadata: Any) -> list[tuple[str, str]]:
    if not isinstance(metadata, dict):
        return []

    preferred_keys = [
        "layer_name",
        "geometry_type",
        "feature_count",
        "crs",
        "projection",
        "bounds",
        "fields",
    ]
    rows: list[tuple[str, str]] = []

    for key in preferred_keys:
        if key not in metadata:
            continue

        value = metadata[key]
        if isinstance(value, list):
            display_value = ", ".join(str(item) for item in value[:5])
            if len(value) > 5:
                display_value += f" (+{len(value) - 5} more)"
        elif isinstance(value, dict):
            display_value = ", ".join(f"{k}: {v}" for k, v in list(value.items())[:4])
            if len(value) > 4:
                display_value += " ..."
        else:
            display_value = str(value)

        rows.append((key.replace("_", " ").title(), display_value))

    return rows


def handle_api_error(prefix: str, error: Exception) -> str:
    if isinstance(error, requests.exceptions.Timeout):
        return f"{prefix} timed out. The backend may still be processing, so please try again."
    if isinstance(error, requests.exceptions.ConnectionError):
        return f"{prefix} failed because the backend could not be reached at `{BACKEND_BASE_URL}`."
    if isinstance(error, requests.exceptions.RequestException):
        return f"{prefix} failed due to a network error: {error}"
    return f"{prefix} failed unexpectedly: {error}"


def render_status_badge(label: str, value: str) -> None:
    st.markdown(f"**{label}:** {value}")


def render_empty_state(title: str, message: str) -> None:
    st.info(f"**{title}**\n\n{message}")


def render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-label">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <h4>{title}</h4>
            <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, copy: str) -> str:
    return (
        f'<div class="metric-card">'
        f"<h4>{title}</h4>"
        f'<div class="metric-value">{value}</div>'
        f"<p>{copy}</p>"
        f"</div>"
    )


def render_legend_card() -> None:
    st.markdown(
        """
        <div class="legend-card">
            <h4>Session Legend</h4>
            <p>Read the page like a lightweight GIS workspace: inputs on the left, map framing beside them, analytical narrative below, then threaded follow-up questions.</p>
            <div class="legend-row"><span class="legend-swatch" style="background:#1f6b5c;"></span> Active analysis and primary actions</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#6f8c51;"></span> Supporting GIS context and metadata</div>
            <div class="legend-row"><span class="legend-swatch" style="background:#87612b;"></span> Workflow guidance and section framing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


initialize_session_state()
inject_styles()

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Academic GIS AI Mapping Workspace</div>
        <h1>SpatioThematicVLM</h1>
        <p>
            Analyze a thematic map, optionally ground it with shapefile context, and turn the result into a guided,
            presentation-ready GIS conversation. The flow below keeps the demo focused: submit inputs, review the report,
            then ask higher-level questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="workflow-grid">
        <div class="workflow-card">
            <div class="workflow-step">1</div>
            <h4>Upload the map</h4>
            <p>Start with a thematic map image that the model can interpret as the main evidence source.</p>
        </div>
        <div class="workflow-card">
            <div class="workflow-step">2</div>
            <h4>Add GIS context</h4>
            <p>Include an optional zipped shapefile package to enrich the analysis with geospatial metadata.</p>
        </div>
        <div class="workflow-card">
            <div class="workflow-step">3</div>
            <h4>Review the report</h4>
            <p>Inspect the generated markdown report and the summarized GIS metadata before continuing.</p>
        </div>
        <div class="workflow-card">
            <div class="workflow-step">4</div>
            <h4>Drive the discussion</h4>
            <p>Use the same analysis thread to ask presentation-ready GIS questions and clarifications.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploader_version = st.session_state.uploader_version
image_file = None
shp_zip = None

with st.sidebar:
    st.header("Session Status")
    render_status_badge("Backend", BACKEND_BASE_URL)
    render_status_badge("Analysis", st.session_state.last_analyze_status)
    render_status_badge("Chat", st.session_state.last_chat_status)

    with st.expander("Current Context", expanded=False):
        render_status_badge("Workflow State", st.session_state.analysis_status.title())
        render_status_badge("Chat State", st.session_state.chat_status.title())
        render_status_badge(
            "Image",
            st.session_state.uploaded_image_name or "No file selected",
        )
        render_status_badge(
            "Shapefile ZIP",
            st.session_state.uploaded_shp_name or "Not provided",
        )
        if st.session_state.thread_id:
            render_status_badge("Thread ID", st.session_state.thread_id)

    if st.button("Clear Session", use_container_width=True):
        reset_app_state()
        st.rerun()


render_section_intro(
    "Step 1",
    "Prepare Inputs",
    "Upload the required thematic map image, optionally add a zipped shapefile package, and confirm the assets you want to analyze in this session.",
)

input_col, preview_col = st.columns([1.1, 0.9], gap="large")

with input_col:
    st.markdown('<div class="panel-title">Upload Panel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Use one map image for the main analysis. Add a shapefile ZIP only when you want the backend to surface GIS metadata alongside the report.</div>',
        unsafe_allow_html=True,
    )
    image_file = st.file_uploader(
        "Thematic map image",
        type=["png", "jpg", "jpeg"],
        help="Required. Use a PNG or JPEG thematic map image for the main analysis.",
        key=f"image_uploader_{uploader_version}",
    )
    shp_zip = st.file_uploader(
        "Zipped shapefile package",
        type=["zip"],
        help="Optional. Upload a single ZIP containing the shapefile set (.shp, .dbf, .shx, and related files).",
        key=f"shp_uploader_{uploader_version}",
    )

    st.session_state.uploaded_image_name = image_file.name if image_file else None
    st.session_state.uploaded_shp_name = shp_zip.name if shp_zip else None

    current_signature = get_file_signature(image_file, shp_zip)
    has_pending_input_change = (
        st.session_state.last_analyzed_signature is not None
        and current_signature != st.session_state.last_analyzed_signature
    )

    if st.session_state.analysis_status not in {"analyzing", "success"}:
        if image_file is not None:
            st.session_state.analysis_status = "ready"
            st.session_state.analysis_error_message = None
        else:
            st.session_state.analysis_status = "idle"

    if has_pending_input_change and st.session_state.analysis_status == "success":
        st.warning("Uploads changed after the last analysis. Run analysis again to refresh the report and chat context.")

    st.markdown(
        "<div class='metric-grid'>"
        + render_metric_card(
            "Map image",
            image_file.name if image_file else "Not selected",
            f"{image_file.size / 1024:.1f} KB uploaded" if image_file else "Required for analysis",
        )
        + render_metric_card(
            "Shapefile ZIP",
            shp_zip.name if shp_zip else "Optional",
            f"{shp_zip.size / 1024:.1f} KB uploaded" if shp_zip else "Attach only if available",
        )
        + render_metric_card(
            "Workflow state",
            st.session_state.analysis_status.title(),
            "Ready for submission" if image_file else "Waiting for map image",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Typical analysis time depends on image size and backend load. For demos, allow a short pause after submission.")

    analyze_disabled = st.session_state.analysis_status == "analyzing"
    if st.button("Run GIS Analysis", type="primary", use_container_width=True, disabled=analyze_disabled):
        if not image_file:
            st.session_state.analysis_status = "error"
            st.session_state.analysis_error_message = "Please upload a thematic map image before starting analysis."
            st.session_state.last_analyze_status = "Blocked: missing image"
        else:
            reset_analysis_state()
            st.session_state.analysis_status = "analyzing"
            st.session_state.last_analyze_status = "Running"

            files = {
                "image_file": (image_file.name, image_file.getvalue(), image_file.type),
            }
            if shp_zip is not None:
                files["shp_zip"] = (shp_zip.name, shp_zip.getvalue(), shp_zip.type)

            try:
                with st.spinner("Analyzing map and preparing the GIS report..."):
                    response = requests.post(
                        ANALYZE_ENDPOINT,
                        files=files,
                        timeout=ANALYZE_TIMEOUT_SECONDS,
                    )

                if response.status_code != 200:
                    st.session_state.analysis_status = "error"
                    st.session_state.analysis_error_message = (
                        f"Analysis failed with status {response.status_code}: {response.text}"
                    )
                    st.session_state.last_analyze_status = f"Failed ({response.status_code})"
                else:
                    try:
                        data = response.json()
                    except ValueError as error:
                        st.session_state.analysis_status = "error"
                        st.session_state.analysis_error_message = (
                            f"Analysis response could not be parsed as JSON: {error}"
                        )
                        st.session_state.last_analyze_status = "Failed (invalid JSON)"
                    else:
                        missing_keys = [
                            key for key in ("thread_id", "report_markdown") if key not in data
                        ]
                        if missing_keys:
                            st.session_state.analysis_status = "error"
                            st.session_state.analysis_error_message = (
                                "Analysis response is missing required fields: "
                                + ", ".join(missing_keys)
                            )
                            st.session_state.last_analyze_status = "Failed (missing fields)"
                        else:
                            st.session_state.thread_id = data["thread_id"]
                            st.session_state.report_markdown = data["report_markdown"]
                            st.session_state.gis_metadata = data.get("gis_metadata", {})
                            st.session_state.analysis_status = "success"
                            st.session_state.chat_status = "idle"
                            st.session_state.analysis_error_message = None
                            st.session_state.last_analyze_status = "Success"
                            st.session_state.last_analyzed_signature = current_signature
            except Exception as error:
                st.session_state.analysis_status = "error"
                st.session_state.analysis_error_message = handle_api_error("Analysis", error)
                st.session_state.last_analyze_status = "Failed (request error)"

with preview_col:
    render_info_card(
        "Map preview",
        "This panel behaves like a compact map frame so you can confirm the selected input before sending it into the analysis pipeline.",
    )
    render_legend_card()
    st.markdown('<div class="map-frame">', unsafe_allow_html=True)
    if image_file is None:
        render_empty_state(
            "No image uploaded",
            "Add a thematic map image to preview it here before analysis.",
        )
    else:
        try:
            preview_image = Image.open(image_file)
            st.image(preview_image, caption="Uploaded thematic map", use_container_width=True)
        except (UnidentifiedImageError, OSError) as error:
            st.warning(
                "The image could not be previewed in Streamlit, but it can still be sent to the backend if the file is valid."
            )
            st.caption(f"Preview detail: {error}")
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.analysis_status == "analyzing":
    st.info("Analysis is in progress. Results and chat will unlock automatically when the backend responds.")
elif st.session_state.analysis_status == "error" and st.session_state.analysis_error_message:
    st.error(st.session_state.analysis_error_message)


render_section_intro(
    "Step 2",
    "Review Analysis Results",
    "The report is the main story for the demo. Supporting GIS metadata and diagnostics remain available without competing for attention.",
)
if st.session_state.analysis_status != "success" or not st.session_state.report_markdown:
    render_empty_state(
        "Results will appear here",
        "Run GIS analysis to generate the report, metadata summary, and a chat-ready thread.",
    )
else:
    st.success("Analysis completed. Review the report first, then continue with follow-up questions below.")

    result_col, metadata_col = st.columns([1.5, 0.9], gap="large")

    with result_col:
        st.markdown('<div class="status-pill">Analysis Ready</div>', unsafe_allow_html=True)
        st.markdown('<div class="report-shell">', unsafe_allow_html=True)
        st.subheader("Analysis Report")
        st.markdown(st.session_state.report_markdown)
        st.markdown("</div>", unsafe_allow_html=True)

    with metadata_col:
        render_info_card(
            "GIS Metadata Summary",
            "Keep this column as supporting evidence during the demo. It is useful for technical questions without stealing focus from the report.",
        )
        metadata_rows = summarize_metadata(st.session_state.gis_metadata)

        if metadata_rows:
            for label, value in metadata_rows:
                st.markdown(f"**{label}:** {value}")
        elif st.session_state.gis_metadata:
            st.caption("Metadata is available below in raw form.")
        else:
            st.caption("No GIS metadata was returned for this analysis.")

        if st.session_state.gis_metadata:
            with st.expander("Raw GIS metadata", expanded=False):
                st.json(st.session_state.gis_metadata)

    with st.expander("Diagnostics", expanded=False):
        render_status_badge("Thread ID", st.session_state.thread_id or "Not available")
        render_status_badge(
            "Last analyzed image",
            st.session_state.uploaded_image_name or "Unknown",
        )
        render_status_badge(
            "Last analyzed ZIP",
            st.session_state.uploaded_shp_name or "Not provided",
        )
        render_status_badge("Analysis request", st.session_state.last_analyze_status)
        render_status_badge("Chat request", st.session_state.last_chat_status)


render_section_intro(
    "Step 3",
    "Continue with the GIS Agent",
    "Use the completed analysis as context for follow-up questions about spatial pattern interpretation, caveats, and presentation-ready takeaways.",
)
chat_locked = (
    st.session_state.analysis_status != "success"
    or not st.session_state.thread_id
    or has_pending_input_change
)

if chat_locked:
    render_empty_state(
        "Chat unlocks after analysis",
        "Complete a successful analysis with the current uploads before sending follow-up questions.",
    )
else:
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.write("Use the report as context, then ask targeted GIS questions about interpretation, uncertainty, or spatial patterns.")
    st.markdown(
        """
        <div class="helper-chip-row">
            <span class="helper-chip">Summarize the strongest spatial pattern.</span>
            <span class="helper-chip">What caveats should I mention in an academic presentation?</span>
            <span class="helper-chip">How does the shapefile context change the interpretation?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.chat_error_message:
        st.error(st.session_state.chat_error_message)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message.get("is_error"):
                st.error(message["content"])
            else:
                st.write(message["content"])

    prompt = st.chat_input(
        "Ask a follow-up question about the current analysis...",
        disabled=st.session_state.chat_status == "sending",
    )

    if prompt:
        st.session_state.chat_status = "sending"
        st.session_state.chat_error_message = None
        st.session_state.last_chat_status = "Running"
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        payload = {"message": prompt, "thread_id": st.session_state.thread_id}

        try:
            with st.spinner("Waiting for the GIS agent..."):
                response = requests.post(
                    CHAT_ENDPOINT,
                    json=payload,
                    timeout=CHAT_TIMEOUT_SECONDS,
                )

            if response.status_code != 200:
                error_message = f"Chat failed with status {response.status_code}: {response.text}"
                st.session_state.chat_error_message = error_message
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": error_message, "is_error": True}
                )
                st.session_state.chat_status = "error"
                st.session_state.last_chat_status = f"Failed ({response.status_code})"
            else:
                try:
                    data = response.json()
                except ValueError as error:
                    error_message = f"Chat response could not be parsed as JSON: {error}"
                    st.session_state.chat_error_message = error_message
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": error_message, "is_error": True}
                    )
                    st.session_state.chat_status = "error"
                    st.session_state.last_chat_status = "Failed (invalid JSON)"
                else:
                    if "response" not in data:
                        error_message = "Chat response is missing the required `response` field."
                        st.session_state.chat_error_message = error_message
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": error_message, "is_error": True}
                        )
                        st.session_state.chat_status = "error"
                        st.session_state.last_chat_status = "Failed (missing fields)"
                
                    else:
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": data["response"]}
                        )
                        st.session_state.chat_status = "idle"
                        st.session_state.last_chat_status = "Success"
            st.rerun()
        except Exception as error:
            error_message = handle_api_error("Chat", error)
            st.session_state.chat_error_message = error_message
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": error_message, "is_error": True}
            )
            st.session_state.chat_status = "error"
            st.session_state.last_chat_status = "Failed (request error)"
    st.markdown("</div>", unsafe_allow_html=True)
