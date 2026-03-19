"""
Universal Agentic AI Resume Chatbot - V7
Features: BULK RESUME MODE, Remove/discard buttons, full-screen preview, agent trace, 100MB upload
Enterprise HR Feature: Multi-resume JD matching with ranked results
"""

import streamlit as st
import os
import time
import base64
from typing import List, Dict
import pandas as pd

from document_processor import process_uploaded_file
from resume_parser import parse_resume_with_llm, get_resume_display_summary
from mcp_tools import create_tool_registry
from agent import ResumeAgent
from bulk_processor import (
    process_bulk_resumes, 
    results_to_dataframe, 
    export_results_to_excel,
    export_results_to_csv,
    BulkProcessingResult,
    CandidateResult
)

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Universal AI Resume Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── MODELS ──
GROQ_MODELS = {
    "Llama 3.1 8B ⚡ (Fast)": {"id": "llama-3.1-8b-instant", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
    "Llama 3.3 70B 🏆 (Best)": {"id": "llama-3.3-70b-versatile", "speed": "🔄", "quality": "⭐⭐⭐⭐⭐"},
    "Llama 4 Scout 🆕": {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "speed": "⚡", "quality": "⭐⭐⭐⭐"},
    "Qwen 3 32B 🧠": {"id": "qwen/qwen3-32b", "speed": "⚡", "quality": "⭐⭐⭐⭐⭐"},
    "Kimi K2 🌙": {"id": "moonshotai/kimi-k2-instruct", "speed": "🔄", "quality": "⭐⭐⭐⭐⭐"},
    "GPT-OSS 120B 💪": {"id": "openai/gpt-oss-120b", "speed": "🔄", "quality": "⭐⭐⭐⭐⭐"},
    "GPT-OSS 20B ⚡": {"id": "openai/gpt-oss-20b", "speed": "⚡", "quality": "⭐⭐⭐⭐"},
}
DEFAULT_MODEL = "Llama 3.1 8B ⚡ (Fast)"


def get_groq_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


# ═══════════════════════════════════════════
#              ENHANCED CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global Text ── */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stText, p, span, li, label, div, td, th {
        color: #e2e8f0 !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }

    /* ── Header ── */
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem; color: #94a3b8 !important;
        text-align: center; margin-bottom: 1.5rem;
    }

    /* ── Upload Card ── */
    .upload-box {
        background: linear-gradient(135deg, #4338ca, #7c3aed);
        color: #fff !important; padding: 1.2rem 1rem;
        border-radius: 12px; margin-bottom: 0.75rem;
        box-shadow: 0 8px 25px rgba(79,70,229,0.35);
    }
    .upload-box h3 { color: #fff !important; margin: 0 0 0.3rem; font-size: 1.05rem; }
    .upload-box p { color: #e0e7ff !important; font-size: 0.82rem; margin: 0.1rem 0; }

    /* ── Bulk Upload Card ── */
    .bulk-upload-box {
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        color: #fff !important; padding: 1.2rem 1rem;
        border-radius: 12px; margin-bottom: 0.75rem;
        box-shadow: 0 8px 25px rgba(236,72,153,0.35);
    }
    .bulk-upload-box h3 { color: #fff !important; margin: 0 0 0.3rem; font-size: 1.05rem; }
    .bulk-upload-box p { color: #fce7f3 !important; font-size: 0.82rem; margin: 0.1rem 0; }

    /* ── Profile Card ── */
    .profile-box {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569; border-radius: 12px;
        padding: 1rem; margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .profile-box p { color: #cbd5e1 !important; font-size: 0.85rem; margin: 0.25rem 0; }
    .profile-box strong { color: #f8fafc !important; font-size: 1rem; }

    /* ── JD Card ── */
    .jd-box {
        background: linear-gradient(135deg, #065f46, #0d9488);
        color: #fff !important; padding: 1rem;
        border-radius: 12px; margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(13,148,136,0.3);
    }
    .jd-box h4 { color: #fff !important; margin: 0 0 0.25rem; font-size: 0.95rem; }
    .jd-box p { color: #ccfbf1 !important; font-size: 0.82rem; }

    /* ── JD Required Card ── */
    .jd-required-box {
        background: linear-gradient(135deg, #dc2626, #f97316);
        color: #fff !important; padding: 1rem;
        border-radius: 12px; margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(220,38,38,0.3);
    }
    .jd-required-box h4 { color: #fff !important; margin: 0 0 0.25rem; font-size: 0.95rem; }
    .jd-required-box p { color: #fef2f2 !important; font-size: 0.82rem; }

    /* ── Model Card ── */
    .model-box {
        background: #1e293b; border: 1px solid #475569;
        padding: 0.75rem; border-radius: 10px; margin-top: 0.4rem;
    }
    .model-box p { color: #94a3b8 !important; font-size: 0.8rem; margin: 0.1rem 0; }
    .model-box .lbl { color: #a78bfa !important; font-weight: 700; }

    /* ── Mode Toggle Container ── */
    .toggle-container {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .mode-label { font-weight: 600; color: #f1f5f9 !important; font-size: 1rem; }
    .mode-description { color: #94a3b8 !important; font-size: 0.85rem; margin-top: 0.25rem; }

    /* ── AGENT TRACE ── */
    .trace-step {
        padding: 14px 18px; margin: 8px 0; border-radius: 8px;
        font-size: 0.95rem; line-height: 1.6;
        background: #d1d5db !important; border-left: 5px solid #6b7280;
    }
    .trace-step strong { color: #000000 !important; }
    .trace-step span.trace-duration { color: #4b5563 !important; opacity: 1 !important; }
    .trace-plan { background: #fef9c3 !important; border-left: 5px solid #facc15; }
    .trace-plan, .trace-plan *, .trace-plan span.trace-duration { color: #713f12 !important; }
    .trace-tool { background: #dcfce7 !important; border-left: 5px solid #4ade80; }
    .trace-tool, .trace-tool *, .trace-tool span.trace-duration { color: #14532d !important; }
    .trace-synth { background: #dbeafe !important; border-left: 5px solid #818cf8; }
    .trace-synth, .trace-synth *, .trace-synth span.trace-duration { color: #1e3a8a !important; }

    /* ── Tool Badges ── */
    .tbadge {
        display: inline-block; background: rgba(167,139,250,0.2);
        color: #c4b5fd !important; padding: 4px 12px; border-radius: 14px;
        font-size: 0.78rem; font-weight: 600; margin: 3px;
        border: 1px solid rgba(167,139,250,0.35);
    }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(167,139,250,0.15); color: #c4b5fd !important;
        padding: 6px 16px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
        border: 1px solid rgba(167,139,250,0.3);
    }
    .pulse-dot {
        width: 9px; height: 9px; border-radius: 50%; background: #22c55e;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }

    /* ── Bulk Mode Styles ── */
    .bulk-stats-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .bulk-stats-card h2 { color: #f1f5f9 !important; font-size: 2.5rem; margin: 0; }
    .bulk-stats-card p { color: #94a3b8 !important; margin: 0.25rem 0 0; font-size: 0.9rem; }

    .candidate-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569; border-radius: 12px;
        padding: 1.25rem; margin: 0.75rem 0; transition: all 0.2s;
    }
    .candidate-card:hover { border-color: #7c3aed; box-shadow: 0 4px 20px rgba(124, 58, 237, 0.2); }

    .candidate-rank {
        display: inline-flex; align-items: center; justify-content: center;
        width: 40px; height: 40px; border-radius: 50%;
        font-weight: bold; font-size: 1.1rem; margin-right: 12px;
    }
    .rank-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #000; }
    .rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: #000; }
    .rank-3 { background: linear-gradient(135deg, #d97706, #b45309); color: #fff; }
    .rank-other { background: #475569; color: #fff; }

    .score-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 0.9rem;
    }
    .score-excellent { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid #22c55e; }
    .score-good { background: rgba(250, 204, 21, 0.2); color: #facc15; border: 1px solid #facc15; }
    .score-moderate { background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid #f97316; }
    .score-low { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }

    .score-breakdown { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .mini-score {
        background: rgba(100, 116, 139, 0.3); color: #cbd5e1;
        padding: 4px 10px; border-radius: 12px; font-size: 0.75rem;
    }

    /* ── Welcome Card ── */
    .welcome-box {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569; padding: 2.5rem 2rem; border-radius: 16px;
        text-align: center; margin: 2rem auto; max-width: 800px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .welcome-box h2 { color: #f1f5f9 !important; font-size: 1.8rem; }
    .welcome-box p { color: #cbd5e1 !important; font-size: 1rem; }
    .welcome-box li { color: #94a3b8 !important; text-align: left; margin: 4px 0; }
    .welcome-box strong { color: #c4b5fd !important; }
    .welcome-box em { color: #93c5fd !important; }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span { color: #e2e8f0 !important; }
    [data-testid="stChatMessage"] strong { color: #c4b5fd !important; }
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 { color: #f1f5f9 !important; }
    [data-testid="stChatMessage"] code { color: #fbbf24 !important; background: #334155 !important; }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader { color: #e2e8f0 !important; font-weight: 600 !important; font-size: 0.9rem !important; }
    details summary span { color: #e2e8f0 !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #0f172a !important; }
    section[data-testid="stSidebar"] p { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] .stCaption { color: #64748b !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%; border-radius: 8px; font-size: 0.82rem;
        padding: 0.5rem; transition: all 0.2s; border: 1px solid #475569;
    }
    .stButton > button:hover { border-color: #7c3aed; box-shadow: 0 0 12px rgba(124,58,237,0.3); }

    /* ── Text Areas ── */
    .stTextArea textarea {
        color: #e2e8f0 !important; background: #1e293b !important;
        border: 1px solid #475569 !important; border-radius: 8px !important;
    }

    /* ── Footer ── */
    .app-footer { text-align: center; font-size: 0.8rem; padding: 1rem 0; }
    .app-footer, .app-footer * { color: #64748b !important; }
    .app-footer a { color: #93c5fd !important; text-decoration: none; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }

    /* ── Hide Streamlit Chrome ── */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: #0f172a !important; }

    /* ── FULL-SCREEN PREVIEW ── */
    .preview-header {
        background: linear-gradient(135deg, #4338ca, #7c3aed);
        padding: 16px 24px; border-radius: 12px 12px 0 0; margin-top: 10px;
    }
    .preview-title { color: #fff !important; font-size: 1.2rem; font-weight: 600; margin: 0; }
    .preview-body {
        background: #0f172a; border: 1px solid #475569; border-top: none;
        border-radius: 0 0 12px 12px; padding: 20px; min-height: 500px; max-height: 75vh; overflow: auto;
    }
    .preview-text-content {
        background: #1e293b; color: #e2e8f0 !important; padding: 24px; border-radius: 8px;
        font-size: 0.92rem; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-height: 65vh; overflow: auto;
    }
    .preview-image-container { background: #1e293b; padding: 20px; border-radius: 8px; text-align: center; }
    .preview-image-container img { max-width: 100%; max-height: 65vh; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }

    /* ── DataFrames ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stDataFrame [data-testid="stDataFrameResizable"] { border-radius: 8px; }

    /* ── Progress bar ── */
    .stProgress > div > div > div { background: linear-gradient(135deg, #7c3aed, #ec4899); }

    /* ── Download Buttons ── */
    .download-btn {
        display: inline-block; background: linear-gradient(135deg, #059669, #10b981);
        color: #fff !important; padding: 10px 20px; border-radius: 8px; font-size: 0.9rem;
        text-decoration: none; font-weight: 600; margin: 5px; transition: all 0.2s;
    }
    .download-btn:hover { box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4); transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE DEFAULTS ──
def get_default_state():
    return {
        # Mode
        "bulk_mode": False,
        
        # Single mode states
        "messages": [],
        "selected_model": DEFAULT_MODEL,
        "show_agent_trace": True,
        "resume_text": "",
        "parsed_resume": None,
        "resume_loaded": False,
        "file_info": None,
        "tool_registry": None,
        "jd_text": "",
        "jd_loaded": False,
        "jd_file_info": None,
        "file_key": None,
        "jd_file_key": None,
        "jd_source": None,
        "show_preview_modal": False,
        "raw_file_bytes": None,
        "raw_file_name": None,
        "raw_file_type": None,
        
        # Bulk mode states
        "bulk_files": [],
        "bulk_results": None,
        "bulk_processing": False,
    }

# Initialize session state
for k, v in get_default_state().items():
    if k not in st.session_state:
        st.session_state[k] = v

GROQ_API_KEY = get_groq_key()


# ═══════════════════════════════════════════
#          HELPER FUNCTIONS
# ═══════════════════════════════════════════

def reset_all_state():
    """Completely reset all session state"""
    default_state = get_default_state()
    for key in list(st.session_state.keys()):
        if key in default_state:
            st.session_state[key] = default_state[key]


def get_score_class(score: float) -> str:
    """Get CSS class based on score"""
    if score >= 80:
        return "score-excellent"
    elif score >= 65:
        return "score-good"
    elif score >= 50:
        return "score-moderate"
    else:
        return "score-low"


def get_rank_class(rank: int) -> str:
    """Get CSS class for rank badge"""
    if rank == 1:
        return "rank-1"
    elif rank == 2:
        return "rank-2"
    elif rank == 3:
        return "rank-3"
    else:
        return "rank-other"


def show_trace(steps, tools_used):
    """Display agent trace in expandable format"""
    with st.expander(
        f"🔍 Agent Trace ({len(steps)} steps · Tools: {', '.join(tools_used)})",
        expanded=False
    ):
        for i, s in enumerate(steps):
            icon = {
                "planning": "🎯", "tool_call": "🔧", "synthesis": "✨"
            }.get(s.step_type, "📌")

            css_class = {
                "planning": "trace-plan",
                "tool_call": "trace-tool",
                "synthesis": "trace-synth"
            }.get(s.step_type, "trace-tool")

            title = f"{icon} Step {i+1}: {s.step_type.upper()}"
            details = ""

            if s.step_type == "planning" and s.output_data:
                reasoning = s.output_data.get('reasoning', 'N/A')
                planned = ', '.join(s.output_data.get('planned_tools', []))
                details = f"<br><b>Plan:</b> {reasoning}<br><b>Tools:</b> {planned}"
            elif s.step_type == "tool_call":
                status = "✅" if s.success else "❌"
                details = f" — <b>{s.tool_name}</b> {status}"

            st.markdown(f"""
            <div class="trace-step {css_class}">
                <strong style="font-size:1rem;">{title}{details}</strong>
                <span class="trace-duration"> ({s.duration}s)</span>
            </div>
            """, unsafe_allow_html=True)


def run_agent(question: str):
    """Run the agentic AI on a question"""
    mi = GROQ_MODELS[st.session_state.selected_model]
    jd = st.session_state.jd_text if st.session_state.jd_loaded else ""
    agent = ResumeAgent(
        st.session_state.tool_registry, GROQ_API_KEY, mi["id"], jd_text=jd
    )
    return agent.run(question, st.session_state.messages), mi


# ═══════════════════════════════════════════
#              SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    # ── Mode Toggle ──
    st.markdown("### 🔄 Processing Mode")
    
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    
    bulk_mode = st.toggle(
        "📚 Bulk Resume Mode (HR)",
        value=st.session_state.bulk_mode,
        key="bulk_mode_toggle",
        help="Enable to upload and compare multiple resumes against a single JD"
    )
    
    if bulk_mode != st.session_state.bulk_mode:
        st.session_state.bulk_mode = bulk_mode
        # Reset relevant states when switching modes
        if bulk_mode:
            st.session_state.resume_text = ""
            st.session_state.parsed_resume = None
            st.session_state.resume_loaded = False
            st.session_state.messages = []
            st.session_state.show_preview_modal = False
        else:
            st.session_state.bulk_files = []
            st.session_state.bulk_results = None
            st.session_state.bulk_processing = False
        st.rerun()
    
    if st.session_state.bulk_mode:
        st.markdown("""
            <p class="mode-description">
                📤 Upload multiple resumes<br>
                📋 Compare all against one JD<br>
                📊 Get ranked results with scores
            </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <p class="mode-description">
                📄 Upload single resume<br>
                💬 Chat with AI assistant<br>
                🔧 Use 7 MCP tools
            </p>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")

    # ═══════════════════════════════════════════
    #         BULK MODE SIDEBAR
    # ═══════════════════════════════════════════
    if st.session_state.bulk_mode:
        # ── JD Upload (REQUIRED for bulk mode) ──
        st.markdown("""<div class="jd-required-box">
            <h4>📋 Job Description (Required)</h4>
            <p>Upload JD to compare resumes against</p>
        </div>""", unsafe_allow_html=True)
        
        if not st.session_state.jd_loaded:
            jd_file = st.file_uploader(
                "Upload JD", type=["pdf", "docx", "doc", "txt"],
                label_visibility="collapsed", key="bulk_jd_uploader"
            )
            
            if jd_file:
                with st.spinner("📋 Processing JD..."):
                    jd_result = process_uploaded_file(jd_file, GROQ_API_KEY)
                if jd_result["success"]:
                    st.session_state.jd_text = jd_result["text"]
                    st.session_state.jd_loaded = True
                    st.session_state.jd_file_info = jd_result
                    st.session_state.jd_source = "file"
                    st.rerun()
                else:
                    st.error(f"❌ {jd_result['error']}")
            
            jd_paste = st.text_area(
                "Or paste JD text:", 
                height=150, 
                key="bulk_jd_paste",
                placeholder="Paste job description here...",
            )
            
            if jd_paste and len(jd_paste.strip()) > 50:
                if st.button("✅ Use This JD", key="use_bulk_jd", use_container_width=True):
                    st.session_state.jd_text = jd_paste
                    st.session_state.jd_loaded = True
                    st.session_state.jd_source = "pasted"
                    st.session_state.jd_file_info = {
                        "file_name": "Pasted JD",
                        "file_size_kb": round(len(jd_paste) / 1024, 1),
                    }
                    st.rerun()
        else:
            jd_info = st.session_state.jd_file_info or {}
            col_jd, col_jd_remove = st.columns([5, 1])
            with col_jd:
                st.success(f"✅ {jd_info.get('file_name', 'JD Loaded')}")
            with col_jd_remove:
                if st.button("❌", key="remove_bulk_jd", help="Remove JD"):
                    st.session_state.jd_text = ""
                    st.session_state.jd_loaded = False
                    st.session_state.jd_file_info = None
                    st.session_state.bulk_results = None
                    st.rerun()
            
            with st.expander("👁️ View JD"):
                st.text_area("JD", st.session_state.jd_text[:3000], height=150, disabled=True, label_visibility="collapsed")
        
        st.markdown("---")
        
        # ── Bulk Resume Upload ──
        st.markdown("""<div class="bulk-upload-box">
            <h3>📚 Upload Multiple Resumes</h3>
            <p>Select multiple files (PDF, DOCX, TXT, Images)</p>
            <p>Up to 50 resumes at once</p>
        </div>""", unsafe_allow_html=True)
        
        bulk_files = st.file_uploader(
            "Upload Resumes",
            type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="bulk_resume_uploader"
        )
        
        if bulk_files:
            st.info(f"📁 {len(bulk_files)} file(s) selected")
            
            # Show first few files
            for f in bulk_files[:5]:
                st.caption(f"📄 {f.name}")
            if len(bulk_files) > 5:
                st.caption(f"... and {len(bulk_files) - 5} more")
        
        st.markdown("---")
        
        # ── Process Button ──
        can_process = st.session_state.jd_loaded and bulk_files and len(bulk_files) > 0
        
        if st.button(
            "🚀 Process & Compare All Resumes",
            disabled=not can_process,
            use_container_width=True,
            type="primary",
            key="process_bulk_btn"
        ):
            st.session_state.bulk_files = bulk_files
            st.session_state.bulk_processing = True
            st.rerun()
        
        if not st.session_state.jd_loaded:
            st.warning("⚠️ Please upload a Job Description first")
        elif not bulk_files:
            st.info("📤 Upload resumes to compare")
        
        # ── Clear Results ──
        if st.session_state.bulk_results:
            st.markdown("---")
            if st.button("🗑️ Clear Results", use_container_width=True, key="clear_bulk_results"):
                st.session_state.bulk_results = None
                st.session_state.bulk_files = []
                st.session_state.bulk_processing = False
                st.rerun()

    # ═══════════════════════════════════════════
    #         SINGLE MODE SIDEBAR
    # ═══════════════════════════════════════════
    else:
        # ── Resume Upload ──
        st.markdown("""<div class="upload-box">
            <h3>📄 Upload Resume</h3>
            <p>PDF · DOCX · TXT · JPG · PNG · Up to 100MB</p>
            <p>Works for any domain · Any format</p>
        </div>""", unsafe_allow_html=True)

        if not st.session_state.resume_loaded:
            resume_file = st.file_uploader(
                "Upload Resume",
                type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key="resume_uploader"
            )

            if resume_file:
                fk = f"{resume_file.name}_{resume_file.size}"
                if not st.session_state.resume_loaded or st.session_state.file_key != fk:
                    resume_file.seek(0)
                    raw_bytes = resume_file.read()
                    resume_file.seek(0)
                    
                    st.session_state.raw_file_bytes = raw_bytes
                    st.session_state.raw_file_name = resume_file.name
                    st.session_state.raw_file_type = os.path.splitext(resume_file.name)[1].lower()
                    
                    with st.spinner("📄 Processing document..."):
                        result = process_uploaded_file(resume_file, GROQ_API_KEY)

                    if result["success"]:
                        st.session_state.resume_text = result["text"]
                        st.session_state.file_info = result
                        st.session_state.file_key = fk

                        with st.spinner("🧠 Parsing resume with AI..."):
                            mid = GROQ_MODELS[st.session_state.selected_model]["id"]
                            parsed = parse_resume_with_llm(result["text"], GROQ_API_KEY, mid)
                            st.session_state.parsed_resume = parsed

                        with st.spinner("🔧 Initializing AI tools..."):
                            reg = create_tool_registry()
                            reg.set_resume_data(parsed, result["text"])
                            st.session_state.tool_registry = reg

                        st.session_state.resume_loaded = True
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
        else:
            fi = st.session_state.file_info
            
            col_file, col_remove = st.columns([5, 1])
            with col_file:
                st.success(f"✅ {fi['file_name']}")
            with col_remove:
                if st.button("❌", key="remove_resume", help="Remove resume"):
                    st.session_state.resume_text = ""
                    st.session_state.parsed_resume = None
                    st.session_state.resume_loaded = False
                    st.session_state.file_info = None
                    st.session_state.tool_registry = None
                    st.session_state.file_key = None
                    st.session_state.raw_file_bytes = None
                    st.session_state.raw_file_name = None
                    st.session_state.raw_file_type = None
                    st.session_state.messages = []
                    st.session_state.show_preview_modal = False
                    st.session_state.jd_text = ""
                    st.session_state.jd_loaded = False
                    st.session_state.jd_file_info = None
                    st.rerun()
            
            st.caption(f"📊 {fi['file_size_kb']} KB · {fi['file_type'].upper().replace('.', '')}")
            
            if st.button("👁️ View Full Resume", key="open_preview", use_container_width=True):
                st.session_state.show_preview_modal = True
                st.rerun()
            
            if st.session_state.raw_file_bytes:
                st.download_button(
                    label="📥 Download Resume",
                    data=st.session_state.raw_file_bytes,
                    file_name=st.session_state.raw_file_name,
                    mime="application/octet-stream",
                    use_container_width=True
                )

        # ── Parsed Profile ──
        if st.session_state.parsed_resume:
            p = st.session_state.parsed_resume
            st.markdown("---")
            st.markdown("### 👤 Detected Profile")

            name = p.get("name", "N/A")
            role = p.get("current_role", "")
            company = p.get("current_company", "")
            loc = p.get("location", "") or p.get("address", "")
            exp = p.get("total_experience_years", 0)
            email = p.get("email", "")
            phone = p.get("phone", "")
            linkedin = p.get("linkedin", "")
            github = p.get("github", "")

            html = f'<div class="profile-box"><p><strong>{name}</strong></p>'
            if role:
                html += f'<p>💼 {role}'
                if company:
                    html += f' at {company}'
                html += '</p>'
            if loc:
                html += f'<p>📍 {loc[:80]}</p>'
            if exp:
                html += f'<p>📅 ~{exp} years exp</p>'
            if email:
                html += f'<p>📧 {email}</p>'
            if phone:
                html += f'<p>📞 {phone}</p>'
            if linkedin:
                html += f'<p>🔗 LinkedIn</p>'
            if github:
                html += f'<p>💻 GitHub</p>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            with st.expander("📋 Full Parsed JSON"):
                st.json(p)

        st.markdown("---")

        # ── JD Upload (Optional for single mode) ──
        st.markdown("""<div class="jd-box">
            <h4>📋 Job Description (Optional)</h4>
            <p>Upload or paste JD to compare</p>
        </div>""", unsafe_allow_html=True)

        if not st.session_state.jd_loaded:
            jd_file = st.file_uploader(
                "Upload JD", type=["pdf", "docx", "doc", "txt"],
                label_visibility="collapsed", key="jd_uploader"
            )

            if jd_file:
                jd_fk = f"jd_{jd_file.name}_{jd_file.size}"
                if not st.session_state.jd_loaded or st.session_state.jd_file_key != jd_fk:
                    with st.spinner("📋 Processing JD..."):
                        jd_result = process_uploaded_file(jd_file, GROQ_API_KEY)
                    if jd_result["success"]:
                        st.session_state.jd_text = jd_result["text"]
                        st.session_state.jd_loaded = True
                        st.session_state.jd_file_key = jd_fk
                        st.session_state.jd_file_info = jd_result
                        st.session_state.jd_source = "file"
                        if st.session_state.tool_registry:
                            st.session_state.tool_registry.set_jd_text(jd_result["text"])
                        st.rerun()
                    else:
                        st.error(f"❌ {jd_result['error']}")

            jd_paste = st.text_area(
                "Or paste JD text:", 
                height=120, 
                key="jd_paste_area",
                placeholder="Paste job description here (optional)...",
                max_chars=50000,
            )
            
            if jd_paste and len(jd_paste.strip()) > 50:
                if st.button("✅ Use This JD", key="use_pasted_jd", use_container_width=True):
                    st.session_state.jd_text = jd_paste
                    st.session_state.jd_loaded = True
                    st.session_state.jd_source = "pasted"
                    st.session_state.jd_file_info = {
                        "file_name": "Pasted JD",
                        "file_size_kb": round(len(jd_paste) / 1024, 1),
                        "file_type": "text"
                    }
                    if st.session_state.tool_registry:
                        st.session_state.tool_registry.set_jd_text(jd_paste)
                    st.rerun()
        else:
            jd_info = st.session_state.jd_file_info or {}
            jd_name = jd_info.get("file_name", "Job Description")
            jd_size = jd_info.get("file_size_kb", 0)
            
            col_jd, col_jd_remove = st.columns([5, 1])
            with col_jd:
                st.success(f"✅ {jd_name}")
            with col_jd_remove:
                if st.button("❌", key="remove_jd", help="Remove JD"):
                    st.session_state.jd_text = ""
                    st.session_state.jd_loaded = False
                    st.session_state.jd_file_info = None
                    st.session_state.jd_file_key = None
                    st.session_state.jd_source = None
                    if st.session_state.tool_registry:
                        st.session_state.tool_registry.set_jd_text("")
                    st.rerun()
            
            if jd_size:
                st.caption(f"📊 {jd_size} KB")
            
            with st.expander("👁️ View Job Description"):
                st.text_area(
                    "JD Content", 
                    st.session_state.jd_text, 
                    height=200,
                    disabled=True, 
                    label_visibility="collapsed"
                )

        st.markdown("---")

        # ── Suggestions (Single Mode) ──
        if st.session_state.resume_loaded:
            st.markdown("### 💡 Try Asking")
            cname = "the candidate"
            if st.session_state.parsed_resume:
                cname = st.session_state.parsed_resume.get("name", "the candidate")

            suggestions = [
                f"What is {cname}'s contact information?",
                "What are the key technical skills?",
                "Give me the complete work experience",
                "Calculate total years of experience",
                "What is the educational background?",
                "Match skills: Python, AWS, Docker, Kubernetes",
                "Write a cover letter for Senior Engineer at Google",
                "What certifications are listed?",
            ]

            if st.session_state.jd_loaded:
                suggestions.insert(0, "Compare this resume against the job description")
                suggestions.insert(1, "How well does this candidate fit the JD?")

            for i, s in enumerate(suggestions[:8]):
                if st.button(f"📌 {s}", key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.pending_question = s
                    st.rerun()

    st.markdown("---")

    # ── Model Selection (Both modes) ──
    st.markdown("### 🧠 AI Model")
    sel = st.selectbox(
        "Model", list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model),
        key="model_selector"
    )
    st.session_state.selected_model = sel
    mi = GROQ_MODELS[sel]
    st.markdown(f"""<div class="model-box">
        <p><span class="lbl">Model:</span> {mi['id'][:30]}...</p>
        <p><span class="lbl">Speed:</span> {mi['speed']}
           <span class="lbl">Quality:</span> {mi['quality']}</p>
    </div>""", unsafe_allow_html=True)

    if GROQ_API_KEY:
        st.success("✅ API Connected")
    else:
        st.error("❌ Add GROQ_API_KEY in Secrets")

    st.markdown("---")

    # ── Settings (Single mode only) ──
    if not st.session_state.bulk_mode:
        st.markdown("### ⚙️ Settings")
        st.session_state.show_agent_trace = st.toggle(
            "🔍 Show Agent Trace", value=st.session_state.show_agent_trace,
            key="trace_toggle"
        )

        st.markdown("---")

        # ── MCP Tools Info ──
        st.markdown("### 🔧 MCP Tools")
        for icon, tname, desc in [
            ("📄", "resume_search", "RAG semantic search"),
            ("📊", "skill_analyzer", "Skill gap analysis"),
            ("💼", "experience_calc", "Experience breakdown"),
            ("📝", "cover_letter", "Cover letters"),
            ("👤", "profile_summary", "Professional bios"),
            ("🎯", "jd_matcher", "JD comparison"),
            ("🎓", "education_extractor", "Education & degrees"),
        ]:
            st.caption(f"{icon} **{tname}**: {desc}")

    st.markdown("---")

    # ── Reset Buttons ──
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("🔄 Reset All", use_container_width=True, key="reset_all_btn"):
            reset_all_state()
            st.rerun()

# ═══════════════════════════════════════════
#        FULL-SCREEN PREVIEW MODAL (Single Mode)
# ═══════════════════════════════════════════

if not st.session_state.bulk_mode and st.session_state.show_preview_modal and st.session_state.resume_loaded:
    fi = st.session_state.file_info
    file_type = st.session_state.raw_file_type
    file_name = st.session_state.raw_file_name
    file_bytes = st.session_state.raw_file_bytes
    
    header_col1, header_col2 = st.columns([8, 2])
    with header_col1:
        st.markdown(f"""
            <div class="preview-header">
                <h3 class="preview-title">
                    📄 {file_name}
                    <span style="color: #e0e7ff; font-size: 0.85rem; font-weight: 400; margin-left: 10px;">
                        {fi['file_size_kb']} KB · {file_type.upper().replace('.', '')}
                    </span>
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    with header_col2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("❌ Close Preview", key="close_preview_btn", use_container_width=True):
            st.session_state.show_preview_modal = False
            st.rerun()
    
    st.markdown('<div class="preview-body">', unsafe_allow_html=True)
    
    if file_type in [".jpg", ".jpeg", ".png", ".webp"]:
        st.markdown('<div class="preview-image-container">', unsafe_allow_html=True)
        b64_img = base64.b64encode(file_bytes).decode()
        mime_type = f"image/{file_type.replace('.', '')}"
        if file_type == ".jpg":
            mime_type = "image/jpeg"
        st.markdown(
            f'<img src="data:{mime_type};base64,{b64_img}" alt="{file_name}">',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("**📄 Extracted Content**")
        st.markdown(
            f'<div class="preview-text-content">{st.session_state.resume_text}</div>',
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
    with col_back2:
        if st.button("⬅️ Back to Chat", key="back_to_chat_btn", use_container_width=True):
            st.session_state.show_preview_modal = False
            st.rerun()
    
    st.stop()


# ═══════════════════════════════════════════
#              MAIN CONTENT - HEADER
# ═══════════════════════════════════════════

col_t, col_b = st.columns([5, 1])
with col_t:
    st.markdown(
        '<h1 class="main-header">🤖 Universal AI Resume Assistant</h1>',
        unsafe_allow_html=True
    )
    if st.session_state.bulk_mode:
        st.markdown(
            '<p class="sub-header">'
            '📚 Bulk Mode: Upload multiple resumes → Compare against JD → Get ranked results'
            '</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="sub-header">'
            'Upload any resume → Ask anything → Agentic AI + MCP Tools'
            '</p>',
            unsafe_allow_html=True
        )
with col_b:
    mode_text = "📚 Bulk Mode" if st.session_state.bulk_mode else "Agentic AI"
    st.markdown(f"""<div style="text-align:right;padding-top:28px;">
        <span class="status-badge">
            <span class="pulse-dot"></span> {mode_text}
        </span>
    </div>""", unsafe_allow_html=True)

# ── API Key Check ──
if not GROQ_API_KEY:
    st.warning("⚠️ Add `GROQ_API_KEY` in Settings → Secrets")
    st.info("🔗 Free: [console.groq.com/keys](https://console.groq.com/keys)")
    st.stop()


# ═══════════════════════════════════════════
#              BULK MODE MAIN CONTENT
# ═══════════════════════════════════════════

if st.session_state.bulk_mode:
    
    # ── Processing State ──
    if st.session_state.bulk_processing and st.session_state.bulk_files:
        st.markdown("### 🔄 Processing Resumes...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(st.session_state.bulk_files)
        
        def update_progress(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(f"({current}/{total}) {message}")
        
        # Process all resumes
        model_id = GROQ_MODELS[st.session_state.selected_model]["id"]
        
        result = process_bulk_resumes(
            st.session_state.bulk_files,
            st.session_state.jd_text,
            GROQ_API_KEY,
            model_id,
            progress_callback=update_progress
        )
        
        st.session_state.bulk_results = result
        st.session_state.bulk_processing = False
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        time.sleep(0.5)
        st.rerun()
    
    # ── Show Results ──
    elif st.session_state.bulk_results:
        result = st.session_state.bulk_results
        
        # ── Summary Stats ──
        st.markdown("### 📊 Processing Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="bulk-stats-card">
                    <h2>{result.total_resumes}</h2>
                    <p>Total Resumes</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="bulk-stats-card">
                    <h2 style="color: #22c55e;">{result.successful}</h2>
                    <p>Successfully Processed</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_score = sum(c.overall_score for c in result.candidates if c.success) / max(result.successful, 1)
            st.markdown(f"""
                <div class="bulk-stats-card">
                    <h2 style="color: #a78bfa;">{avg_score:.1f}%</h2>
                    <p>Average Match Score</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="bulk-stats-card">
                    <h2>{result.processing_time}s</h2>
                    <p>Processing Time</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ── Export Buttons ──
        st.markdown("### 📥 Export Results")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            excel_data = export_results_to_excel(result)
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"resume_comparison_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with export_col2:
            csv_data = export_results_to_csv(result)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"resume_comparison_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col3:
            # JSON export
            json_data = pd.DataFrame([{
                "rank": c.rank,
                "name": c.candidate_name,
                "email": c.email,
                "overall_score": c.overall_score,
                "skills_score": c.skills_score,
                "experience_score": c.experience_score,
                "recommendation": c.recommendation
            } for c in result.candidates if c.success]).to_json(orient="records", indent=2)
            
            st.download_button(
                label="🔧 Download JSON",
                data=json_data,
                file_name=f"resume_comparison_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # ── Ranked Candidates ──
        st.markdown("### 🏆 Ranked Candidates")
        
        # Filter tabs
        tab_all, tab_excellent, tab_good, tab_moderate = st.tabs([
            f"📋 All ({result.successful})",
            f"🟢 Excellent (≥80%)",
            f"🟡 Good (65-79%)",
            f"🟠 Moderate (<65%)"
        ])
        
        # In the display_candidate_cards function, update the education display:

def display_candidate_cards(candidates_list):
    if not candidates_list:
        st.info("No candidates in this category")
        return
    
    for candidate in candidates_list:
        if not candidate.success:
            continue
        
        score_class = get_score_class(candidate.overall_score)
        rank_class = get_rank_class(candidate.rank)
        
        # Get deduplicated highest education for display
        highest_edu = candidate.highest_education
        if len(highest_edu) > 80:
            display_edu = highest_edu[:80] + "..."
        else:
            display_edu = highest_edu
        
        with st.container():
            st.markdown(f"""
                <div class="candidate-card">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <span class="candidate-rank {rank_class}">#{candidate.rank}</span>
                        <div style="flex: 1;">
                            <strong style="font-size: 1.1rem; color: #f1f5f9;">{candidate.candidate_name}</strong>
                            <p style="margin: 2px 0; color: #94a3b8; font-size: 0.85rem;">
                                {candidate.current_role} {f'at {candidate.current_company}' if candidate.current_company else ''}
                            </p>
                        </div>
                        <span class="score-badge {score_class}">{candidate.overall_score}%</span>
                    </div>
                    <div class="score-breakdown">
                        <span class="mini-score">🎯 Skills: {candidate.skills_score}%</span>
                        <span class="mini-score">💼 Exp: {candidate.experience_score}%</span>
                        <span class="mini-score">🎓 Edu: {candidate.education_score}%</span>
                        <span class="mini-score">📍 Location: {candidate.location_score}%</span>
                        <span class="mini-score">📅 {candidate.total_experience}y exp</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📋 Details - {candidate.candidate_name}"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**📞 Contact:**")
                    if candidate.email:
                        st.write(f"📧 {candidate.email}")
                    if candidate.phone:
                        st.write(f"📱 {candidate.phone}")
                    if candidate.location:
                        st.write(f"📍 {candidate.location}")
                    
                    # Show only highest education (deduplicated)
                    st.markdown("**🎓 Highest Education:**")
                    st.write(highest_edu or "Not specified")
                
                with detail_col2:
                    # Deduplicate matched skills for display
                    unique_matched = list(set(candidate.matched_skills))
                    st.markdown("**✅ Matched Skills:**")
                    if unique_matched:
                        st.write(", ".join(unique_matched[:12]))
                        if len(unique_matched) > 12:
                            st.caption(f"... and {len(unique_matched) - 12} more")
                    else:
                        st.write("No specific matches")
                    
                    # Deduplicate missing skills
                    unique_missing = list(set(candidate.missing_skills))
                    st.markdown("**❌ Missing Skills:**")
                    if unique_missing:
                        st.write(", ".join(unique_missing[:8]))
                    else:
                        st.write("None identified")
                
                st.markdown("**💡 Recommendation:**")
                st.info(candidate.recommendation)
                
                # Deduplicate strengths
                if candidate.strengths:
                    unique_strengths = list(set(candidate.strengths))
                    st.markdown("**✨ Strengths:**")
                    for s in unique_strengths:
                        st.write(f"• {s}")
                
                # Deduplicate gaps
                if candidate.gaps:
                    unique_gaps = list(set(candidate.gaps))
                    st.markdown("**⚠️ Gaps:**")
                    for g in unique_gaps:
                        st.write(f"• {g}")
                
                st.caption(f"📁 File: {candidate.file_name} | ⏱️ Processed in {candidate.processing_time}s")
        
        with tab_all:
            display_candidate_cards([c for c in result.candidates if c.success])
        
        with tab_excellent:
            display_candidate_cards([c for c in result.candidates if c.success and c.overall_score >= 80])
        
        with tab_good:
            display_candidate_cards([c for c in result.candidates if c.success and 65 <= c.overall_score < 80])
        
        with tab_moderate:
            display_candidate_cards([c for c in result.candidates if c.success and c.overall_score < 65])
        
        # ── Failed Files ──
        failed = [c for c in result.candidates if not c.success]
        if failed:
            st.markdown("---")
            with st.expander(f"❌ Failed Files ({len(failed)})"):
                for c in failed:
                    st.error(f"**{c.file_name}**: {c.error}")
        
        # ── JD Requirements Summary ──
        st.markdown("---")
        with st.expander("📋 JD Requirements Analysis"):
            jd_summary = result.jd_summary
            
            col_jd1, col_jd2 = st.columns(2)
            
            with col_jd1:
                st.markdown("**🎯 Required Skills Detected:**")
                if jd_summary.get("required_skills"):
                    st.write(", ".join(jd_summary["required_skills"][:20]))
                else:
                    st.write("None specifically identified")
                
                st.markdown("**📅 Experience Requirement:**")
                st.write(f"{jd_summary.get('min_experience_years', 0)}+ years")
            
            with col_jd2:
                st.markdown("**🎓 Education Requirements:**")
                if jd_summary.get("required_education"):
                    st.write(", ".join(jd_summary["required_education"][:10]))
                else:
                    st.write("Not specified")
                
                st.markdown("**📍 Preferred Locations:**")
                if jd_summary.get("preferred_locations"):
                    st.write(", ".join(jd_summary["preferred_locations"][:10]))
                else:
                    st.write("Not specified")
    
    # ── Welcome State (No results yet) ──
    else:
        st.markdown("""
        <div class="welcome-box">
            <h2>📚 Bulk Resume Comparison Mode</h2>
            <p>Compare multiple resumes against a single Job Description</p>
            <br>
            <p><strong>How it works:</strong></p>
            <ul>
                <li>📋 <strong>Step 1:</strong> Upload a Job Description (required)</li>
                <li>📤 <strong>Step 2:</strong> Upload multiple resumes (PDF, DOCX, TXT, Images)</li>
                <li>🚀 <strong>Step 3:</strong> Click "Process & Compare All Resumes"</li>
                <li>📊 <strong>Step 4:</strong> View ranked results with detailed scores</li>
            </ul>
            <br>
            <p><strong>Scoring Breakdown:</strong></p>
            <ul>
                <li>🎯 <strong>Skills Match (35%)</strong> — Technical & soft skills alignment</li>
                <li>💼 <strong>Experience Match (25%)</strong> — Years of experience vs requirements</li>
                <li>🎓 <strong>Education Match (15%)</strong> — Degree & qualifications</li>
                <li>📍 <strong>Location Match (10%)</strong> — Geographic compatibility</li>
                <li>🔑 <strong>Keyword Match (15%)</strong> — JD keyword coverage</li>
            </ul>
            <br>
            <p><em>👈 Start by uploading a JD in the sidebar!</em></p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#              SINGLE MODE MAIN CONTENT
# ═══════════════════════════════════════════

else:
    # ── Welcome State ──
    if not st.session_state.resume_loaded:
        st.markdown("""
        <div class="welcome-box">
            <h2>📄 Upload a Resume to Get Started</h2>
            <p>Upload <strong>any resume</strong> in any format.
            Optionally add a <strong>Job Description</strong> for comparison.</p>
            <br>
            <p><strong>Supported Formats:</strong></p>
            <ul>
                <li>📕 <strong>PDF</strong> — Most common resume format</li>
                <li>📘 <strong>DOCX</strong> — Microsoft Word documents</li>
                <li>📝 <strong>TXT</strong> — Plain text files</li>
                <li>🖼️ <strong>Images</strong> — JPG, PNG, WEBP (AI Vision OCR)</li>
            </ul>
            <br>
            <p><strong>🔧 7 MCP Tools:</strong></p>
            <ul>
                <li>📄 Resume Search — RAG semantic search</li>
                <li>📊 Skill Analyzer — Match skills vs requirements</li>
                <li>💼 Experience Calculator — Years breakdown</li>
                <li>📝 Cover Letter Generator — Tailored cover letters</li>
                <li>👤 Profile Summary — LinkedIn / portfolio bios</li>
                <li>🎯 JD Matcher — Resume vs Job Description scoring</li>
                <li>🎓 Education Extractor — Degrees, GPA, certifications</li>
            </ul>
            <br>
            <p><em>👈 Upload a resume using the sidebar!</em></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Chat History ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                meta = []
                if msg.get("model"):
                    meta.append(f"🧠 {msg['model']}")
                if msg.get("total_time"):
                    meta.append(f"⚡ {msg['total_time']}s")
                if msg.get("tools_used"):
                    meta.append(f"🔧 {len(msg['tools_used'])} tools")
                if meta:
                    st.caption(" | ".join(meta))

                if msg.get("tools_used"):
                    st.markdown(
                        " ".join(
                            f'<span class="tbadge">{t}</span>'
                            for t in msg["tools_used"]
                        ),
                        unsafe_allow_html=True
                    )

                if st.session_state.show_agent_trace and msg.get("steps"):
                    show_trace(msg["steps"], msg.get("tools_used", []))

    # ── Handle Pending Question (from suggestions) ──
    if "pending_question" in st.session_state:
        q = st.session_state.pending_question
        del st.session_state.pending_question

        st.session_state.messages.append({"role": "user", "content": q})

        with st.spinner("🤖 Agent thinking..."):
            result, mi = run_agent(q)

        st.session_state.messages.append({
            "role": "assistant", "content": result.answer,
            "tools_used": result.tools_used, "steps": result.steps,
            "total_time": result.total_time, "model": mi["id"]
        })
        st.rerun()

    # ── Chat Input ──
    if prompt := st.chat_input("Ask anything about the uploaded resume..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        mi = GROQ_MODELS[st.session_state.selected_model]

        with st.chat_message("assistant"):
            with st.spinner(f"🤖 Agent working with **{mi['id']}**..."):
                result, _ = run_agent(prompt)

            st.markdown(result.answer)
            st.caption(
                f"🧠 {mi['id']} | ⚡ {result.total_time}s | "
                f"🔧 {len(result.tools_used)} tools"
            )

            if result.tools_used:
                st.markdown(
                    " ".join(
                        f'<span class="tbadge">{t}</span>'
                        for t in result.tools_used
                    ),
                    unsafe_allow_html=True
                )

            if st.session_state.show_agent_trace:
                show_trace(result.steps, result.tools_used)

        st.session_state.messages.append({
            "role": "assistant", "content": result.answer,
            "tools_used": result.tools_used, "steps": result.steps,
            "total_time": result.total_time, "model": mi["id"]
        })


# ═══════════════════════════════════════════
#              FOOTER
# ═══════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<p class="app-footer">'
    'Built with ❤️ using Agentic AI + MCP + RAG + Groq · '
    '100% Free · '
    '<a href="https://github.com/VijaiVenkatesan/Agentic_AI_Resume_Chatbot" target="_blank">GitHub</a>'
    '</p>',
    unsafe_allow_html=True
)
