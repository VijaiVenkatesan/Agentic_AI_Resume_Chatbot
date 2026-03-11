"""
Universal Agentic AI Resume Chatbot
Upload ANY resume → Ask ANY question → Agent uses MCP tools
"""

import streamlit as st
import os
import time
from typing import List, Dict

from document_processor import process_uploaded_file
from resume_parser import parse_resume_with_llm, get_resume_display_summary
from mcp_tools import create_tool_registry
from agent import ResumeAgent

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Universal AI Resume Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODELS
# ============================================================

GROQ_MODELS = {
    "Llama 3.1 8B ⚡ (Fast)": {"id": "llama-3.1-8b-instant", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
    "Llama 3.3 70B 🏆 (Best)": {"id": "llama-3.3-70b-versatile", "speed": "🔄", "quality": "⭐⭐⭐⭐⭐"},
    "Llama 4 Scout 🆕 (Latest)": {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "speed": "⚡", "quality": "⭐⭐⭐⭐"},
    "Qwen 3 32B 🧠 (Alibaba)": {"id": "qwen/qwen3-32b", "speed": "⚡", "quality": "⭐⭐⭐⭐⭐"},
    "Kimi K2 🌙 (Moonshot)": {"id": "moonshotai/kimi-k2-instruct", "speed": "🔄", "quality": "⭐⭐⭐⭐⭐"},
    "GPT-OSS 120B 💪 (OpenAI)": {"id": "openai/gpt-oss-120b", "speed": "🔄", "quality": "⭐⭐⭐⭐⭐"},
    "GPT-OSS 20B ⚡ (Light)": {"id": "openai/gpt-oss-20b", "speed": "⚡", "quality": "⭐⭐⭐⭐"},
}

DEFAULT_MODEL = "Llama 3.1 8B ⚡ (Fast)"

def get_groq_key() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
    .main-header { font-size:2.5rem; font-weight:bold; background:linear-gradient(90deg,#1E88E5,#7C4DFF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-align:center; }
    .sub-header { font-size:1.1rem; color:#666; text-align:center; margin-bottom:2rem; }
    .upload-card { background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:1.5rem; border-radius:1rem; margin-bottom:1rem; }
    .upload-card h3 { margin:0 0 .5rem; }
    .upload-card p { margin:.3rem 0; font-size:.9rem; }
    .resume-card { background:#1e293b; border:1px solid #334155; padding:1rem; border-radius:.75rem; margin:.5rem 0; }
    .model-info { background:linear-gradient(135deg,#1a1a2e,#16213e); color:#e0e0e0; padding:1rem; border-radius:.75rem; margin-top:.5rem; border:1px solid #334155; font-size:.85rem; }
    .model-info .label { color:#7C4DFF; font-weight:600; }
    .source-card { background-color:#f0f7ff; border-left:3px solid #1E88E5; padding:.75rem; margin:.5rem 0; border-radius:0 .5rem .5rem 0; font-size:.85rem; }
    .source-section { color:#1E88E5; font-weight:bold; }
    .agent-step { background:#f8f9fa; border-left:3px solid #7C4DFF; padding:.5rem .75rem; margin:.3rem 0; border-radius:0 .5rem .5rem 0; font-size:.8rem; }
    .agent-step-planning { border-left-color:#FF9800; }
    .agent-step-tool { border-left-color:#4CAF50; }
    .agent-step-synthesis { border-left-color:#2196F3; }
    .tool-badge { display:inline-block; background:rgba(124,77,255,0.1); color:#7C4DFF; padding:2px 8px; border-radius:10px; font-size:.75rem; font-weight:600; margin:2px; }
    .rag-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(124,77,255,0.1); color:#7C4DFF; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
    .status-dot { width:8px; height:8px; border-radius:50%; background:#22c55e; display:inline-block; animation:pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
    .stButton>button { width:100%; border-radius:.5rem; font-size:.85rem; padding:.5rem; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "selected_model": DEFAULT_MODEL,
    "show_agent_trace": True,
    "resume_text": "",
    "parsed_resume": None,
    "resume_loaded": False,
    "file_info": None,
    "tool_registry": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

GROQ_API_KEY = get_groq_key()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    # Upload Section
    st.markdown("""
    <div class="upload-card">
        <h3>📄 Upload Resume</h3>
        <p>Supports: PDF, DOCX, TXT, JPG, PNG</p>
        <p>Any domain. Any format.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="Upload any resume in PDF, DOCX, TXT, or image format"
    )

    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        if (
            not st.session_state.resume_loaded
            or st.session_state.get("file_key") != file_key
        ):
            with st.spinner("📄 Processing document..."):
                result = process_uploaded_file(uploaded_file, GROQ_API_KEY)

            if result["success"]:
                st.session_state.resume_text = result["text"]
                st.session_state.file_info = result
                st.session_state.file_key = file_key

                with st.spinner("🧠 Parsing resume with AI..."):
                    model_id = GROQ_MODELS[
                        st.session_state.selected_model
                    ]["id"]
                    parsed = parse_resume_with_llm(
                        result["text"], GROQ_API_KEY, model_id
                    )
                    st.session_state.parsed_resume = parsed

                with st.spinner("🔧 Initializing AI tools..."):
                    registry = create_tool_registry()
                    registry.set_resume_data(parsed, result["text"])
                    st.session_state.tool_registry = registry

                st.session_state.resume_loaded = True
                st.session_state.messages = []
                st.success(
                    f"✅ Loaded: {result['file_name']} "
                    f"({result['file_size_kb']} KB)"
                )
            else:
                st.error(f"❌ {result['error']}")

    # Show parsed resume info
    if st.session_state.parsed_resume:
        parsed = st.session_state.parsed_resume
        st.markdown("---")
        st.markdown("### 👤 Detected Profile")
        summary = get_resume_display_summary(parsed)
        st.markdown(summary)

        with st.expander("📋 Parsed Details"):
            st.json(parsed)

    st.markdown("---")

    # Model Selection
    st.markdown("### 🧠 AI Model")
    selected = st.selectbox(
        "Model", list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(
            st.session_state.selected_model
        ),
    )
    st.session_state.selected_model = selected
    mi = GROQ_MODELS[selected]
    st.markdown(f"""
    <div class="model-info">
        <span class="label">Model:</span> {mi['id']}<br>
        <span class="label">Speed:</span> {mi['speed']}
        <span class="label">Quality:</span> {mi['quality']}
    </div>
    """, unsafe_allow_html=True)

    if GROQ_API_KEY:
        st.success("✅ API Connected")
    else:
        st.error("❌ Add GROQ_API_KEY in Secrets")

    st.markdown("---")

    # Agent Settings
    st.markdown("### ⚙️ Settings")
    st.session_state.show_agent_trace = st.toggle(
        "Show Agent Trace", value=st.session_state.show_agent_trace
    )

    st.markdown("---")

    # Tools
    st.markdown("### 🔧 MCP Tools")
    for icon, name, desc in [
        ("📄", "resume_search", "RAG search"),
        ("📊", "skill_analyzer", "Skill gap analysis"),
        ("💼", "experience_calc", "Experience breakdown"),
        ("📝", "cover_letter", "Cover letters"),
        ("👤", "profile_summary", "Summaries/bios"),
        ("🎯", "job_matcher", "Job fit scoring"),
    ]:
        st.caption(f"{icon} {name}: {desc}")

    st.markdown("---")

    # Suggestions
    if st.session_state.resume_loaded:
        st.markdown("### 💡 Try Asking")
        name = st.session_state.parsed_resume.get("name", "this person")
        suggestions = [
            f"What is {name}'s professional summary?",
            "What are the key technical skills?",
            "Give me the complete work experience",
            "Match skills: Python, AWS, Docker, Kubernetes",
            "Generate a cover letter for Senior Engineer at Google",
            "Write a LinkedIn summary",
            "What certifications are listed?",
            "Calculate total years of experience",
            "How well does this resume fit a Data Scientist role?",
            "What are the key achievements?",
        ]
        for i, s in enumerate(suggestions):
            if st.button(f"📌 {s}", key=f"s_{i}", use_container_width=True):
                st.session_state.pending_question = s

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("📄 Upload New Resume", use_container_width=True):
        for k in ["resume_text", "parsed_resume", "resume_loaded",
                   "file_info", "tool_registry", "messages", "file_key"]:
            if k in st.session_state:
                st.session_state[k] = defaults.get(k)
        st.rerun()


# ============================================================
# MAIN
# ============================================================

col_t, col_b = st.columns([5, 1])
with col_t:
    st.markdown(
        '<h1 class="main-header">🤖 Universal AI Resume Assistant</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        "Upload any resume → Ask anything → Agentic AI + MCP Tools"
        "</p>",
        unsafe_allow_html=True
    )
with col_b:
    st.markdown("""
    <div style="text-align:right;padding-top:30px;">
        <span class="rag-badge"><span class="status-dot"></span> Agentic AI</span>
    </div>
    """, unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.warning("⚠️ Add `GROQ_API_KEY` in Settings → Secrets")
    st.info("🔗 Free key: [console.groq.com/keys](https://console.groq.com/keys)")
    st.stop()


def display_trace(steps, tools_used):
    with st.expander(
        f"🔍 Agent Trace ({len(steps)} steps, Tools: {', '.join(tools_used)})",
        expanded=False
    ):
        for i, s in enumerate(steps):
            cls = f"agent-step-{s.step_type.split('_')[0]}"
            icon = {"planning": "🎯", "tool_call": "🔧", "synthesis": "✨"}.get(
                s.step_type, "📌"
            )
            label = s.step_type.upper()
            extra = ""
            if s.step_type == "planning" and s.output_data:
                extra = f"<br>Plan: {s.output_data.get('reasoning', '')}"
                extra += f"<br>Tools: {', '.join(s.output_data.get('planned_tools', []))}"
            elif s.step_type == "tool_call":
                status = "✅" if s.success else "❌"
                extra = f" — {s.tool_name} {status}"
            st.markdown(f"""
            <div class="agent-step {cls}">
                <strong>{icon} Step {i+1}: {label}{extra}</strong> ({s.duration}s)
            </div>
            """, unsafe_allow_html=True)


# Not loaded yet
if not st.session_state.resume_loaded:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f5f7fa,#e4e8ec);padding:3rem;border-radius:1rem;text-align:center;margin:2rem 0;">
        <h2 style="color:#1e3a8a;">📄 Upload a Resume to Get Started</h2>
        <p style="font-size:1.1rem;color:#1f2937;">
            Upload <strong>any resume</strong> in any format and ask questions using
            <strong>Agentic AI</strong> with <strong>MCP tools</strong>.
        </p>
        <br>
        <p style="color:#1f2937;"><strong>Supported Formats:</strong></p>
        <ul style="text-align:left;display:inline-block;color:#374151;">
            <li>📕 <strong>PDF</strong> — Most common resume format</li>
            <li>📘 <strong>DOCX</strong> — Microsoft Word documents</li>
            <li>📝 <strong>TXT</strong> — Plain text files</li>
            <li>🖼️ <strong>Images</strong> — JPG, PNG, WEBP (OCR via AI Vision)</li>
        </ul>
        <br><br>
        <p style="color:#1f2937;"><strong>🔧 6 MCP Tools:</strong></p>
        <ul style="text-align:left;display:inline-block;color:#374151;">
            <li>📄 <strong>Resume Search</strong> — RAG semantic search</li>
            <li>📊 <strong>Skill Analyzer</strong> — Match skills vs job requirements</li>
            <li>💼 <strong>Experience Calculator</strong> — Breakdown by years/domain</li>
            <li>📝 <strong>Cover Letter Generator</strong> — Tailored cover letters</li>
            <li>👤 <strong>Profile Summary</strong> — LinkedIn/portfolio bios</li>
            <li>🎯 <strong>Job Matcher</strong> — Job fit scoring</li>
        </ul>
        <br><br>
        <p style="color:#1f2937;"><em>👈 Upload a resume using the sidebar to begin!</em></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# Display history
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
                        f'<span class="tool-badge">{t}</span>'
                        for t in msg["tools_used"]
                    ),
                    unsafe_allow_html=True
                )
            if st.session_state.show_agent_trace and msg.get("steps"):
                display_trace(msg["steps"], msg.get("tools_used", []))


# Sidebar suggestion
if "pending_question" in st.session_state:
    q = st.session_state.pending_question
    del st.session_state.pending_question

    st.session_state.messages.append({"role": "user", "content": q})

    mi = GROQ_MODELS[st.session_state.selected_model]
    agent = ResumeAgent(
        st.session_state.tool_registry, GROQ_API_KEY, mi["id"]
    )

    with st.spinner(f"🤖 Agent working with {mi['id']}..."):
        result = agent.run(q, st.session_state.messages)

    st.session_state.messages.append({
        "role": "assistant", "content": result.answer,
        "tools_used": result.tools_used, "steps": result.steps,
        "total_time": result.total_time, "model": mi["id"]
    })
    st.rerun()


# Chat input
if prompt := st.chat_input("Ask anything about the uploaded resume..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    mi = GROQ_MODELS[st.session_state.selected_model]
    agent = ResumeAgent(
        st.session_state.tool_registry, GROQ_API_KEY, mi["id"]
    )

    with st.chat_message("assistant"):
        with st.spinner(f"🤖 Agent planning + executing with **{mi['id']}**..."):
            result = agent.run(prompt, st.session_state.messages)

        st.markdown(result.answer)
        st.caption(
            f"🧠 {mi['id']} | ⚡ {result.total_time}s | "
            f"🔧 {len(result.tools_used)} tools"
        )
        if result.tools_used:
            st.markdown(
                " ".join(
                    f'<span class="tool-badge">{t}</span>'
                    for t in result.tools_used
                ),
                unsafe_allow_html=True
            )
        if st.session_state.show_agent_trace:
            display_trace(result.steps, result.tools_used)

    st.session_state.messages.append({
        "role": "assistant", "content": result.answer,
        "tools_used": result.tools_used, "steps": result.steps,
        "total_time": result.total_time, "model": mi["id"]
    })


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:.85rem;">
    Built with ❤️ using Agentic AI + MCP + RAG + Groq | 100% Free
</div>

""", unsafe_allow_html=True)
