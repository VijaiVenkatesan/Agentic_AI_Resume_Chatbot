"""
Universal Agentic AI Resume Chatbot - Full Featured
File preview, JD comparison, high contrast UI, better extraction
"""

import streamlit as st
import os
import time
from typing import List, Dict

from document_processor import process_uploaded_file
from resume_parser import parse_resume_with_llm, get_resume_display_summary
from mcp_tools import create_tool_registry
from agent import ResumeAgent

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Universal AI Resume Assistant",
    page_icon="🤖", layout="wide",
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


# ── HIGH CONTRAST CSS ──
st.markdown("""
<style>
    /* Force readable text everywhere */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stText, p, span, li, label, div {
        color: #e2e8f0 !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }

    .main-header {
        font-size: 2.3rem; font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.3rem;
    }
    .sub-header { font-size: 1rem; color: #94a3b8 !important; text-align: center; margin-bottom: 1.5rem; }

    /* Upload card */
    .upload-box {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #fff !important; padding: 1.2rem; border-radius: 0.75rem;
        margin-bottom: 0.75rem; box-shadow: 0 4px 12px rgba(79,70,229,0.3);
    }
    .upload-box h3 { color: #fff !important; margin: 0 0 0.3rem; font-size: 1.1rem; }
    .upload-box p { color: #e0e7ff !important; font-size: 0.85rem; margin: 0.15rem 0; }

    /* Profile card */
    .profile-box {
        background: #1e293b; border: 1px solid #475569; border-radius: 0.75rem;
        padding: 1rem; margin: 0.5rem 0;
    }
    .profile-box p { color: #cbd5e1 !important; font-size: 0.85rem; margin: 0.2rem 0; }
    .profile-box strong { color: #f8fafc !important; }

    /* JD card */
    .jd-box {
        background: linear-gradient(135deg, #0f766e, #0d9488);
        color: #fff !important; padding: 1rem; border-radius: 0.75rem;
        margin: 0.5rem 0;
    }
    .jd-box h4 { color: #fff !important; margin: 0 0 0.3rem; }
    .jd-box p { color: #ccfbf1 !important; font-size: 0.85rem; }

    /* Model card */
    .model-box {
        background: #1e293b; border: 1px solid #475569; padding: 0.75rem;
        border-radius: 0.5rem; margin-top: 0.4rem;
    }
    .model-box p { color: #94a3b8 !important; font-size: 0.8rem; margin: 0.1rem 0; }
    .model-box .lbl { color: #a78bfa !important; font-weight: 600; }

    /* Agent trace steps */
    .trace-step {
        padding: 10px 14px; margin: 6px 0; border-radius: 6px;
        font-size: 0.85rem; line-height: 1.5;
    }
    .trace-plan { background: #fef3c7; border-left: 4px solid #f59e0b; color: #78350f !important; }
    .trace-tool { background: #d1fae5; border-left: 4px solid #10b981; color: #064e3b !important; }
    .trace-synth { background: #dbeafe; border-left: 4px solid #3b82f6; color: #1e3a5f !important; }
    .trace-step strong, .trace-step span { color: inherit !important; }

    /* Tool badges */
    .tbadge {
        display: inline-block; background: rgba(167,139,250,0.15);
        color: #c4b5fd !important; padding: 3px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin: 2px;
        border: 1px solid rgba(167,139,250,0.3);
    }

    /* Status badge */
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(167,139,250,0.15); color: #c4b5fd !important;
        padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
    .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

    /* Welcome card */
    .welcome-box {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569; padding: 2.5rem; border-radius: 1rem;
        text-align: center; margin: 2rem 0;
    }
    .welcome-box h2 { color: #f1f5f9 !important; }
    .welcome-box p { color: #cbd5e1 !important; }
    .welcome-box li { color: #94a3b8 !important; text-align: left; }
    .welcome-box strong { color: #c4b5fd !important; }
    .welcome-box em { color: #93c5fd !important; }

    /* Chat messages */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span { color: #e2e8f0 !important; }
    [data-testid="stChatMessage"] strong { color: #c4b5fd !important; }
    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 { color: #f1f5f9 !important; }

    /* Metrics */
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.3rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

    /* Expander */
    .streamlit-expanderHeader { color: #cbd5e1 !important; font-weight: 600 !important; }
    details summary span { color: #cbd5e1 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0f172a !important; }
    section[data-testid="stSidebar"] p { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] .stCaption { color: #64748b !important; }

    /* Buttons */
    .stButton > button { width: 100%; border-radius: 0.5rem; font-size: 0.82rem; }

    /* Footer */
    .app-footer { text-align: center; font-size: 0.8rem; }
    .app-footer, .app-footer a { color: #64748b !important; }
    .app-footer a:hover { color: #93c5fd !important; }

    /* Hide streamlit chrome */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──
defaults = {
    "messages": [], "selected_model": DEFAULT_MODEL, "show_agent_trace": True,
    "resume_text": "", "parsed_resume": None, "resume_loaded": False,
    "file_info": None, "tool_registry": None,
    "jd_text": "", "jd_loaded": False, "jd_file_info": None,
    "file_key": None, "jd_file_key": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

GROQ_API_KEY = get_groq_key()

# ══════════════════════════════════════
#           SIDEBAR
# ══════════════════════════════════════
with st.sidebar:
    # ── Resume Upload ──
    st.markdown("""<div class="upload-box">
        <h3>📄 Upload Resume</h3>
        <p>PDF, DOCX, TXT, JPG, PNG — Any domain</p>
    </div>""", unsafe_allow_html=True)

    resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed", key="resume_up"
    )

    if resume_file:
        fk = f"{resume_file.name}_{resume_file.size}"
        if not st.session_state.resume_loaded or st.session_state.file_key != fk:
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
                st.success(f"✅ {result['file_name']} ({result['file_size_kb']} KB)")
            else:
                st.error(f"❌ {result['error']}")

    # ── File Preview ──
    if st.session_state.resume_loaded and st.session_state.file_info:
        fi = st.session_state.file_info
        preview = fi.get("preview", {})

        with st.expander("👁️ View Uploaded Resume", expanded=False):
            if preview.get("can_preview"):
                ptype = preview["type"]
                if ptype in [".jpg", ".jpeg", ".png", ".webp"]:
                    st.image(
                        f"data:{preview['mime_type']};base64,{preview['preview_data']}",
                        caption=fi["file_name"], use_container_width=True
                    )
                elif ptype == ".pdf":
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{preview["preview_data"]}" '
                        f'width="100%" height="400px" style="border:none;border-radius:8px;"></iframe>',
                        unsafe_allow_html=True
                    )
                    if preview.get("page_count"):
                        st.caption(f"📄 {preview['page_count']} pages")
                elif ptype in [".txt", ".md", ".text"]:
                    st.code(preview["preview_data"][:2000], language=None)
                elif ptype in [".docx", ".doc"]:
                    st.text_area("Content Preview", preview["preview_data"][:2000],
                                height=250, disabled=True)
            st.caption(f"📁 {fi['file_name']} · {fi['file_size_kb']} KB · {fi['file_type'].upper()}")

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

        profile_html = f'<div class="profile-box"><p><strong>{name}</strong></p>'
        if role:
            profile_html += f'<p>💼 {role}'
            if company:
                profile_html += f' at {company}'
            profile_html += '</p>'
        if loc:
            profile_html += f'<p>📍 {loc[:60]}</p>'
        if exp:
            profile_html += f'<p>📅 ~{exp} years exp (as of 2026)</p>'
        if email:
            profile_html += f'<p>📧 {email}</p>'
        if phone:
            profile_html += f'<p>📞 {phone}</p>'
        profile_html += '</div>'
        st.markdown(profile_html, unsafe_allow_html=True)

        with st.expander("📋 Full Parsed Data"):
            st.json(p)

    st.markdown("---")

    # ── JD Upload (Optional) ──
    st.markdown("""<div class="jd-box">
        <h4>📋 Job Description (Optional)</h4>
        <p>Upload or paste a JD to compare</p>
    </div>""", unsafe_allow_html=True)

    jd_file = st.file_uploader(
        "Upload JD", type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed", key="jd_up"
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
                if st.session_state.tool_registry:
                    st.session_state.tool_registry.set_jd_text(jd_result["text"])
                st.success(f"✅ JD: {jd_result['file_name']}")
            else:
                st.error(f"❌ {jd_result['error']}")

    if st.session_state.jd_loaded:
        with st.expander("👁️ View Job Description"):
            st.text_area("JD", st.session_state.jd_text[:2000], height=200, disabled=True)

    if not st.session_state.jd_loaded:
        jd_paste = st.text_area(
            "Or paste JD text:", height=100, key="jd_paste",
            placeholder="Paste job description here (optional)..."
        )
        if jd_paste and len(jd_paste.strip()) > 50:
            st.session_state.jd_text = jd_paste
            st.session_state.jd_loaded = True
            if st.session_state.tool_registry:
                st.session_state.tool_registry.set_jd_text(jd_paste)
            st.success("✅ JD text loaded")

    st.markdown("---")

    # ── Model Selection ──
    st.markdown("### 🧠 AI Model")
    sel = st.selectbox(
        "Model", list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = sel
    mi = GROQ_MODELS[sel]
    st.markdown(f"""<div class="model-box">
        <p><span class="lbl">Model:</span> {mi['id']}</p>
        <p><span class="lbl">Speed:</span> {mi['speed']}
        <span class="lbl">Quality:</span> {mi['quality']}</p>
    </div>""", unsafe_allow_html=True)

    if GROQ_API_KEY:
        st.success("✅ API Connected")
    else:
        st.error("❌ Add GROQ_API_KEY in Secrets")

    st.markdown("---")

    # ── Settings ──
    st.session_state.show_agent_trace = st.toggle(
        "🔍 Show Agent Trace", value=st.session_state.show_agent_trace
    )

    st.markdown("---")

    # ── MCP Tools ──
    st.markdown("### 🔧 MCP Tools")
    for icon, name, desc in [
        ("📄", "resume_search", "RAG search"),
        ("📊", "skill_analyzer", "Skill gaps"),
        ("💼", "experience_calc", "Experience"),
        ("📝", "cover_letter", "Cover letters"),
        ("👤", "profile_summary", "Summaries"),
        ("🎯", "jd_matcher", "JD comparison"),
    ]:
        st.caption(f"{icon} **{name}**: {desc}")

    st.markdown("---")

    # ── Suggestions ──
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
            "Match skills: Python, AWS, Docker, Kubernetes",
            "Write a cover letter for Senior Engineer at Google",
            "Write a LinkedIn summary",
            "What certifications are listed?",
            "What are the key achievements with numbers?",
        ]

        if st.session_state.jd_loaded:
            suggestions.insert(0, "Compare this resume against the job description")
            suggestions.insert(1, "How well does this candidate fit the JD?")

        for i, s in enumerate(suggestions):
            if st.button(f"📌 {s}", key=f"s_{i}", use_container_width=True):
                st.session_state.pending_question = s

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("📄 New Resume", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()


# ══════════════════════════════════════
#           MAIN CONTENT
# ══════════════════════════════════════

col_t, col_b = st.columns([5, 1])
with col_t:
    st.markdown(
        '<h1 class="main-header">🤖 Universal AI Resume Assistant</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Upload any resume → Ask anything → Agentic AI + MCP Tools</p>',
        unsafe_allow_html=True
    )
with col_b:
    st.markdown("""<div style="text-align:right;padding-top:28px;">
        <span class="status-badge"><span class="pulse-dot"></span> Agentic AI</span>
    </div>""", unsafe_allow_html=True)

# API key check
if not GROQ_API_KEY:
    st.warning("⚠️ Add `GROQ_API_KEY` in Settings → Secrets")
    st.info("🔗 Free key: [console.groq.com/keys](https://console.groq.com/keys)")
    st.stop()


# ── Agent Trace Display ──
def show_trace(steps, tools_used):
    with st.expander(
        f"🔍 Agent Trace ({len(steps)} steps · Tools: {', '.join(tools_used)})",
        expanded=False
    ):
        for i, s in enumerate(steps):
            icon = {"planning": "🎯", "tool_call": "🔧", "synthesis": "✨"}.get(s.step_type, "📌")
            css = {"planning": "trace-plan", "tool_call": "trace-tool", "synthesis": "trace-synth"}.get(
                s.step_type, "trace-tool")
            extra = ""
            if s.step_type == "planning" and s.output_data:
                extra = f"<br>Plan: {s.output_data.get('reasoning', '')}"
                extra += f"<br>Tools: {', '.join(s.output_data.get('planned_tools', []))}"
            elif s.step_type == "tool_call":
                extra = f" — {s.tool_name} {'✅' if s.success else '❌'}"

            st.markdown(f"""<div class="trace-step {css}">
                <strong>{icon} Step {i+1}: {s.step_type.upper()}{extra}</strong>
                <span>({s.duration}s)</span>
            </div>""", unsafe_allow_html=True)


# ── Welcome Screen ──
if not st.session_state.resume_loaded:
    st.markdown("""
    <div class="welcome-box">
        <h2>📄 Upload a Resume to Get Started</h2>
        <p>Upload <strong>any resume</strong> in any format. Optionally add a <strong>Job Description</strong> for comparison.</p>
        <br>
        <p><strong>Supported Formats:</strong></p>
        <ul>
            <li>📕 <strong>PDF</strong> — Most common resume format</li>
            <li>📘 <strong>DOCX</strong> — Microsoft Word documents</li>
            <li>📝 <strong>TXT</strong> — Plain text files</li>
            <li>🖼️ <strong>Images</strong> — JPG, PNG, WEBP (AI Vision OCR)</li>
        </ul>
        <br>
        <p><strong>🔧 6 MCP Tools + Optional JD Matching:</strong></p>
        <ul>
            <li>📄 Resume Search — RAG semantic search</li>
            <li>📊 Skill Analyzer — Match skills vs requirements</li>
            <li>💼 Experience Calculator — Years breakdown (as of 2026)</li>
            <li>📝 Cover Letter Generator — Tailored cover letters</li>
            <li>👤 Profile Summary — LinkedIn / portfolio bios</li>
            <li>🎯 JD Matcher — Resume vs Job Description comparison</li>
        </ul>
        <br>
        <p><em>👈 Upload a resume using the sidebar to begin!</em></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Display Chat History ──
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
                    " ".join(f'<span class="tbadge">{t}</span>' for t in msg["tools_used"]),
                    unsafe_allow_html=True
                )

            if st.session_state.show_agent_trace and msg.get("steps"):
                show_trace(msg["steps"], msg.get("tools_used", []))


# ── Agent Runner ──
def run_agent(question: str):
    mi = GROQ_MODELS[st.session_state.selected_model]
    jd = st.session_state.jd_text if st.session_state.jd_loaded else ""
    agent = ResumeAgent(
        st.session_state.tool_registry, GROQ_API_KEY, mi["id"], jd_text=jd
    )
    return agent.run(question, st.session_state.messages), mi


# ── Handle Sidebar Suggestion Click ──
if "pending_question" in st.session_state:
    q = st.session_state.pending_question
    del st.session_state.pending_question

    st.session_state.messages.append({"role": "user", "content": q})
    with st.spinner("🤖 Agent working..."):
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
            f"🧠 {mi['id']} | ⚡ {result.total_time}s | 🔧 {len(result.tools_used)} tools"
        )
        if result.tools_used:
            st.markdown(
                " ".join(f'<span class="tbadge">{t}</span>' for t in result.tools_used),
                unsafe_allow_html=True
            )
        if st.session_state.show_agent_trace:
            show_trace(result.steps, result.tools_used)

    st.session_state.messages.append({
        "role": "assistant", "content": result.answer,
        "tools_used": result.tools_used, "steps": result.steps,
        "total_time": result.total_time, "model": mi["id"]
    })


# ── Footer ──
st.markdown("---")
st.markdown(
    '<p class="app-footer">'
    'Built with ❤️ using Agentic AI + MCP + RAG + Groq · 100% Free · Year: 2026'
    '</p>',
    unsafe_allow_html=True
)
