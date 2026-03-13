"""
Universal Agentic AI Resume Chatbot - V4
Features: Full-screen preview modal, open in new tab, agent trace, 100MB upload, better UI
"""

import streamlit as st
import os
import time
import base64
from typing import List, Dict

from document_processor import process_uploaded_file
from resume_parser import parse_resume_with_llm, get_resume_display_summary
from mcp_tools import create_tool_registry
from agent import ResumeAgent

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

    /* ── Model Card ── */
    .model-box {
        background: #1e293b; border: 1px solid #475569;
        padding: 0.75rem; border-radius: 10px; margin-top: 0.4rem;
    }
    .model-box p { color: #94a3b8 !important; font-size: 0.8rem; margin: 0.1rem 0; }
    .model-box .lbl { color: #a78bfa !important; font-weight: 700; }

    /* ── AGENT TRACE (HIGH CONTRAST FIX) ── */
    .trace-step {
        padding: 14px 18px; margin: 8px 0; border-radius: 8px;
        font-size: 0.95rem; line-height: 1.6;
    }
    .trace-plan {
        background: #fef9c3 !important;
        border-left: 5px solid #ca8a04;
    }
    .trace-plan strong, .trace-plan span, .trace-plan br,
    .trace-plan { color: #713f12 !important; }

    .trace-tool {
        background: #bbf7d0 !important;
        border-left: 5px solid #16a34a;
    }
    .trace-tool strong, .trace-tool span, .trace-tool br,
    .trace-tool { color: #14532d !important; }

    .trace-synth {
        background: #bfdbfe !important;
        border-left: 5px solid #2563eb;
    }
    .trace-synth strong, .trace-synth span, .trace-synth br,
    .trace-synth { color: #1e3a8a !important; }

    /* ── Tool Badges ── */
    .tbadge {
        display: inline-block;
        background: rgba(167,139,250,0.2);
        color: #c4b5fd !important;
        padding: 4px 12px; border-radius: 14px;
        font-size: 0.78rem; font-weight: 600; margin: 3px;
        border: 1px solid rgba(167,139,250,0.35);
    }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(167,139,250,0.15);
        color: #c4b5fd !important;
        padding: 6px 16px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
        border: 1px solid rgba(167,139,250,0.3);
    }
    .pulse-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: #22c55e;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }

    /* ── File Preview Box ── */
    .preview-box {
        background: #1e293b; border: 1px solid #475569;
        border-radius: 10px; padding: 12px; margin: 8px 0;
        max-height: 450px; overflow-y: auto;
    }
    .preview-box p, .preview-box span { color: #cbd5e1 !important; font-size: 0.85rem; }
    .preview-label {
        color: #94a3b8 !important; font-size: 0.75rem;
        margin-top: 6px; display: block;
    }

    /* ── Welcome Card ── */
    .welcome-box {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        padding: 2.5rem 2rem; border-radius: 16px;
        text-align: center; margin: 2rem auto; max-width: 800px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .welcome-box h2 { color: #f1f5f9 !important; font-size: 1.8rem; }
    .welcome-box p { color: #cbd5e1 !important; font-size: 1rem; }
    .welcome-box li { color: #94a3b8 !important; text-align: left; margin: 4px 0; }
    .welcome-box strong { color: #c4b5fd !important; }
    .welcome-box em { color: #93c5fd !important; }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
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
        padding: 0.5rem; transition: all 0.2s;
        border: 1px solid #475569;
    }
    .stButton > button:hover {
        border-color: #7c3aed; box-shadow: 0 0 12px rgba(124,58,237,0.3);
    }

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

    /* ── FULL-SCREEN PREVIEW MODAL ── */
    .preview-modal-header {
        background: linear-gradient(135deg, #4338ca, #7c3aed);
        padding: 16px 24px;
        border-radius: 12px 12px 0 0;
        margin-top: 10px;
    }
    
    .preview-modal-title {
        color: #fff !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .preview-modal-body {
        background: #0f172a;
        border: 1px solid #475569;
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 20px;
        min-height: 500px;
    }
    
    .preview-content {
        background: #1e293b;
        color: #e2e8f0 !important;
        padding: 20px;
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.7;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 600px;
        overflow: auto;
    }
    
    .newtab-link {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: #fff !important;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .newtab-link:hover {
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ──
defaults = {
    "messages": [], "selected_model": DEFAULT_MODEL, "show_agent_trace": True,
    "resume_text": "", "parsed_resume": None, "resume_loaded": False,
    "file_info": None, "tool_registry": None,
    "jd_text": "", "jd_loaded": False, "jd_file_info": None,
    "file_key": None, "jd_file_key": None,
    "show_preview_modal": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

GROQ_API_KEY = get_groq_key()


# ═══════════════════════════════════════════
#              SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    # ── Resume Upload ──
    st.markdown("""<div class="upload-box">
        <h3>📄 Upload Resume</h3>
        <p>PDF · DOCX · TXT · JPG · PNG · Up to 100MB</p>
        <p>Works for any domain · Any format</p>
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
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")

    # ── Show success after rerun ──
    if st.session_state.resume_loaded and st.session_state.file_info:
        fi = st.session_state.file_info
        st.success(f"✅ {fi['file_name']} ({fi['file_size_kb']} KB)")

    # ── File Preview Buttons ──
    if st.session_state.resume_loaded and st.session_state.file_info:
        fi = st.session_state.file_info
        preview = fi.get("preview", {})
        ptype = preview.get("type", "")
        
        col_prev1, col_prev2 = st.columns([1, 1])
        with col_prev1:
            if st.button("👁️ View Resume", key="open_preview", use_container_width=True):
                st.session_state.show_preview_modal = True
                st.rerun()
        with col_prev2:
            # Open in new tab button (for PDFs and images)
            if ptype in [".pdf", ".jpg", ".jpeg", ".png", ".webp"] and preview.get("preview_data"):
                mime = preview.get("mime_type", "application/pdf")
                b64_data = preview["preview_data"]
                st.markdown(
                    f'''<a href="data:{mime};base64,{b64_data}" 
                        target="_blank" 
                        class="newtab-link"
                        style="
                            display: inline-block;
                            width: 100%;
                            text-align: center;
                            background: linear-gradient(135deg, #059669, #10b981);
                            color: #fff;
                            padding: 8px 12px;
                            border-radius: 8px;
                            font-size: 0.82rem;
                            text-decoration: none;
                            font-weight: 500;
                        ">🔗 New Tab</a>''',
                    unsafe_allow_html=True
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
            html += f'<p>📅 ~{exp} years exp (as of 2026)</p>'
        if email:
            html += f'<p>📧 {email}</p>'
        if phone:
            html += f'<p>📞 {phone}</p>'
        if linkedin:
            html += f'<p>🔗 {linkedin}</p>'
        if github:
            html += f'<p>💻 {github}</p>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        with st.expander("📋 Full Parsed JSON"):
            st.json(p)

    st.markdown("---")

    # ── JD Upload (Optional) ──
    st.markdown("""<div class="jd-box">
        <h4>📋 Job Description (Optional)</h4>
        <p>Upload or paste JD to compare · Up to 100MB</p>
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
                st.rerun()
            else:
                st.error(f"❌ {jd_result['error']}")

    if st.session_state.jd_loaded:
        st.success("✅ Job Description loaded")
        with st.expander("👁️ View Job Description"):
            st.text_area("JD", st.session_state.jd_text[:5000], height=200,
                        disabled=True, label_visibility="collapsed")

    if not st.session_state.jd_loaded:
        jd_paste = st.text_area(
            "Or paste JD text:", height=120, key="jd_paste",
            placeholder="Paste job description here (optional)...",
            max_chars=50000,
        )
        if jd_paste and len(jd_paste.strip()) > 50:
            st.session_state.jd_text = jd_paste
            st.session_state.jd_loaded = True
            if st.session_state.tool_registry:
                st.session_state.tool_registry.set_jd_text(jd_paste)
            st.rerun()

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
    st.markdown("### ⚙️ Settings")
    st.session_state.show_agent_trace = st.toggle(
        "🔍 Show Agent Trace", value=st.session_state.show_agent_trace
    )

    st.markdown("---")

    # ── MCP Tools ──
    st.markdown("### 🔧 MCP Tools")
    for icon, tname, desc in [
        ("📄", "resume_search", "RAG semantic search"),
        ("📊", "skill_analyzer", "Skill gap analysis"),
        ("💼", "experience_calc", "Experience breakdown"),
        ("📝", "cover_letter", "Cover letters"),
        ("👤", "profile_summary", "Professional bios"),
        ("🎯", "jd_matcher", "JD comparison"),
    ]:
        st.caption(f"{icon} **{tname}**: {desc}")

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
            "What is the educational background?",
        ]

        if st.session_state.jd_loaded:
            suggestions.insert(0, "Compare this resume against the job description")
            suggestions.insert(1, "How well does this candidate fit the JD?")

        for i, s in enumerate(suggestions):
            if st.button(f"📌 {s}", key=f"s_{i}", use_container_width=True):
                st.session_state.pending_question = s
                st.rerun()

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


# ═══════════════════════════════════════════
#        FULL-SCREEN PREVIEW MODAL
# ═══════════════════════════════════════════

if st.session_state.show_preview_modal and st.session_state.file_info:
    fi = st.session_state.file_info
    preview = fi.get("preview", {})
    ptype = preview.get("type", "")
    
    # Modal container
    st.markdown("""
        <div style="
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 16px;
            margin-bottom: 20px;
        ">
    """, unsafe_allow_html=True)
    
    # Header with close button
    header_col1, header_col2 = st.columns([8, 2])
    with header_col1:
        st.markdown(f"""
            <div class="preview-modal-header">
                <h3 class="preview-modal-title">
                    📄 {fi['file_name']}
                    <span style="color: #e0e7ff; font-size: 0.85rem; font-weight: 400; margin-left: 10px;">
                        {fi['file_size_kb']} KB · {fi['file_type'].upper()}
                    </span>
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    with header_col2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("❌ Close Preview", key="close_preview", use_container_width=True):
            st.session_state.show_preview_modal = False
            st.rerun()
    
    # Preview content container
    st.markdown("""
        <div class="preview-modal-body">
    """, unsafe_allow_html=True)
    
    # Render preview based on file type
    if ptype in [".jpg", ".jpeg", ".png", ".webp"] and preview.get("preview_data"):
        # Image preview
        mime = preview.get("mime_type", "image/png")
        st.image(
            f"data:{mime};base64,{preview['preview_data']}",
            caption=fi["file_name"],
            use_container_width=True
        )
    
    elif ptype == ".pdf" and preview.get("preview_data"):
        # PDF preview using iframe
        b64_pdf = preview["preview_data"]
        pdf_display = f'''
            <iframe 
                src="data:application/pdf;base64,{b64_pdf}#toolbar=1&navpanes=1&scrollbar=1" 
                width="100%" 
                height="650px" 
                style="border: none; border-radius: 8px; background: #fff;"
            ></iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Page count info
        if preview.get("page_count"):
            st.caption(f"📑 {preview['page_count']} pages")
    
    elif ptype in [".docx", ".doc"] and preview.get("preview_data"):
        # DOCX preview as formatted text
        st.markdown(f"""
            <pre class="preview-content">{preview['preview_data']}</pre>
        """, unsafe_allow_html=True)
    
    elif ptype in [".txt", ".md", ".text"] and preview.get("preview_data"):
        # Text preview
        st.markdown(f"""
            <pre class="preview-content">{preview['preview_data']}</pre>
        """, unsafe_allow_html=True)
    
    else:
        # Fallback - show extracted text
        st.markdown(f"""
            <pre class="preview-content">{st.session_state.resume_text}</pre>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer with actions
    footer_col1, footer_col2, footer_col3 = st.columns([4, 4, 4])
    
    with footer_col1:
        st.caption(f"📁 **File:** {fi['file_name']}")
    
    with footer_col2:
        st.caption(f"📊 **Size:** {fi['file_size_kb']} KB")
    
    with footer_col3:
        # New tab link for PDFs and images
        if ptype in [".pdf", ".jpg", ".jpeg", ".png", ".webp"] and preview.get("preview_data"):
            mime = preview.get("mime_type", "application/pdf")
            b64_data = preview["preview_data"]
            st.markdown(
                f'''<a href="data:{mime};base64,{b64_data}" 
                    target="_blank" 
                    class="newtab-link">🔗 Open in New Tab</a>''',
                unsafe_allow_html=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Back to chat button
    if st.button("⬅️ Back to Chat", key="back_to_chat", use_container_width=True):
        st.session_state.show_preview_modal = False
        st.rerun()
    
    # Stop here to show only the preview
    st.stop()


# ═══════════════════════════════════════════
#              MAIN CONTENT
# ═══════════════════════════════════════════

col_t, col_b = st.columns([5, 1])
with col_t:
    st.markdown(
        '<h1 class="main-header">🤖 Universal AI Resume Assistant</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        'Upload any resume → Ask anything → Agentic AI + MCP Tools'
        '</p>',
        unsafe_allow_html=True
    )
with col_b:
    st.markdown("""<div style="text-align:right;padding-top:28px;">
        <span class="status-badge">
            <span class="pulse-dot"></span> Agentic AI
        </span>
    </div>""", unsafe_allow_html=True)

# API key check
if not GROQ_API_KEY:
    st.warning("⚠️ Add `GROQ_API_KEY` in Settings → Secrets")
    st.info("🔗 Free: [console.groq.com/keys](https://console.groq.com/keys)")
    st.stop()


# ── Agent Trace Display (HIGH CONTRAST) ──
def show_trace(steps, tools_used):
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

            # Build content with explicit dark colors
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
                <span style="opacity:0.7;"> ({s.duration}s)</span>
            </div>
            """, unsafe_allow_html=True)


# ── Welcome Screen ──
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
        <p><strong>🔧 6 MCP Tools + Optional JD Matching:</strong></p>
        <ul>
            <li>📄 Resume Search — RAG semantic search</li>
            <li>📊 Skill Analyzer — Match skills vs requirements</li>
            <li>💼 Experience Calculator — Years breakdown (as of 2026)</li>
            <li>📝 Cover Letter Generator — Tailored cover letters</li>
            <li>👤 Profile Summary — LinkedIn / portfolio bios</li>
            <li>🎯 JD Matcher — Resume vs Job Description scoring</li>
        </ul>
        <br>
        <p><em>👈 Upload a resume using the sidebar!</em></p>
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
                    " ".join(
                        f'<span class="tbadge">{t}</span>'
                        for t in msg["tools_used"]
                    ),
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


# ── Handle Sidebar Suggestion ──
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


# ── Footer ──
st.markdown("---")
st.markdown(
    '<p class="app-footer">'
    'Built with ❤️ using Agentic AI + MCP + RAG + Groq · '
    '100% Free · '
    '<a href="https://github.com" target="_blank">GitHub</a>'
    '</p>',
    unsafe_allow_html=True
)
