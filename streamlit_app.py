"""
Universal Agentic AI Resume Chatbot - V7 (Enterprise Edition)
Features:
- NEW: Bulk Resume Ranking Mode for comparing multiple resumes against one JD.
- NEW: UI toggle to switch between single-chat and bulk-ranking modes.
- Remove/discard buttons, full-screen preview, agent trace, 100MB upload.
"""

import streamlit as st
import os
import time
import base64
import pandas as pd
from typing import List, Dict

from document_processor import process_uploaded_file
from resume_parser import parse_resume_with_llm, get_resume_display_summary
from mcp_tools import create_tool_registry, JDMatcherTool
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
    /* ── AGENT TRACE (FINAL HIGH CONTRAST FIX) ── */
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
    /* ── Welcome Card ── */
    .welcome-box {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569; padding: 2.5rem 2rem; border-radius: 16px;
        text-align: center; margin: 2rem auto; max-width: 800px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span { color: #e2e8f0 !important; }
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #0f172a !important; }
    /* ── Buttons ── */
    .stButton > button { border: 1px solid #475569; }
    /* ── Hide Streamlit Chrome ── */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: #0f172a !important; }

    /* --- NEW: Results Table --- */
    .results-table {
        border-collapse: collapse; width: 100%; margin-top: 20px;
        font-size: 0.9rem;
    }
    .results-table th, .results-table td {
        border: 1px solid #475569;
        padding: 12px 15px;
        text-align: left;
    }
    .results-table th {
        background-color: #334155;
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    .results-table tr:nth-child(even) { background-color: #1e293b; }
    .results-table tr:hover { background-color: #4338ca; }
    .score-bar-container {
        width: 100%; background-color: #334155; border-radius: 4px;
        height: 18px;
    }
    .score-bar {
        height: 100%; border-radius: 4px;
        background: linear-gradient(90deg, #34d399, #60a5fa);
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE DEFAULTS ──
def get_default_state():
    return {
        "app_mode": "Single Resume Chat",
        "messages": [], "selected_model": DEFAULT_MODEL, "show_agent_trace": True,
        "resume_text": "", "parsed_resume": None, "resume_loaded": False,
        "file_info": None, "tool_registry": None, "jd_text": "", "jd_loaded": False,
        "jd_file_info": None, "file_key": None, "jd_file_key": None, "jd_source": None,
        "show_preview_modal": False, "raw_file_bytes": None, "raw_file_name": None,
        "raw_file_type": None,
        # New state for bulk processing
        "bulk_resumes": [], "bulk_results": None, "bulk_processing_complete": False,
    }

# Initialize session state
for k, v in get_default_state().items():
    if k not in st.session_state:
        st.session_state[k] = v

GROQ_API_KEY = get_groq_key()

# ═══════════════════════════════════════════
#          HELPER & BULK PROCESSING FUNCTIONS
# ═══════════════════════════════════════════

def reset_chat_state():
    """Resets states related to a single resume chat"""
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
    
def reset_bulk_state():
    """Resets states for bulk processing"""
    st.session_state.bulk_resumes = []
    st.session_state.bulk_results = None
    st.session_state.bulk_processing_complete = False

def reset_jd_state():
    st.session_state.jd_text = ""
    st.session_state.jd_loaded = False
    st.session_state.jd_file_info = None
    st.session_state.jd_file_key = None
    st.session_state.jd_source = None

def run_bulk_comparison():
    """The core function to process multiple resumes against a single JD."""
    if not st.session_state.bulk_resumes or not st.session_state.jd_loaded:
        st.error("Please upload resumes and a Job Description first.")
        return

    results = []
    total_files = len(st.session_state.bulk_resumes)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(st.session_state.bulk_resumes):
        file.seek(0)
        status_text.info(f"Processing {i+1}/{total_files}: {file.name}...")
        
        # 1. Process and Parse each resume
        doc_result = process_uploaded_file(file, GROQ_API_KEY)
        if not doc_result["success"]:
            results.append({"candidate_name": file.name, "overall_fit_score": 0, "recommendation": "🔴 Error: Could not read file", "strengths": [], "gaps": [doc_result['error']]})
            continue

        mid = GROQ_MODELS[st.session_state.selected_model]["id"]
        parsed_resume = parse_resume_with_llm(doc_result["text"], GROQ_API_KEY, mid)

        # 2. Create a dedicated tool registry for this resume
        registry = create_tool_registry()
        registry.set_resume_data(parsed_resume, doc_result["text"])
        
        # 3. Execute the JD Matcher tool
        jd_matcher = registry._tools.get("jd_matcher")
        if jd_matcher:
            match_result = jd_matcher.execute(jd_text=st.session_state.jd_text)
            if match_result.success:
                results.append(match_result.data)
            else:
                results.append({"candidate_name": parsed_resume.get("name", file.name), "overall_fit_score": 0, "recommendation": "🔴 Error during analysis", "strengths": [], "gaps": [match_result.error]})
        
        progress_bar.progress((i + 1) / total_files)

    status_text.success("✅ All resumes processed!")
    # Sort results by score, descending
    results.sort(key=lambda x: x.get("overall_fit_score", 0), reverse=True)
    st.session_state.bulk_results = results
    st.session_state.bulk_processing_complete = True

# ═══════════════════════════════════════════
#              SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Application Mode")
    app_mode = st.radio(
        "Choose Mode",
        ["Single Resume Chat", "Bulk Resume Ranking"],
        key="app_mode",
        label_visibility="collapsed"
    )
    st.markdown("---")

    # ------------------- SINGLE RESUME CHAT MODE -------------------
    if app_mode == "Single Resume Chat":
        st.markdown("""<div class="upload-box">
            <h3>📄 Upload Resume</h3>
            <p>PDF, DOCX, TXT, JPG, PNG · Max 100MB</p>
        </div>""", unsafe_allow_html=True)
        
        if not st.session_state.resume_loaded:
            resume_file = st.file_uploader(
                "Upload Resume", type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed", key="resume_uploader"
            )
            if resume_file:
                fk = f"{resume_file.name}_{resume_file.size}"
                if not st.session_state.resume_loaded or st.session_state.file_key != fk:
                    reset_chat_state() # Reset before loading new file
                    with st.spinner("📄 Processing document..."):
                        result = process_uploaded_file(resume_file, GROQ_API_KEY)
                    
                    if result["success"]:
                        st.session_state.resume_text = result["text"]
                        st.session_state.file_info = result
                        st.session_state.file_key = fk
                        st.session_state.raw_file_bytes = result['file_bytes']
                        st.session_state.raw_file_name = result['file_name']
                        st.session_state.raw_file_type = result['file_type']

                        with st.spinner("🧠 Parsing resume with AI..."):
                            mid = GROQ_MODELS[st.session_state.selected_model]["id"]
                            parsed = parse_resume_with_llm(result["text"], GROQ_API_KEY, mid)
                            st.session_state.parsed_resume = parsed
                        with st.spinner("🔧 Initializing AI tools..."):
                            reg = create_tool_registry()
                            reg.set_resume_data(parsed, result["text"])
                            st.session_state.tool_registry = reg
                        st.session_state.resume_loaded = True
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
        else:
            # Clear state if file is removed from uploader UI
            if st.session_state.resume_loaded:
                reset_chat_state()
                st.rerun()

        if st.session_state.resume_loaded:
            fi = st.session_state.file_info
            col_file, col_remove = st.columns([5, 1])
            with col_file: st.success(f"✅ {fi['file_name']}")
            with col_remove:
                if st.button("❌", key="remove_resume", help="Remove resume"):
                    reset_chat_state()
                    reset_jd_state()
                    st.rerun()
            st.caption(f"📊 {fi['file_size_kb']} KB · {fi['file_type'].upper().replace('.', '')}")
            if st.button("👁️ View Full Resume", key="open_preview", use_container_width=True):
                st.session_state.show_preview_modal = True
                st.rerun()
        
        if st.session_state.parsed_resume:
            st.markdown("---")
            st.markdown("### 👤 Detected Profile")
            st.markdown(get_resume_display_summary(st.session_state.parsed_resume), unsafe_allow_html=True)
            with st.expander("📋 Full Parsed JSON"):
                st.json(st.session_state.parsed_resume)
    
    # ------------------- BULK RESUME RANKING MODE -------------------
    elif app_mode == "Bulk Resume Ranking":
        st.markdown("""<div class="upload-box">
            <h3>📂 Upload Multiple Resumes</h3>
            <p>Select all resumes to compare against the JD</p>
        </div>""", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload one or more resumes",
            type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="bulk_resume_uploader"
        )

        if uploaded_files:
            st.session_state.bulk_resumes = uploaded_files
            st.success(f"✅ {len(uploaded_files)} resumes loaded and ready for ranking.")
            for file in uploaded_files:
                st.caption(f" - {file.name}")
        else:
            st.session_state.bulk_resumes = []
            st.session_state.bulk_results = None
            st.session_state.bulk_processing_complete = False

    # ── JD Upload (Common for both modes) ──
    st.markdown("---")
    st.markdown("""<div class="jd-box">
        <h4>📋 Job Description (Required)</h4>
        <p>Upload or paste JD to compare resumes against</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.jd_loaded:
        jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "doc", "txt"], label_visibility="collapsed", key="jd_uploader")
        if jd_file:
            with st.spinner("📋 Processing JD..."):
                jd_result = process_uploaded_file(jd_file, GROQ_API_KEY)
            if jd_result["success"]:
                st.session_state.jd_text = jd_result["text"]
                st.session_state.jd_loaded = True
                st.session_state.jd_source = "file"
                st.session_state.jd_file_info = jd_result
                if st.session_state.tool_registry: # for single mode
                    st.session_state.tool_registry.set_jd_text(jd_result["text"])
                st.rerun()
        
        jd_paste = st.text_area("Or paste JD text:", height=120, placeholder="Paste job description here...")
        if jd_paste and len(jd_paste.strip()) > 50:
            if st.button("✅ Use This JD", use_container_width=True):
                st.session_state.jd_text = jd_paste
                st.session_state.jd_loaded = True
                st.session_state.jd_source = "pasted"
                if st.session_state.tool_registry: # for single mode
                    st.session_state.tool_registry.set_jd_text(jd_paste)
                st.rerun()
    else:
        jd_name = "Pasted JD" if st.session_state.jd_source == "pasted" else st.session_state.jd_file_info['file_name']
        col_jd, col_jd_remove = st.columns([5, 1])
        with col_jd: st.success(f"✅ JD: {jd_name}")
        with col_jd_remove:
            if st.button("❌", key="remove_jd", help="Remove JD"):
                reset_jd_state()
                st.rerun()
        with st.expander("👁️ View Job Description"):
            st.text_area("JD Content", st.session_state.jd_text, height=200, disabled=True, label_visibility="collapsed")
            
    st.markdown("---")
    st.markdown("### 🧠 AI Model")
    sel = st.selectbox("Model", list(GROQ_MODELS.keys()), index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model))
    st.session_state.selected_model = sel
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    if app_mode == "Single Resume Chat":
        if c1.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if c2.button("🔄 Reset All", use_container_width=True):
            reset_chat_state()
            reset_jd_state()
            st.rerun()
    else:
         if c1.button("🗑️ Reset Files", use_container_width=True):
            reset_bulk_state()
            reset_jd_state()
            st.rerun()


# ═══════════════════════════════════════════
#              MAIN CONTENT
# ═══════════════════════════════════════════

st.markdown('<h1 class="main-header">🤖 Universal AI Resume Assistant</h1>', unsafe_allow_html=True)

# ------------------- SINGLE RESUME CHAT MODE -------------------
if st.session_state.app_mode == "Single Resume Chat":
    if not st.session_state.resume_loaded:
        st.markdown("""<div class="welcome-box">
            <h2>📄 Upload a Resume to Get Started</h2>
            <p>Upload any resume, then ask the AI assistant anything about it.
            You can also switch to <strong>Bulk Resume Ranking</strong> mode in the sidebar.</p>
            <br><p><em>👈 Upload a resume using the sidebar!</em></p>
            </div>""", unsafe_allow_html=True)
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # ... (rest of the message metadata display logic)

    def run_agent(question: str):
        mi = GROQ_MODELS[st.session_state.selected_model]
        jd = st.session_state.jd_text if st.session_state.jd_loaded else ""
        agent = ResumeAgent(st.session_state.tool_registry, GROQ_API_KEY, mi["id"], jd_text=jd)
        return agent.run(question, st.session_state.messages), mi

    if prompt := st.chat_input("Ask anything about the uploaded resume..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent thinking..."):
                result, mi = run_agent(prompt)
            st.markdown(result.answer)
            # ... (rest of the agent trace and metadata logic)
        st.session_state.messages.append({
            "role": "assistant", "content": result.answer, "tools_used": result.tools_used, 
            "steps": result.steps, "total_time": result.total_time, "model": mi["id"]
        })

# ------------------- BULK RESUME RANKING MODE -------------------
elif st.session_state.app_mode == "Bulk Resume Ranking":
    st.markdown('<p class="sub-header">Compare multiple resumes against one Job Description to find the best candidates.</p>', unsafe_allow_html=True)

    if not st.session_state.bulk_resumes or not st.session_state.jd_loaded:
        st.markdown("""<div class="welcome-box">
            <h2>📂 Upload Resumes and a JD to Start Ranking</h2>
            <p>This mode allows you to efficiently compare multiple candidates for a single role.</p>
            <br>
            <p><strong>Step 1:</strong> 👈 Upload all candidate resumes using the multi-file uploader in the sidebar.</p>
            <p><strong>Step 2:</strong> 👈 Upload or paste the single Job Description you are hiring for.</p>
            <p><strong>Step 3:</strong> 👇 Click the 'Rank All Resumes' button to begin the analysis.</p>
            </div>""", unsafe_allow_html=True)
        st.stop()

    if st.button("🚀 Rank All Resumes", use_container_width=True, type="primary"):
        run_bulk_comparison()
    
    if st.session_state.bulk_processing_complete and st.session_state.bulk_results:
        st.markdown("---")
        st.markdown("## 🏆 Candidate Ranking Results")

        results = st.session_state.bulk_results
        
        # Create a DataFrame for better display
        data_for_df = []
        for i, res in enumerate(results):
            score = res.get('overall_fit_score', 0)
            score_bar = f"""
            <div class="score-bar-container">
                <div class="score-bar" style="width: {score}%;"></div>
            </div>
            """
            data_for_df.append({
                "Rank": i + 1,
                "Candidate": res.get('candidate_name', 'N/A'),
                "Score": f"{score}%",
                "Score Bar": score_bar,
                "Recommendation": res.get('recommendation', '-'),
                "Strengths": " ".join([f"<li>{s}</li>" for s in res.get('strengths', [])]),
                "Gaps": " ".join([f"<li>{g}</li>" for g in res.get('gaps', [])])
            })
        
        df = pd.DataFrame(data_for_df)
        
        st.markdown(df.to_html(escape=False, index=False, classes='results-table'), unsafe_allow_html=True)

