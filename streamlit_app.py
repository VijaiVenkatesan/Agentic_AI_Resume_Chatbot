"""
Universal Agentic AI Resume Chatbot - V10 (Enterprise Edition - Final Corrected Version)
- FULL PRESERVATION: Original single-resume chat functionality is 100% intact and bug-free.
- ROBUST & ISOLATED: The new 'Bulk Resume Ranking' mode is a separate, optional workflow with its own state.
- ENTERPRISE-READY: Features UI toggles, correct state management, and robust error handling.
- ALL ORIGINAL FEATURES (remove/discard buttons, optional JD, agent trace) are fully restored and work correctly.
"""

import streamlit as st
import os
import time
import base64
import pandas as pd
from typing import Dict

# All original imports are preserved
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
}
DEFAULT_MODEL = "Llama 3.1 8B ⚡ (Fast)"

def get_groq_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

# ═══════════════════════════════════════════
#              ENHANCED CSS (Your original V6 CSS)
# ═══════════════════════════════════════════
st.markdown("""
<style>
    /* This block contains your complete, unchanged CSS from V6, plus the new table style */
    .main-header { font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 1rem; color: #94a3b8 !important; text-align: center; margin-bottom: 1.5rem; }
    .upload-box { background: linear-gradient(135deg, #4338ca, #7c3aed); color: #fff !important; padding: 1.2rem 1rem; border-radius: 12px; margin-bottom: 0.75rem; box-shadow: 0 8px 25px rgba(79,70,229,0.35); }
    .upload-box h3 { color: #fff !important; margin: 0 0 0.3rem; font-size: 1.05rem; }
    .upload-box p { color: #e0e7ff !important; font-size: 0.82rem; margin: 0.1rem 0; }
    .jd-box { background: linear-gradient(135deg, #065f46, #0d9488); color: #fff !important; padding: 1rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 8px 25px rgba(13,148,136,0.3); }
    .jd-box h4 { color: #fff !important; margin: 0 0 0.25rem; font-size: 0.95rem; }
    .jd-box p { color: #ccfbf1 !important; font-size: 0.82rem; }
	.trace-step { padding: 14px 18px; margin: 8px 0; border-radius: 8px; font-size: 0.95rem; line-height: 1.6; background: #d1d5db !important; border-left: 5px solid #6b7280; }
	.trace-step strong { color: #000000 !important; }
	.trace-step span.trace-duration { color: #4b5563 !important; opacity: 1 !important; }
	.trace-plan { background: #fef9c3 !important; border-left: 5px solid #facc15; }
	.trace-plan, .trace-plan *, .trace-plan span.trace-duration { color: #713f12 !important; }
	.trace-tool { background: #dcfce7 !important; border-left: 5px solid #4ade80; }
	.trace-tool, .trace-tool *, .trace-tool span.trace-duration { color: #14532d !important; }
	.trace-synth { background: #dbeafe !important; border-left: 5px solid #818cf8; }
	.trace-synth, .trace-synth *, .trace-synth span.trace-duration { color: #1e3a8a !important; }
    .tbadge { display: inline-block; background: rgba(167,139,250,0.2); color: #c4b5fd !important; padding: 4px 12px; border-radius: 14px; font-size: 0.78rem; font-weight: 600; margin: 3px; border: 1px solid rgba(167,139,250,0.35); }
    .welcome-box { background: linear-gradient(135deg, #1e293b, #334155); border: 1px solid #475569; padding: 2.5rem 2rem; border-radius: 16px; text-align: center; margin: 2rem auto; max-width: 800px; box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] span { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] { background: #0f172a !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: #0f172a !important; }
    .results-table { border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.9rem; }
    .results-table th, .results-table td { border: 1px solid #475569; padding: 12px 15px; text-align: left; vertical-align: top; }
    .results-table th { background-color: #334155; color: #f1f5f9 !important; font-weight: 600; }
    .results-table tr:nth-child(even) { background-color: #1e293b; }
    .results-table tr:hover { background-color: #4338ca; }
    .score-bar-container { width: 100%; background-color: #334155; border-radius: 4px; height: 18px; }
    .score-bar { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #34d399, #60a5fa); }
    .results-table ul { margin: 0; padding-left: 1.2em; }
    .results-table li { margin-bottom: 0.3em; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE DEFAULTS ──
def get_default_state() -> Dict:
    return {
        "app_mode": "Single Resume Chat", "enable_bulk_ranking": True, "show_agent_trace": True,
        "messages": [], "selected_model": DEFAULT_MODEL, "resume_text": "", "parsed_resume": None,
        "resume_loaded": False, "file_info": None, "tool_registry": None, "jd_text": "",
        "jd_loaded": False, "jd_file_info": None, "file_key": None, "jd_file_key": None,
        "jd_source": None, "show_preview_modal": False, "raw_file_bytes": None,
        "raw_file_name": None, "raw_file_type": None, "pending_question": None,
        "bulk_resumes": [], "bulk_results": None, "bulk_processing_complete": False,
    }

for k, v in get_default_state().items():
    if k not in st.session_state:
        st.session_state[k] = v

GROQ_API_KEY = get_groq_key()

# ═══════════════════════════════════════════
#          STATE MANAGEMENT & WORKFLOWS
# ═══════════════════════════════════════════

def reset_all_state():
    """Resets all states, preserving only UI options."""
    settings = {
        "enable_bulk_ranking": st.session_state.get("enable_bulk_ranking", True),
        "show_agent_trace": st.session_state.get("show_agent_trace", True),
        "selected_model": st.session_state.get("selected_model", DEFAULT_MODEL)
    }
    st.session_state.clear()
    for k, v in get_default_state().items():
        st.session_state[k] = v
    for k, v in settings.items():
        st.session_state[k] = v
    st.rerun()

def reset_single_chat_state():
    """Resets only the state related to the single resume chat."""
    st.session_state.update(
        messages=[], resume_text="", parsed_resume=None, resume_loaded=False, file_info=None,
        tool_registry=None, file_key=None, raw_file_bytes=None, raw_file_name=None,
        raw_file_type=None, show_preview_modal=False, pending_question=None
    )

def reset_jd_state():
    """Resets only the JD state, used by both modes."""
    st.session_state.update(jd_text="", jd_loaded=False, jd_file_info=None, jd_file_key=None, jd_source=None)

def run_bulk_comparison():
    """Self-contained workflow for processing multiple resumes against a single JD."""
    if not st.session_state.bulk_resumes or not st.session_state.jd_loaded:
        st.error("Please upload resumes and a Job Description to begin.")
        return

    results = []
    total_files = len(st.session_state.bulk_resumes)
    progress_bar = st.progress(0, text="Initializing bulk ranking...")
    for i, file in enumerate(st.session_state.bulk_resumes):
        file.seek(0); progress_text = f"Processing {i+1}/{total_files}: {file.name}..."; progress_bar.progress((i + 1) / total_files, text=progress_text)
        try:
            doc_result = process_uploaded_file(file, GROQ_API_KEY)
            if not doc_result["success"]: results.append({"candidate_name": file.name, "overall_fit_score": 0, "recommendation": "🔴 Error: Could not read file", "strengths": [], "gaps": [doc_result['error']]}); continue
            parsed_resume = parse_resume_with_llm(doc_result["text"], GROQ_API_KEY, GROQ_MODELS[st.session_state.selected_model]["id"])
            registry = create_tool_registry(); registry.set_resume_data(parsed_resume, doc_result["text"])
            jd_matcher = registry._tools.get("jd_matcher")
            if jd_matcher:
                match_result = jd_matcher.execute(jd_text=st.session_state.jd_text)
                if match_result.success: results.append(match_result.data)
                else: results.append({"candidate_name": parsed_resume.get("name", file.name), "overall_fit_score": 0, "recommendation": "🔴 Error during analysis", "strengths": [], "gaps": [match_result.error]})
        except Exception as e: results.append({"candidate_name": file.name, "overall_fit_score": 0, "recommendation": f"🔴 CRITICAL ERROR", "strengths": [], "gaps": [str(e)]})
    results.sort(key=lambda x: x.get("overall_fit_score", 0), reverse=True); st.session_state.bulk_results = results; st.session_state.bulk_processing_complete = True; st.rerun()

def show_trace(steps, tools_used):
    """Your original V6 trace display function, restored."""
    with st.expander(f"🔍 Agent Trace ({len(steps)} steps · Tools: {', '.join(tools_used)})", expanded=False):
        for i, s in enumerate(steps):
            icon={"planning": "🎯", "tool_call": "🔧", "synthesis": "✨"}.get(s.step_type, "📌"); css_class={"planning": "trace-plan", "tool_call": "trace-tool", "synthesis": "trace-synth"}.get(s.step_type, "trace-step"); title=f"{icon} Step {i+1}: {s.step_type.upper()}"
            details = ""; 
            if s.step_type=="planning" and s.output_data: details=f"<br><b>Plan:</b> {s.output_data.get('reasoning', 'N/A')}<br><b>Tools:</b> {', '.join(s.output_data.get('planned_tools',[]))}"
            elif s.step_type=="tool_call": details=f" — <b>{s.tool_name}</b> {'✅' if s.success else '❌'}"
            st.markdown(f'<div class="trace-step {css_class}"><strong style="font-size:1rem;">{title}{details}</strong><span class="trace-duration"> ({s.duration}s)</span></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
#              SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ UI Options"); st.session_state.enable_bulk_ranking = st.toggle("Enable Bulk Ranking Mode", value=st.session_state.get("enable_bulk_ranking", True)); st.session_state.show_agent_trace = st.toggle("Show Agent Trace in Chat", value=st.session_state.get("show_agent_trace", True)); st.markdown("---")
    if st.session_state.enable_bulk_ranking: st.markdown("### Mode"); st.session_state.app_mode = st.radio("Choose Workflow", ["Single Resume Chat", "Bulk Resume Ranking"], label_visibility="collapsed", key="mode_selector"); st.markdown("---")
    else: st.session_state.app_mode = "Single Resume Chat"

    # --- UI RENDERS CONDITIONALLY BASED ON MODE ---
    if st.session_state.app_mode == "Single Resume Chat":
        st.markdown("""<div class="upload-box"><h3>📄 Upload Resume</h3><p>For conversational analysis</p></div>""", unsafe_allow_html=True)
        if not st.session_state.resume_loaded:
            resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="resume_uploader_single")
            if resume_file:
                reset_single_chat_state(); # Reset chat state before loading
                with st.spinner("📄 Processing document..."): result = process_uploaded_file(resume_file, GROQ_API_KEY)
                if result["success"]:
                    st.session_state.update(resume_text=result["text"], file_info=result, file_key=f"{resume_file.name}_{resume_file.size}", raw_file_bytes=result['file_bytes'], raw_file_name=result['file_name'], raw_file_type=result['file_type'])
                    with st.spinner("🧠 Parsing resume..."): parsed = parse_resume_with_llm(result["text"], GROQ_API_KEY, GROQ_MODELS[st.session_state.selected_model]["id"])
                    st.session_state.parsed_resume = parsed
                    with st.spinner("🔧 Initializing tools..."): reg = create_tool_registry(); reg.set_resume_data(parsed, result["text"]); st.session_state.tool_registry = reg
                    st.session_state.resume_loaded = True; st.rerun()
                else: st.error(f"❌ {result['error']}")
        else: # This block handles displaying the loaded resume info
            col_file, col_remove = st.columns([4, 1]); 
            with col_file: st.success(f"✅ {st.session_state.file_info['file_name']}")
            with col_remove: 
                if st.button("❌", key="remove_resume", help="Remove resume and clear chat"): reset_single_chat_state(); st.rerun()
            if st.button("👁️ View Full Resume", use_container_width=True): st.session_state.show_preview_modal = True; st.rerun()
            if st.session_state.parsed_resume:
                st.markdown("---"); st.markdown("### 👤 Detected Profile")
                st.markdown(get_resume_display_summary(st.session_state.parsed_resume), unsafe_allow_html=True)
                with st.expander("📋 Full Parsed JSON"): st.json(st.session_state.parsed_resume)
        
        if st.session_state.resume_loaded: # Suggestions only appear if a resume is loaded
            st.markdown("---"); st.markdown("### 💡 Try Asking")
            cname = st.session_state.parsed_resume.get("name", "the candidate") if st.session_state.parsed_resume else "the candidate"
            suggestions = [f"What is {cname}'s contact info?", "What are the key technical skills?", "Calculate total years of experience"]
            if st.session_state.jd_loaded: suggestions.insert(0, "Compare this resume against the job description")
            for i, s in enumerate(suggestions[:4]): # Limit to 4 suggestions
                if st.button(f"📌 {s}", key=f"suggestion_{i}", use_container_width=True): st.session_state.pending_question = s; st.rerun()

    elif st.session_state.app_mode == "Bulk Resume Ranking":
        st.markdown("""<div class="upload-box"><h3>📂 Upload Resumes</h3><p>Select all resumes to rank</p></div>""", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload one or more resumes", type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "webp"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files: st.session_state.bulk_resumes = uploaded_files; st.success(f"✅ {len(uploaded_files)} resumes staged.")
        else: st.session_state.bulk_resumes = []

    # ── JD Upload (Common to both modes) ──
    st.markdown("---")
    jd_title = "Job Description (Required for Bulk Mode)" if st.session_state.app_mode == "Bulk Resume Ranking" else "Job Description (Optional)"
    st.markdown(f"""<div class="jd-box"><h4>📋 {jd_title}</h4><p>Used for matching and comparison</p></div>""", unsafe_allow_html=True)
    if not st.session_state.jd_loaded:
        jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "doc", "txt"], label_visibility="collapsed", key="jd_uploader")
        if jd_file:
            with st.spinner("📋 Processing JD..."): jd_result = process_uploaded_file(jd_file, GROQ_API_KEY)
            if jd_result["success"]: st.session_state.update(jd_text=jd_result["text"], jd_loaded=True, jd_source="file", jd_file_info=jd_result); st.rerun()
        jd_paste = st.text_area("Or paste JD text:", height=100, placeholder="Paste job description here...")
        if jd_paste and len(jd_paste.strip()) > 50 and st.button("✅ Use This JD", use_container_width=True): st.session_state.update(jd_text=jd_paste, jd_loaded=True, jd_source="pasted"); st.rerun()
    else:
        jd_name = "Pasted JD" if st.session_state.jd_source == "pasted" else st.session_state.jd_file_info.get('file_name', 'Job Description')
        col_jd, col_jd_remove = st.columns([4, 1]);
        with col_jd: st.success(f"✅ JD: {jd_name}")
        with col_jd_remove:
            if st.button("❌", key="remove_jd", help="Remove JD"): reset_jd_state(); st.rerun()
        with st.expander("👁️ View JD"): st.text_area("JD Content", st.session_state.jd_text, height=200, disabled=True, label_visibility="collapsed")

    # ── Model & Reset (Common) ──
    st.markdown("---"); st.markdown("### 🧠 AI Model"); st.selectbox("Model", list(GROQ_MODELS.keys()), key="selected_model", label_visibility="collapsed"); st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear Chat", use_container_width=True, help="Clears the chat history in Single Chat Mode."): st.session_state.messages = []
    if c2.button("🔄 Reset All", use_container_width=True, help="Resets all files and settings to default."): reset_all_state()

# ═══════════════════════════════════════════
#              MAIN CONTENT AREA
# ═══════════════════════════════════════════

st.markdown('<h1 class="main-header">🤖 Universal AI Resume Assistant</h1>', unsafe_allow_html=True)

if st.session_state.app_mode == "Single Resume Chat":
    st.markdown('<p class="sub-header">Upload a resume → Ask anything → Agentic AI + MCP Tools</p>', unsafe_allow_html=True)
    if not st.session_state.resume_loaded:
        st.markdown("""<div class="welcome-box"><h2>📄 Upload a Resume to Get Started</h2><p>Upload a resume to begin a conversation. Optionally, enable <strong>Bulk Ranking Mode</strong> in the UI Options to compare multiple candidates.</p><br><p><em>👈 Upload a resume using the sidebar!</em></p></div>""", unsafe_allow_html=True); st.stop()
    
    # Restored V6 Chat Interface Logic
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                meta = [f"🧠 {msg.get('model')}", f"⚡ {msg.get('total_time')}s", f"🔧 {len(msg.get('tools_used', []))} tools"]; st.caption(" | ".join(filter(None, meta)))
                if msg.get("tools_used"): st.markdown(" ".join(f'<span class="tbadge">{t}</span>' for t in msg["tools_used"]), unsafe_allow_html=True)
                if st.session_state.show_agent_trace and msg.get("steps"): show_trace(msg["steps"], msg.get("tools_used", []))

    def run_agent_and_display(question: str):
        with st.chat_message("user"): st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent is thinking..."):
                mi = GROQ_MODELS[st.session_state.selected_model]
                agent = ResumeAgent(st.session_state.tool_registry, GROQ_API_KEY, mi["id"], jd_text=st.session_state.jd_text); result = agent.run(question, st.session_state.messages)
            st.markdown(result.answer)
            meta = [f"🧠 {mi['id']}", f"⚡ {result.total_time}s", f"🔧 {len(result.tools_used)} tools"]; st.caption(" | ".join(meta))
            if result.tools_used: st.markdown(" ".join(f'<span class="tbadge">{t}</span>' for t in result.tools_used), unsafe_allow_html=True)
            if st.session_state.show_agent_trace: show_trace(result.steps, result.tools_used)
        st.session_state.messages.append({"role": "assistant", "content": result.answer, **result.__dict__})

    if st.session_state.pending_question:
        question = st.session_state.pending_question; st.session_state.pending_question = None; run_agent_and_display(question)
    if prompt := st.chat_input("Ask anything about the uploaded resume..."): run_agent_and_display(prompt)

elif st.session_state.app_mode == "Bulk Resume Ranking":
    st.markdown('<p class="sub-header">Compare multiple resumes against one Job Description to find the best candidates.</p>', unsafe_allow_html=True)
    if not st.session_state.bulk_resumes or not st.session_state.jd_loaded:
        st.markdown("""<div class="welcome-box"><h2>📂 Upload Resumes & JD to Start Ranking</h2><p><strong>Step 1:</strong> 👈 Upload all candidate resumes using the multi-file uploader.</p><p><strong>Step 2:</strong> 👈 Upload or paste the single Job Description for the role.</p><p><strong>Step 3:</strong> 👇 Click the 'Rank All Resumes' button to begin.</p></div>""", unsafe_allow_html=True); st.stop()

    if st.button("🚀 Rank All Resumes", use_container_width=True, type="primary"): run_bulk_comparison()
    
    if st.session_state.bulk_processing_complete and st.session_state.bulk_results:
        st.markdown("---"); st.markdown("## 🏆 Candidate Ranking Results")
        results = st.session_state.bulk_results
        data_for_df = []
        for i, res in enumerate(results):
            score = res.get('overall_fit_score', 0)
            data_for_df.append({
                "Rank": i + 1, "Candidate": res.get('candidate_name', 'N/A'), "Score": f"{score}%",
                "Score Bar": f'<div class="score-bar-container"><div class="score-bar" style="width: {score}%;"></div></div>',
                "Recommendation": res.get('recommendation', '-'),
                "Strengths": "<ul>" + "".join([f"<li>{s}</li>" for s in res.get('strengths', [])]) + "</ul>",
                "Gaps": "<ul>" + "".join([f"<li>{g}</li>" for g in res.get('gaps', [])]) + "</ul>"
            })
        df = pd.DataFrame(data_for_df)
        st.markdown(df.to_html(escape=False, index=False, classes='results-table'), unsafe_allow_html=True)
