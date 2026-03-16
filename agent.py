"""
Agentic AI Orchestrator - V3
STRICT: No hallucination, no duplicates, only original data
JD-aware, Education-aware, 2026 experience
Plan → Execute MCP Tools → Synthesize
"""

import json
import time
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from mcp_tools import MCPToolRegistry, ToolResult

PLANNING_PROMPT = """You are an intelligent agent. Current year is 2026.

Available tools:
{tools_description}

{jd_context}

Respond ONLY in this JSON format (no markdown, no code blocks):
{{
    "reasoning": "Brief plan",
    "tools": [
        {{"tool_name": "name", "parameters": {{"param": "value"}}}}
    ]
}}

STRICT RULES:
1. General resume questions → "resume_search"
2. Skill listing/extraction → "skill_analyzer" (WITHOUT required_skills parameter if no JD)
3. Experience/timeline/work history → "experience_calculator"
4. Cover letter requests → "cover_letter_generator" + "resume_search"
5. Summary/bio/profile/contact → "profile_summary"
6. Education/degree/university/college/GPA/qualifications → "education_extractor" + "resume_search"
7. Certifications → "education_extractor"
8. Contact info → "profile_summary" + "resume_search"

⚠️ CRITICAL - JD MATCHING RULES:
- "jd_matcher" tool → ONLY use if JD is uploaded (check jd_context below)
- "skill_analyzer" with required_skills → ONLY use if JD is uploaded OR user explicitly provides skills to match
- If NO JD uploaded and user asks for job fit/matching → Return message that JD is required
- NEVER compare or match skills against anything unless explicitly provided

⚠️ NEVER HALLUCINATE:
- Only use tools to retrieve ACTUAL data from the resume
- Do NOT invent or assume any information
- If information is not found, say "Not found in resume"

USER QUESTION: {question}"""

SYNTHESIS_PROMPT = """You are a professional AI resume assistant. Current year is 2026.

⚠️ CRITICAL RULES - READ CAREFULLY:

1. **NO HALLUCINATION**: 
   - ONLY use information from the tool results below
   - NEVER invent, assume, or make up ANY information
   - If data is not in tool results, say "Not mentioned in resume" or "Information not available"
   - Do NOT add details that are not explicitly in the results

2. **NO DUPLICATES**:
   - List each skill, achievement, or detail ONLY ONCE
   - Do NOT repeat the same information in different sections
   - Consolidate similar items

3. **JD MATCHING RULES**:
   - If no JD matcher results are present, do NOT compare or score against any job
   - If user asked for job fit but no JD was uploaded, inform them: "Please upload a Job Description to compare"
   - Do NOT assume or create skill match percentages without actual JD comparison

4. **ACCURACY**:
   - Use EXACT names, titles, dates, and numbers from the data
   - Do NOT paraphrase or modify factual information
   - For experience: clearly state "as of 2026"

5. **FORMATTING**:
   - Use **bold** for key highlights
   - Use bullet points for lists (no duplicates)
   - Use ### headers for sections in long answers
   - Be concise but complete

Tool results:
{tool_results}

QUESTION: {question}

Generate response following ALL rules above. If information is not available in tool results, explicitly state that."""


@dataclass
class AgentStep:
    step_type: str
    tool_name: Optional[str] = None
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentResponse:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_time: float = 0.0
    model_used: str = ""


class ResumeAgent:
    def __init__(self, tool_registry: MCPToolRegistry, groq_api_key: str,
                 model_id: str = "llama-3.1-8b-instant", jd_text: str = ""):
        self.registry = tool_registry
        self.api_key = groq_api_key
        self.model_id = model_id
        self.jd_text = jd_text
        self.has_jd = bool(jd_text and len(jd_text.strip()) > 50)

    def _call_groq(self, system: str, user: str, temp: float = 0.3) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": temp,
            "max_tokens": 3500,
        }

        models = [self.model_id, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        seen = set()
        unique = [m for m in models if m not in seen and not seen.add(m)]

        for m in unique:
            payload["model"] = m
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=payload, timeout=30
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                elif r.status_code == 400:
                    continue
            except Exception:
                continue
        return "Error: Could not get response from any model."

    def _is_jd_comparison_query(self, question: str) -> bool:
        """Check if the question requires JD comparison"""
        jd_keywords = [
            "job description", "jd", "job fit", "fit score", "match score",
            "compare", "comparison", "job match", "suitable", "qualified",
            "requirements match", "skill gap", "missing skills", "gaps",
            "how well", "fit for", "good fit", "match against", "versus job"
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in jd_keywords)

    def _is_skill_match_query(self, question: str) -> bool:
        """Check if the question asks for skill matching against specific skills"""
        match_keywords = [
            "match skills", "compare skills", "skill match", "skills:",
            "match:", "against:", "required skills", "check skills"
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in match_keywords)

    def _plan(self, question: str) -> Tuple[List[Dict], AgentStep]:
        start = time.time()
        
        # Check if JD comparison is needed but JD not available
        needs_jd = self._is_jd_comparison_query(question)
        if needs_jd and not self.has_jd:
            # Return a special response indicating JD is needed
            step = AgentStep(
                "planning",
                input_data={"question": question},
                output_data={
                    "reasoning": "JD comparison requested but no JD uploaded",
                    "planned_tools": ["profile_summary"],
                    "jd_required": True
                },
                duration=round(time.time() - start, 3)
            )
            return [{"tool_name": "profile_summary", "parameters": {"context": "detailed"}}], step
        
        jd_ctx = ""
        if self.has_jd:
            jd_ctx = (
                "✅ A Job Description HAS BEEN UPLOADED. You can use 'jd_matcher' for job fit analysis. "
                f"JD preview: {self.jd_text[:200]}..."
            )
        else:
            jd_ctx = (
                "❌ NO Job Description uploaded. "
                "DO NOT use 'jd_matcher' tool. "
                "DO NOT compare skills against any job requirements. "
                "If user asks for job fit/matching, inform them to upload a JD first."
            )

        prompt = PLANNING_PROMPT.format(
            tools_description=self.registry.get_tools_description(),
            jd_context=jd_ctx,
            question=question
        )

        response = self._call_groq(
            "Return ONLY valid JSON. No markdown. No code blocks. Follow JD rules strictly.", 
            prompt, 
            0.1
        )

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            plan = json.loads(cleaned.strip())
            tools = plan.get("tools", [])
            reasoning = plan.get("reasoning", "")
        except json.JSONDecodeError:
            tools = self._get_fallback_tools(question)
            reasoning = "Fallback planning based on question keywords"

        # Filter out JD matcher if no JD uploaded
        if not self.has_jd:
            tools = [t for t in tools if t.get("tool_name") != "jd_matcher"]
            # Also remove skill_analyzer with required_skills if no JD and not explicit skill match
            if not self._is_skill_match_query(question):
                for t in tools:
                    if t.get("tool_name") == "skill_analyzer":
                        t["parameters"] = {"required_skills": ""}  # Clear required skills

        # Auto-inject JD text into jd_matcher calls only if JD exists
        if self.has_jd:
            for t in tools:
                if t["tool_name"] == "jd_matcher":
                    t["parameters"]["jd_text"] = self.jd_text

        step = AgentStep(
            "planning",
            input_data={"question": question},
            output_data={
                "reasoning": reasoning,
                "planned_tools": [t["tool_name"] for t in tools],
                "has_jd": self.has_jd
            },
            duration=round(time.time() - start, 3)
        )
        return tools, step

    def _get_fallback_tools(self, question: str) -> List[Dict]:
        """Get fallback tools based on question keywords"""
        q_lower = question.lower()
        tools = []
        
        # Education keywords
        edu_keywords = ["education", "degree", "university", "college", "school", 
                       "gpa", "cgpa", "qualification", "bachelor", "master", "phd",
                       "b.tech", "m.tech", "graduated", "studied", "academic"]
        if any(kw in q_lower for kw in edu_keywords):
            tools.append({"tool_name": "education_extractor", "parameters": {"include_certifications": True}})
        
        # Experience keywords
        exp_keywords = ["experience", "work", "job", "career", "years", "timeline", "history", "company"]
        if any(kw in q_lower for kw in exp_keywords):
            tools.append({"tool_name": "experience_calculator", "parameters": {"category": "all"}})
        
        # Skills keywords (without matching if no JD)
        skill_keywords = ["skill", "technology", "tech stack", "proficient", "expertise", "tools"]
        if any(kw in q_lower for kw in skill_keywords):
            tools.append({"tool_name": "skill_analyzer", "parameters": {"required_skills": ""}})
        
        # Contact keywords
        contact_keywords = ["contact", "email", "phone", "linkedin", "github", "reach", "address"]
        if any(kw in q_lower for kw in contact_keywords):
            tools.append({"tool_name": "profile_summary", "parameters": {"context": "detailed"}})
        
        # Cover letter
        if "cover letter" in q_lower:
            tools.append({"tool_name": "cover_letter_generator", "parameters": {"job_title": "", "company_name": ""}})
        
        # Summary/profile
        summary_keywords = ["summary", "profile", "bio", "about", "linkedin", "introduction"]
        if any(kw in q_lower for kw in summary_keywords):
            tools.append({"tool_name": "profile_summary", "parameters": {"context": "linkedin"}})
        
        # JD matching - only if JD is uploaded
        if self.has_jd and self._is_jd_comparison_query(q_lower):
            tools.append({"tool_name": "jd_matcher", "parameters": {"jd_text": self.jd_text}})
        
        # Always add resume search for context
        tools.append({"tool_name": "resume_search", "parameters": {"query": question}})
        
        return tools if tools else [{"tool_name": "resume_search", "parameters": {"query": question}}]

    def _execute_tools(self, tools: List[Dict]) -> Tuple[List[ToolResult], List[AgentStep]]:
        results, steps = [], []
        executed_tools = set()  # Prevent duplicate tool execution
        
        for tc in tools:
            name = tc.get("tool_name", "")
            params = tc.get("parameters", {})
            
            # Skip duplicate tools
            tool_key = f"{name}_{json.dumps(params, sort_keys=True)}"
            if tool_key in executed_tools:
                continue
            executed_tools.add(tool_key)
            
            start = time.time()

            if name not in self.registry.tool_names:
                name = "resume_search"
                if "query" not in params:
                    params["query"] = str(params)

            result = self.registry.execute_tool(name, **params)
            results.append(result)

            steps.append(AgentStep(
                "tool_call", tool_name=name,
                input_data=params,
                output_data=(
                    {"preview": str(result.data)[:500]}
                    if result.success else {"error": result.error}
                ),
                duration=round(time.time() - start, 3),
                success=result.success,
                error=result.error
            ))
        return results, steps

    def _synthesize(self, question: str, results: List[ToolResult],
                    history: List[Dict] = None, jd_required: bool = False) -> Tuple[str, AgentStep]:
        start = time.time()

        # Check if JD was required but not available
        if jd_required and not self.has_jd:
            no_jd_response = (
                "## ⚠️ Job Description Required\n\n"
                "To compare the resume against job requirements or calculate a fit score, "
                "please **upload a Job Description** first.\n\n"
                "You can:\n"
                "1. Upload a JD file (PDF, DOCX, TXT) in the sidebar\n"
                "2. Paste the JD text in the text area provided\n\n"
                "Once uploaded, I can:\n"
                "- Calculate overall fit score\n"
                "- Identify matching skills\n"
                "- Find skill gaps\n"
                "- Provide detailed comparison"
            )
            step = AgentStep("synthesis", duration=round(time.time() - start, 3))
            return no_jd_response, step

        results_text = ""
        for r in results:
            results_text += f"\n### Tool: {r.tool_name}\n"
            results_text += f"Status: {'OK' if r.success else 'FAIL'}\n"
            if r.success:
                results_text += f"Data:\n{json.dumps(r.data, indent=2, default=str)[:5000]}\n"
            else:
                results_text += f"Error: {r.error}\n"

        # Add JD status to context
        jd_status = ""
        if not self.has_jd:
            jd_status = "\n\n⚠️ NOTE: No Job Description is uploaded. Do NOT create any skill match percentages or job fit scores."

        prompt = SYNTHESIS_PROMPT.format(
            tool_results=results_text + jd_status, 
            question=question
        )

        history_ctx = ""
        if history:
            for msg in history[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_ctx += f"\n{role}: {msg['content'][:200]}"

        system_prompt = (
            "You are a precise resume assistant. "
            "CRITICAL: Only use information from tool results. "
            "NEVER hallucinate or invent information. "
            "NEVER duplicate information. "
            "If data is not available, say 'Not found in resume'. "
            "Do NOT create match scores without actual JD comparison data."
        )

        answer = self._call_groq(system_prompt, prompt + history_ctx, 0.5)
        step = AgentStep("synthesis", duration=round(time.time() - start, 3))
        return answer, step

    def run(self, question: str, history: List[Dict] = None) -> AgentResponse:
        total_start = time.time()
        all_steps = []

        tools_to_call, plan_step = self._plan(question)
        all_steps.append(plan_step)
        
        # Check if JD was required
        jd_required = plan_step.output_data.get("jd_required", False) if plan_step.output_data else False

        results, exec_steps = self._execute_tools(tools_to_call)
        all_steps.extend(exec_steps)
        tools_used = list(set([s.tool_name for s in exec_steps if s.tool_name]))  # Deduplicate

        answer, synth_step = self._synthesize(question, results, history, jd_required)
        all_steps.append(synth_step)

        return AgentResponse(
            answer=answer,
            steps=all_steps,
            tools_used=tools_used,
            total_time=round(time.time() - total_start, 2),
            model_used=self.model_id
        )
