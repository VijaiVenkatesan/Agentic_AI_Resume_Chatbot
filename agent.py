"""
Agentic AI Orchestrator - JD-aware, 2026 experience
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

RULES:
1. General resume questions → "resume_search"
2. Skill matching/comparison → "skill_analyzer" + "resume_search"
3. Experience/timeline → "experience_calculator"
4. Cover letter requests → "cover_letter_generator" + "resume_search"
5. Summary/bio/profile → "profile_summary"
6. JD comparison/job fit → "jd_matcher" (ONLY if JD is uploaded)
7. Contact info → "profile_summary" + "resume_search"
8. Complex questions → MULTIPLE tools
9. Always include "resume_search" for extra context
10. Greetings → "profile_summary" with context="elevator_pitch"

USER QUESTION: {question}"""

SYNTHESIS_PROMPT = """You are a professional AI resume assistant. Current year is 2026.

Tool results:
{tool_results}

QUESTION: {question}

Create a comprehensive answer:
- Use **bold** for key highlights, numbers, percentages
- Use bullet points for lists
- Use ### headers for sections in long answers
- Include ALL relevant details from tool results
- For experience: clearly state "as of 2026"
- For contact info: list ALL available details (name, email, phone, address, LinkedIn, GitHub)
- For JD matching: present scores clearly with recommendation
- Be specific - use actual numbers, dates, company names
- Only use info from tool results"""


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
            "max_tokens": 3000,
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

    def _plan(self, question: str) -> Tuple[List[Dict], AgentStep]:
        start = time.time()
        jd_ctx = ""
        if self.jd_text:
            jd_ctx = (
                "A Job Description has been uploaded. If user asks about job fit, "
                "matching, or comparison, use 'jd_matcher'. "
                f"JD preview: {self.jd_text[:200]}..."
            )

        prompt = PLANNING_PROMPT.format(
            tools_description=self.registry.get_tools_description(),
            jd_context=jd_ctx,
            question=question
        )

        response = self._call_groq(
            "Return ONLY valid JSON. No markdown. No code blocks.", prompt, 0.1
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
            tools = [{"tool_name": "resume_search", "parameters": {"query": question}}]
            reasoning = "Defaulting to resume search"

        # Auto-inject JD text into jd_matcher calls
        for t in tools:
            if t["tool_name"] == "jd_matcher" and self.jd_text:
                t["parameters"]["jd_text"] = self.jd_text

        step = AgentStep(
            "planning",
            input_data={"question": question},
            output_data={
                "reasoning": reasoning,
                "planned_tools": [t["tool_name"] for t in tools]
            },
            duration=round(time.time() - start, 3)
        )
        return tools, step

    def _execute_tools(self, tools: List[Dict]) -> Tuple[List[ToolResult], List[AgentStep]]:
        results, steps = [], []
        for tc in tools:
            name = tc.get("tool_name", "")
            params = tc.get("parameters", {})
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
                    {"preview": str(result.data)[:300]}
                    if result.success else {"error": result.error}
                ),
                duration=round(time.time() - start, 3),
                success=result.success,
                error=result.error
            ))
        return results, steps

    def _synthesize(self, question: str, results: List[ToolResult],
                    history: List[Dict] = None) -> Tuple[str, AgentStep]:
        start = time.time()

        results_text = ""
        for r in results:
            results_text += f"\n### Tool: {r.tool_name}\n"
            results_text += f"Status: {'OK' if r.success else 'FAIL'}\n"
            if r.success:
                results_text += f"Data:\n{json.dumps(r.data, indent=2, default=str)[:4000]}\n"
            else:
                results_text += f"Error: {r.error}\n"

        prompt = SYNTHESIS_PROMPT.format(
            tool_results=results_text, question=question
        )

        history_ctx = ""
        if history:
            for msg in history[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_ctx += f"\n{role}: {msg['content'][:200]}"

        answer = self._call_groq(prompt, question + history_ctx, 0.7)
        step = AgentStep("synthesis", duration=round(time.time() - start, 3))
        return answer, step

    def run(self, question: str, history: List[Dict] = None) -> AgentResponse:
        total_start = time.time()
        all_steps = []

        tools_to_call, plan_step = self._plan(question)
        all_steps.append(plan_step)

        results, exec_steps = self._execute_tools(tools_to_call)
        all_steps.extend(exec_steps)
        tools_used = [s.tool_name for s in exec_steps if s.tool_name]

        answer, synth_step = self._synthesize(question, results, history)
        all_steps.append(synth_step)

        return AgentResponse(
            answer=answer,
            steps=all_steps,
            tools_used=tools_used,
            total_time=round(time.time() - total_start, 2),
            model_used=self.model_id
        )
