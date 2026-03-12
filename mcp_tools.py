"""
Universal MCP Tool Server - Enhanced
6 tools including JD Matcher
Experience calculated with CURRENT_YEAR = 2026
"""

import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

CURRENT_YEAR = 2026


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class MCPTool:
    name: str = ""
    description: str = ""
    parameters: List[ToolParameter] = []

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def to_schema(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {"type": p.type, "description": p.description}
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }


# ━━━ TOOL 1: RESUME SEARCH (RAG) ━━━

class ResumeSearchTool(MCPTool):
    name = "resume_search"
    description = "Semantic search through uploaded resume. Use for any question about the resume content."
    parameters = [
        ToolParameter("query", "string", "Search query"),
        ToolParameter("num_results", "integer", "Number of results", False, 4)
    ]

    def __init__(self):
        self._collection = None

    def initialize(self, resume_text: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        chunks = splitter.split_text(resume_text)
        client = chromadb.Client()
        ef = embedding_functions.DefaultEmbeddingFunction()
        try:
            client.delete_collection("universal_resume")
        except Exception:
            pass

        self._collection = client.create_collection(
            "universal_resume", embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

        section_map = {
            "SUMMARY": "Summary", "PROFILE": "Summary", "OBJECTIVE": "Summary",
            "ABOUT": "Summary", "EXPERIENCE": "Work Experience",
            "EMPLOYMENT": "Work Experience", "WORK": "Work Experience",
            "SKILL": "Skills", "TECHNICAL": "Skills", "COMPETENC": "Skills",
            "EDUCATION": "Education", "ACADEMIC": "Education",
            "CERTIF": "Certifications", "LICENS": "Certifications",
            "AWARD": "Awards", "HONOR": "Awards", "ACHIEV": "Awards",
            "PROJECT": "Projects", "CONTACT": "Contact",
            "PUBLICATION": "Publications", "RESEARCH": "Research",
        }

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            upper = chunk.upper()
            section = "General"
            for kw, sec in section_map.items():
                if kw in upper:
                    section = sec
                    break
            ids.append(str(i))
            docs.append(chunk)
            metas.append({"section": section})

        if ids:
            self._collection.add(ids=ids, documents=docs, metadatas=metas)

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        query = kwargs.get("query", "")
        k = kwargs.get("num_results", 4)

        if not self._collection:
            return ToolResult(self.name, False, None, error="Resume not loaded")

        try:
            results = self._collection.query(
                query_texts=[query], n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            retrieved = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    dist = results["distances"][0][i] if results["distances"] else 0
                    retrieved.append({
                        "content": doc,
                        "section": meta.get("section", "General"),
                        "relevance": round(max(0, 1 - dist), 3)
                    })

            return ToolResult(self.name, True, {
                "query": query,
                "results": retrieved,
                "context": "\n\n".join(r["content"] for r in retrieved)
            }, metadata={"count": len(retrieved)},
               execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━ TOOL 2: SKILL ANALYZER ━━━

class SkillAnalyzerTool(MCPTool):
    name = "skill_analyzer"
    description = "Analyze skill match between resume and job requirements with gap analysis."
    parameters = [
        ToolParameter("required_skills", "string",
                      "Comma-separated skills or job description text")
    ]

    def __init__(self):
        self._parsed = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            req_input = kwargs.get("required_skills", "")
            required = [s.strip().lower() for s in req_input.replace("\n", ",").split(",") if s.strip()]

            all_skills = []
            for cat in self._parsed.get("skills", {}).values():
                if isinstance(cat, list):
                    all_skills.extend(s.lower() for s in cat)
            all_skills.extend(s.lower() for s in self._parsed.get("specializations", []))

            # Also search work history achievements for implicit skills
            for job in self._parsed.get("work_history", []):
                for ach in job.get("key_achievements", []):
                    all_skills.append(ach.lower())
                for tech in job.get("technologies_used", []):
                    all_skills.append(tech.lower())

            matched, partial, missing = [], [], []
            related = {
                "machine learning": ["ml", "scikit-learn", "tensorflow", "keras", "pytorch"],
                "deep learning": ["tensorflow", "keras", "pytorch", "neural", "cnn", "rnn"],
                "nlp": ["natural language", "ner", "bert", "text", "spacy", "transformers"],
                "cloud": ["aws", "gcp", "azure", "cloud computing"],
                "api": ["rest api", "django", "flask", "fastapi"],
                "llm": ["large language", "generative ai", "gpt", "transformer", "langchain"],
                "docker": ["container", "kubernetes", "k8s", "devops"],
                "python": ["python", "django", "flask", "pandas", "numpy"],
                "sql": ["mysql", "postgresql", "database", "mongodb"],
                "javascript": ["js", "react", "node", "angular", "vue", "typescript"],
                "java": ["java", "spring", "jvm", "maven"],
            }

            for req in required:
                if not req:
                    continue
                if any(req in s or s in req for s in all_skills):
                    matched.append(req)
                else:
                    found = False
                    for key, aliases in related.items():
                        check = aliases + [key]
                        if any(req in a or a in req for a in check):
                            if any(any(a in s or s in a for a in check) for s in all_skills):
                                partial.append(req)
                                found = True
                                break
                    if not found:
                        missing.append(req)

            total = max(len(required), 1)
            pct = round(((len(matched) + len(partial) * 0.5) / total) * 100, 1)

            return ToolResult(self.name, True, {
                "match_percentage": pct,
                "matched_skills": matched,
                "partial_matches": partial,
                "missing_skills": missing,
                "total_required": len(required),
                "candidate_name": self._parsed.get("name", ""),
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━ TOOL 3: EXPERIENCE CALCULATOR ━━━

class ExperienceCalculatorTool(MCPTool):
    name = "experience_calculator"
    description = f"Calculate experience breakdown by years, roles, timeline. Current year: {CURRENT_YEAR}."
    parameters = [
        ToolParameter("category", "string", "total, timeline, or all", False, "all")
    ]

    def __init__(self):
        self._parsed = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            work = self._parsed.get("work_history", [])
            total = self._parsed.get("total_experience_years", 0)

            timeline = []
            for j in work:
                start_d = j.get("start_date", "")
                end_d = j.get("end_date", "")
                duration = j.get("duration", "")
                if not duration and start_d:
                    duration = f"{start_d} - {end_d}"

                timeline.append({
                    "role": j.get("title", ""),
                    "company": j.get("company", ""),
                    "duration": duration,
                    "years": j.get("duration_years", j.get("years", 0)),
                    "type": j.get("type", ""),
                    "achievements": j.get("key_achievements", [])
                })

            return ToolResult(self.name, True, {
                "candidate_name": self._parsed.get("name", ""),
                "total_years": round(total, 1),
                "calculated_as_of": f"July {CURRENT_YEAR}",
                "total_positions": len(work),
                "current_role": self._parsed.get("current_role", ""),
                "current_company": self._parsed.get("current_company", ""),
                "timeline": timeline,
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━ TOOL 4: COVER LETTER GENERATOR ━━━

class CoverLetterTool(MCPTool):
    name = "cover_letter_generator"
    description = "Generate data for a tailored cover letter for a specific role/company."
    parameters = [
        ToolParameter("job_title", "string", "Target job title"),
        ToolParameter("company_name", "string", "Target company", False, "the company")
    ]

    def __init__(self):
        self._parsed = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            r = self._parsed
            achievements = []
            for w in r.get("work_history", [])[:3]:
                achievements.extend(w.get("key_achievements", [])[:3])

            return ToolResult(self.name, True, {
                "candidate_name": r.get("name", ""),
                "target_role": kwargs.get("job_title", ""),
                "target_company": kwargs.get("company_name", "the company"),
                "experience_years": r.get("total_experience_years", 0),
                "current_role": r.get("current_role", ""),
                "current_company": r.get("current_company", ""),
                "key_skills": r.get("specializations", []),
                "achievements": achievements,
                "education": r.get("education", []),
                "certifications_count": len(r.get("certifications", [])),
                "contact": {
                    "email": r.get("email", ""),
                    "phone": r.get("phone", ""),
                },
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━ TOOL 5: PROFILE SUMMARY ━━━

class ProfileSummaryTool(MCPTool):
    name = "profile_summary"
    description = "Generate professional summary/bio for different contexts (linkedin, portfolio, elevator_pitch)."
    parameters = [
        ToolParameter("context", "string",
                      "Context: linkedin, portfolio, elevator_pitch, detailed", False, "detailed")
    ]

    def __init__(self):
        self._parsed = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            r = self._parsed
            all_skills = []
            for cat in r.get("skills", {}).values():
                if isinstance(cat, list):
                    all_skills.extend(cat[:5])

            return ToolResult(self.name, True, {
                "name": r.get("name", ""),
                "context": kwargs.get("context", "detailed"),
                "experience_years": r.get("total_experience_years", 0),
                "current_role": r.get("current_role", ""),
                "current_company": r.get("current_company", ""),
                "summary": r.get("professional_summary", ""),
                "specializations": r.get("specializations", []),
                "top_skills": all_skills[:12],
                "education": r.get("education", []),
                "certifications": r.get("certifications", [])[:5],
                "awards": r.get("awards", []),
                "contact": {
                    "email": r.get("email", ""),
                    "phone": r.get("phone", ""),
                    "address": r.get("address", ""),
                    "linkedin": r.get("linkedin", ""),
                    "github": r.get("github", ""),
                }
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━ TOOL 6: JD MATCHER ━━━

class JDMatcherTool(MCPTool):
    name = "jd_matcher"
    description = "Compare resume against a Job Description. Provides fit score and detailed analysis."
    parameters = [
        ToolParameter("jd_text", "string", "Job description text to compare against")
    ]

    def __init__(self):
        self._parsed = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        jd = kwargs.get("jd_text", "").lower()

        if not jd or len(jd.strip()) < 20:
            return ToolResult(self.name, False, None,
                            error="No job description provided or too short")
        try:
            r = self._parsed

            # All candidate skills flattened
            all_skills = []
            for cat in r.get("skills", {}).values():
                if isinstance(cat, list):
                    all_skills.extend(s.lower() for s in cat)
            all_skills.extend(s.lower() for s in r.get("specializations", []))

            # Skills found in JD
            skill_hits = [s for s in set(all_skills) if any(w in jd for w in s.split())]
            skill_score = min(100, (len(skill_hits) / max(len(set(all_skills)), 1)) * 150)

            # Experience comparison
            exp = r.get("total_experience_years", 0)
            exp_matches = re.findall(r'(\d+)\+?\s*years?', jd)
            exp_required = max([int(x) for x in exp_matches]) if exp_matches else 3
            exp_score = min(100, (exp / max(exp_required, 1)) * 100)

            # Education
            edu = r.get("education", [])
            edu_keywords = ["bachelor", "master", "phd", "b.tech", "m.tech", "mba",
                           "b.sc", "m.sc", "degree", "b.e", "m.e"]
            edu_in_resume = any(
                any(k in json.dumps(e).lower() for k in edu_keywords)
                for e in edu
            ) if edu else False
            jd_needs_edu = any(k in jd for k in edu_keywords)
            edu_score = 90 if edu_in_resume else (60 if not jd_needs_edu else 30)

            # Certifications
            certs = r.get("certifications", [])
            cert_score = min(100, len(certs) * 15) if certs else 20

            # Keyword overlap
            resume_blob = r.get("professional_summary", "").lower()
            for job in r.get("work_history", []):
                resume_blob += " " + " ".join(job.get("key_achievements", [])).lower()
                resume_blob += " " + " ".join(job.get("technologies_used", [])).lower()
            jd_words = [w for w in set(re.findall(r'\b\w+\b', jd)) if len(w) > 4 and w.isalpha()]
            kw_hits = sum(1 for w in jd_words if w in resume_blob)
            kw_score = min(100, (kw_hits / max(len(jd_words), 1)) * 200)

            overall = round(
                skill_score * 0.35 + exp_score * 0.25 +
                edu_score * 0.10 + cert_score * 0.10 +
                kw_score * 0.20, 1
            )
            overall = min(overall, 100)

            strengths, gaps = [], []
            if skill_score >= 60:
                strengths.append(f"Strong skill alignment ({skill_score:.0f}%)")
            if exp_score >= 80:
                strengths.append("Experience meets/exceeds requirements")
            if cert_score >= 50:
                strengths.append("Relevant certifications present")
            if kw_score >= 50:
                strengths.append("Good keyword match with JD")

            if skill_score < 40:
                gaps.append("Significant skill gaps detected")
            if exp_score < 60:
                gaps.append(f"Experience gap ({exp}y vs {exp_required}y needed)")
            if not edu_in_resume and jd_needs_edu:
                gaps.append("Education qualification mismatch")
            if kw_score < 30:
                gaps.append("Low keyword overlap with JD")

            recommendation = (
                "🟢 Strong fit — highly recommended" if overall >= 75 else
                "🟡 Good fit with some gaps — worth considering" if overall >= 55 else
                "🟠 Moderate fit — consider upskilling" if overall >= 35 else
                "🔴 Low fit for this specific role"
            )

            return ToolResult(self.name, True, {
                "candidate_name": r.get("name", ""),
                "overall_fit_score": overall,
                "breakdown": {
                    "skills_match": round(skill_score, 1),
                    "experience_match": round(exp_score, 1),
                    "education_match": round(edu_score, 1),
                    "certifications_match": round(cert_score, 1),
                    "keyword_match": round(kw_score, 1),
                },
                "matched_skills": list(set(skill_hits))[:15],
                "experience_comparison": {
                    "candidate_years": exp,
                    "jd_requirement_years": exp_required
                },
                "strengths": strengths,
                "gaps": gaps,
                "recommendation": recommendation
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━ REGISTRY ━━━

class MCPToolRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    def get_tools_description(self) -> str:
        return "\n".join(
            f"- **{t.name}**: {t.description}\n  Params: "
            + ", ".join(f"{p.name}({p.type})" for p in t.parameters)
            for t in self._tools.values()
        )

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(name, False, None, error=f"Tool '{name}' not found")
        return tool.execute(**kwargs)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def set_resume_data(self, parsed: Dict, text: str):
        for tool in self._tools.values():
            if hasattr(tool, 'set_resume'):
                tool.set_resume(parsed)
            if hasattr(tool, 'initialize') and isinstance(tool, ResumeSearchTool):
                tool.initialize(text)

    def set_jd_text(self, jd_text: str):
        tool = self._tools.get("jd_matcher")
        if tool:
            tool._jd_text = jd_text


def create_tool_registry() -> MCPToolRegistry:
    reg = MCPToolRegistry()
    reg.register(ResumeSearchTool())
    reg.register(SkillAnalyzerTool())
    reg.register(ExperienceCalculatorTool())
    reg.register(CoverLetterTool())
    reg.register(ProfileSummaryTool())
    reg.register(JDMatcherTool())
    return reg
