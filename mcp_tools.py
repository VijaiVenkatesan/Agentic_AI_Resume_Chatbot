"""
Universal MCP Tool Server
Works with ANY resume — tools adapt based on parsed resume data.
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# MCP BASE CLASSES
# ============================================================

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
                    p.name: {
                        "type": p.type,
                        "description": p.description
                    }
                    for p in self.parameters
                },
                "required": [
                    p.name for p in self.parameters if p.required
                ]
            }
        }


# ============================================================
# TOOL 1: RESUME SEARCH (RAG)
# ============================================================

class ResumeSearchTool(MCPTool):
    name = "resume_search"
    description = (
        "Semantic search through the uploaded resume. "
        "Use for any question about experience, projects, "
        "responsibilities, skills, education, or achievements."
    )
    parameters = [
        ToolParameter("query", "string", "Search query about the resume"),
        ToolParameter(
            "num_results", "integer",
            "Number of results", required=False, default=4
        )
    ]

    def __init__(self):
        self._collection = None
        self._resume_text = ""

    def initialize(self, resume_text: str):
        """Build vector store from resume text"""
        self._resume_text = resume_text

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        chunks = splitter.split_text(resume_text)

        client = chromadb.Client()
        ef = embedding_functions.DefaultEmbeddingFunction()

        # Delete existing collection if any
        try:
            client.delete_collection("universal_resume")
        except Exception:
            pass

        self._collection = client.create_collection(
            name="universal_resume",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

        # Detect sections
        section_kw = {
            "SUMMARY": "Summary", "PROFILE": "Summary",
            "OBJECTIVE": "Summary", "ABOUT": "Summary",
            "EXPERIENCE": "Work Experience",
            "EMPLOYMENT": "Work Experience",
            "WORK HISTORY": "Work Experience",
            "SKILLS": "Skills", "TECHNICAL": "Skills",
            "COMPETENC": "Skills",
            "EDUCATION": "Education",
            "ACADEMIC": "Education",
            "CERTIF": "Certifications",
            "LICENS": "Certifications",
            "AWARD": "Awards", "HONOR": "Awards",
            "ACHIEV": "Awards",
            "PROJECT": "Projects",
            "CONTACT": "Contact",
            "PUBLICATION": "Publications",
            "RESEARCH": "Research",
            "VOLUNTEER": "Volunteer",
        }

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            upper = chunk.upper()
            section = "General"
            for kw, sec in section_kw.items():
                if kw in upper:
                    section = sec
                    break
            ids.append(str(i))
            docs.append(chunk)
            metas.append({"section": section})

        self._collection.add(ids=ids, documents=docs, metadatas=metas)

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        query = kwargs.get("query", "")
        k = kwargs.get("num_results", 4)

        if not self._collection:
            return ToolResult(
                self.name, False, None,
                error="Resume not loaded yet"
            )

        try:
            results = self._collection.query(
                query_texts=[query], n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = (
                        results["metadatas"][0][i]
                        if results["metadatas"] else {}
                    )
                    dist = (
                        results["distances"][0][i]
                        if results["distances"] else 0
                    )
                    retrieved.append({
                        "content": doc,
                        "section": meta.get("section", "General"),
                        "relevance": round(max(0, 1 - dist), 3)
                    })

            return ToolResult(
                self.name, True,
                {
                    "query": query,
                    "results": retrieved,
                    "context": "\n\n".join(
                        r["content"] for r in retrieved
                    )
                },
                metadata={"num_results": len(retrieved)},
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(
                self.name, False, None,
                error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ============================================================
# TOOL 2: SKILL ANALYZER
# ============================================================

class SkillAnalyzerTool(MCPTool):
    name = "skill_analyzer"
    description = (
        "Analyze skill match between resume and job requirements. "
        "Provides match percentage and gap analysis."
    )
    parameters = [
        ToolParameter(
            "required_skills", "string",
            "Comma-separated required skills or job description"
        )
    ]

    def __init__(self):
        self._parsed_resume = {}

    def set_resume(self, parsed: Dict):
        self._parsed_resume = parsed

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        required_input = kwargs.get("required_skills", "")

        try:
            required = [
                s.strip().lower()
                for s in required_input.replace("\n", ",").split(",")
                if s.strip()
            ]

            # Flatten all resume skills
            all_skills = []
            skills_dict = self._parsed_resume.get("skills", {})
            for cat_skills in skills_dict.values():
                if isinstance(cat_skills, list):
                    all_skills.extend(s.lower() for s in cat_skills)

            specs = self._parsed_resume.get("specializations", [])
            all_skills.extend(s.lower() for s in specs)

            # Also check work history for implicit skills
            for job in self._parsed_resume.get("work_history", []):
                for ach in job.get("key_achievements", []):
                    all_skills.append(ach.lower())

            matched, partial, missing = [], [], []

            related_terms = {
                "machine learning": ["ml", "scikit-learn", "tensorflow", "keras"],
                "deep learning": ["tensorflow", "keras", "pytorch", "neural"],
                "nlp": ["natural language", "ner", "bert", "text"],
                "cloud": ["aws", "gcp", "azure", "cloud"],
                "api": ["rest api", "django", "flask", "fastapi"],
                "llm": ["large language", "generative ai", "gpt", "transformer"],
                "docker": ["container", "devops", "kubernetes"],
                "git": ["github", "version control", "gitlab"],
                "data science": ["pandas", "numpy", "analysis", "statistics"],
                "ai": ["artificial intelligence", "machine learning"],
                "sql": ["mysql", "postgresql", "database", "query"],
                "python": ["python", "django", "flask", "pandas"],
                "java": ["java", "spring", "jvm"],
                "javascript": ["js", "react", "node", "angular", "vue"],
            }

            for req in required:
                if not req:
                    continue

                if any(req in s or s in req for s in all_skills):
                    matched.append(req)
                else:
                    found_partial = False
                    for key, aliases in related_terms.items():
                        all_check = aliases + [key]
                        if any(req in a or a in req for a in all_check):
                            if any(
                                any(a in s or s in a for a in all_check)
                                for s in all_skills
                            ):
                                partial.append(req)
                                found_partial = True
                                break
                    if not found_partial:
                        missing.append(req)

            total = len(required) or 1
            pct = round(
                ((len(matched) + len(partial) * 0.5) / total) * 100, 1
            )

            return ToolResult(
                self.name, True,
                {
                    "match_percentage": pct,
                    "matched_skills": matched,
                    "partial_matches": partial,
                    "missing_skills": missing,
                    "total_required": len(required),
                    "candidate_name": self._parsed_resume.get("name", ""),
                },
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(
                self.name, False, None, error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ============================================================
# TOOL 3: EXPERIENCE CALCULATOR
# ============================================================

class ExperienceCalculatorTool(MCPTool):
    name = "experience_calculator"
    description = (
        "Calculate experience breakdown by years, domain, roles."
    )
    parameters = [
        ToolParameter(
            "category", "string",
            "Category: total, domain, timeline, or all",
            required=False, default="all"
        )
    ]

    def __init__(self):
        self._parsed_resume = {}

    def set_resume(self, parsed: Dict):
        self._parsed_resume = parsed

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            work = self._parsed_resume.get("work_history", [])
            total = sum(j.get("years", 0) for j in work)

            timeline = []
            for job in work:
                timeline.append({
                    "role": job.get("title", ""),
                    "company": job.get("company", ""),
                    "duration": job.get("duration", ""),
                    "years": job.get("years", 0),
                    "type": job.get("type", ""),
                    "achievements": job.get("key_achievements", [])
                })

            return ToolResult(
                self.name, True,
                {
                    "candidate_name": self._parsed_resume.get("name", ""),
                    "total_years": round(total, 1),
                    "total_positions": len(work),
                    "current_role": self._parsed_resume.get(
                        "current_role", ""
                    ),
                    "current_company": self._parsed_resume.get(
                        "current_company", ""
                    ),
                    "timeline": timeline,
                },
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(
                self.name, False, None, error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ============================================================
# TOOL 4: COVER LETTER GENERATOR
# ============================================================

class CoverLetterTool(MCPTool):
    name = "cover_letter_generator"
    description = (
        "Generate data for a tailored cover letter for a "
        "specific job role or company."
    )
    parameters = [
        ToolParameter("job_title", "string", "Target job title"),
        ToolParameter(
            "company_name", "string", "Target company",
            required=False, default="the company"
        )
    ]

    def __init__(self):
        self._parsed_resume = {}

    def set_resume(self, parsed: Dict):
        self._parsed_resume = parsed

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        job = kwargs.get("job_title", "")
        company = kwargs.get("company_name", "the company")

        try:
            r = self._parsed_resume
            achievements = []
            for w in r.get("work_history", [])[:3]:
                achievements.extend(
                    w.get("key_achievements", [])[:3]
                )

            return ToolResult(
                self.name, True,
                {
                    "candidate_name": r.get("name", ""),
                    "target_role": job,
                    "target_company": company,
                    "experience_years": r.get(
                        "total_experience_years", 0
                    ),
                    "current_role": r.get("current_role", ""),
                    "current_company": r.get("current_company", ""),
                    "key_skills": r.get("specializations", []),
                    "achievements": achievements,
                    "education": r.get("education", []),
                    "certifications_count": len(
                        r.get("certifications", [])
                    ),
                    "contact": {
                        "email": r.get("email", ""),
                        "phone": r.get("phone", ""),
                    }
                },
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(
                self.name, False, None, error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ============================================================
# TOOL 5: PROFILE SUMMARY
# ============================================================

class ProfileSummaryTool(MCPTool):
    name = "profile_summary"
    description = (
        "Generate professional summary or bio for different contexts."
    )
    parameters = [
        ToolParameter(
            "context", "string",
            "Context: linkedin, portfolio, elevator_pitch, detailed",
            required=False, default="detailed"
        )
    ]

    def __init__(self):
        self._parsed_resume = {}

    def set_resume(self, parsed: Dict):
        self._parsed_resume = parsed

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        context = kwargs.get("context", "detailed")

        try:
            r = self._parsed_resume
            all_skills = []
            for cat in r.get("skills", {}).values():
                if isinstance(cat, list):
                    all_skills.extend(cat[:5])

            return ToolResult(
                self.name, True,
                {
                    "name": r.get("name", ""),
                    "context": context,
                    "experience_years": r.get(
                        "total_experience_years", 0
                    ),
                    "current_role": r.get("current_role", ""),
                    "current_company": r.get("current_company", ""),
                    "summary": r.get("professional_summary", ""),
                    "specializations": r.get("specializations", []),
                    "top_skills": all_skills[:10],
                    "education": r.get("education", []),
                    "certifications": r.get("certifications", [])[:3],
                    "awards": r.get("awards", []),
                },
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(
                self.name, False, None, error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ============================================================
# TOOL 6: JOB MATCHER
# ============================================================

class JobMatcherTool(MCPTool):
    name = "job_matcher"
    description = (
        "Score how well the candidate fits a job description. "
        "Provides overall fit score and detailed breakdown."
    )
    parameters = [
        ToolParameter(
            "job_description", "string",
            "Job description or requirements text"
        )
    ]

    def __init__(self):
        self._parsed_resume = {}

    def set_resume(self, parsed: Dict):
        self._parsed_resume = parsed

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        jd = kwargs.get("job_description", "").lower()

        try:
            r = self._parsed_resume

            # Score components
            scores = {}

            # Skills match
            all_skills = []
            for cat in r.get("skills", {}).values():
                if isinstance(cat, list):
                    all_skills.extend(s.lower() for s in cat)

            skill_hits = sum(1 for s in all_skills if s in jd)
            scores["skills"] = min(100, (skill_hits / max(len(all_skills), 1)) * 200)

            # Experience relevance
            exp = r.get("total_experience_years", 0)
            scores["experience"] = min(100, exp * 15)

            # Education
            edu = r.get("education", [])
            scores["education"] = 70 if edu else 30

            # Certifications
            certs = r.get("certifications", [])
            scores["certifications"] = min(100, len(certs) * 15)

            overall = round(
                scores["skills"] * 0.4 +
                scores["experience"] * 0.3 +
                scores["education"] * 0.15 +
                scores["certifications"] * 0.15,
                1
            )

            return ToolResult(
                self.name, True,
                {
                    "candidate_name": r.get("name", ""),
                    "overall_fit_score": overall,
                    "breakdown": scores,
                    "strengths": [
                        k for k, v in scores.items() if v >= 70
                    ],
                    "gaps": [
                        k for k, v in scores.items() if v < 50
                    ],
                },
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(
                self.name, False, None, error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ============================================================
# TOOL REGISTRY
# ============================================================

class MCPToolRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[MCPTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        return [t.to_schema() for t in self._tools.values()]

    def get_tools_description(self) -> str:
        parts = []
        for t in self._tools.values():
            params = ", ".join(
                f"{p.name} ({p.type})" for p in t.parameters
            )
            parts.append(
                f"- **{t.name}**: {t.description}\n"
                f"  Parameters: {params}"
            )
        return "\n".join(parts)

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(name, False, None, error=f"Tool '{name}' not found")
        return tool.execute(**kwargs)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def set_resume_data(self, parsed_resume: Dict, resume_text: str):
        """Update all tools with new resume data"""
        for tool in self._tools.values():
            if hasattr(tool, 'set_resume'):
                tool.set_resume(parsed_resume)
            if hasattr(tool, 'initialize') and isinstance(tool, ResumeSearchTool):
                tool.initialize(resume_text)


def create_tool_registry() -> MCPToolRegistry:
    registry = MCPToolRegistry()
    registry.register(ResumeSearchTool())
    registry.register(SkillAnalyzerTool())
    registry.register(ExperienceCalculatorTool())
    registry.register(CoverLetterTool())
    registry.register(ProfileSummaryTool())
    registry.register(JobMatcherTool())
    return registry