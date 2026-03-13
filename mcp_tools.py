"""
Universal MCP Tool Server - Enhanced V2
7 tools including Education Extractor
Better section detection, education extraction, experience calculation
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 1: RESUME SEARCH (ENHANCED RAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResumeSearchTool(MCPTool):
    name = "resume_search"
    description = "Semantic search through uploaded resume. Use for any question about resume content including education, skills, experience, contact, certifications, projects."
    parameters = [
        ToolParameter("query", "string", "Search query - be specific"),
        ToolParameter("num_results", "integer", "Number of results", False, 5)
    ]

    def __init__(self):
        self._collection = None
        self._full_text = ""
        self._section_chunks = {}

    def _detect_section(self, text: str) -> str:
        """Improved section detection with comprehensive keywords"""
        upper = text.upper()
        
        section_patterns = [
            # Education - comprehensive patterns
            (["EDUCATION", "ACADEMIC", "QUALIFICATION", "DEGREE", "UNIVERSITY",
              "COLLEGE", "BACHELOR", "MASTER", "MBA", "PHD", "DIPLOMA", "SCHOOL",
              "B.TECH", "B.E.", "B.SC", "M.TECH", "M.E.", "M.SC", "B.A.", "M.A.",
              "BTECH", "MTECH", "BSC", "MSC", "GRADUATE", "POST GRADUATE", "POSTGRADUATE",
              "CGPA", "GPA", "PERCENTAGE", "CLASS OF", "GRADUATED", "ALUMNUS",
              "BACHELOR OF", "MASTER OF", "DOCTOR OF", "B.COM", "M.COM", "BBA", "BCA", "MCA",
              "ENGINEERING", "COMPUTER SCIENCE", "INFORMATION TECHNOLOGY"], "Education"),
            
            # Experience
            (["EXPERIENCE", "EMPLOYMENT", "WORK HISTORY", "PROFESSIONAL EXPERIENCE",
              "CAREER", "WORK EXPERIENCE", "JOB HISTORY", "POSITIONS HELD",
              "PROFESSIONAL BACKGROUND", "CAREER HISTORY"], "Work Experience"),
            
            # Skills
            (["SKILL", "TECHNICAL SKILL", "COMPETENC", "TECHNOLOGIES", "TOOLS",
              "PROGRAMMING", "LANGUAGES", "FRAMEWORKS", "EXPERTISE", "PROFICIENC",
              "TECH STACK", "CORE COMPETENCIES"], "Skills"),
            
            # Certifications
            (["CERTIF", "LICENS", "CREDENTIAL", "ACCREDITATION", "TRAINING",
              "PROFESSIONAL DEVELOPMENT", "COURSES"], "Certifications"),
            
            # Projects
            (["PROJECT", "PORTFOLIO", "PERSONAL PROJECT", "ACADEMIC PROJECT",
              "KEY PROJECTS", "NOTABLE PROJECTS"], "Projects"),
            
            # Contact
            (["CONTACT", "EMAIL", "PHONE", "ADDRESS", "LINKEDIN", "GITHUB",
              "PORTFOLIO", "WEBSITE", "MOBILE", "REACH ME"], "Contact"),
            
            # Summary
            (["SUMMARY", "PROFILE", "OBJECTIVE", "ABOUT ME", "PROFESSIONAL SUMMARY",
              "CAREER OBJECTIVE", "OVERVIEW", "INTRODUCTION"], "Summary"),
            
            # Awards
            (["AWARD", "HONOR", "ACHIEV", "RECOGNITION", "ACCOMPLISHMENT",
              "ACCOLADE"], "Awards"),
            
            # Publications
            (["PUBLICATION", "RESEARCH", "PAPER", "JOURNAL", "CONFERENCE",
              "THESIS", "DISSERTATION"], "Publications"),
        ]
        
        for keywords, section in section_patterns:
            for kw in keywords:
                if kw in upper:
                    return section
        
        return "General"

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract text by sections for direct access"""
        sections = {}
        lines = text.split('\n')
        current_section = "General"
        current_content = []
        
        for line in lines:
            detected = self._detect_section(line)
            
            # If short line matches a section, it's likely a header
            if len(line.strip()) < 60 and detected != "General":
                if current_content:
                    if current_section not in sections:
                        sections[current_section] = ""
                    sections[current_section] += "\n".join(current_content) + "\n"
                
                current_section = detected
                current_content = [line]
            else:
                current_content.append(line)
        
        if current_content:
            if current_section not in sections:
                sections[current_section] = ""
            sections[current_section] += "\n".join(current_content)
        
        return sections

    def initialize(self, resume_text: str):
        """Initialize search index with improved chunking"""
        self._full_text = resume_text
        self._section_chunks = self._extract_sections(resume_text)
        
        # Create chunks with better settings
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            keep_separator=True
        )
        chunks = splitter.split_text(resume_text)
        
        # Also create section-aware chunks
        section_aware_chunks = []
        for section, content in self._section_chunks.items():
            if content.strip():
                section_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=350,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", ", ", " "]
                )
                for chunk in section_splitter.split_text(content):
                    section_aware_chunks.append((chunk, section))
        
        # Initialize ChromaDB
        client = chromadb.Client()
        ef = embedding_functions.DefaultEmbeddingFunction()
        
        try:
            client.delete_collection("universal_resume")
        except Exception:
            pass

        self._collection = client.create_collection(
            "universal_resume",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

        ids, docs, metas = [], [], []
        
        # Add regular chunks
        for i, chunk in enumerate(chunks):
            section = self._detect_section(chunk)
            ids.append(f"chunk_{i}")
            docs.append(chunk)
            metas.append({"section": section, "type": "regular"})
        
        # Add section-aware chunks
        for i, (chunk, section) in enumerate(section_aware_chunks):
            ids.append(f"section_{i}")
            docs.append(chunk)
            metas.append({"section": section, "type": "section_aware"})
        
        if ids:
            self._collection.add(ids=ids, documents=docs, metadatas=metas)

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        query = kwargs.get("query", "")
        k = kwargs.get("num_results", 5)

        if not self._collection:
            return ToolResult(self.name, False, None, error="Resume not loaded")

        try:
            # Check if query is asking for a specific section
            query_section = self._detect_section(query)
            
            # Get section content if available
            section_content = ""
            if query_section != "General" and query_section in self._section_chunks:
                section_content = self._section_chunks[query_section]
            
            # Perform semantic search
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k + 3, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved = []
            seen_content = set()
            
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    doc_key = doc[:80].strip()
                    if doc_key in seen_content:
                        continue
                    seen_content.add(doc_key)
                    
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    dist = results["distances"][0][i] if results["distances"] else 0
                    relevance = round(max(0, 1 - dist), 3)
                    
                    # Boost section-matched results
                    if query_section != "General" and meta.get("section") == query_section:
                        relevance = min(1.0, relevance + 0.2)
                    
                    retrieved.append({
                        "content": doc,
                        "section": meta.get("section", "General"),
                        "relevance": relevance
                    })
            
            retrieved.sort(key=lambda x: x["relevance"], reverse=True)
            retrieved = retrieved[:k]
            
            # Build context
            context_parts = []
            if section_content:
                context_parts.append(f"=== {query_section} Section ===\n{section_content}")
            
            for r in retrieved:
                if r["content"] not in section_content:
                    context_parts.append(r["content"])
            
            context = "\n\n".join(context_parts)

            return ToolResult(
                self.name, True,
                {
                    "query": query,
                    "results": retrieved,
                    "section_content": section_content,
                    "context": context
                },
                metadata={"count": len(retrieved), "section_found": query_section},
                execution_time=round(time.time() - start, 3)
            )
            
        except Exception as e:
            return ToolResult(
                self.name, False, None,
                error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 2: SKILL ANALYZER (ENHANCED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SkillAnalyzerTool(MCPTool):
    name = "skill_analyzer"
    description = "Analyze skill match between resume and job requirements with gap analysis."
    parameters = [
        ToolParameter("required_skills", "string",
                      "Comma-separated skills or job description text")
    ]

    def __init__(self):
        self._parsed = {}
        self._resume_text = ""

    def set_resume(self, p: Dict):
        self._parsed = p

    def set_text(self, text: str):
        self._resume_text = text

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills directly from text"""
        skills = set()
        patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|Scala|R|MATLAB)\b',
            r'\b(React|Angular|Vue|Next\.?js|Node\.?js|Express|Django|Flask|FastAPI|Spring|Laravel|jQuery)\b',
            r'\b(AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|K8s|Terraform|Jenkins|CI/CD|DevOps)\b',
            r'\b(SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|DynamoDB|Cassandra|Oracle|SQLite)\b',
            r'\b(TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|OpenCV|NLP|ML|AI|Deep Learning)\b',
            r'\b(Git|GitHub|GitLab|Bitbucket|JIRA|Confluence|Agile|Scrum)\b',
            r'\b(HTML|CSS|SASS|LESS|Bootstrap|Tailwind|REST|GraphQL|API)\b',
            r'\b(Linux|Unix|Windows Server|Bash|PowerShell|Shell)\b',
            r'\b(Power BI|Tableau|Excel|SAP|Salesforce|ServiceNow)\b',
            r'\b(Microservices|Serverless|Lambda|S3|EC2|RDS|CloudFormation)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update([m.strip() for m in matches])
        
        return list(skills)

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            req_input = kwargs.get("required_skills", "")
            required = [s.strip().lower() for s in req_input.replace("\n", ",").split(",") if s.strip()]

            # Get skills from parsed resume
            all_skills = []
            skills_data = self._parsed.get("skills", {})
            if isinstance(skills_data, dict):
                for cat in skills_data.values():
                    if isinstance(cat, list):
                        all_skills.extend(s.lower() for s in cat)
            elif isinstance(skills_data, list):
                all_skills.extend(s.lower() for s in skills_data)
            
            all_skills.extend(s.lower() for s in self._parsed.get("specializations", []))

            # Extract from work history
            work_history = self._parsed.get("work_history", self._parsed.get("experience", []))
            if isinstance(work_history, list):
                for job in work_history:
                    if isinstance(job, dict):
                        for ach in job.get("key_achievements", job.get("highlights", [])):
                            if isinstance(ach, str):
                                all_skills.append(ach.lower())
                        for tech in job.get("technologies_used", job.get("technologies", [])):
                            if isinstance(tech, str):
                                all_skills.append(tech.lower())

            # Also extract from raw text
            if self._resume_text:
                text_skills = self._extract_skills_from_text(self._resume_text)
                all_skills.extend([s.lower() for s in text_skills])

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
                "all_detected_skills": list(set(all_skills))[:30]
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 3: EXPERIENCE CALCULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
            # Try both work_history and experience keys
            work = self._parsed.get("work_history", self._parsed.get("experience", []))
            if not isinstance(work, list):
                work = []
            
            total = self._parsed.get("total_experience_years", 0)

            timeline = []
            for j in work:
                if not isinstance(j, dict):
                    continue
                    
                start_d = j.get("start_date", j.get("from", ""))
                end_d = j.get("end_date", j.get("to", ""))
                duration = j.get("duration", "")
                if not duration and start_d:
                    duration = f"{start_d} - {end_d}"

                timeline.append({
                    "role": j.get("title", j.get("role", j.get("position", ""))),
                    "company": j.get("company", j.get("organization", "")),
                    "duration": duration,
                    "years": j.get("duration_years", j.get("years", 0)),
                    "type": j.get("type", ""),
                    "achievements": j.get("key_achievements", j.get("highlights", j.get("responsibilities", [])))[:5]
                })

            return ToolResult(self.name, True, {
                "candidate_name": self._parsed.get("name", ""),
                "total_years": round(total, 1),
                "calculated_as_of": f"July {CURRENT_YEAR}",
                "total_positions": len(timeline),
                "current_role": self._parsed.get("current_role", ""),
                "current_company": self._parsed.get("current_company", ""),
                "timeline": timeline,
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 4: COVER LETTER GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
            work_history = r.get("work_history", r.get("experience", []))
            if isinstance(work_history, list):
                for w in work_history[:3]:
                    if isinstance(w, dict):
                        achs = w.get("key_achievements", w.get("highlights", w.get("responsibilities", [])))
                        if isinstance(achs, list):
                            achievements.extend(achs[:3])

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 5: PROFILE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
            skills_data = r.get("skills", {})
            if isinstance(skills_data, dict):
                for cat in skills_data.values():
                    if isinstance(cat, list):
                        all_skills.extend(cat[:5])
            elif isinstance(skills_data, list):
                all_skills = skills_data[:15]

            return ToolResult(self.name, True, {
                "name": r.get("name", ""),
                "context": kwargs.get("context", "detailed"),
                "experience_years": r.get("total_experience_years", 0),
                "current_role": r.get("current_role", ""),
                "current_company": r.get("current_company", ""),
                "summary": r.get("professional_summary", r.get("summary", "")),
                "specializations": r.get("specializations", []),
                "top_skills": all_skills[:12],
                "education": r.get("education", []),
                "certifications": r.get("certifications", [])[:5],
                "awards": r.get("awards", []),
                "contact": {
                    "email": r.get("email", ""),
                    "phone": r.get("phone", ""),
                    "address": r.get("address", r.get("location", "")),
                    "linkedin": r.get("linkedin", ""),
                    "github": r.get("github", ""),
                }
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                            execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 6: JD MATCHER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
            skills_data = r.get("skills", {})
            if isinstance(skills_data, dict):
                for cat in skills_data.values():
                    if isinstance(cat, list):
                        all_skills.extend(s.lower() for s in cat)
            elif isinstance(skills_data, list):
                all_skills.extend(s.lower() for s in skills_data)
            
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
                           "b.sc", "m.sc", "degree", "b.e", "m.e", "bca", "mca"]
            edu_in_resume = any(
                any(k in json.dumps(e).lower() for k in edu_keywords)
                for e in edu if isinstance(e, dict)
            ) if edu else False
            jd_needs_edu = any(k in jd for k in edu_keywords)
            edu_score = 90 if edu_in_resume else (60 if not jd_needs_edu else 30)

            # Certifications
            certs = r.get("certifications", [])
            cert_score = min(100, len(certs) * 15) if certs else 20

            # Keyword overlap
            resume_blob = r.get("professional_summary", r.get("summary", "")).lower()
            work_history = r.get("work_history", r.get("experience", []))
            if isinstance(work_history, list):
                for job in work_history:
                    if isinstance(job, dict):
                        resume_blob += " " + " ".join(job.get("key_achievements", job.get("highlights", []))).lower()
                        resume_blob += " " + " ".join(job.get("technologies_used", job.get("technologies", []))).lower()
            
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 7: EDUCATION EXTRACTOR (NEW)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EducationExtractorTool(MCPTool):
    name = "education_extractor"
    description = "Extract and analyze education details including degrees, institutions, years, GPA, majors, and certifications."
    parameters = [
        ToolParameter("include_certifications", "boolean", "Include certifications", False, True)
    ]

    def __init__(self):
        self._parsed = {}
        self._resume_text = ""

    def set_resume(self, p: Dict):
        self._parsed = p

    def set_text(self, text: str):
        self._resume_text = text

    def _extract_education_from_text(self, text: str) -> List[Dict]:
        """Extract education directly from text using patterns"""
        education = []
        
        # Degree patterns
        degree_patterns = [
            (r'(Bachelor[\'s]?\s*(?:of\s*)?(?:Science|Arts|Engineering|Technology|Commerce|Business Administration)?)', 'Bachelor'),
            (r'(B\.?(?:Tech|E|Sc|A|Com|B\.?A|S|CA|BA)\.?)\s*(?:in\s*)?([\w\s,]+)?', 'Bachelor'),
            (r'(Master[\'s]?\s*(?:of\s*)?(?:Science|Arts|Engineering|Technology|Business Administration)?)', 'Master'),
            (r'(M\.?(?:Tech|E|Sc|A|B\.?A|S|CA|BA)\.?|MBA)\s*(?:in\s*)?([\w\s,]+)?', 'Master'),
            (r'(Ph\.?D\.?|Doctorate)\s*(?:in\s*)?([\w\s,]+)?', 'PhD'),
            (r'(Diploma)\s*(?:in\s*)?([\w\s,]+)?', 'Diploma'),
            (r'(High School|HSC|SSC|12th|10th|Secondary|Higher Secondary)', 'School'),
        ]
        
        # GPA patterns
        gpa_patterns = [
            r'(?:GPA|CGPA|CPI)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?',
            r'(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?\s*(?:GPA|CGPA|CPI)',
            r'(\d{1,2}(?:\.\d+)?)\s*%\s*(?:marks?|score)?',
            r'(?:First\s*Class|Distinction|Honors?)',
        ]
        
        # Institution patterns
        institution_patterns = [
            r'([A-Z][A-Za-z\s\.]+(?:University|College|Institute|School|Academy|IIT|NIT|BITS|IIIT))',
            r'(?:from|at)\s+([A-Z][A-Za-z\s\.]+)',
        ]
        
        # Year patterns
        year_patterns = [
            r'(19|20)\d{2}\s*[-–to]\s*(19|20)\d{2}',
            r'(?:class of|graduated?|batch|year)[:\s]*(19|20)\d{2}',
            r'(19|20)\d{2}',
        ]
        
        # Split text into potential education entries
        lines = text.split('\n')
        current_edu = {}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_edu and any(current_edu.values()):
                    education.append(current_edu)
                    current_edu = {}
                continue
            
            # Check for degree
            for pattern, degree_type in degree_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    current_edu['degree'] = match.group(1).strip()
                    current_edu['degree_type'] = degree_type
                    if match.lastindex and match.lastindex >= 2 and match.group(2):
                        current_edu['field'] = match.group(2).strip()
                    break
            
            # Check for institution
            for pattern in institution_patterns:
                match = re.search(pattern, line)
                if match:
                    inst = match.group(1).strip()
                    if len(inst) > 3 and inst not in ['The', 'And', 'For']:
                        current_edu['institution'] = inst
                    break
            
            # Check for year
            for pattern in year_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    current_edu['year'] = match.group(0)
                    break
            
            # Check for GPA
            for pattern in gpa_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    current_edu['gpa'] = match.group(0)
                    break
        
        if current_edu and any(current_edu.values()):
            education.append(current_edu)
        
        return education

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        include_certs = kwargs.get("include_certifications", True)

        try:
            # Get from parsed resume
            education_list = []
            edu_data = self._parsed.get("education", [])
            if isinstance(edu_data, list):
                education_list = edu_data
            elif isinstance(edu_data, dict):
                education_list = [edu_data]
            
            # Also extract from text
            text_education = []
            if self._resume_text:
                text_education = self._extract_education_from_text(self._resume_text)
            
            # Merge and format
            all_education = []
            seen = set()
            
            for edu in education_list + text_education:
                if not isinstance(edu, dict):
                    continue
                
                degree = edu.get("degree", edu.get("title", ""))
                institution = edu.get("institution", edu.get("university", edu.get("school", edu.get("college", ""))))
                
                # Skip duplicates
                key = f"{degree}_{institution}".lower()[:50]
                if key in seen or not (degree or institution):
                    continue
                seen.add(key)
                
                all_education.append({
                    "degree": degree,
                    "field": edu.get("field", edu.get("major", edu.get("specialization", edu.get("branch", "")))),
                    "institution": institution,
                    "year": edu.get("year", edu.get("graduation_year", edu.get("end_date", edu.get("passing_year", "")))),
                    "gpa": edu.get("gpa", edu.get("cgpa", edu.get("grade", edu.get("percentage", "")))),
                    "honors": edu.get("honors", edu.get("distinction", ""))
                })
            
            # Get certifications
            certifications = []
            if include_certs:
                certs = self._parsed.get("certifications", [])
                if isinstance(certs, list):
                    certifications = certs
            
            # Determine highest degree
            degree_rank = {
                "phd": 5, "doctorate": 5, "doctor": 5,
                "master": 4, "mba": 4, "m.tech": 4, "mtech": 4, "m.e": 4, "m.sc": 4, "m.s": 4, "mca": 4,
                "bachelor": 3, "b.tech": 3, "btech": 3, "b.e": 3, "b.sc": 3, "b.s": 3, "b.a": 3, "bca": 3, "b.com": 3,
                "diploma": 2,
                "high school": 1, "secondary": 1, "12th": 1, "hsc": 1, "10th": 1, "ssc": 1
            }
            
            highest_degree = ""
            highest_rank = 0
            for edu in all_education:
                degree_lower = edu.get("degree", "").lower()
                for deg, rank in degree_rank.items():
                    if deg in degree_lower and rank > highest_rank:
                        highest_rank = rank
                        highest_degree = edu.get("degree", "")

            return ToolResult(
                self.name, True,
                {
                    "education": all_education,
                    "highest_degree": highest_degree,
                    "total_qualifications": len(all_education),
                    "certifications": certifications,
                    "total_certifications": len(certifications),
                    "candidate_name": self._parsed.get("name", "")
                },
                metadata={"edu_count": len(all_education), "cert_count": len(certifications)},
                execution_time=round(time.time() - start, 3)
            )
            
        except Exception as e:
            return ToolResult(
                self.name, False, None,
                error=str(e),
                execution_time=round(time.time() - start, 3)
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                      REGISTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MCPToolRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._resume_text = ""

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
        self._resume_text = text
        for tool in self._tools.values():
            if hasattr(tool, 'set_resume'):
                tool.set_resume(parsed)
            if hasattr(tool, 'set_text'):
                tool.set_text(text)
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
    reg.register(EducationExtractorTool())  # NEW TOOL
    return reg
