"""
Universal MCP Tool Server - V4
STRICT: No hallucination, only original data, no duplicates
7 tools including Education Extractor
Experience calculated with CURRENT_YEAR = 2026
Enhanced: Shared education validation, JD fallback fix, dedup fixes
"""

import json
import time
import re
from typing import Dict, List, Any, Optional, Set
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


# ═══════════════════════════════════════════════════════════════
#                    SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════

def _deduplicate_list(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order"""
    seen: Set[str] = set()
    result = []
    for item in items:
        if not item:
            continue
        item_lower = str(item).lower().strip()
        if item_lower and item_lower not in seen:
            seen.add(item_lower)
            result.append(str(item).strip())
    return result


def _clean_value(value: str) -> str:
    """Return empty string for invalid/placeholder values"""
    if not value:
        return ""
    value_str = str(value).strip()
    invalid = [
        'n/a', 'na', 'none', 'null', '-', '—', 'unknown',
        'not specified', 'not available',
        'your name', 'your email', 'your phone',
        'enter here', 'xxx', '000',
    ]
    if value_str.lower() in invalid:
        return ""
    return value_str


def _normalize_degree_key(degree: str) -> str:
    """Normalize degree name for deduplication — matches resume_parser.py"""
    if not degree:
        return ""
    d = degree.lower().strip()
    d = re.sub(r'[^a-z0-9]', '', d)

    normalizations = {
        r'bacheloroftechnology|btech|be': 'btech',
        r'masteroftechnology|mtech|me': 'mtech',
        r'bachelorofscience|bsc|bs': 'bsc',
        r'masterofscience|msc|ms': 'msc',
        r'bachelorofarts|ba': 'ba',
        r'masterofarts|ma': 'ma',
        r'bachelorofcommerce|bcom': 'bcom',
        r'masterofcommerce|mcom': 'mcom',
        r'bachelorofcomputerapplications|bca': 'bca',
        r'masterofcomputerapplications|mca': 'mca',
        r'bachelorofbusinessadministration|bba': 'bba',
        r'masterofbusinessadministration|mba': 'mba',
        r'doctorofphilosophy|phd|doctorate': 'phd',
        r'highersecondary|hsc|12th|xii|intermediate': 'hsc',
        r'secondary|ssc|10th|matriculation': 'ssc',
        r'diploma': 'diploma',
    }

    for pattern, replacement in normalizations.items():
        if re.fullmatch(pattern, d):
            return replacement
    return d


def _has_repetition_pattern(text: str) -> bool:
    """Detect repeated words/phrases indicating parsing errors."""
    if not text:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    for i in range(len(words) - 1):
        w1 = words[i].lower().strip(".,;:()")
        w2 = words[i + 1].lower().strip(".,;:()")
        if w1 == w2 and len(w1) > 1:
            return True
    counts: Dict[str, int] = {}
    for w in words:
        key = w.lower().strip(".,;:()")
        if len(key) > 1:
            counts[key] = counts.get(key, 0) + 1
    if any(c >= 3 for c in counts.values()):
        return True
    return False


def _has_garbage_pattern(text: str) -> bool:
    """Detect generic garbage patterns in any text field."""
    if not text:
        return False
    if text.count('. .') >= 2 or text.count(' . ') >= 2:
        return True
    stripped = text.strip()
    if stripped and stripped[0].islower():
        first_word = stripped.split()[0] if stripped.split() else ""
        allowed = {"in", "at", "of", "for", "the", "and", "or", "with", "to", "from"}
        if first_word and first_word not in allowed and len(first_word) < 5:
            return True
    header_words = ["education", "experience", "skills", "summary", "objective", "profile"]
    for hw in header_words:
        if text.lower().count(hw) >= 2:
            return True
    return False


def _is_valid_education_entry(edu: Dict) -> bool:
    """
    Validate education entry — matches resume_parser.py logic.
    Rejects awards, company names, fragments, repetitions.
    """
    if not isinstance(edu, dict):
        return False

    degree = str(edu.get("degree", "")).strip()
    field_of_study = str(
        edu.get("field", "")
        or edu.get("major", "")
        or edu.get("branch", "")
        or edu.get("specialization", "")
    ).strip()
    institution = str(
        edu.get("institution", "")
        or edu.get("university", "")
        or edu.get("college", "")
    ).strip()

    if not degree or len(degree) < 2:
        return False

    # Check alpha ratio in degree
    alpha_count = sum(1 for c in degree if c.isalpha())
    if alpha_count < len(degree) * 0.5:
        return False

    combined = f"{degree} {field_of_study} {institution}".lower()

    # 1. Reject award / performance / company text
    garbage_keywords = [
        "award", "performer", "performance", "star performer",
        "received", "recognition", "appreciation", "appreciated",
        "good performances", "best employee", "employee of",
        "delivered", "achievements", "achieved",
        "client project", "client ems", "project delivery",
        "team lead", "team member", "responsible for",
        "worked on", "developed", "implemented",
    ]
    for kw in garbage_keywords:
        if kw in combined:
            return False

    # 2. Reject repetition patterns
    if _has_repetition_pattern(institution):
        return False
    if _has_repetition_pattern(field_of_study):
        return False
    if _has_repetition_pattern(degree):
        return False

    # 3. Reject garbage patterns
    if _has_garbage_pattern(institution):
        return False
    if _has_garbage_pattern(field_of_study):
        return False

    # 4. Reject section-header leaks
    inst_lower = institution.lower().strip()
    if inst_lower.count("education") >= 1 and len(inst_lower) < 30:
        return False
    header_values = {
        "education", "qualifications", "academic details",
        "academic qualifications", "educational details",
        "educational qualifications", "academic background",
        "educational background",
    }
    if inst_lower in header_values:
        return False

    # 5. Reject truncated / fragment fields
    allowed_short_fields = {"it", "cs", "ai", "ml", "ee", "ec", "me", "ce"}
    if field_of_study and len(field_of_study) == 1:
        return False
    if field_of_study and len(field_of_study) < 3:
        if field_of_study.lower() not in allowed_short_fields:
            return False

    if institution and len(re.sub(r'[^a-zA-Z]', '', institution)) < 3:
        return False

    if institution and institution.split():
        first_word = institution.split()[0]
        if first_word and first_word[0].islower() and len(first_word) < 6:
            return False

    if field_of_study and field_of_study[0].islower() and len(field_of_study) < 5:
        if field_of_study.lower() not in allowed_short_fields:
            return False

    # 6. Ambiguous 2-letter degrees need school-like institution
    ambiguous_short_degrees = {"ma", "me", "ms", "ba", "be", "bs"}
    if degree.lower().strip('.') in ambiguous_short_degrees:
        school_keywords = [
            "university", "college", "institute", "school",
            "academy", "polytechnic", "iit", "nit", "iiit",
            "bits", "vit", "mit", "anna", "delhi", "mumbai",
            "vidyalaya", "vidyapeeth",
        ]
        if institution:
            if not any(kw in inst_lower for kw in school_keywords):
                return False
        else:
            return False

    # 7. Reject company names masquerading as institutions
    company_indicators = [
        "mahindra", "infosys", "wipro", "tcs", "cognizant",
        "accenture", "capgemini", "hcl", "tech mahindra",
        "client", "pvt", "ltd", "inc", "llc", "corp",
        "solutions", "technologies", "services", "consulting",
        "private limited", "limited", "software",
    ]
    if institution:
        school_kw_check = [
            "university", "college", "institute", "school",
            "academy", "polytechnic",
        ]
        has_school_kw = any(kw in inst_lower for kw in school_kw_check)
        has_company_kw = any(kw in inst_lower for kw in company_indicators)
        if has_company_kw and not has_school_kw:
            return False

    # 8. Reject suspiciously long institution text
    if institution and len(institution) > 150:
        return False

    # 9. Placeholder values
    invalid_values = [
        'n/a', 'na', 'none', 'null', 'unknown', 'not specified',
        'your degree', 'degree name', 'enter degree', 'degree here',
        'your university', 'university name', 'enter university',
        'school name', 'college name', 'institution name',
        '-', '—', 'tbd', 'pending', 'xxx', '000',
    ]
    if degree.lower() in invalid_values:
        return False

    return True


def _deduplicate_education(education_list: List[Dict]) -> List[Dict]:
    """
    Validate, then aggressively deduplicate education entries.
    Groups by normalized degree and keeps the BEST entry per group.
    """
    if not education_list:
        return []

    # Filter valid entries first
    valid = [e for e in education_list if isinstance(e, dict) and _is_valid_education_entry(e)]

    # Group by normalized degree
    degree_groups: Dict[str, List[Dict]] = {}
    for edu in valid:
        degree = str(edu.get("degree", "")).strip()
        if not degree:
            continue
        key = _normalize_degree_key(degree)
        if not key:
            key = re.sub(r'[^a-z0-9]', '', degree.lower())
        if key not in degree_groups:
            degree_groups[key] = []
        degree_groups[key].append(edu)

    # For each degree, pick the best entry (most complete)
    result: List[Dict] = []
    for _dk, entries in degree_groups.items():
        if len(entries) == 1:
            result.append(entries[0])
        else:
            best_entry = entries[0]
            best_score = -999
            for entry in entries:
                score = 0
                inst = str(
                    entry.get("institution", "")
                    or entry.get("university", "")
                    or entry.get("college", "")
                ).strip()
                fv = str(
                    entry.get("field", "")
                    or entry.get("major", "")
                    or entry.get("branch", "")
                ).strip()
                yr = str(
                    entry.get("year", "")
                    or entry.get("end_year", "")
                    or entry.get("graduation_year", "")
                ).strip()
                gpa = str(
                    entry.get("gpa", "")
                    or entry.get("cgpa", "")
                    or entry.get("grade", "")
                ).strip()

                if inst and len(inst) > 5:
                    score += 3
                if fv and len(fv) > 2:
                    score += 2
                if yr:
                    score += 2
                if gpa:
                    score += 1
                if entry.get("location"):
                    score += 1
                if _has_garbage_pattern(inst):
                    score -= 5
                if _has_repetition_pattern(inst):
                    score -= 5

                if score > best_score:
                    best_score = score
                    best_entry = entry

            result.append(best_entry)

    return result


def _deduplicate_dicts(items: List, key_fields: List[str]) -> List:
    """
    Generic deduplication for list of dicts using specified key fields.
    Falls back to string representation for non-dict items.
    """
    if not items:
        return []

    seen: Set[str] = set()
    unique = []

    for item in items:
        if not item:
            continue

        if isinstance(item, dict):
            key_parts = []
            for kf in key_fields:
                val = str(item.get(kf, "")).lower().strip()
                if val and val not in ('n/a', 'na', 'none', 'null', '-', '—'):
                    key_parts.append(val)
            item_key = "_".join(key_parts) if key_parts else ""
        else:
            item_key = str(item).lower().strip()

        if not item_key or item_key in ('n/a', 'na', 'none', 'null', '-', '—'):
            continue

        if item_key not in seen:
            seen.add(item_key)
            unique.append(item)

    return unique


def _deduplicate_certifications(cert_list: List) -> List[Dict]:
    """Remove duplicate certifications"""
    if not cert_list:
        return []

    seen: Set[str] = set()
    unique = []

    for cert in cert_list:
        if isinstance(cert, dict):
            name = str(cert.get("name", "")).strip()
        elif isinstance(cert, str):
            name = cert.strip()
        else:
            continue

        if not name or len(name) < 2:
            continue
        if name.lower() in ('n/a', 'na', 'none', 'null', '-', '—', 'unknown'):
            continue

        name_norm = re.sub(r'[^a-z0-9]', '', name.lower())[:50]
        if name_norm in seen:
            continue
        seen.add(name_norm)

        if isinstance(cert, dict):
            unique.append(cert)
        else:
            unique.append({"name": name})

    return unique


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 1: RESUME SEARCH (ENHANCED RAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResumeSearchTool(MCPTool):
    name = "resume_search"
    description = "Semantic search through uploaded resume. Returns ONLY original content from the resume. Use for any question about resume content."
    parameters = [
        ToolParameter("query", "string", "Search query - be specific"),
        ToolParameter("num_results", "integer", "Number of results", False, 5)
    ]

    def __init__(self):
        self._collection = None
        self._full_text = ""
        self._section_chunks: Dict[str, str] = {}

    def _detect_section(self, text: str) -> str:
        upper = text.upper()

        section_patterns = [
            (["EDUCATION", "ACADEMIC", "QUALIFICATION", "DEGREE", "UNIVERSITY",
              "COLLEGE", "BACHELOR", "MASTER", "MBA", "PHD", "DIPLOMA", "SCHOOL",
              "B.TECH", "B.E.", "B.SC", "M.TECH", "M.E.", "M.SC", "B.A.", "M.A.",
              "BTECH", "MTECH", "BSC", "MSC", "GRADUATE", "POST GRADUATE", "POSTGRADUATE",
              "CGPA", "GPA", "PERCENTAGE", "CLASS OF", "GRADUATED", "ALUMNUS"], "Education"),
            (["EXPERIENCE", "EMPLOYMENT", "WORK HISTORY", "PROFESSIONAL EXPERIENCE",
              "CAREER", "WORK EXPERIENCE", "JOB HISTORY", "POSITIONS HELD"], "Work Experience"),
            (["SKILL", "TECHNICAL SKILL", "COMPETENC", "TECHNOLOGIES", "TOOLS",
              "PROGRAMMING", "LANGUAGES", "FRAMEWORKS", "EXPERTISE", "PROFICIENC",
              "TECH STACK", "CORE COMPETENCIES"], "Skills"),
            (["CERTIF", "LICENS", "CREDENTIAL", "ACCREDITATION", "TRAINING",
              "PROFESSIONAL DEVELOPMENT", "COURSES"], "Certifications"),
            (["PROJECT", "PORTFOLIO", "PERSONAL PROJECT", "ACADEMIC PROJECT",
              "KEY PROJECTS", "NOTABLE PROJECTS"], "Projects"),
            (["CONTACT", "EMAIL", "PHONE", "ADDRESS", "LINKEDIN", "GITHUB",
              "PORTFOLIO", "WEBSITE", "MOBILE", "REACH ME"], "Contact"),
            (["SUMMARY", "PROFILE", "OBJECTIVE", "ABOUT ME", "PROFESSIONAL SUMMARY",
              "CAREER OBJECTIVE", "OVERVIEW", "INTRODUCTION"], "Summary"),
            (["AWARD", "HONOR", "ACHIEV", "RECOGNITION", "ACCOMPLISHMENT"], "Awards"),
            (["PUBLICATION", "RESEARCH", "PAPER", "JOURNAL", "CONFERENCE"], "Publications"),
        ]

        for keywords, section in section_patterns:
            for kw in keywords:
                if kw in upper:
                    return section
        return "General"

    def _extract_sections(self, text: str) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        lines = text.split('\n')
        current_section = "General"
        current_content: List[str] = []

        for line in lines:
            detected = self._detect_section(line)
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
        self._full_text = resume_text
        self._section_chunks = self._extract_sections(resume_text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            keep_separator=True
        )
        chunks = splitter.split_text(resume_text)

        section_aware_chunks: List[tuple] = []
        for section, content in self._section_chunks.items():
            if content.strip():
                sec_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=350, chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", ", ", " "]
                )
                for chunk in sec_splitter.split_text(content):
                    section_aware_chunks.append((chunk, section))

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
        for i, chunk in enumerate(chunks):
            section = self._detect_section(chunk)
            ids.append(f"chunk_{i}")
            docs.append(chunk)
            metas.append({"section": section, "type": "regular"})

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
            query_section = self._detect_section(query)

            section_content = ""
            if query_section != "General" and query_section in self._section_chunks:
                section_content = self._section_chunks[query_section]

            results = self._collection.query(
                query_texts=[query],
                n_results=min(k + 3, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            retrieved: List[Dict] = []
            seen_content: Set[str] = set()

            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    doc_key = doc[:80].strip().lower()
                    if doc_key in seen_content:
                        continue
                    seen_content.add(doc_key)

                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    dist = results["distances"][0][i] if results["distances"] else 0
                    relevance = round(max(0, 1 - dist), 3)

                    if query_section != "General" and meta.get("section") == query_section:
                        relevance = min(1.0, relevance + 0.2)

                    retrieved.append({
                        "content": doc,
                        "section": meta.get("section", "General"),
                        "relevance": relevance
                    })

            retrieved.sort(key=lambda x: x["relevance"], reverse=True)
            retrieved = retrieved[:k]

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
                    "context": context,
                    "note": "All content retrieved directly from uploaded resume"
                },
                metadata={"count": len(retrieved), "section_found": query_section},
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                              execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 2: SKILL ANALYZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SkillAnalyzerTool(MCPTool):
    name = "skill_analyzer"
    description = "Extract and list ALL skills from resume. Only compares against required_skills if explicitly provided. Returns ONLY skills actually found in the resume."
    parameters = [
        ToolParameter("required_skills", "string",
                      "Comma-separated skills to match (ONLY if JD uploaded or user provides skills)", False, "")
    ]

    def __init__(self):
        self._parsed: Dict = {}
        self._resume_text = ""

    def set_resume(self, p: Dict):
        self._parsed = p

    def set_text(self, text: str):
        self._resume_text = text

    def _extract_skills_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        skills: Set[str] = set()
        text_lower = text.lower()
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
            skills.update([m.strip() for m in matches if m.strip()])
        return list(skills)

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            req_input = kwargs.get("required_skills", "").strip()

            # STEP 1: Get skills from parsed resume
            all_skills: List[str] = []
            skills_data = self._parsed.get("skills", {})
            if isinstance(skills_data, dict):
                for cat in skills_data.values():
                    if isinstance(cat, list):
                        all_skills.extend(str(s) for s in cat if s)
            elif isinstance(skills_data, list):
                all_skills.extend(str(s) for s in skills_data if s)

            specs = self._parsed.get("specializations", [])
            if isinstance(specs, list):
                all_skills.extend(str(s) for s in specs if s)

            work_history = self._parsed.get("work_history", self._parsed.get("experience", []))
            if isinstance(work_history, list):
                for job in work_history:
                    if isinstance(job, dict):
                        for tech in job.get("technologies_used", job.get("technologies", [])):
                            if isinstance(tech, str) and tech:
                                all_skills.append(tech)

            if self._resume_text:
                all_skills.extend(self._extract_skills_from_text(self._resume_text))

            # STEP 2: Deduplicate
            all_skills = _deduplicate_list(all_skills)

            # STEP 3: Categorize
            categorized: Dict[str, List[str]] = {
                "programming_languages": [], "frameworks_libraries": [],
                "cloud_devops": [], "databases": [], "ai_ml": [],
                "tools_platforms": [], "other": []
            }

            for skill in all_skills:
                sl = skill.lower()
                if any(x in sl for x in ['python', 'java', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r', 'matlab', 'javascript', 'typescript']):
                    categorized["programming_languages"].append(skill)
                elif any(x in sl for x in ['react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'laravel', 'next', 'fastapi']):
                    categorized["frameworks_libraries"].append(skill)
                elif any(x in sl for x in ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s', 'terraform', 'jenkins', 'ci/cd', 'devops']):
                    categorized["cloud_devops"].append(skill)
                elif any(x in sl for x in ['sql', 'mysql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'dynamodb', 'oracle', 'cassandra']):
                    categorized["databases"].append(skill)
                elif any(x in sl for x in ['tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy', 'opencv', 'nlp', 'ml', 'ai', 'machine learning', 'deep learning']):
                    categorized["ai_ml"].append(skill)
                elif any(x in sl for x in ['git', 'jira', 'confluence', 'agile', 'scrum', 'linux', 'unix', 'power bi', 'tableau']):
                    categorized["tools_platforms"].append(skill)
                else:
                    categorized["other"].append(skill)

            for cat in categorized:
                categorized[cat] = _deduplicate_list(categorized[cat])

            # STEP 4: Build result
            result_data: Dict[str, Any] = {
                "candidate_name": self._parsed.get("name", ""),
                "total_skills_found": len(all_skills),
                "all_skills": all_skills,
                "skills_by_category": categorized,
                "note": "All skills extracted directly from resume." if all_skills else "No skills found in the resume.",
            }

            # STEP 5: Comparison
            if req_input:
                required = [s.strip().lower() for s in req_input.replace("\n", ",").split(",") if s.strip()]
                matched, partial, missing = [], [], []
                related = {
                    "machine learning": ["ml", "scikit-learn", "tensorflow", "keras", "pytorch"],
                    "deep learning": ["tensorflow", "keras", "pytorch", "neural", "cnn", "rnn"],
                    "nlp": ["natural language", "ner", "bert", "text", "spacy", "transformers"],
                    "cloud": ["aws", "gcp", "azure"],
                    "docker": ["container", "kubernetes", "k8s"],
                    "python": ["django", "flask", "pandas", "numpy"],
                    "sql": ["mysql", "postgresql", "database", "mongodb"],
                    "javascript": ["js", "react", "node", "angular", "vue", "typescript"],
                }
                all_skills_lower = [s.lower() for s in all_skills]

                for req in required:
                    if not req:
                        continue
                    if any(req in s or s in req for s in all_skills_lower):
                        matched.append(req)
                    else:
                        found = False
                        for key, aliases in related.items():
                            check = aliases + [key]
                            if any(req in a or a in req for a in check):
                                if any(any(a in s or s in a for a in check) for s in all_skills_lower):
                                    partial.append(req)
                                    found = True
                                    break
                        if not found:
                            missing.append(req)

                total = max(len(required), 1)
                pct = round(((len(matched) + len(partial) * 0.5) / total) * 100, 1)

                result_data["skill_comparison"] = {
                    "match_percentage": pct,
                    "matched_skills": _deduplicate_list(matched),
                    "partial_matches": _deduplicate_list(partial),
                    "missing_skills": _deduplicate_list(missing),
                    "total_required": len(required),
                }
            else:
                result_data["skill_comparison"] = None
                result_data["info"] = "No required skills provided for comparison."

            return ToolResult(self.name, True, result_data,
                              execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                              execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 3: EXPERIENCE CALCULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExperienceCalculatorTool(MCPTool):
    name = "experience_calculator"
    description = f"Calculate experience breakdown from resume. Returns ONLY data from the resume. Current year: {CURRENT_YEAR}."
    parameters = [
        ToolParameter("category", "string", "total, timeline, or all", False, "all")
    ]

    def __init__(self):
        self._parsed: Dict = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            work = self._parsed.get("work_history", self._parsed.get("experience", []))
            if not isinstance(work, list):
                work = []

            total = self._parsed.get("total_experience_years", 0)

            timeline: List[Dict] = []
            seen_jobs: Set[str] = set()

            for j in work:
                if not isinstance(j, dict):
                    continue
                role = j.get("title", j.get("role", j.get("position", "")))
                company = j.get("company", j.get("organization", ""))
                job_key = f"{role}_{company}".lower()
                if job_key in seen_jobs:
                    continue
                seen_jobs.add(job_key)

                start_d = j.get("start_date", j.get("from", ""))
                end_d = j.get("end_date", j.get("to", ""))
                duration = j.get("duration", "")
                if not duration and start_d:
                    duration = f"{start_d} - {end_d}"

                achievements = j.get("key_achievements", j.get("highlights", j.get("responsibilities", [])))
                if isinstance(achievements, list):
                    achievements = _deduplicate_list(achievements)[:5]

                timeline.append({
                    "role": role, "company": company,
                    "duration": duration,
                    "years": j.get("duration_years", j.get("years", 0)),
                    "type": j.get("type", "Full-time"),
                    "achievements": achievements
                })

            return ToolResult(self.name, True, {
                "candidate_name": self._parsed.get("name", ""),
                "total_years": round(total, 1),
                "calculated_as_of": f"March {CURRENT_YEAR}",
                "total_positions": len(timeline),
                "current_role": self._parsed.get("current_role", ""),
                "current_company": self._parsed.get("current_company", ""),
                "timeline": timeline,
                "note": "All experience data extracted directly from resume"
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                              execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 4: COVER LETTER GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CoverLetterTool(MCPTool):
    name = "cover_letter_generator"
    description = "Generate cover letter data using ONLY information from the resume. No fabricated achievements."
    parameters = [
        ToolParameter("job_title", "string", "Target job title"),
        ToolParameter("company_name", "string", "Target company", False, "the company")
    ]

    def __init__(self):
        self._parsed: Dict = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            r = self._parsed
            achievements: List[str] = []
            work_history = r.get("work_history", r.get("experience", []))
            if isinstance(work_history, list):
                for w in work_history[:3]:
                    if isinstance(w, dict):
                        achs = w.get("key_achievements", w.get("highlights", w.get("responsibilities", [])))
                        if isinstance(achs, list):
                            achievements.extend(str(a) for a in achs[:3] if a)
            achievements = _deduplicate_list(achievements)

            all_skills: List[str] = []
            skills_data = r.get("skills", {})
            if isinstance(skills_data, dict):
                for cat in skills_data.values():
                    if isinstance(cat, list):
                        all_skills.extend(str(s) for s in cat if s)
            elif isinstance(skills_data, list):
                all_skills = [str(s) for s in skills_data if s]
            all_skills = _deduplicate_list(all_skills)

            # Validated education
            edu = _deduplicate_education(r.get("education", []))

            return ToolResult(self.name, True, {
                "candidate_name": r.get("name", ""),
                "target_role": kwargs.get("job_title", ""),
                "target_company": kwargs.get("company_name", "the company"),
                "experience_years": r.get("total_experience_years", 0),
                "current_role": r.get("current_role", ""),
                "current_company": r.get("current_company", ""),
                "key_skills": all_skills[:10],
                "achievements": achievements[:5],
                "education": edu,
                "certifications_count": len(r.get("certifications", [])),
                "contact": {
                    "email": r.get("email", ""),
                    "phone": r.get("phone", ""),
                },
                "note": "All data extracted from resume — use this to write cover letter"
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                              execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 5: PROFILE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProfileSummaryTool(MCPTool):
    name = "profile_summary"
    description = "Extract profile summary and contact info. Returns ONLY data actually found in the resume."
    parameters = [
        ToolParameter("context", "string",
                      "Context: linkedin, portfolio, elevator_pitch, detailed", False, "detailed")
    ]

    def __init__(self):
        self._parsed: Dict = {}

    def set_resume(self, p: Dict):
        self._parsed = p

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        try:
            r = self._parsed

            # Skills (deduplicated)
            all_skills: List[str] = []
            skills_data = r.get("skills", {})
            if isinstance(skills_data, dict):
                for cat in skills_data.values():
                    if isinstance(cat, list):
                        all_skills.extend(str(s) for s in cat if s)
            elif isinstance(skills_data, list):
                all_skills = [str(s) for s in skills_data if s]
            all_skills = _deduplicate_list(all_skills)

            # Certifications (deduplicated)
            certs = _deduplicate_certifications(r.get("certifications", []))[:10]

            # Education (validated + deduplicated)
            education = _deduplicate_education(r.get("education", []))

            # Awards (deduplicated)
            awards = _deduplicate_dicts(r.get("awards", []), ["name"])

            # Specializations (deduplicated)
            specializations = _deduplicate_list(
                [str(s) for s in r.get("specializations", []) if s]
            )

            # Clean contact values
            name = _clean_value(r.get("name", ""))
            email = _clean_value(r.get("email", ""))
            phone = _clean_value(r.get("phone", ""))
            address = _clean_value(r.get("address", "")) or _clean_value(r.get("location", ""))
            linkedin = _clean_value(r.get("linkedin", ""))
            github = _clean_value(r.get("github", ""))
            portfolio = _clean_value(r.get("portfolio", ""))
            current_role = _clean_value(r.get("current_role", ""))
            current_company = _clean_value(r.get("current_company", ""))
            summary = _clean_value(r.get("professional_summary", "")) or _clean_value(r.get("summary", ""))

            experience = r.get("total_experience_years", 0)
            try:
                experience = float(experience)
            except (ValueError, TypeError):
                experience = 0

            return ToolResult(self.name, True, {
                "name": name or "Not found in resume",
                "context": kwargs.get("context", "detailed"),
                "experience_years": experience,
                "current_role": current_role or "Not specified",
                "current_company": current_company or "Not specified",
                "summary": summary or "No professional summary found in resume",
                "specializations": specializations,
                "top_skills": all_skills[:15],
                "education": education,
                "certifications": certs,
                "awards": awards,
                "contact": {
                    "email": email or "Not found",
                    "phone": phone or "Not found",
                    "address": address or "Not found",
                    "linkedin": linkedin or "Not found",
                    "github": github or "Not found",
                    "portfolio": portfolio or "Not found",
                },
                "note": "All data extracted directly from resume. Fields marked 'Not found' are not present in the document."
            }, execution_time=round(time.time() - start, 3))
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                              execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 6: JD MATCHER (ENHANCED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JDMatcherTool(MCPTool):
    name = "jd_matcher"
    description = "Compare resume against a Job Description with detailed scoring. REQUIRES JD to be uploaded. Returns skills, experience, education, location, and overall match scores."
    parameters = [
        ToolParameter("jd_text", "string",
                      "Job description text - REQUIRED for comparison. Leave empty to use uploaded JD.",
                      False, "")
    ]

    def __init__(self):
        self._parsed: Dict = {}
        self._resume_text = ""
        self._jd_text = ""  # Stored JD from sidebar upload

    def set_resume(self, p: Dict):
        self._parsed = p

    def set_text(self, text: str):
        self._resume_text = text

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()

        # Try kwargs first, then fall back to stored JD
        jd = kwargs.get("jd_text", "").strip()
        if not jd or len(jd) < 50:
            jd = self._jd_text.strip() if self._jd_text else ""

        if not jd or len(jd) < 50:
            return ToolResult(
                self.name, False, None,
                error="Job Description is required for comparison. Please upload a JD first.",
                execution_time=round(time.time() - start, 3)
            )

        try:
            jd_lower = jd.lower()
            r = self._parsed

            # ═══ SKILLS ANALYSIS ═══
            all_skills: List[str] = []
            skills_data = r.get("skills", {})
            if isinstance(skills_data, dict):
                for cat in skills_data.values():
                    if isinstance(cat, list):
                        all_skills.extend(s.lower() for s in cat if s)
            elif isinstance(skills_data, list):
                all_skills.extend(s.lower() for s in skills_data if s)
            all_skills.extend(s.lower() for s in r.get("specializations", []) if s)

            work_history = r.get("work_history", r.get("experience", []))
            if isinstance(work_history, list):
                for job in work_history:
                    if isinstance(job, dict):
                        techs = job.get("technologies_used", job.get("technologies", []))
                        if isinstance(techs, list):
                            all_skills.extend(t.lower() for t in techs if t)

            all_skills = list(set(all_skills))

            skill_hits: List[str] = []
            for s in all_skills:
                skill_words = s.split()
                if any(w in jd_lower for w in skill_words if len(w) > 2):
                    skill_hits.append(s)
            skill_hits = _deduplicate_list(skill_hits)
            skill_score = min(100, (len(skill_hits) / max(len(all_skills), 1)) * 140)

            # ═══ EXPERIENCE ANALYSIS ═══
            exp = r.get("total_experience_years", 0)
            try:
                exp = float(exp)
            except (ValueError, TypeError):
                exp = 0

            exp_matches = re.findall(r'(\d+)\+?\s*years?', jd_lower)
            exp_required = max([int(x) for x in exp_matches]) if exp_matches else 3

            if exp >= exp_required:
                exp_score = 100
            elif exp > 0:
                exp_score = min(95, (exp / max(exp_required, 1)) * 100)
            else:
                exp_score = 20

            # ═══ EDUCATION ANALYSIS ═══
            edu = r.get("education", [])
            edu_keywords = [
                "bachelor", "master", "phd", "b.tech", "m.tech", "mba",
                "b.sc", "m.sc", "degree", "b.e", "m.e", "bca", "mca",
                "graduate", "post-graduate", "diploma",
            ]

            edu_in_resume = False
            edu_level = 0
            if isinstance(edu, list):
                for e in edu:
                    if isinstance(e, dict):
                        edu_text = json.dumps(e).lower()
                        for kw in edu_keywords:
                            if kw in edu_text:
                                edu_in_resume = True
                                if kw in ["phd", "doctorate"]:
                                    edu_level = max(edu_level, 5)
                                elif kw in ["master", "mba", "m.tech", "m.sc", "m.e", "mca"]:
                                    edu_level = max(edu_level, 4)
                                elif kw in ["bachelor", "b.tech", "b.sc", "b.e", "bca", "degree", "graduate"]:
                                    edu_level = max(edu_level, 3)
                                elif kw == "diploma":
                                    edu_level = max(edu_level, 2)

            jd_needs_edu = any(k in jd_lower for k in edu_keywords)
            if edu_in_resume:
                edu_score = min(100, 70 + edu_level * 6)
            else:
                edu_score = 50 if not jd_needs_edu else 30

            # ═══ LOCATION ANALYSIS ═══
            candidate_location = (r.get("location", "") or r.get("address", "")).lower()
            location_patterns = [
                r'\b(remote|hybrid|on-?site|work from home|wfh)\b',
                r'\b(bangalore|bengaluru|hyderabad|chennai|mumbai|delhi|pune|noida|gurgaon|gurugram|kolkata)\b',
                r'\b(new york|san francisco|seattle|austin|boston|chicago|los angeles|denver)\b',
                r'\b(london|berlin|amsterdam|dublin|singapore|tokyo|sydney|toronto)\b',
            ]
            jd_locations: List[str] = []
            for pattern in location_patterns:
                matches = re.findall(pattern, jd_lower)
                jd_locations.extend(matches)
            jd_locations = list(set(jd_locations))

            if not jd_locations:
                location_score = 85
            elif "remote" in jd_locations or "work from home" in jd_locations:
                location_score = 100
            elif candidate_location:
                location_score = 100 if any(loc in candidate_location for loc in jd_locations) else 55
            else:
                location_score = 50

            # ═══ CERTIFICATIONS ═══
            certs = r.get("certifications", [])
            cert_count = len(certs) if isinstance(certs, list) else 0
            cert_score = min(100, cert_count * 18 + 20)

            # ═══ KEYWORD ANALYSIS ═══
            resume_blob = (r.get("professional_summary", r.get("summary", "")) or "").lower()
            if isinstance(work_history, list):
                for job in work_history:
                    if isinstance(job, dict):
                        resume_blob += " " + " ".join(
                            str(a) for a in job.get("key_achievements", job.get("highlights", []))
                        ).lower()
                        resume_blob += " " + " ".join(
                            str(t) for t in job.get("technologies_used", job.get("technologies", []))
                        ).lower()
            resume_blob += " " + self._resume_text.lower()

            stopwords = {'their', 'about', 'which', 'would', 'there', 'should', 'could',
                         'these', 'those', 'other', 'being', 'having', 'doing'}
            jd_words = [w for w in set(re.findall(r'\b\w+\b', jd_lower))
                        if len(w) > 4 and w.isalpha() and w not in stopwords]
            kw_hits = sum(1 for w in jd_words if w in resume_blob)
            kw_score = min(100, (kw_hits / max(len(jd_words), 1)) * 180)

            # ═══ OVERALL ═══
            overall = round(min(100,
                skill_score * 0.35 + exp_score * 0.25 + edu_score * 0.12 +
                location_score * 0.10 + cert_score * 0.08 + kw_score * 0.10
            ), 1)

            # ═══ STRENGTHS & GAPS ═══
            strengths, gaps = [], []
            if skill_score >= 70:
                strengths.append(f"Strong skill alignment ({skill_score:.0f}%)")
            elif skill_score < 45:
                gaps.append("Significant skill gaps detected")
            if exp_score >= 85:
                strengths.append(f"Experience exceeds requirements ({exp}y vs {exp_required}y)")
            elif exp_score >= 70:
                strengths.append(f"Experience meets requirements ({exp}y)")
            elif exp_score < 60:
                gaps.append(f"Experience gap ({exp}y vs {exp_required}y required)")
            if edu_score >= 80:
                strengths.append("Education qualifications match well")
            elif edu_score < 50:
                gaps.append("Education qualification concerns")
            if location_score >= 90:
                strengths.append("Location compatible")
            elif location_score < 60:
                gaps.append("Location may not align with job requirements")
            if cert_count >= 2:
                strengths.append(f"Has {cert_count} relevant certifications")
            if kw_score >= 60:
                strengths.append("Good keyword/context match with JD")
            elif kw_score < 35:
                gaps.append("Low keyword overlap — consider tailoring resume")

            recommendation = (
                "🟢 Excellent Match — Highly Recommended" if overall >= 80 else
                "🟡 Good Match — Worth Considering" if overall >= 65 else
                "🟠 Moderate Match — Review Needed" if overall >= 50 else
                "🔴 Below Average — Significant Gaps" if overall >= 35 else
                "⚫ Low Match — May Not Fit"
            )

            return ToolResult(self.name, True, {
                "candidate_name": r.get("name", ""),
                "overall_fit_score": overall,
                "breakdown": {
                    "skills_match": round(skill_score, 1),
                    "experience_match": round(exp_score, 1),
                    "education_match": round(edu_score, 1),
                    "location_match": round(location_score, 1),
                    "certifications_match": round(cert_score, 1),
                    "keyword_match": round(kw_score, 1),
                },
                "matched_skills": skill_hits[:15],
                "total_skills_found": len(all_skills),
                "experience_comparison": {
                    "candidate_years": exp,
                    "jd_requirement_years": exp_required
                },
                "location_analysis": {
                    "candidate_location": r.get("location", "") or r.get("address", ""),
                    "jd_locations": jd_locations,
                    "score": round(location_score, 1)
                },
                "strengths": _deduplicate_list(strengths),
                "gaps": _deduplicate_list(gaps),
                "recommendation": recommendation,
                "note": "Comparison based on uploaded JD and resume data"
            }, execution_time=round(time.time() - start, 3))

        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                              execution_time=round(time.time() - start, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#              TOOL 7: EDUCATION EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EducationExtractorTool(MCPTool):
    name = "education_extractor"
    description = "Extract education details from resume. Returns ONLY valid education information actually found in the resume."
    parameters = [
        ToolParameter("include_certifications", "boolean", "Include certifications", False, True)
    ]

    def __init__(self):
        self._parsed: Dict = {}
        self._resume_text = ""

    def set_resume(self, p: Dict):
        self._parsed = p

    def set_text(self, text: str):
        self._resume_text = text

    def _get_highest_degree(self, education_list: List[Dict]) -> str:
        degree_priority = {
            "phd": 10, "ph.d": 10, "doctorate": 10, "doctor": 10,
            "master": 8, "mba": 8, "m.tech": 8, "mtech": 8,
            "m.e": 8, "m.sc": 8, "mca": 8,
            "bachelor": 6, "b.tech": 6, "btech": 6, "b.e": 6,
            "b.sc": 6, "bca": 6, "bba": 6,
            "associate": 4,
            "diploma": 3, "polytechnic": 3,
            "12th": 2, "hsc": 2, "higher secondary": 2,
            "10th": 1, "ssc": 1, "secondary": 1, "matriculation": 1,
        }

        highest = ""
        highest_priority = -1

        for edu in education_list:
            if not isinstance(edu, dict):
                continue
            degree = str(edu.get("degree", "")).lower()
            for key, priority in degree_priority.items():
                if key in degree and priority > highest_priority:
                    highest_priority = priority
                    highest = edu.get("degree", "")
                    break

        return highest

    def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        include_certs = kwargs.get("include_certifications", True)

        try:
            # STEP 1: Get education
            edu_data = self._parsed.get("education", [])
            if isinstance(edu_data, dict):
                edu_data = [edu_data]
            elif not isinstance(edu_data, list):
                edu_data = []

            # STEP 2: Validate + deduplicate (uses shared functions)
            unique_education = _deduplicate_education(edu_data)

            # STEP 3: Format
            formatted: List[Dict] = []
            for edu in unique_education:
                entry = {
                    "degree": edu.get("degree", ""),
                    "field": (edu.get("field", "") or edu.get("major", "")
                              or edu.get("branch", "") or edu.get("specialization", "")),
                    "institution": (edu.get("institution", "") or edu.get("university", "")
                                    or edu.get("college", "")),
                    "location": edu.get("location", ""),
                    "year": (edu.get("year", "") or edu.get("end_year", "")
                             or edu.get("graduation_year", "")),
                    "start_year": edu.get("start_year", ""),
                    "gpa": (edu.get("gpa", "") or edu.get("cgpa", "")
                            or edu.get("grade", "") or edu.get("percentage", "")),
                    "achievements": edu.get("achievements", []) if isinstance(edu.get("achievements"), list) else []
                }
                if entry["degree"] or entry["institution"]:
                    formatted.append(entry)

            # STEP 4: Certifications
            certifications = []
            if include_certs:
                certifications = _deduplicate_certifications(self._parsed.get("certifications", []))

            # STEP 5: Highest degree
            highest_degree = self._get_highest_degree(formatted)

            return ToolResult(
                self.name, True,
                {
                    "candidate_name": self._parsed.get("name", ""),
                    "education": formatted,
                    "highest_degree": highest_degree,
                    "total_qualifications": len(formatted),
                    "certifications": certifications,
                    "total_certifications": len(certifications),
                    "note": "Education data extracted directly from resume."
                    if formatted else "No education details found in the resume.",
                },
                metadata={"edu_count": len(formatted), "cert_count": len(certifications)},
                execution_time=round(time.time() - start, 3)
            )
        except Exception as e:
            return ToolResult(self.name, False, None, error=str(e),
                              execution_time=round(time.time() - start, 3))


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
        """Store JD text in JD matcher tool for fallback access."""
        tool = self._tools.get("jd_matcher")
        if tool and hasattr(tool, '_jd_text'):
            tool._jd_text = jd_text


def create_tool_registry() -> MCPToolRegistry:
    reg = MCPToolRegistry()
    reg.register(ResumeSearchTool())
    reg.register(SkillAnalyzerTool())
    reg.register(ExperienceCalculatorTool())
    reg.register(CoverLetterTool())
    reg.register(ProfileSummaryTool())
    reg.register(JDMatcherTool())
    reg.register(EducationExtractorTool())
    return reg
