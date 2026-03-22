"""
Bulk Resume Processor - Enterprise HR Feature
Handles multiple resume uploads and batch JD matching
Version: 5.0
  - Imports shared validation from resume_parser.py (single-mode parity)
  - Robust name cleaning (strips trailing junk, handles Initial.Name)
  - Education formatting fix (no degree-in-institution duplication)
  - Independent experience recalculation from dates
  - Consistent email / location cleaning across modes
  - Deduplication everywhere
"""

import time
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import pandas as pd

from document_processor import process_uploaded_file
from resume_parser import parse_resume_with_llm

# ═══════════════════════════════════════════════════════════════
#  Import shared utilities from resume_parser for single-mode
#  parity.  If the import fails (e.g. refactor), local fallbacks
#  are used — the processor never breaks.
# ═══════════════════════════════════════════════════════════════
try:
    from resume_parser import (
        _clean_email,
        _is_valid_location,
        _clean_location,
        _extract_name_from_text  as _rp_extract_name,
        _is_valid_name           as _rp_is_valid_name,
        _clean_name              as _rp_clean_name,
        _calc_duration_from_dates,
    )
    _RP_AVAILABLE = True
except ImportError:
    _RP_AVAILABLE = False

    # ── Minimal local fallbacks ──
    def _clean_email(e):
        return e.strip() if e else ""

    def _is_valid_location(loc):
        return bool(loc and len(loc.strip()) >= 2)

    def _clean_location(loc):
        return loc.strip() if loc else ""

    def _rp_extract_name(text):
        return ""

    def _rp_is_valid_name(name):
        return bool(name and len(name.split()) >= 2)

    def _rp_clean_name(name):
        return name.strip() if name else ""

    def _calc_duration_from_dates(start, end):
        return 0.0


CURRENT_YEAR = 2026


# ═══════════════════════════════════════════════════════════════
#                       DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class CandidateResult:
    """Result for a single candidate in bulk processing"""
    file_name: str
    candidate_name: str
    email: str
    phone: str
    location: str
    current_role: str
    current_company: str
    total_experience: float
    highest_education: str

    # Match scores (0-100)
    overall_score: float = 0.0
    skills_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    location_score: float = 0.0
    keyword_score: float = 0.0

    # Details — DEDUPLICATED
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendation: str = ""

    # Processing info
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    rank: int = 0

    # Raw data for detailed view
    parsed_resume: Dict = field(default_factory=dict)
    resume_text: str = ""


@dataclass
class BulkProcessingResult:
    """Result of bulk resume processing"""
    total_resumes: int
    successful: int
    failed: int
    processing_time: float
    candidates: List[CandidateResult] = field(default_factory=list)
    jd_summary: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
#                  DEDUPLICATION UTILITIES
# ═══════════════════════════════════════════════════════════════

def _deduplicate_list(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order"""
    if not items:
        return []
    seen: Set[str] = set()
    result = []
    for item in items:
        if not item:
            continue
        item_normalized = str(item).strip().lower()
        if item_normalized and item_normalized not in seen:
            seen.add(item_normalized)
            result.append(str(item).strip())
    return result


def _normalize_degree(degree: str) -> str:
    """Normalize degree name for comparison"""
    if not degree:
        return ""
    degree_lower = degree.lower().strip()
    degree_lower = re.sub(r'^(degree|diploma|certificate)\s*(in|of)?\s*', '', degree_lower)
    degree_lower = re.sub(r'\s*(degree|diploma|certificate)$', '', degree_lower)

    normalizations = {
        r'b\.?tech|bachelor\s*of\s*technology|btech': 'btech',
        r'm\.?tech|master\s*of\s*technology|mtech': 'mtech',
        r'b\.?e\.?|bachelor\s*of\s*engineering': 'be',
        r'm\.?e\.?|master\s*of\s*engineering': 'me',
        r'b\.?sc\.?|bachelor\s*of\s*science|bsc': 'bsc',
        r'm\.?sc\.?|master\s*of\s*science|msc': 'msc',
        r'b\.?a\.?|bachelor\s*of\s*arts': 'ba',
        r'm\.?a\.?|master\s*of\s*arts': 'ma',
        r'b\.?com\.?|bachelor\s*of\s*commerce|bcom': 'bcom',
        r'm\.?com\.?|master\s*of\s*commerce|mcom': 'mcom',
        r'b\.?c\.?a\.?|bachelor\s*of\s*computer\s*applications?|bca': 'bca',
        r'm\.?c\.?a\.?|master\s*of\s*computer\s*applications?|mca': 'mca',
        r'b\.?b\.?a\.?|bachelor\s*of\s*business\s*administration|bba': 'bba',
        r'm\.?b\.?a\.?|master\s*of\s*business\s*administration|mba': 'mba',
        r'ph\.?d\.?|doctorate|doctor\s*of\s*philosophy': 'phd',
    }
    for pattern, replacement in normalizations.items():
        if re.search(pattern, degree_lower):
            return replacement
    return re.sub(r'[^a-z0-9]', '', degree_lower)


# ═══════════════════════════════════════════════════════════════
#         EDUCATION ENTRY VALIDATION  (aligned with resume_parser)
# ═══════════════════════════════════════════════════════════════

def _has_repetition_pattern(text: str) -> bool:
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
    return any(c >= 3 for c in counts.values())


def _has_garbage_pattern(text: str) -> bool:
    if not text:
        return False
    if text.count('. .') >= 2 or text.count(' . ') >= 2:
        return True
    s = text.strip()
    if s and s[0].islower():
        fw = s.split()[0] if s.split() else ""
        allowed = {"in", "at", "of", "for", "the", "and", "or", "with", "to", "from"}
        if fw and fw not in allowed and len(fw) < 5:
            return True
    for hw in ["education", "experience", "skills", "summary", "objective", "profile"]:
        if text.lower().count(hw) >= 2:
            return True
    return False


def _is_valid_education_entry(edu: Dict) -> bool:
    """
    Return True only when the education dict genuinely describes a
    degree / diploma.  Rejects entries built from award text, company
    names, section-header contamination, or truncated fragments.
    Aligned with resume_parser.py logic.
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

    # Alpha ratio check
    if sum(1 for c in degree if c.isalpha()) < len(degree) * 0.5:
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

    # 2. Reject repetition / garbage patterns
    if _has_repetition_pattern(institution) or _has_repetition_pattern(field_of_study) or _has_repetition_pattern(degree):
        return False
    if _has_garbage_pattern(institution) or _has_garbage_pattern(field_of_study):
        return False

    # 3. Reject section-header leaks
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

    # 4. Reject truncated / fragment fields
    allowed_short = {"it", "cs", "ai", "ml", "ee", "ec", "me", "ce"}
    if field_of_study and len(field_of_study) == 1:
        return False
    if field_of_study and len(field_of_study) < 3 and field_of_study.lower() not in allowed_short:
        return False
    if institution and len(re.sub(r'[^a-zA-Z]', '', institution)) < 3:
        return False
    if institution and institution.split():
        first_word = institution.split()[0]
        if first_word and first_word[0].islower() and len(first_word) < 6:
            return False
    if field_of_study and field_of_study[0].islower() and len(field_of_study) < 5:
        if field_of_study.lower() not in allowed_short:
            return False

    # 5. Ambiguous 2-letter degrees need school-like institution
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

    # 6. Reject company names masquerading as institutions
    company_indicators = [
        "mahindra", "infosys", "wipro", "tcs", "cognizant",
        "accenture", "capgemini", "hcl", "tech mahindra",
        "client", "pvt", "ltd", "inc", "llc", "corp",
        "solutions", "technologies", "services", "consulting",
        "private limited", "limited", "software",
    ]
    if institution:
        school_kw_check = ["university", "college", "institute", "school", "academy", "polytechnic"]
        has_school_kw = any(kw in inst_lower for kw in school_kw_check)
        has_company_kw = any(kw in inst_lower for kw in company_indicators)
        if has_company_kw and not has_school_kw:
            return False

    # 7. Too-long institution
    if institution and len(institution) > 150:
        return False

    # 8. Placeholder values
    invalid_values = [
        'n/a', 'na', 'none', 'null', 'unknown', 'not specified',
        '-', '—', 'tbd', 'pending', 'xxx', '000',
    ]
    if degree.lower() in invalid_values:
        return False

    return True


# ═══════════════════════════════════════════════════════════════
#         CANDIDATE NAME VALIDATION  (ENHANCED)
# ═══════════════════════════════════════════════════════════════

# ANY single word match → reject the candidate name
_STRONG_NAME_BLACKLIST: Set[str] = {
    # Resume / document labels
    "resume", "cv", "curriculum", "vitae", "biodata",
    # Section headers
    "objective", "summary", "profile", "overview",
    "education", "experience", "skills", "certifications",
    "references", "declaration", "signature",
    "qualifications", "achievements", "accomplishments",
    "responsibilities", "duties",
    # HR terms
    "candidate", "applicant", "recruitment", "hiring",
    "position", "vacancy",
    # Section labels
    "contact", "details", "information", "personal",
    "professional", "technical", "academic",
    "employment", "employer",
    # Job titles
    "engineer", "developer", "manager", "analyst",
    "consultant", "designer", "architect", "administrator",
    "senior", "junior", "lead", "head", "director",
    "executive", "intern", "trainee",
    # Industry / org words
    "software", "hardware", "technology", "technologies",
    "university", "college", "school", "institute",
    "company", "corporation", "organization",
    "private", "limited", "ltd", "inc", "corp", "llc", "pvt",
    # Degree words
    "bachelor", "master", "doctor", "phd", "mba",
    "btech", "mtech", "degree", "diploma",
    # Other
    "project", "projects", "work", "history",
    "reference", "present", "current",
    # Months
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # Days
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    # Locations (common cities that appear as headers)
    "bangalore", "mumbai", "delhi", "chennai", "hyderabad",
    "pune", "kolkata", "india", "singapore",
    # ── Contact / junk words that leak into names ──
    "mobile", "phone", "email", "tel", "telephone",
    "cell", "fax", "website", "linkedin", "github",
    # ── Time / context words ──
    "period", "duration", "tenure", "since", "from",
    # ── Misc ──
    "about", "more", "regarding", "section", "introduction",
    "career", "table", "contents", "index", "appendix",
    "voice", "message", "routing", "optimization",
}

# 2 or more word matches → reject
_SOFT_NAME_BLACKLIST: Set[str] = {
    "voice", "message", "support", "service", "process",
    "system", "application", "development",
    "client", "customer", "business",
    "department", "team",
    "specialist", "coordinator", "associate",
    "responsible", "working", "developing", "managing",
    "leading", "building", "creating",
    "page", "date", "address", "city", "state", "country",
    "number", "no",
}


def _clean_candidate_name(name: str) -> str:
    """
    Remove trailing / leading contact-keywords and junk words from a
    candidate name before validation.

    "M.SREEKANTH Mobile"       → "M.SREEKANTH"
    "Mobile M.SREEKANTH"       → "M.SREEKANTH"
    "SupportLogic Period"      → "SupportLogic"   (then fails 1-word check → fallback)
    "John Doe Resume"          → "John Doe"
    """
    if not name:
        return ""

    trailing_junk = {
        'mobile', 'phone', 'email', 'tel', 'telephone',
        'cell', 'fax', 'contact', 'number', 'period',
        'duration', 'address', 'website', 'linkedin', 'github',
        'portfolio', 'resume', 'cv', 'tenure', 'since',
    }

    words = name.strip().split()

    # Remove trailing junk words
    while words and words[-1].lower().strip('.,;:()') in trailing_junk:
        words.pop()

    # Remove leading junk words
    while words and words[0].lower().strip('.,;:()') in trailing_junk:
        words.pop(0)

    cleaned = ' '.join(words).strip()

    # Also delegate to resume_parser's _clean_name for ALL-CAPS / Initial.Name
    if _RP_AVAILABLE and cleaned:
        cleaned = _rp_clean_name(cleaned)

    return cleaned


def _is_valid_candidate_name(name: str) -> bool:
    """
    Strict validation that *name* looks like a real person name.
    Rejects section headers, job titles, random phrases, and junk.
    Handles Initial.Name format: "M.SREEKANTH", "K.Ravi".
    """
    if not name:
        return False

    name_clean = name.strip()
    name_lower = name_clean.lower()

    # Exact-match rejects
    exact_rejects = {
        "", "n/a", "na", "none", "unknown", "candidate",
        "your name", "first last", "firstname lastname",
        "name", "full name", "[name]", "<name>", "(name)",
        "enter name", "type name", "not available",
        "curriculum vitae", "resume", "cv",
        "more about me", "about me",
    }
    if name_lower in exact_rejects:
        return False

    words = name_clean.split()

    # Handle "M.SREEKANTH" or "K.Ravi" — initial.name counts as 2 words
    if len(words) == 1:
        dot_split = re.split(r'\.', name_clean)
        dot_split = [p.strip() for p in dot_split if p.strip()]
        if len(dot_split) >= 2:
            words = dot_split
        else:
            return False

    if not (2 <= len(words) <= 4):
        if not (len(words) == 2):
            return False

    # Each word: reasonable length, mostly alphabetic
    for w in words:
        stripped = w.strip(".-'")
        if not stripped or len(stripped) > 20:
            return False
        # Allow single-letter initials
        if len(stripped) == 1:
            if not stripped.isalpha():
                return False
            continue
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.8:
            return False

    # Overall alpha ratio
    alpha_chars = sum(1 for c in name_clean if c.isalpha() or c.isspace() or c in '.-')
    if alpha_chars / max(len(name_clean), 1) < 0.80:
        return False

    # Strong blacklist: ANY hit → reject
    check_words = [w.strip(".-'") for w in words]
    for w in check_words:
        if w.lower().strip(".,;:()") in _STRONG_NAME_BLACKLIST and len(w) > 1:
            return False

    # Soft blacklist: 2+ hits → reject
    soft_hits = sum(
        1 for w in check_words
        if w.lower().strip(".,;:()") in _SOFT_NAME_BLACKLIST and len(w) > 1
    )
    if soft_hits >= 2:
        return False

    # Names are usually Title Case or ALL CAPS, not all lowercase
    if name_clean == name_clean.lower():
        return False

    return True


# ═══════════════════════════════════════════════════════════════
#                  JD REQUIREMENTS EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_jd_requirements(jd_text: str) -> Dict:
    """Extract key requirements from Job Description"""
    jd_lower = jd_text.lower()

    requirements: Dict[str, Any] = {
        "required_skills": [],
        "preferred_skills": [],
        "min_experience_years": 0,
        "max_experience_years": 99,
        "required_education": [],
        "preferred_locations": [],
        "job_title": "",
        "company": "",
        "keywords": []
    }

    # Experience
    exp_patterns = [
        r'(\d+)\s*\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\s*-\s*(\d+)\s*years?',
        r'minimum\s*(\d+)\s*years?',
        r'at\s*least\s*(\d+)\s*years?',
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, jd_lower)
        if match:
            if len(match.groups()) == 2:
                requirements["min_experience_years"] = int(match.group(1))
                requirements["max_experience_years"] = int(match.group(2))
            else:
                requirements["min_experience_years"] = int(match.group(1))
            break

    # Skills
    skill_patterns = [
        r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|golang|rust|swift|kotlin|php|scala|r|matlab|perl)\b',
        r'\b(react|reactjs|angular|angularjs|vue|vuejs|nextjs|next\.js|nodejs|node\.js|express|expressjs)\b',
        r'\b(django|flask|fastapi|spring|springboot|laravel|rails|ruby on rails|asp\.net|\.net)\b',
        r'\b(aws|amazon web services|azure|microsoft azure|gcp|google cloud|cloud computing)\b',
        r'\b(docker|kubernetes|k8s|terraform|ansible|jenkins|ci/cd|cicd|devops|gitops)\b',
        r'\b(sql|mysql|postgresql|postgres|mongodb|redis|elasticsearch|dynamodb|cassandra|oracle|sqlite)\b',
        r'\b(tensorflow|pytorch|keras|scikit-learn|sklearn|pandas|numpy|opencv|nltk|spacy)\b',
        r'\b(machine learning|deep learning|artificial intelligence|ai|ml|nlp|natural language processing)\b',
        r'\b(git|github|gitlab|bitbucket|jira|confluence|agile|scrum|kanban)\b',
        r'\b(html|css|sass|less|bootstrap|tailwind|tailwindcss|webpack|babel)\b',
        r'\b(rest|restful|graphql|api|microservices|serverless|lambda)\b',
        r'\b(linux|unix|bash|shell|powershell|windows server)\b',
        r'\b(power bi|tableau|excel|sap|salesforce|servicenow)\b',
        r'\b(data science|data analysis|data engineering|etl|data pipeline)\b',
        r'\b(cybersecurity|security|networking|tcp/ip|dns|http|https)\b',
    ]
    skills_found: Set[str] = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, jd_lower)
        skills_found.update([m.strip() for m in matches if m.strip()])
    requirements["required_skills"] = list(skills_found)

    # Education
    edu_patterns = [
        r"(bachelor'?s?|master'?s?|phd|ph\.d|doctorate|mba|m\.b\.a)",
        r'(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?sc|m\.?sc|b\.?ca|m\.?ca|b\.?com|m\.?com)',
        r'(computer science|information technology|software engineering|engineering|mathematics|statistics|physics)',
        r'(degree|graduate|post.?graduate|undergraduate)',
    ]
    edu_found: Set[str] = set()
    for pattern in edu_patterns:
        matches = re.findall(pattern, jd_lower)
        edu_found.update([m.strip() for m in matches if m.strip()])
    requirements["required_education"] = list(edu_found)

    # Locations
    location_patterns = [
        r'\b(remote|hybrid|on-?site|work from home|wfh|telecommute)\b',
        r'\b(bangalore|bengaluru|hyderabad|chennai|mumbai|delhi|ncr|pune|noida|gurgaon|gurugram|kolkata)\b',
        r'\b(new york|nyc|san francisco|sf|seattle|austin|boston|chicago|los angeles|la|denver|atlanta)\b',
        r'\b(london|berlin|amsterdam|dublin|singapore|tokyo|sydney|toronto|vancouver)\b',
        r'\b(india|usa|us|uk|united states|united kingdom|canada|australia|germany|netherlands)\b',
    ]
    locations_found: Set[str] = set()
    for pattern in location_patterns:
        matches = re.findall(pattern, jd_lower)
        locations_found.update([m.strip() for m in matches if m.strip()])
    requirements["preferred_locations"] = list(locations_found)

    # Keywords
    words = re.findall(r'\b[a-z]{4,}\b', jd_lower)
    word_freq: Dict[str, int] = {}
    stopwords = {
        'with', 'have', 'will', 'that', 'this', 'from', 'they', 'been', 'were',
        'said', 'each', 'which', 'their', 'would', 'there', 'could', 'other',
        'into', 'more', 'some', 'than', 'them', 'these', 'then', 'only', 'over',
        'such', 'about', 'should', 'your', 'work', 'experience', 'years', 'team',
        'ability', 'skills', 'role', 'position', 'looking', 'strong', 'good',
        'excellent', 'required', 'preferred', 'must', 'including', 'working',
        'understanding', 'knowledge', 'using', 'responsibilities',
        'requirements', 'qualifications', 'company', 'join', 'opportunity',
        'environment', 'culture', 'benefits', 'salary', 'compensation',
        'what', 'when', 'where', 'while', 'being', 'having', 'doing', 'make',
        'like', 'just', 'also', 'well', 'back', 'after', 'before', 'between',
    }
    for word in words:
        if word not in stopwords and len(word) >= 4:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    requirements["keywords"] = [w[0] for w in sorted_words[:50]]

    return requirements


# ═══════════════════════════════════════════════════════════════
#                  MATCH SCORE CALCULATION
# ═══════════════════════════════════════════════════════════════

def calculate_match_scores(
    parsed_resume: Dict,
    resume_text: str,
    jd_text: str,
    jd_requirements: Dict
) -> Dict:
    """Calculate detailed match scores between resume and JD"""
    scores: Dict[str, Any] = {
        "overall_score": 0.0, "skills_score": 0.0, "experience_score": 0.0,
        "education_score": 0.0, "location_score": 0.0, "keyword_score": 0.0,
        "matched_skills": [], "missing_skills": [],
        "strengths": [], "gaps": [], "recommendation": ""
    }

    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()

    # ─── 1. SKILLS MATCH (35 %) ───
    all_skills: List[str] = []
    skills_data = parsed_resume.get("skills", {})
    if isinstance(skills_data, dict):
        for category in skills_data.values():
            if isinstance(category, list):
                all_skills.extend([s.lower().strip() for s in category if s])
    elif isinstance(skills_data, list):
        all_skills.extend([s.lower().strip() for s in skills_data if s])

    specs = parsed_resume.get("specializations", [])
    if isinstance(specs, list):
        all_skills.extend([s.lower().strip() for s in specs if s])

    work_history = parsed_resume.get("work_history", parsed_resume.get("experience", []))
    if isinstance(work_history, list):
        for job in work_history:
            if isinstance(job, dict):
                techs = job.get("technologies_used", job.get("technologies", []))
                if isinstance(techs, list):
                    all_skills.extend([t.lower().strip() for t in techs if t])

    all_skills = list(set(all_skills))
    required_skills = [s.lower().strip() for s in jd_requirements.get("required_skills", []) if s]

    if required_skills:
        matched: List[str] = []
        missing: List[str] = []
        skill_aliases = {
            "javascript": ["js", "ecmascript"], "typescript": ["ts"],
            "python": ["py"], "kubernetes": ["k8s"],
            "amazon web services": ["aws"], "google cloud": ["gcp"],
            "microsoft azure": ["azure"], "machine learning": ["ml"],
            "artificial intelligence": ["ai"],
            "natural language processing": ["nlp"], "deep learning": ["dl"],
            "react": ["reactjs", "react.js"], "node": ["nodejs", "node.js"],
            "angular": ["angularjs"], "vue": ["vuejs", "vue.js"],
            "postgres": ["postgresql"], "mongo": ["mongodb"],
        }
        for req_skill in required_skills:
            found = False
            for resume_skill in all_skills:
                if req_skill in resume_skill or resume_skill in req_skill:
                    matched.append(req_skill); found = True; break
            if not found:
                aliases = skill_aliases.get(req_skill, [])
                for alias in aliases:
                    for resume_skill in all_skills:
                        if alias in resume_skill or resume_skill in alias:
                            matched.append(req_skill); found = True; break
                    if found:
                        break
            if not found:
                if req_skill in resume_lower:
                    matched.append(req_skill)
                else:
                    missing.append(req_skill)
        scores["matched_skills"] = _deduplicate_list(matched)
        scores["missing_skills"] = _deduplicate_list(missing)
        match_ratio = len(scores["matched_skills"]) / len(required_skills) if required_skills else 0
        scores["skills_score"] = round(min(100, match_ratio * 110), 1)
    else:
        scores["skills_score"] = 70

    # ─── 2. EXPERIENCE MATCH (25 %) ───
    candidate_exp = parsed_resume.get("total_experience_years", 0)
    try:
        candidate_exp = float(candidate_exp)
    except (ValueError, TypeError):
        candidate_exp = 0

    min_exp = jd_requirements.get("min_experience_years", 0)
    max_exp = jd_requirements.get("max_experience_years", 99)

    if min_exp > 0:
        if candidate_exp >= min_exp:
            scores["experience_score"] = 100 if candidate_exp <= max_exp else max(65, 100 - ((candidate_exp - max_exp) * 3))
        else:
            scores["experience_score"] = round(min(95, (candidate_exp / min_exp) * 100), 1) if candidate_exp > 0 else 20
    else:
        scores["experience_score"] = 80 if candidate_exp > 0 else 50

    # ─── 3. EDUCATION MATCH (15 %) ───
    education = parsed_resume.get("education", [])
    required_edu = jd_requirements.get("required_education", [])
    edu_text = json.dumps(education).lower() if isinstance(education, (list, dict)) else ""

    degree_levels = {
        "phd": 6, "ph.d": 6, "doctorate": 6, "doctoral": 6,
        "master": 5, "mba": 5, "m.tech": 5, "mtech": 5,
        "m.e": 5, "m.sc": 5, "mca": 5, "m.com": 5,
        "bachelor": 4, "b.tech": 4, "btech": 4, "b.e": 4,
        "b.sc": 4, "bca": 4, "b.com": 4,
        "graduate": 4, "degree": 3, "diploma": 2, "certificate": 1,
    }
    candidate_level = 0
    for key, level in degree_levels.items():
        if key in edu_text:
            candidate_level = max(candidate_level, level)
    required_level = 0
    for req in required_edu:
        for key, level in degree_levels.items():
            if key in req.lower():
                required_level = max(required_level, level)

    if required_level > 0:
        if candidate_level >= required_level:
            scores["education_score"] = 100
        elif candidate_level > 0:
            scores["education_score"] = round((candidate_level / required_level) * 85, 1)
        else:
            scores["education_score"] = 35
    else:
        scores["education_score"] = 80 if candidate_level > 0 else 50

    # ─── 4. LOCATION MATCH (10 %) ───
    candidate_location = (parsed_resume.get("location", "") or parsed_resume.get("address", "")).lower()
    preferred_locations = [loc.lower() for loc in jd_requirements.get("preferred_locations", [])]

    if preferred_locations:
        remote_kw = ["remote", "work from home", "wfh", "telecommute", "hybrid"]
        if any(kw in preferred_locations for kw in remote_kw):
            scores["location_score"] = 100
        elif candidate_location:
            scores["location_score"] = 100 if any(loc in candidate_location or candidate_location in loc for loc in preferred_locations if loc not in remote_kw) else 55
        else:
            scores["location_score"] = 50
    else:
        scores["location_score"] = 85

    # ─── 5. KEYWORD MATCH (15 %) ───
    keywords = jd_requirements.get("keywords", [])
    if keywords:
        matched_kw = sum(1 for kw in keywords if kw in resume_lower)
        scores["keyword_score"] = round(min(100, (matched_kw / len(keywords)) * 140), 1)
    else:
        scores["keyword_score"] = 70

    # ─── OVERALL (weighted) ───
    scores["overall_score"] = round(min(100,
        scores["skills_score"] * 0.35
        + scores["experience_score"] * 0.25
        + scores["education_score"] * 0.15
        + scores["location_score"] * 0.10
        + scores["keyword_score"] * 0.15
    ), 1)

    # ─── STRENGTHS & GAPS ───
    strengths: List[str] = []
    gaps: List[str] = []

    if scores["skills_score"] >= 75:
        strengths.append(f"Strong skill match ({scores['skills_score']:.0f}%)")
    elif scores["skills_score"] >= 60:
        strengths.append(f"Good skill alignment ({scores['skills_score']:.0f}%)")
    if scores["experience_score"] >= 90:
        strengths.append(f"Experience exceeds requirements ({candidate_exp}y)")
    elif scores["experience_score"] >= 75:
        strengths.append(f"Experience meets requirements ({candidate_exp}y)")
    if scores["education_score"] >= 85:
        strengths.append("Education qualifications match well")
    if scores["location_score"] >= 90:
        strengths.append("Location compatible")
    if scores["keyword_score"] >= 70:
        strengths.append("Good keyword/context match with JD")

    if scores["skills_score"] < 50:
        gaps.append(f"Significant skill gaps ({len(scores['missing_skills'])} missing skills)")
    elif scores["skills_score"] < 65:
        gaps.append("Some skill gaps detected")
    if scores["experience_score"] < 60:
        gaps.append(f"Experience gap ({candidate_exp}y vs {min_exp}y required)")
    if scores["education_score"] < 50:
        gaps.append("Education may not meet requirements")
    if scores["location_score"] < 60:
        gaps.append("Location may not align with job requirements")
    if scores["keyword_score"] < 50:
        gaps.append("Low keyword match — consider tailoring resume")

    scores["strengths"] = _deduplicate_list(strengths)
    scores["gaps"] = _deduplicate_list(gaps)

    overall = scores["overall_score"]
    scores["recommendation"] = (
        "🟢 Excellent Match — Highly Recommended" if overall >= 80 else
        "🟡 Good Match — Worth Considering" if overall >= 65 else
        "🟠 Moderate Match — Review Needed" if overall >= 50 else
        "🔴 Below Average — Significant Gaps" if overall >= 35 else
        "⚫ Low Match — May Not Fit"
    )
    return scores


# ═══════════════════════════════════════════════════════════════
#       EDUCATION EXTRACTION  (VALIDATED + DEDUPLICATED)
# ═══════════════════════════════════════════════════════════════

def _format_education_entry(edu: Dict) -> str:
    """Format a single education dict into a readable string.
    Prevents "B.Tech from B.Tech from …" duplication."""
    parts: List[str] = []

    degree = edu.get("degree", "")
    if degree:
        parts.append(degree)

    field_val = (
        edu.get("field") or edu.get("major")
        or edu.get("branch") or edu.get("specialization", "")
    )
    if field_val:
        parts.append(f"in {field_val}")

    institution = (
        edu.get("institution") or edu.get("university")
        or edu.get("college", "")
    )
    if institution:
        # ── Strip degree prefix from institution to prevent duplication ──
        if degree:
            inst_lower = institution.lower().strip()
            degree_lower = degree.lower().strip()
            if inst_lower.startswith(degree_lower):
                institution = institution[len(degree):].strip()
                institution = re.sub(r'^(?:from|in|at)\s+', '', institution, flags=re.IGNORECASE).strip()
                institution = institution.lstrip(' ,.;:-\u2013\u2014')
        if institution:
            parts.append(f"from {institution}")

    location = edu.get("location", "")
    if location:
        parts.append(f"({location})")

    year = edu.get("year") or edu.get("end_year") or edu.get("graduation_year", "")
    if year:
        parts.append(f"[{year}]")

    gpa = edu.get("gpa") or edu.get("cgpa") or edu.get("grade") or edu.get("percentage", "")
    if gpa:
        parts.append(f"- GPA/Grade: {gpa}")

    return " ".join(parts) if parts else "Not specified"


def get_highest_education(parsed_resume: Dict) -> str:
    """Return a single string for the HIGHEST valid education entry."""
    education = parsed_resume.get("education", [])
    if not education:
        return "Not specified"

    degree_priority = {
        "phd": 10, "ph.d": 10, "ph.d.": 10, "doctorate": 10,
        "doctor": 10, "dphil": 10, "d.phil": 10,
        "master": 8, "masters": 8, "mba": 8, "m.b.a": 8, "m.b.a.": 8,
        "m.tech": 8, "mtech": 8, "m.e": 8, "m.e.": 8,
        "m.sc": 8, "m.sc.": 8, "msc": 8, "m.s": 8, "m.s.": 8, "ms": 8,
        "m.a": 8, "m.a.": 8, "mca": 8, "m.c.a": 8, "m.c.a.": 8,
        "m.com": 8, "m.com.": 8, "mcom": 8, "m.phil": 8, "mphil": 8,
        "pgdm": 7, "pg diploma": 7, "post graduate diploma": 7, "pgd": 7,
        "bachelor": 6, "bachelors": 6,
        "b.tech": 6, "btech": 6, "b.e": 6, "b.e.": 6,
        "b.sc": 6, "b.sc.": 6, "bsc": 6, "b.s": 6, "b.s.": 6, "bs": 6,
        "b.a": 6, "b.a.": 6, "bca": 6, "b.c.a": 6, "b.c.a.": 6,
        "b.com": 6, "b.com.": 6, "bcom": 6,
        "bba": 6, "b.b.a": 6, "b.b.a.": 6,
        "b.arch": 6, "barch": 6, "b.pharm": 6, "bpharm": 6,
        "b.ed": 6, "bed": 6, "llb": 6, "l.l.b": 6,
        "associate": 4, "associates": 4,
        "diploma": 3, "polytechnic": 3, "iti": 2, "i.t.i": 2,
        "12th": 1, "hsc": 1, "higher secondary": 1,
        "intermediate": 1, "+2": 1, "plus two": 1, "xii": 1,
        "10th": 0, "ssc": 0, "secondary": 0,
        "matriculation": 0, "matric": 0,
    }

    highest_edu = None
    highest_priority = -1
    seen_degrees: Set[str] = set()

    if isinstance(education, list):
        for edu in education:
            if not isinstance(edu, dict):
                continue
            if not _is_valid_education_entry(edu):
                continue
            degree = str(edu.get("degree", "")).strip()
            if not degree:
                continue
            normalized_key = _normalize_degree(degree.lower())
            if normalized_key in seen_degrees:
                continue
            seen_degrees.add(normalized_key)

            current_priority = -1
            for key, priority in degree_priority.items():
                if key in degree.lower():
                    current_priority = max(current_priority, priority)
            if current_priority > highest_priority:
                highest_priority = current_priority
                highest_edu = edu

    return _format_education_entry(highest_edu) if highest_edu else "Not specified"


def get_unique_education_list(parsed_resume: Dict) -> List[Dict]:
    """Deduplicated AND validated list of education entries."""
    education = parsed_resume.get("education", [])
    if not education or not isinstance(education, list):
        return []

    seen: Set[str] = set()
    unique_education: List[Dict] = []

    for edu in education:
        if not isinstance(edu, dict):
            continue
        if not _is_valid_education_entry(edu):
            continue
        degree = str(edu.get("degree", "")).strip()
        if not degree:
            continue
        institution = str(
            edu.get("institution", "") or edu.get("university", "") or edu.get("college", "")
        ).strip()
        degree_normalized = _normalize_degree(degree.lower())
        institution_normalized = re.sub(r'[^a-z0-9]', '', institution.lower())[:30]
        key = f"{degree_normalized}_{institution_normalized}"
        if key in seen:
            continue
        seen.add(key)
        unique_education.append(edu)

    return unique_education


def get_all_education_details(parsed_resume: Dict) -> str:
    """All VALID, UNIQUE education entries as a formatted string."""
    unique_education = get_unique_education_list(parsed_resume)
    if not unique_education:
        return "Not specified"

    edu_entries: List[str] = []
    for edu in unique_education:
        parts: List[str] = []
        degree = edu.get("degree", "")
        if degree:
            parts.append(degree)
        field_val = edu.get("field") or edu.get("major") or edu.get("branch", "")
        if field_val:
            parts.append(f"in {field_val}")

        institution = edu.get("institution") or edu.get("university") or edu.get("college", "")
        if institution:
            # ── Strip degree prefix from institution ──
            if degree:
                inst_lower = institution.lower().strip()
                degree_lower = degree.lower().strip()
                if inst_lower.startswith(degree_lower):
                    institution = institution[len(degree):].strip()
                    institution = re.sub(r'^(?:from|in|at)\s+', '', institution, flags=re.IGNORECASE).strip()
                    institution = institution.lstrip(' ,.;:-\u2013\u2014')
            if institution:
                parts.append(f"from {institution}")

        location = edu.get("location", "")
        if location:
            parts.append(f"({location})")
        year = edu.get("year") or edu.get("end_year") or edu.get("graduation_year", "")
        start_year = edu.get("start_year", "")
        if start_year and year:
            parts.append(f"[{start_year} - {year}]")
        elif year:
            parts.append(f"[{year}]")
        gpa = edu.get("gpa") or edu.get("cgpa") or edu.get("grade") or edu.get("percentage", "")
        if gpa:
            parts.append(f"- GPA/Grade: {gpa}")
        if parts:
            edu_entries.append(" ".join(parts))

    return " || ".join(edu_entries) if edu_entries else "Not specified"


# ═══════════════════════════════════════════════════════════════
#                  WORK HISTORY EXTRACTION
# ═══════════════════════════════════════════════════════════════

def get_all_work_history(parsed_resume: Dict) -> str:
    """Get ALL unique work history entries as a formatted string"""
    work_history = parsed_resume.get("work_history", parsed_resume.get("experience", []))
    if not work_history or not isinstance(work_history, list):
        return "Not specified"

    work_entries: List[str] = []
    seen: Set[str] = set()

    for job in work_history:
        if not isinstance(job, dict):
            continue
        title = job.get("title") or job.get("role") or job.get("position", "")
        company = job.get("company") or job.get("organization", "")
        key = f"{title.lower().strip()}_{company.lower().strip()}"[:60]
        if key in seen:
            continue
        seen.add(key)

        parts: List[str] = []
        if title:
            parts.append(title)
        if company:
            parts.append(f"at {company}")
        location = job.get("location", "")
        if location:
            parts.append(f"({location})")
        start_date = job.get("start_date") or job.get("from", "")
        end_date = job.get("end_date") or job.get("to", "")
        duration = job.get("duration", "")
        if start_date and end_date:
            parts.append(f"[{start_date} - {end_date}]")
        elif duration:
            parts.append(f"[{duration}]")
        duration_years = job.get("duration_years") or job.get("years", "")
        if duration_years:
            parts.append(f"({duration_years} years)")
        job_type = job.get("type", "")
        if job_type:
            parts.append(f"- {job_type}")
        if parts:
            work_entries.append(" ".join(parts))

    return " || ".join(work_entries) if work_entries else "Not specified"


# ═══════════════════════════════════════════════════════════════
#                  SKILLS EXTRACTION
# ═══════════════════════════════════════════════════════════════

def get_all_skills(parsed_resume: Dict) -> str:
    """Get ALL unique skills as a formatted string"""
    all_skills: List[str] = []
    skills_data = parsed_resume.get("skills", {})
    if isinstance(skills_data, dict):
        for _category, skills_list in skills_data.items():
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
    elif isinstance(skills_data, list):
        all_skills.extend(skills_data)

    specs = parsed_resume.get("specializations", [])
    if isinstance(specs, list):
        all_skills.extend(specs)

    work_history = parsed_resume.get("work_history", parsed_resume.get("experience", []))
    if isinstance(work_history, list):
        for job in work_history:
            if isinstance(job, dict):
                techs = job.get("technologies_used", job.get("technologies", []))
                if isinstance(techs, list):
                    all_skills.extend(techs)

    unique_skills = _deduplicate_list(all_skills)
    return ", ".join(unique_skills) if unique_skills else "Not specified"


# ═══════════════════════════════════════════════════════════════
#                  CERTIFICATIONS EXTRACTION
# ═══════════════════════════════════════════════════════════════

def get_all_certifications(parsed_resume: Dict) -> str:
    """Get ALL unique certifications as a formatted string"""
    certifications = parsed_resume.get("certifications", [])
    if not certifications or not isinstance(certifications, list):
        return "Not specified"

    cert_entries: List[str] = []
    seen: Set[str] = set()

    for cert in certifications:
        if isinstance(cert, dict):
            name = cert.get("name", "")
            if not name:
                continue
            key = name.lower().strip()[:50]
            if key in seen:
                continue
            seen.add(key)
            parts = [name]
            provider = cert.get("provider") or cert.get("issuer") or cert.get("organization", "")
            if provider:
                parts.append(f"by {provider}")
            date = cert.get("date") or cert.get("issue_date") or cert.get("year", "")
            if date:
                parts.append(f"({date})")
            credential_id = cert.get("credential_id") or cert.get("id", "")
            if credential_id:
                parts.append(f"[ID: {credential_id}]")
            cert_entries.append(" ".join(parts))
        elif isinstance(cert, str) and cert:
            key = cert.lower().strip()[:50]
            if key not in seen:
                seen.add(key)
                cert_entries.append(cert)

    return " || ".join(cert_entries) if cert_entries else "Not specified"


# ═══════════════════════════════════════════════════════════════
#            NAME EXTRACTION  (ENHANCED — multi-source)
# ═══════════════════════════════════════════════════════════════

def _get_best_name(
    parsed: Dict,
    doc_contacts: Dict,
    file_name: str,
    resume_text: str,
) -> str:
    """
    Return the best candidate name from multiple sources.
    Cleans trailing junk (Mobile, Period, …) before validating.
    Falls back to resume_parser's 8-strategy extraction.
    """

    # 1. Try parsed name (clean → validate)
    parsed_name = _clean_candidate_name(parsed.get("name", "").strip())
    if _is_valid_candidate_name(parsed_name):
        return parsed_name

    # 2. Try doc_contacts name (clean → validate)
    doc_name = _clean_candidate_name(doc_contacts.get("name", "").strip())
    if _is_valid_candidate_name(doc_name):
        return doc_name

    # 3. Try resume_parser's 8-strategy name extraction
    if _RP_AVAILABLE:
        rp_name = _rp_extract_name(resume_text)
        if rp_name:
            rp_name = _clean_candidate_name(rp_name)
            if _is_valid_candidate_name(rp_name):
                return rp_name

    # 4. Try document_processor helper (separate from resume_parser)
    try:
        from document_processor import extract_name_from_text
        extracted_name = extract_name_from_text(resume_text)
        if extracted_name:
            extracted_name = _clean_candidate_name(extracted_name)
            if _is_valid_candidate_name(extracted_name):
                return extracted_name
    except ImportError:
        pass

    # 5. Heuristic: first lines of resume
    lines = resume_text.strip().split('\n')[:15]
    for line in lines:
        line = line.strip()
        if not line or len(line) < 4 or len(line) > 45:
            continue
        if '@' in line or 'http' in line.lower():
            continue
        if re.search(r'\d{5,}', line):
            continue
        if re.search(r'^[\d\.\-\(\)+\s]+$', line):
            continue

        # Split by large gaps to isolate name from contact info
        parts_by_space = re.split(r'\s{3,}', line)
        candidate_line = parts_by_space[0].strip() if parts_by_space else line
        if not candidate_line or len(candidate_line) < 4:
            continue

        candidate_line = _clean_candidate_name(candidate_line)
        words = candidate_line.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w):
                if _is_valid_candidate_name(candidate_line):
                    return candidate_line

        # Handle Initial.Name: "M.SREEKANTH"
        if len(words) == 1 and '.' in candidate_line:
            if _is_valid_candidate_name(candidate_line):
                return candidate_line

    # 6. Last resort: derive from filename
    name_from_file = file_name.rsplit(".", 1)[0]
    name_from_file = re.sub(r'[_\-]', ' ', name_from_file)
    name_from_file = re.sub(r'\s+', ' ', name_from_file).strip().title()
    name_from_file = re.sub(
        r'(?i)\s*(resume|cv|curriculum|vitae|updated|final|new|latest|\d+)\s*',
        ' ', name_from_file
    )
    name_from_file = re.sub(r'\s+', ' ', name_from_file).strip()
    if _is_valid_candidate_name(name_from_file):
        return name_from_file

    return "Unknown Candidate"


def _get_best_value(*values, validator=None) -> str:
    """Get the best non-empty value from multiple sources"""
    for value in values:
        if value and isinstance(value, str):
            value = value.strip()
            if value and value.lower() not in (
                'n/a', 'na', 'none', 'null', '-', '—', 'unknown',
                'not specified', 'not available',
            ):
                if validator is None or validator(value):
                    return value
    return ""


# ═══════════════════════════════════════════════════════════════
#                  SINGLE RESUME PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_single_resume_for_bulk(
    file_data: Dict,
    jd_text: str,
    jd_requirements: Dict,
    groq_api_key: str,
    model_id: str,
) -> CandidateResult:
    """Process a single resume in bulk mode with enhanced validation.
    Uses same parsing pipeline as single-resume mode."""
    start_time = time.time()
    file_name = file_data.get("file_name", "Unknown")

    try:
        resume_text = file_data.get("text", "")

        if not resume_text or len(resume_text.strip()) < 50:
            return CandidateResult(
                file_name=file_name,
                candidate_name="[Error - No Text]",
                email="", phone="", location="",
                current_role="", current_company="",
                total_experience=0, highest_education="",
                success=False,
                error="Could not extract text from resume. File may be corrupted, image-based without OCR, or empty.",
                processing_time=round(time.time() - start_time, 2)
            )

        # Pre-extracted contacts from document processor
        doc_contacts = file_data.get("extracted_contacts", {})
        if not doc_contacts:
            try:
                from document_processor import extract_contacts_from_text
                doc_contacts = extract_contacts_from_text(resume_text)
            except ImportError:
                doc_contacts = {}

        # Parse resume with LLM (same pipeline as single mode)
        try:
            parsed = parse_resume_with_llm(resume_text, groq_api_key, model_id)
        except Exception:
            parsed = {
                "name": doc_contacts.get("name", ""),
                "email": doc_contacts.get("email", ""),
                "phone": doc_contacts.get("phone", ""),
                "location": doc_contacts.get("location", ""),
                "address": doc_contacts.get("address", ""),
                "linkedin": doc_contacts.get("linkedin", ""),
                "github": doc_contacts.get("github", ""),
                "current_role": "", "current_company": "",
                "total_experience_years": 0,
                "education": [], "skills": {},
                "work_history": [], "certifications": [],
                "professional_summary": resume_text[:500] if resume_text else "",
            }

        # ── Enhanced contact merging ──
        candidate_name = _get_best_name(parsed, doc_contacts, file_name, resume_text)

        # Email — cleaned consistently via resume_parser
        raw_emails = [
            parsed.get("email", ""),
            doc_contacts.get("email", ""),
        ]
        email = ""
        for raw_e in raw_emails:
            cleaned = _clean_email(raw_e)
            if cleaned and "@" in cleaned and "." in cleaned.split("@")[-1]:
                lp = cleaned.split("@")[0]
                if any(c.isalpha() for c in lp) and len(lp) >= 2:
                    email = cleaned
                    break

        # Phone
        phone = _get_best_value(
            parsed.get("phone", ""),
            doc_contacts.get("phone", ""),
            validator=lambda x: len(re.sub(r'[^\d]', '', x)) >= 10,
        )

        # Location — cleaned consistently via resume_parser
        raw_locations = [
            parsed.get("location", ""),
            doc_contacts.get("location", ""),
            parsed.get("address", ""),
            doc_contacts.get("address", ""),
        ]
        location = ""
        for raw_loc in raw_locations:
            if not raw_loc or not raw_loc.strip():
                continue
            cleaned_loc = _clean_location(raw_loc)
            if cleaned_loc and _is_valid_location(cleaned_loc):
                location = cleaned_loc
                break
        if not location:
            location = _get_best_value(*raw_locations)

        parsed["name"] = candidate_name
        parsed["email"] = email
        parsed["phone"] = phone
        parsed["location"] = location

        # Calculate match scores
        scores = calculate_match_scores(parsed, resume_text, jd_text, jd_requirements)

        # Current role / company
        current_role = parsed.get("current_role", "")
        current_company = parsed.get("current_company", "")

        if not current_role or not current_company:
            wh = parsed.get("work_history", parsed.get("experience", []))
            if isinstance(wh, list) and wh:
                first_job = wh[0] if isinstance(wh[0], dict) else {}
                if not current_role:
                    current_role = (
                        first_job.get("title", "")
                        or first_job.get("role", "")
                        or first_job.get("position", "")
                    )
                if not current_company:
                    current_company = (
                        first_job.get("company", "")
                        or first_job.get("organization", "")
                        or first_job.get("employer", "")
                    )

        # ── Total experience — independent recalculation ──
        total_exp = 0.0

        # (a) Try parsed total
        try:
            total_exp = float(parsed.get("total_experience_years", 0) or 0)
        except (ValueError, TypeError):
            pass

        # (b) If 0, recalculate from work_history dates
        if total_exp == 0:
            wh = parsed.get("work_history", parsed.get("experience", []))
            if isinstance(wh, list):
                for job in wh:
                    if not isinstance(job, dict):
                        continue
                    ss = str(job.get("start_date", job.get("from", ""))).strip()
                    es = str(job.get("end_date", job.get("to", ""))).strip()
                    if ss:
                        dur = _calc_duration_from_dates(ss, es)
                        if dur > 0:
                            total_exp += dur
                            continue
                    try:
                        total_exp += float(job.get("duration_years", job.get("years", 0)) or 0)
                    except (ValueError, TypeError):
                        pass

        # (c) If still 0, extract from summary text
        if total_exp == 0:
            for ctx in [
                parsed.get("professional_summary", "") or parsed.get("summary", "") or "",
                resume_text[:3000],
            ]:
                if not ctx:
                    continue
                for ep in [
                    r'(\d+)\s*\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
                    r'over\s+(\d+)\s*years?',
                    r'more\s+than\s+(\d+)\s*years?',
                    r'nearly\s+(\d+)\s*years?',
                    r'around\s+(\d+)\s*years?',
                    r'experience\s*(?:of\s*)?(\d+)\s*\+?\s*years?',
                ]:
                    em = re.search(ep, ctx.lower())
                    if em:
                        yrs = int(em.group(1))
                        if 1 <= yrs <= 40:
                            total_exp = float(yrs)
                            break
                if total_exp > 0:
                    break

        # Validated highest education
        highest_education = get_highest_education(parsed)

        return CandidateResult(
            file_name=file_name,
            candidate_name=candidate_name,
            email=email,
            phone=phone,
            location=location,
            current_role=current_role,
            current_company=current_company,
            total_experience=round(total_exp, 1),
            highest_education=highest_education,
            overall_score=scores["overall_score"],
            skills_score=scores["skills_score"],
            experience_score=scores["experience_score"],
            education_score=scores["education_score"],
            location_score=scores["location_score"],
            keyword_score=scores["keyword_score"],
            matched_skills=scores["matched_skills"],
            missing_skills=scores["missing_skills"],
            strengths=scores["strengths"],
            gaps=scores["gaps"],
            recommendation=scores["recommendation"],
            processing_time=round(time.time() - start_time, 2),
            success=True,
            parsed_resume=parsed,
            resume_text=resume_text,
        )

    except Exception as e:
        return CandidateResult(
            file_name=file_name,
            candidate_name="[Error]",
            email="", phone="", location="",
            current_role="", current_company="",
            total_experience=0, highest_education="",
            success=False, error=str(e),
            processing_time=round(time.time() - start_time, 2),
        )


# ═══════════════════════════════════════════════════════════════
#                  BULK PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_bulk_resumes(
    uploaded_files: List,
    jd_text: str,
    groq_api_key: str,
    model_id: str,
    progress_callback=None,
) -> BulkProcessingResult:
    """Process multiple resumes and match against JD"""
    start_time = time.time()
    total = len(uploaded_files)

    jd_requirements = extract_jd_requirements(jd_text)

    candidates: List[CandidateResult] = []
    successful = 0
    failed = 0

    for i, uploaded_file in enumerate(uploaded_files):
        if progress_callback:
            progress_callback(i + 1, total, f"Processing {uploaded_file.name}...")

        try:
            uploaded_file.seek(0)
            file_result = process_uploaded_file(uploaded_file, groq_api_key)

            if not file_result.get("success", False):
                candidates.append(CandidateResult(
                    file_name=uploaded_file.name,
                    candidate_name="[Error]",
                    email="", phone="", location="",
                    current_role="", current_company="",
                    total_experience=0, highest_education="",
                    success=False,
                    error=file_result.get("error", "Failed to process file"),
                ))
                failed += 1
                continue

            file_result["file_name"] = uploaded_file.name

            result = process_single_resume_for_bulk(
                file_result, jd_text, jd_requirements,
                groq_api_key, model_id,
            )

            candidates.append(result)
            if result.success:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            candidates.append(CandidateResult(
                file_name=uploaded_file.name,
                candidate_name="[Error]",
                email="", phone="", location="",
                current_role="", current_company="",
                total_experience=0, highest_education="",
                success=False, error=str(e),
            ))
            failed += 1

        if i < total - 1:
            time.sleep(0.3)

    # Sort by (success, overall_score) descending
    candidates.sort(key=lambda x: (x.success, x.overall_score), reverse=True)

    rank = 1
    for candidate in candidates:
        if candidate.success:
            candidate.rank = rank
            rank += 1
        else:
            candidate.rank = 0

    return BulkProcessingResult(
        total_resumes=total,
        successful=successful,
        failed=failed,
        processing_time=round(time.time() - start_time, 2),
        candidates=candidates,
        jd_summary=jd_requirements,
    )


# ═══════════════════════════════════════════════════════════════
#                  EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def results_to_dataframe(result: BulkProcessingResult) -> pd.DataFrame:
    """Convert bulk processing results to a pandas DataFrame — DEDUPLICATED"""
    data = []
    for candidate in result.candidates:
        education_text = candidate.highest_education if candidate.success else ""
        matched_skills = _deduplicate_list(candidate.matched_skills)
        missing_skills = _deduplicate_list(candidate.missing_skills)
        strengths = _deduplicate_list(candidate.strengths)
        gaps = _deduplicate_list(candidate.gaps)

        row = {
            "Rank": candidate.rank if candidate.success else "-",
            "Candidate Name": candidate.candidate_name,
            "Overall Score": f"{candidate.overall_score}%" if candidate.success else "Error",
            "Skills Match": f"{candidate.skills_score}%" if candidate.success else "-",
            "Experience Match": f"{candidate.experience_score}%" if candidate.success else "-",
            "Education Match": f"{candidate.education_score}%" if candidate.success else "-",
            "Location Match": f"{candidate.location_score}%" if candidate.success else "-",
            "Keyword Match": f"{candidate.keyword_score}%" if candidate.success else "-",
            "Experience (Years)": candidate.total_experience if candidate.success else "-",
            "Current Role": candidate.current_role,
            "Company": candidate.current_company,
            "Email": candidate.email,
            "Phone": candidate.phone,
            "Location": candidate.location,
            "Highest Education": education_text,
            "Matched Skills": ", ".join(matched_skills),
            "Missing Skills": ", ".join(missing_skills),
            "Strengths": " | ".join(strengths),
            "Gaps": " | ".join(gaps),
            "Recommendation": candidate.recommendation if candidate.success else f"Error: {candidate.error}",
            "File Name": candidate.file_name,
            "Processing Time (s)": candidate.processing_time if candidate.success else "-",
            "Status": "Success" if candidate.success else "Failed",
        }
        data.append(row)

    return pd.DataFrame(data)


def export_results_to_excel(result: BulkProcessingResult) -> bytes:
    """Export bulk processing results to Excel file — DEDUPLICATED"""
    import io

    df = results_to_dataframe(result)
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # ── SHEET 1: CANDIDATE RANKINGS ──
        df.to_excel(writer, sheet_name='Candidate Rankings', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Candidate Rankings']

        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4338ca', 'font_color': 'white',
            'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True,
        })
        wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        for i, col in enumerate(df.columns):
            max_len = df[col].astype(str).apply(len).max()
            width = max(len(col), min(max_len, 100)) + 2
            worksheet.set_column(i, i, width, wrap_format)
        worksheet.freeze_panes(1, 1)

        # ── SHEET 2: DETAILED CANDIDATE INFO ──
        detailed_data = []
        for candidate in result.candidates:
            if not candidate.success:
                continue
            unique_skills = get_all_skills(candidate.parsed_resume)
            work_summary = get_all_work_history(candidate.parsed_resume)
            education_full = get_all_education_details(candidate.parsed_resume)
            certifications = get_all_certifications(candidate.parsed_resume)
            specializations = _deduplicate_list(candidate.parsed_resume.get("specializations", []))
            languages = _deduplicate_list(candidate.parsed_resume.get("languages", []))

            awards_raw = candidate.parsed_resume.get("awards", [])
            awards: List[str] = []
            for a in awards_raw:
                awards.append(a.get("name", str(a)) if isinstance(a, dict) else str(a))
            awards = _deduplicate_list(awards)

            detailed_data.append({
                "Rank": candidate.rank,
                "Candidate Name": candidate.candidate_name,
                "Overall Score (%)": candidate.overall_score,
                "Skills Score (%)": candidate.skills_score,
                "Experience Score (%)": candidate.experience_score,
                "Education Score (%)": candidate.education_score,
                "Location Score (%)": candidate.location_score,
                "Keyword Score (%)": candidate.keyword_score,
                "Email": candidate.email,
                "Phone": candidate.phone,
                "Address/Location": candidate.parsed_resume.get("address", "") or candidate.location,
                "LinkedIn": candidate.parsed_resume.get("linkedin", ""),
                "GitHub": candidate.parsed_resume.get("github", ""),
                "Portfolio": candidate.parsed_resume.get("portfolio", ""),
                "Professional Summary": candidate.parsed_resume.get(
                    "professional_summary", candidate.parsed_resume.get("summary", "")
                ),
                "Total Experience (Years)": candidate.total_experience,
                "Current Role": candidate.current_role,
                "Current Company": candidate.current_company,
                "All Skills (Unique)": unique_skills,
                "Work History": work_summary,
                "Education (Full - Unique)": education_full,
                "Certifications": certifications,
                "Specializations": ", ".join(specializations),
                "Awards": ", ".join(awards),
                "Languages": ", ".join(languages),
                "Matched Skills": ", ".join(_deduplicate_list(candidate.matched_skills)),
                "Missing Skills": ", ".join(_deduplicate_list(candidate.missing_skills)),
                "Strengths": " | ".join(_deduplicate_list(candidate.strengths)),
                "Gaps": " | ".join(_deduplicate_list(candidate.gaps)),
                "Recommendation": candidate.recommendation,
                "File Name": candidate.file_name,
                "Processing Time (s)": candidate.processing_time,
            })

        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Candidate Info', index=False)
            dws = writer.sheets['Detailed Candidate Info']
            for col_num, value in enumerate(detailed_df.columns.values):
                dws.write(0, col_num, value, header_format)
            for i, col in enumerate(detailed_df.columns):
                max_len = detailed_df[col].astype(str).apply(len).max()
                width = max(len(col), min(max_len, 120)) + 2
                dws.set_column(i, i, width, wrap_format)
            dws.freeze_panes(1, 1)

        # ── SHEET 3: JD REQUIREMENTS ──
        jd_skills = _deduplicate_list(result.jd_summary.get("required_skills", []))
        jd_edu = _deduplicate_list(result.jd_summary.get("required_education", []))
        jd_locations = _deduplicate_list(result.jd_summary.get("preferred_locations", []))
        jd_keywords = _deduplicate_list(result.jd_summary.get("keywords", []))
        max_exp_val = result.jd_summary.get("max_experience_years", 99)

        jd_data = [
            {"Category": "Required Skills", "Details": ", ".join(jd_skills) or "Not specified"},
            {"Category": "Min Experience (Years)", "Details": str(result.jd_summary.get("min_experience_years", 0))},
            {"Category": "Max Experience (Years)", "Details": str(max_exp_val) if max_exp_val < 99 else "Not specified"},
            {"Category": "Required Education", "Details": ", ".join(jd_edu) or "Not specified"},
            {"Category": "Preferred Locations", "Details": ", ".join(jd_locations) or "Not specified"},
            {"Category": "Key Keywords", "Details": ", ".join(jd_keywords) or "Not specified"},
        ]
        jd_df = pd.DataFrame(jd_data)
        jd_df.to_excel(writer, sheet_name='JD Requirements', index=False)
        jd_ws = writer.sheets['JD Requirements']
        jd_ws.set_column(0, 0, 25)
        jd_ws.set_column(1, 1, 150, wrap_format)
        for col_num, value in enumerate(jd_df.columns.values):
            jd_ws.write(0, col_num, value, header_format)

        # ── SHEET 4: SUMMARY ──
        avg_score = (
            sum(c.overall_score for c in result.candidates if c.success)
            / max(result.successful, 1)
        )
        top_candidate = (
            result.candidates[0]
            if result.candidates and result.candidates[0].success else None
        )
        min_score = min(
            (c.overall_score for c in result.candidates if c.success), default=0
        )

        summary_data = [
            {"Metric": "Report Generated", "Value": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Metric": "", "Value": ""},
            {"Metric": "=== PROCESSING STATS ===", "Value": ""},
            {"Metric": "Total Resumes Processed", "Value": result.total_resumes},
            {"Metric": "Successfully Processed", "Value": result.successful},
            {"Metric": "Failed", "Value": result.failed},
            {"Metric": "Processing Time (seconds)", "Value": result.processing_time},
            {"Metric": "", "Value": ""},
            {"Metric": "=== SCORE SUMMARY ===", "Value": ""},
            {"Metric": "Average Match Score", "Value": f"{avg_score:.1f}%"},
            {"Metric": "Highest Score", "Value": f"{top_candidate.overall_score}%" if top_candidate else "N/A"},
            {"Metric": "Lowest Score (successful)", "Value": f"{min_score:.1f}%" if result.successful > 0 else "N/A"},
            {"Metric": "", "Value": ""},
            {"Metric": "=== TOP CANDIDATE ===", "Value": ""},
            {"Metric": "Name", "Value": top_candidate.candidate_name if top_candidate else "N/A"},
            {"Metric": "Score", "Value": f"{top_candidate.overall_score}%" if top_candidate else "N/A"},
            {"Metric": "Email", "Value": top_candidate.email if top_candidate else "N/A"},
            {"Metric": "Phone", "Value": top_candidate.phone if top_candidate else "N/A"},
            {"Metric": "Current Role", "Value": top_candidate.current_role if top_candidate else "N/A"},
            {"Metric": "Experience", "Value": f"{top_candidate.total_experience} years" if top_candidate else "N/A"},
            {"Metric": "", "Value": ""},
            {"Metric": "=== SCORE DISTRIBUTION ===", "Value": ""},
            {"Metric": "Excellent (≥80%)", "Value": sum(1 for c in result.candidates if c.success and c.overall_score >= 80)},
            {"Metric": "Good (65-79%)", "Value": sum(1 for c in result.candidates if c.success and 65 <= c.overall_score < 80)},
            {"Metric": "Moderate (50-64%)", "Value": sum(1 for c in result.candidates if c.success and 50 <= c.overall_score < 65)},
            {"Metric": "Below Average (<50%)", "Value": sum(1 for c in result.candidates if c.success and c.overall_score < 50)},
        ]
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        s_ws = writer.sheets['Summary']
        s_ws.set_column(0, 0, 35)
        s_ws.set_column(1, 1, 50)
        for col_num, value in enumerate(summary_df.columns.values):
            s_ws.write(0, col_num, value, header_format)

        # ── SHEET 5: FAILED FILES ──
        failed_candidates = [c for c in result.candidates if not c.success]
        if failed_candidates:
            failed_data = [
                {"File Name": c.file_name, "Error": c.error or "Unknown error"}
                for c in failed_candidates
            ]
            failed_df = pd.DataFrame(failed_data)
            failed_df.to_excel(writer, sheet_name='Failed Files', index=False)
            f_ws = writer.sheets['Failed Files']
            f_ws.set_column(0, 0, 40)
            f_ws.set_column(1, 1, 100, wrap_format)
            for col_num, value in enumerate(failed_df.columns.values):
                f_ws.write(0, col_num, value, header_format)

    output.seek(0)
    return output.getvalue()


def export_results_to_csv(result: BulkProcessingResult) -> str:
    """Export bulk processing results to CSV string — DEDUPLICATED"""
    df = results_to_dataframe(result)
    return df.to_csv(index=False)


def export_detailed_results_to_csv(result: BulkProcessingResult) -> str:
    """Export detailed results to CSV with all fields — DEDUPLICATED"""
    detailed_data = []
    for candidate in result.candidates:
        if not candidate.success:
            continue
        detailed_data.append({
            "Rank": candidate.rank,
            "Candidate Name": candidate.candidate_name,
            "Overall Score (%)": candidate.overall_score,
            "Skills Score (%)": candidate.skills_score,
            "Experience Score (%)": candidate.experience_score,
            "Education Score (%)": candidate.education_score,
            "Location Score (%)": candidate.location_score,
            "Keyword Score (%)": candidate.keyword_score,
            "Email": candidate.email,
            "Phone": candidate.phone,
            "Location": candidate.location,
            "LinkedIn": candidate.parsed_resume.get("linkedin", ""),
            "GitHub": candidate.parsed_resume.get("github", ""),
            "Professional Summary": candidate.parsed_resume.get("professional_summary", ""),
            "Total Experience (Years)": candidate.total_experience,
            "Current Role": candidate.current_role,
            "Current Company": candidate.current_company,
            "All Skills": get_all_skills(candidate.parsed_resume),
            "Work History": get_all_work_history(candidate.parsed_resume),
            "Education (Full)": get_all_education_details(candidate.parsed_resume),
            "Certifications": get_all_certifications(candidate.parsed_resume),
            "Matched Skills": ", ".join(_deduplicate_list(candidate.matched_skills)),
            "Missing Skills": ", ".join(_deduplicate_list(candidate.missing_skills)),
            "Strengths": " | ".join(_deduplicate_list(candidate.strengths)),
            "Gaps": " | ".join(_deduplicate_list(candidate.gaps)),
            "Recommendation": candidate.recommendation,
            "File Name": candidate.file_name,
        })
    df = pd.DataFrame(detailed_data)
    return df.to_csv(index=False)
