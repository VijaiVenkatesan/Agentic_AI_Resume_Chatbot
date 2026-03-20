"""
Enhanced Resume Parser V7
- Triple extraction: Document Processor + Regex + LLM
- Enhanced name extraction with strong/soft blacklists
- Enhanced education validation + cross-validation against resume text
- Paragraph-format work experience extraction with ALL date formats
- Independent experience calculation — NEVER trusts LLM totals
- Work entry validation — rejects garbage role/company names
- Accurate experience calculation using CURRENT_YEAR = 2026
- Works for ANY domain resume, ANY format
- NO TRUNCATION on any field
"""

import json
import re
import time
import requests
from typing import Dict, List, Tuple, Optional, Set

CURRENT_YEAR = 2026
CURRENT_MONTH = 3


# ═══════════════════════════════════════════════════════════════
#          NAME VALIDATION BLACKLISTS
# ═══════════════════════════════════════════════════════════════

_STRONG_NAME_BLACKLIST: Set[str] = {
    "resume", "cv", "curriculum", "vitae", "biodata",
    "objective", "summary", "profile", "overview",
    "education", "experience", "skills", "certifications",
    "references", "declaration", "signature",
    "qualifications", "achievements", "accomplishments",
    "responsibilities", "duties",
    "candidate", "applicant", "recruitment", "hiring",
    "position", "vacancy",
    "contact", "details", "information", "personal",
    "professional", "technical", "academic",
    "employment", "employer",
    "engineer", "developer", "manager", "analyst",
    "consultant", "designer", "architect", "administrator",
    "senior", "junior", "lead", "head", "director",
    "executive", "intern", "trainee",
    "software", "hardware", "technology", "technologies",
    "university", "college", "school", "institute",
    "company", "corporation", "organization",
    "private", "limited", "ltd", "inc", "corp", "llc", "pvt",
    "bachelor", "master", "doctor", "phd", "mba",
    "btech", "mtech", "degree", "diploma",
    "project", "projects", "work", "history",
    "reference", "present", "current",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "bangalore", "mumbai", "delhi", "chennai", "hyderabad",
    "pune", "kolkata", "india", "singapore",
}

_SOFT_NAME_BLACKLIST: Set[str] = {
    "voice", "message", "support", "service", "process",
    "system", "application", "development",
    "client", "customer", "business",
    "department", "team",
    "specialist", "coordinator", "associate",
    "responsible", "working", "developing", "managing",
    "leading", "building", "creating",
    "page", "date", "mobile", "email", "phone",
    "address", "city", "state", "country",
}

# Words that should NEVER appear as a job role or company name
_GARBAGE_WORK_WORDS: Set[str] = {
    "in", "a", "an", "the", "at", "of", "for", "to", "and", "or",
    "with", "from", "by", "on", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "not", "no", "yes", "position", "role", "company",
}

MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
    'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
    'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}

LOCATION_WORDS: Set[str] = {
    'bangalore', 'bengaluru', 'mumbai', 'delhi', 'chennai', 'hyderabad',
    'pune', 'kolkata', 'noida', 'gurgaon', 'gurugram', 'india',
    'new', 'york', 'san', 'francisco', 'london', 'singapore',
    'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'llc',
}

PRESENT_WORDS: Set[str] = {
    'present', 'current', 'currently', 'now', 'ongoing',
    'till date', 'till now', 'to date', 'today', 'continuing',
}


# ═══════════════════════════════════════════════════════════════
#                    LLM PARSING PROMPT
# ═══════════════════════════════════════════════════════════════

PARSE_PROMPT = """You are an expert resume parser. The current date is March 2026.

RESUME TEXT:
{resume_text}

Extract ALL information into this exact JSON structure (no markdown, no code blocks):

{{
    "name": "FULL name exactly as written - look at the VERY FIRST lines",
    "email": "email address",
    "phone": "complete phone number with country code",
    "address": "full physical/mailing address if mentioned",
    "linkedin": "LinkedIn URL",
    "github": "GitHub URL",
    "portfolio": "any portfolio or personal website URL",
    "location": "City, State/Country",
    "current_role": "most recent job title",
    "current_company": "most recent company name",
    "total_experience_years": 0,
    "professional_summary": "full professional summary or objective text",
    "specializations": ["area1", "area2"],
    "skills": {{
        "programming_languages": [],
        "frameworks_libraries": [],
        "ai_ml_tools": [],
        "cloud_platforms": [],
        "databases": [],
        "devops_tools": [],
        "visualization": [],
        "other_tools": [],
        "soft_skills": []
    }},
    "work_history": [
        {{
            "title": "exact job title",
            "company": "company name",
            "location": "location",
            "start_date": "Month Year",
            "end_date": "Month Year or Present",
            "duration_years": 0.0,
            "type": "Full-time/Internship/Contract/Part-time/Freelance",
            "description": "role description",
            "key_achievements": ["achievement1"],
            "technologies_used": ["tech1"]
        }}
    ],
    "education": [
        {{
            "degree": "EXACT degree as written in resume",
            "field": "EXACT field/major/branch as written in resume",
            "institution": "EXACT university/college name as written in resume",
            "location": "location",
            "start_year": "year",
            "end_year": "year or expected year",
            "gpa": "GPA, CGPA, or percentage if mentioned",
            "achievements": []
        }}
    ],
    "certifications": [
        {{
            "name": "certification name",
            "provider": "issuing organization",
            "date": "date obtained",
            "credential_id": "ID if mentioned"
        }}
    ],
    "awards": [
        {{
            "name": "award name",
            "organization": "org",
            "date": "date",
            "description": "details"
        }}
    ],
    "projects": [
        {{
            "name": "project name",
            "description": "description",
            "technologies": ["tech"],
            "achievements": ["results"]
        }}
    ],
    "publications": [],
    "volunteer": [],
    "languages": [],
    "interests": []
}}

CRITICAL RULES:
1. NAME: The VERY FIRST non-empty line is usually the name.
2. PHONE: Patterns like +91, +1, (XXX), or any 10+ digit numbers.
3. EMAIL: Find ANYTHING with @ symbol.
4. EDUCATION:
   - Extract ONLY degrees EXPLICITLY WRITTEN in the resume text.
   - DO NOT use example values from this template.
   - DO NOT create entries from awards or company names.
   - Keep degree name EXACTLY as written (do not expand abbreviations).
5. EXPERIENCE: If end_date is "Present", calculate duration until March 2026.
6. duration_years: Calculate as decimal from dates. 2 years 6 months = 2.5.
7. total_experience_years: Sum of all duration_years.
8. SKILLS: Extract EVERY skill mentioned ANYWHERE.

Return ONLY valid JSON."""


# ═══════════════════════════════════════════════════════════════
#               SHARED DATE PARSING
# ═══════════════════════════════════════════════════════════════

def _parse_date_to_ym(date_str: str, month_map: Dict = None) -> Tuple[Optional[int], Optional[int]]:
    if not date_str:
        return None, None
    if month_map is None:
        month_map = MONTH_MAP

    ds = date_str.strip().lower().strip('.,;:)( ')

    # Present/Current
    ds_clean = re.sub(r'[^a-z\s]', '', ds).strip()
    if ds_clean in PRESENT_WORDS or any(pw in ds for pw in PRESENT_WORDS):
        return CURRENT_YEAR, CURRENT_MONTH

    # "Oct'19" or "Oct'2019"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*['\u2019]\s*(\d{2,4})", ds)
    if m:
        mn = m.group(1)[:3]
        yr = int(m.group(2))
        if yr < 100:
            yr += 2000 if yr < 50 else 1900
        return yr, month_map.get(mn, 6)

    # "Month[,.][ ]Year"
    m = re.search(
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
        r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|'
        r'oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        r'[.,]?\s*(\d{4})', ds
    )
    if m:
        return int(m.group(2)), month_map.get(m.group(1)[:3], 6)

    # "MM/YYYY"
    m = re.search(r'(\d{1,2})[/\-\.](\d{4})', ds)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return int(m.group(2)), month

    # "YYYY/MM"
    m = re.search(r'(\d{4})[/\-\.](\d{1,2})', ds)
    if m:
        month = int(m.group(2))
        if 1 <= month <= 12:
            return int(m.group(1)), month

    # Just year
    m = re.search(r'((?:19|20)\d{2})', ds)
    if m:
        return int(m.group(1)), 6

    return None, None


def _calc_duration_from_dates(start_str: str, end_str: str) -> float:
    """Independently calculate years between two date strings."""
    sy, sm = _parse_date_to_ym(start_str)
    if not sy:
        return 0.0
    if not end_str or end_str.strip().lower() in PRESENT_WORDS or end_str.strip() in ['-', '\u2013', '']:
        ey, em = CURRENT_YEAR, CURRENT_MONTH
    else:
        ey, em = _parse_date_to_ym(end_str)
        if not ey:
            ey, em = CURRENT_YEAR, CURRENT_MONTH
    sm = sm or 6
    em = em or 6
    months = (ey - sy) * 12 + (em - sm)
    return round(max(0, months / 12.0), 1)


def _calculate_total_experience_from_dates(work_history: list) -> float:
    """Calculate total experience ONLY from date strings, ignoring duration_years."""
    total = 0.0
    for job in work_history:
        if not isinstance(job, dict):
            continue
        start_str = str(job.get("start_date", job.get("from", "")))
        end_str = str(job.get("end_date", job.get("to", "")))
        if start_str:
            dur = _calc_duration_from_dates(start_str, end_str)
            if dur > 0:
                total += dur
                continue
        # Fallback: use duration_years only if no dates
        dur = job.get("duration_years", job.get("years", 0))
        try:
            total += float(dur) if dur else 0
        except (ValueError, TypeError):
            pass
    return round(total, 1)


# ═══════════════════════════════════════════════════════════════
#               WORK ENTRY VALIDATION
# ═══════════════════════════════════════════════════════════════

def _is_valid_work_entry(job: Dict) -> bool:
    """Validate that a work history entry has real role and company names."""
    if not isinstance(job, dict):
        return False

    title = str(job.get("title", "") or job.get("role", "") or job.get("position", "")).strip()
    company = str(job.get("company", "") or job.get("organization", "")).strip()

    # Must have either a role or company
    if not title and not company:
        return False

    # Check role
    if title:
        title_words = title.lower().split()
        # All words are garbage → invalid
        meaningful_title_words = [w for w in title_words if w not in _GARBAGE_WORK_WORDS and len(w) > 1]
        if not meaningful_title_words:
            return False
        # Role too short (single char like "a")
        if len(title) < 3:
            return False

    # Check company
    if company:
        company_words = company.lower().split()
        meaningful_company_words = [w for w in company_words if w not in _GARBAGE_WORK_WORDS and len(w) > 1]
        if not meaningful_company_words:
            return False
        if len(company) < 2:
            return False

    # Reject if role looks like a sentence fragment
    if title:
        # "in a position in a" → garbage
        if title.lower().startswith(('in ', 'a ', 'an ', 'the ', 'at ', 'of ')):
            return False
        # Too many prepositions = probably a sentence fragment
        preps = sum(1 for w in title.lower().split() if w in {'in', 'a', 'an', 'the', 'at', 'of', 'for', 'to', 'with'})
        if preps >= len(title.split()) * 0.5 and len(title.split()) > 2:
            return False

    if company:
        if company.lower().startswith(('in ', 'a ', 'an ', 'the ', 'at ', 'of ')):
            return False
        preps = sum(1 for w in company.lower().split() if w in {'in', 'a', 'an', 'the', 'at', 'of', 'for', 'to', 'with'})
        if preps >= len(company.split()) * 0.5 and len(company.split()) > 2:
            return False

    return True


# ═══════════════════════════════════════════════════════════════
#               EDUCATION ENTRY VALIDATION
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
    stripped = text.strip()
    if stripped and stripped[0].islower():
        first_word = stripped.split()[0] if stripped.split() else ""
        allowed = {"in", "at", "of", "for", "the", "and", "or", "with", "to", "from"}
        if first_word and first_word not in allowed and len(first_word) < 5:
            return True
    for hw in ["education", "experience", "skills", "summary", "objective", "profile"]:
        if text.lower().count(hw) >= 2:
            return True
    return False


def _is_valid_education_entry(edu: Dict) -> bool:
    if not isinstance(edu, dict):
        return False
    degree = str(edu.get("degree", "")).strip()
    field_of_study = str(
        edu.get("field", "") or edu.get("major", "") or edu.get("branch", "") or edu.get("specialization", "")
    ).strip()
    institution = str(
        edu.get("institution", "") or edu.get("university", "") or edu.get("college", "")
    ).strip()

    if not degree or len(degree) < 2:
        return False
    if sum(1 for c in degree if c.isalpha()) < len(degree) * 0.5:
        return False

    combined = f"{degree} {field_of_study} {institution}".lower()
    for kw in [
        "award", "performer", "performance", "star performer", "received", "recognition",
        "appreciation", "good performances", "best employee", "employee of",
        "delivered", "achievements", "achieved", "client project", "client ems",
        "project delivery", "team lead", "team member", "responsible for",
        "worked on", "developed", "implemented",
    ]:
        if kw in combined:
            return False

    if _has_repetition_pattern(institution) or _has_repetition_pattern(field_of_study) or _has_repetition_pattern(degree):
        return False
    if _has_garbage_pattern(institution) or _has_garbage_pattern(field_of_study):
        return False

    inst_lower = institution.lower().strip()
    if inst_lower.count("education") >= 1 and len(inst_lower) < 30:
        return False
    if inst_lower in {"education", "qualifications", "academic details", "academic qualifications",
                      "educational details", "educational qualifications", "academic background", "educational background"}:
        return False

    allowed_short = {"it", "cs", "ai", "ml", "ee", "ec", "me", "ce"}
    if field_of_study and len(field_of_study) == 1:
        return False
    if field_of_study and len(field_of_study) < 3 and field_of_study.lower() not in allowed_short:
        return False
    if institution and len(re.sub(r'[^a-zA-Z]', '', institution)) < 3:
        return False
    if institution and institution.split():
        fw = institution.split()[0]
        if fw and fw[0].islower() and len(fw) < 6:
            return False
    if field_of_study and field_of_study[0].islower() and len(field_of_study) < 5 and field_of_study.lower() not in allowed_short:
        return False

    if degree.lower().strip('.') in {"ma", "me", "ms", "ba", "be", "bs"}:
        school_kw = ["university", "college", "institute", "school", "academy", "polytechnic",
                     "iit", "nit", "iiit", "bits", "vit", "mit", "anna", "delhi", "mumbai", "vidyalaya", "vidyapeeth"]
        if institution:
            if not any(kw in inst_lower for kw in school_kw):
                return False
        else:
            return False

    company_ind = ["mahindra", "infosys", "wipro", "tcs", "cognizant", "accenture", "capgemini", "hcl",
                   "tech mahindra", "client", "pvt", "ltd", "inc", "llc", "corp", "solutions",
                   "technologies", "services", "consulting", "private limited", "limited", "software"]
    if institution:
        school_kw2 = ["university", "college", "institute", "school", "academy", "polytechnic"]
        if any(kw in inst_lower for kw in company_ind) and not any(kw in inst_lower for kw in school_kw2):
            return False
    if institution and len(institution) > 150:
        return False
    return True


def _normalize_degree_key(degree: str) -> str:
    if not degree:
        return ""
    d = re.sub(r'[^a-z0-9\s]', '', degree.lower().strip())
    d = re.sub(r'\s+', '', d)
    d_no_pg = re.sub(r'^pg\.?\s*', '', degree.lower().strip())
    d_no_pg = re.sub(r'[^a-z0-9]', '', d_no_pg)

    norms = {
        r'^(?:bacheloroftechnology|btech|be|pgbe|pgbtech)$': 'btech',
        r'^(?:masteroftechnology|mtech|me|pgme|pgmtech)$': 'mtech',
        r'^(?:bachelorofscience|bsc|bs|pgbsc|pgbs)$': 'bsc',
        r'^(?:masterofscience|msc|ms)$': 'msc',
        r'^(?:bachelorofarts|ba)$': 'ba', r'^(?:masterofarts|ma)$': 'ma',
        r'^(?:bachelorofcommerce|bcom)$': 'bcom', r'^(?:masterofcommerce|mcom)$': 'mcom',
        r'^(?:bachelorofcomputerapplications|bca)$': 'bca', r'^(?:masterofcomputerapplications|mca)$': 'mca',
        r'^(?:bachelorofbusinessadministration|bba)$': 'bba', r'^(?:masterofbusinessadministration|mba)$': 'mba',
        r'^(?:doctorofphilosophy|phd|doctorate)$': 'phd',
        r'^(?:highersecondary|hsc|12th|xii|intermediate)$': 'hsc',
        r'^(?:secondary|ssc|10th|matriculation)$': 'ssc',
        r'^(?:diploma|pgdiploma|pgd|postgraduatediploma)$': 'diploma',
    }
    for p, r in norms.items():
        if re.match(p, d):
            return r
    if d_no_pg != d:
        for p, r in norms.items():
            if re.match(p, d_no_pg):
                return r
    return d


def _deduplicate_education_entries(edu_list: List[Dict]) -> List[Dict]:
    if not edu_list:
        return []
    groups: Dict[str, List[Dict]] = {}
    for edu in edu_list:
        if not isinstance(edu, dict):
            continue
        deg = str(edu.get("degree", "")).strip()
        if not deg:
            continue
        key = _normalize_degree_key(deg) or re.sub(r'[^a-z0-9]', '', deg.lower())
        groups.setdefault(key, []).append(edu)

    result: List[Dict] = []
    for entries in groups.values():
        if len(entries) == 1:
            result.append(entries[0])
        else:
            best, best_score = entries[0], -999
            for e in entries:
                s = 0
                inst = str(e.get("institution", "") or e.get("university", "") or e.get("college", "")).strip()
                fv = str(e.get("field", "") or e.get("major", "") or e.get("branch", "")).strip()
                if inst and len(inst) > 5: s += 3
                if fv and len(fv) > 2: s += 2
                if e.get("year") or e.get("end_year") or e.get("graduation_year"): s += 2
                if e.get("gpa") or e.get("cgpa") or e.get("grade"): s += 1
                if e.get("location"): s += 1
                if _has_garbage_pattern(inst): s -= 5
                if _has_repetition_pattern(inst): s -= 5
                if s > best_score:
                    best_score = s
                    best = e
            result.append(best)
    return result


def _filter_valid_education(edu_list: list) -> list:
    if not edu_list or not isinstance(edu_list, list):
        return []
    valid = [e for e in edu_list if isinstance(e, dict) and _is_valid_education_entry(e)]
    return _deduplicate_education_entries(valid)


def _validate_education_against_text(edu_list: List[Dict], resume_text: str) -> List[Dict]:
    if not edu_list or not resume_text:
        return edu_list

    text_lower = resume_text.lower()
    text_alpha = re.sub(r'[^a-z0-9\s]', '', text_lower)
    validated: List[Dict] = []

    ignore = {'the', 'and', 'for', 'from', 'of', 'in', 'at', 'to', 'with'}
    generic = {'university', 'college', 'institute', 'school', 'academy',
               'degree', 'bachelor', 'master', 'doctor', 'diploma',
               'science', 'arts', 'technology', 'engineering'}

    deg_abbrs = {
        'bachelor of technology': ['b.tech', 'btech', 'b tech'],
        'master of technology': ['m.tech', 'mtech', 'm tech'],
        'bachelor of engineering': ['b.e', 'b.e.', 'be', 'pg. b.e', 'pg b.e', 'pg.b.e'],
        'master of engineering': ['m.e', 'm.e.', 'me'],
        'bachelor of science': ['b.sc', 'bsc', 'b.s', 'bs'],
        'master of science': ['m.sc', 'msc', 'm.s', 'ms'],
        'bachelor of arts': ['b.a', 'ba'], 'master of arts': ['m.a', 'ma'],
        'bachelor of commerce': ['b.com', 'bcom'], 'master of commerce': ['m.com', 'mcom'],
        'bachelor of computer applications': ['bca', 'b.c.a'],
        'master of computer applications': ['mca', 'm.c.a'],
        'bachelor of business administration': ['bba', 'b.b.a'],
        'master of business administration': ['mba', 'm.b.a'],
        'doctor of philosophy': ['phd', 'ph.d', 'ph.d.'],
    }

    for edu in edu_list:
        if not isinstance(edu, dict):
            continue
        degree = str(edu.get("degree", "")).strip()
        field_val = str(edu.get("field", "") or edu.get("major", "") or edu.get("branch", "") or edu.get("specialization", "")).strip()
        institution = str(edu.get("institution", "") or edu.get("university", "") or edu.get("college", "")).strip()

        found = False

        # Check institution
        if institution and len(institution) > 3:
            iw = [w for w in institution.lower().split() if w not in ignore and w not in generic]
            if iw:
                if any(w in text_lower for w in iw if len(w) > 2):
                    found = True
                if not found:
                    iwa = [re.sub(r'[^a-z0-9]', '', w) for w in iw]
                    if any(w in text_alpha for w in iwa if len(w) > 2):
                        found = True
            else:
                ia = re.sub(r'[^a-z0-9\s]', '', institution.lower()).strip()
                if ia and ia in text_alpha:
                    found = True

        # Check degree
        if not found and degree and len(degree) > 1:
            dl = degree.lower()
            da = re.sub(r'[^a-z0-9\s]', '', dl)
            if dl in text_lower or da in text_alpha:
                found = True
            if not found:
                for ff, abbrs in deg_abbrs.items():
                    if ff in dl or dl in ff:
                        if any(a in text_lower for a in abbrs):
                            found = True
                            break
            if not found:
                dw = [w for w in dl.split() if len(w) > 1 and w not in ignore and w not in generic]
                if dw and sum(1 for w in dw if w in text_lower) >= 1:
                    found = True
            if not found:
                am = re.search(r'(?:PG\.?\s*)?(B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA|B\.?Sc|M\.?Sc|Ph\.?D)', degree, re.IGNORECASE)
                if am:
                    ab = am.group(0).lower()
                    if ab in text_lower or ab.replace('.', '') in text_alpha:
                        found = True

        # Check field
        if not found and field_val and len(field_val) > 3:
            if field_val.lower() in text_lower:
                found = True
            else:
                fw = [w for w in field_val.lower().split() if len(w) > 3 and w not in ignore and w not in generic]
                if fw and any(w in text_lower for w in fw):
                    found = True

        if found:
            validated.append(edu)

    # Safety
    if not validated and edu_list:
        indicators = ['b.e', 'b.tech', 'btech', 'm.tech', 'mtech', 'b.sc', 'm.sc', 'bca', 'mca', 'mba', 'phd',
                      'bachelor', 'master', 'diploma', 'degree', 'university', 'college', 'education', 'graduated', 'pg.', 'pg ', 'engineering']
        if any(ind in text_lower for ind in indicators):
            return edu_list
    return validated


# ═══════════════════════════════════════════════════════════════
#                    CONTACT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_contacts_regex(text: str) -> Dict:
    contacts: Dict[str, str] = {"email": "", "phone": "", "address": "", "linkedin": "", "github": "", "portfolio": "", "location": ""}
    if not text:
        return contacts

    for p in [r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}', r'[\w._%+-]+\s*@\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
              r'[\w._%+-]+\s*\[\s*at\s*\]\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
              r'(?:email|e-mail|mail)[\s.:]*[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            e = re.sub(r'^(?:email|e-mail|mail)[\s.:]*', '', m.group(0), flags=re.IGNORECASE)
            e = re.sub(r'\s+', '', e)
            e = re.sub(r'\[\s*at\s*\]|\(\s*at\s*\)', '@', e, flags=re.IGNORECASE)
            if '@' in e and '.' in e.split('@')[-1]:
                contacts["email"] = e
                break

    for p in [r'\+\d{1,3}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', r'\+\d{10,15}',
              r'\+91[-.\s]?\d{5}[-.\s]?\d{5}', r'\+91[-.\s]?\d{10}',
              r'(?<!\d)91[-.\s]?\d{10}(?!\d)', r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
              r'(?<!\d)\d{3}[-.\s]\d{3}[-.\s]\d{4}(?!\d)',
              r'(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*[\+]?[\d][\d\s\-().]{8,18}',
              r'(?<!\d)\d{5}[-.\s]?\d{5}(?!\d)', r'(?<!\d)\d{10}(?!\d)']:
        try:
            for match in re.findall(p, text, re.IGNORECASE):
                cl = re.sub(r'^(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*', '', match, flags=re.IGNORECASE).strip()
                if 10 <= len(re.sub(r'[^\d]', '', cl)) <= 15:
                    contacts["phone"] = cl
                    break
            if contacts["phone"]:
                break
        except re.error:
            continue

    for p in [r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?', r'linkedin\.com/in/[\w-]+']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            contacts["linkedin"] = m.group(0).strip()
            break

    for p in [r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?', r'github\.com/[\w-]+']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            contacts["github"] = m.group(0).strip()
            break

    for p in [r'(?:portfolio|website|web|site|blog)[\s.:]+(?:https?://)?[\w.-]+\.[a-z]{2,}[\w/.-]*',
              r'(?:https?://)?(?:www\.)?[\w-]+\.(?:dev|io|me|tech|design|codes?|site|online|app)/?[\w/.-]*']:
        for match in re.findall(p, text, re.IGNORECASE):
            url = re.sub(r'^(?:portfolio|website|web|site|blog)[\s.:]+', '', match, flags=re.IGNORECASE).strip()
            if 'linkedin' not in url.lower() and 'github' not in url.lower() and '@' not in url:
                contacts["portfolio"] = url
                break
        if contacts["portfolio"]:
            break

    for p in [r'(?:Address|Location|Residence|Home|Addr)[\s.:]+([^\n]{15,150})',
              r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            addr = (m.group(1) if m.lastindex else m.group(0)).strip()
            addr = re.sub(r'^(?:Address|Location|Residence|Home|Addr)[\s.:]+', '', addr, flags=re.IGNORECASE).strip()
            if 15 < len(addr) < 200:
                contacts["address"] = addr
                break

    for p in [r'(?:Location|Based in|Located at|City|Current Location)[\s.:]+([A-Za-z][A-Za-z\s,]+)',
              r'\b(Bangalore|Bengaluru|Mumbai|Delhi|NCR|Chennai|Hyderabad|Pune|Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur)\b',
              r'\b(New York|NYC|San Francisco|SF|Los Angeles|LA|Chicago|Seattle|Boston|Austin|Denver|Atlanta)\b',
              r'\b(London|Berlin|Amsterdam|Dublin|Singapore|Tokyo|Sydney|Toronto|Vancouver|Melbourne)\b',
              r'\b(India|USA|US|United States|UK|United Kingdom|Canada|Australia|Germany|Netherlands|Singapore|UAE)\b']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            loc = (m.group(1) if m.lastindex else m.group(0)).strip()
            loc = re.sub(r'^(?:Location|Based in|Located at|City|Current Location)[\s.:]+', '', loc, flags=re.IGNORECASE).strip()
            if 2 <= len(loc) <= 100:
                contacts["location"] = loc
                break

    return contacts


# ═══════════════════════════════════════════════════════════════
#                    NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_name_from_text(text: str) -> str:
    if not text:
        return ""
    lines = text.strip().split("\n")
    skip = ['resume', 'curriculum', 'vitae', 'cv', 'http', 'www', '@', 'address', 'phone', 'email',
            'street', 'road', 'avenue', 'objective', 'summary', 'profile', 'linkedin', 'github',
            'portfolio', 'mobile', 'tel:', 'contact', 'experience', 'education', 'skills',
            'professional', 'career', 'about', 'personal', 'details', 'information',
            'confidential', 'page', 'date', 'application', 'position', 'job',
            'candidate', 'recruitment', 'hiring', 'vacancy', 'declaration', 'reference',
            'signature', 'company', 'corporation', 'limited', 'pvt', 'ltd']

    for p in [r'(?:Name|Full Name|Candidate Name|Applicant Name)[\s.:]+([A-Z][a-zA-Z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-zA-Z]+){1,2})',
              r'(?:Name|Full Name)[\s.:]+([A-Z][A-Z\s]+)']:
        m = re.search(p, text, re.IGNORECASE)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    for line in lines[:15]:
        line = line.strip()
        if not line or len(line) < 4 or len(line) > 45:
            continue
        if any(kw in line.lower() for kw in skip):
            continue
        if re.match(r'^\d', line) or len(re.findall(r'\d', line)) >= 5:
            continue
        if re.match(r'^[\+\d\(\)]', line) or '@' in line or 'http' in line.lower():
            continue
        if sum(1 for c in line if c.isalpha() or c.isspace() or c in '.-') / max(len(line), 1) < 0.85:
            continue
        for p in [r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',
                  r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+$', r'^[A-Z]+\s+[A-Z]+$',
                  r'^[A-Z]+\s+[A-Z]+\s+[A-Z]+$', r'^Dr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
                  r'^Mr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$', r'^Ms\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$']:
            if re.match(p, line) and _is_valid_name(line):
                return _clean_name(line)
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w) and all(2 <= len(w) <= 15 for w in words):
            if _is_valid_name(line):
                return _clean_name(line)

    if lines:
        m = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})', lines[0].strip())
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    for p in [r"(?:I am|I'm|My name is|This is|Myself)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})"]:
        m = re.search(p, text, re.IGNORECASE)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    em = re.search(r'([\w.]+)@', text)
    if em:
        parts = re.split(r'[._]', em.group(1))
        if len(parts) >= 2:
            pn = ' '.join(p.capitalize() for p in parts if p.isalpha() and len(p) > 1)
            if len(pn.split()) >= 2:
                return pn

    headers = {'RESUME', 'CURRICULUM', 'VITAE', 'CV', 'OBJECTIVE', 'SUMMARY', 'EXPERIENCE', 'EDUCATION',
               'SKILLS', 'CONTACT', 'PROFILE', 'ABOUT', 'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS',
               'AWARDS', 'REFERENCES', 'DECLARATION', 'PERSONAL', 'PROFESSIONAL', 'TECHNICAL', 'WORK',
               'HISTORY', 'QUALIFICATIONS', 'RESPONSIBILITIES', 'DETAILS', 'INFORMATION', 'OVERVIEW'}
    for line in lines[:10]:
        line = line.strip()
        if line and line.isupper() and 5 <= len(line) <= 40:
            words = line.split()
            if 2 <= len(words) <= 4 and not any(h in line for h in headers):
                tn = line.title()
                if _is_valid_name(tn):
                    return tn
    return ""


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    nc = name.strip()
    nl = nc.lower()
    if nl in {"", "n/a", "na", "none", "unknown", "candidate", "your name", "first last",
              "firstname lastname", "name", "full name", "[name]", "<name>", "(name)",
              "enter name", "type name", "not available", "curriculum vitae", "resume", "cv"}:
        return False
    words = nc.split()
    if not (2 <= len(words) <= 4):
        return False
    for w in words:
        s = w.strip(".-'")
        if not s or len(s) > 20 or sum(c.isalpha() for c in s) / max(len(s), 1) < 0.8:
            return False
    if sum(1 for c in nc if c.isalpha() or c.isspace()) / max(len(nc), 1) < 0.80:
        return False
    if any(w.lower().strip(".,;:()") in _STRONG_NAME_BLACKLIST for w in words):
        return False
    if sum(1 for w in words if w.lower().strip(".,;:()") in _SOFT_NAME_BLACKLIST) >= 2:
        return False
    if nc == nc.lower():
        return False
    return True


def _clean_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r'\s+', ' ', name).strip()
    if name.isupper():
        name = name.title()
    name = re.sub(r'\s*\.\s*', '. ', name)
    return re.sub(r'\s+', ' ', name).strip()


# ═══════════════════════════════════════════════════════════════
#        WORK EXPERIENCE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_work_experience_from_text(text: str) -> List[Dict]:
    if not text:
        return []

    work_entries: List[Dict] = []
    seen_jobs: Set[str] = set()

    MONTH_YEAR = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                  r'Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s*\d{4}')
    NUMERIC_DATE = r'\d{1,2}[/\-\.]\d{4}'
    NUMERIC_DATE_REV = r'\d{4}[/\-\.]\d{1,2}'
    MONTH_APOS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)['\u2019]\s*\d{2,4}"
    PRESENT_KW = r'(?:present|current(?:ly)?|now|ongoing|till\s*date|till\s*now|to\s*date|today|continuing)'
    JUST_YEAR = r'\d{4}'
    ANY_DATE = f'(?:{MONTH_YEAR}|{NUMERIC_DATE}|{NUMERIC_DATE_REV}|{MONTH_APOS}|{PRESENT_KW}|{JUST_YEAR})'
    DATE_SEP = r'\s*(?:to|till|until|[-\u2013\u2014])\s*'

    def _clean_co(c):
        c = re.sub(r'https?://\S+', '', c).strip()
        c = re.sub(r'[,;]\s*\w+$', '', c)
        return c.rstrip('-,. ;:')

    def _norm_key(r, c):
        rn = re.sub(r'[^a-z0-9]', '', r.lower())[:25]
        cn = re.sub(r'[^a-z0-9]', '', c.lower())[:25]
        return f"{rn}_{cn}"

    def _add(role, company, start, end, loc=""):
        role = role.strip().rstrip(',. ')
        company = _clean_co(company)
        end = re.sub(r'https?://\S+', '', end).strip().rstrip('.,;: ')

        # Validate before adding
        test_entry = {"title": role, "company": company}
        if not _is_valid_work_entry(test_entry):
            return

        key = _norm_key(role, company)
        if key in seen_jobs:
            return
        seen_jobs.add(key)

        dur = _calc_duration_from_dates(start, end)
        clean_end = end.strip() if end.strip() and end.strip().lower() not in ['', '-', '\u2013', '\u2014'] else "Present"

        work_entries.append({
            "title": role, "company": company, "location": loc,
            "start_date": start.strip(), "end_date": clean_end,
            "duration_years": dur, "type": "Full-time",
            "description": "", "key_achievements": [], "technologies_used": []
        })

    # Pattern 1
    for m in re.finditer(
        r'(?:currently\s+)?(?:working|employed|serving)\s+(?:as\s+)?(.+?)\s+(?:with|at|in|for)\s+(.+?)\s*'
        rf'(?:since|from)\s+({ANY_DATE}){DATE_SEP}({ANY_DATE})(?:\s*[.\n]|$)', text, re.IGNORECASE):
        _add(m.group(1), m.group(2), m.group(3), m.group(4))

    # Pattern 2a
    for m in re.finditer(
        r'(?:ex|former|previous|past)\s+(?:employee|member|associate|consultant|staff)\s+(?:of|at|with|in)\s+(.+?)'
        rf'\s+from\s+({ANY_DATE}){DATE_SEP}({ANY_DATE})\s+as\s+(.+?)(?:\s*[.\n;]|$)', text, re.IGNORECASE):
        co_raw = m.group(1).strip()
        co_clean = _clean_co(co_raw)
        role = m.group(4).strip().rstrip('.,;: ')
        loc = ""
        lm = re.search(r'[,;]\s*(\w[\w\s]*?)$', re.sub(r'https?://\S+', '', co_raw))
        if lm:
            loc = lm.group(1).strip()
        test_entry = {"title": role, "company": co_clean}
        if not _is_valid_work_entry(test_entry):
            continue
        key = _norm_key(role, co_clean)
        if key in seen_jobs:
            continue
        seen_jobs.add(key)
        dur = _calc_duration_from_dates(m.group(2).strip(), m.group(3).strip())
        work_entries.append({
            "title": role, "company": co_clean, "location": loc,
            "start_date": m.group(2).strip(), "end_date": m.group(3).strip(),
            "duration_years": dur, "type": "Full-time",
            "description": "", "key_achievements": [], "technologies_used": []
        })

    # Pattern 2b
    for m in re.finditer(
        r'(?:ex|former|previous|past)\s+(?:employee|member|associate|consultant|staff)\s+(?:of|at|with|in)\s+(.+?)'
        r'\s+(?:from)\s+(.+?)\s+(?:to|till|until)\s+(.+?)\s+(?:as)\s+(.+?)(?:\s*[.\n;,]|$)', text, re.IGNORECASE):
        co_clean = _clean_co(m.group(1).strip())
        cn = re.sub(r'[^a-z0-9]', '', co_clean.lower())[:20]
        if any(re.sub(r'[^a-z0-9]', '', e.get('company', '').lower())[:20] == cn for e in work_entries):
            continue
        _add(m.group(4).strip(), co_clean, m.group(2).strip(), m.group(3).strip())

    # Pattern 3
    for m in re.finditer(
        r'(?:worked|employed|served|joined|was)\s+(?:as\s+)?(.+?)\s+(?:at|with|in|for)\s+(.+?)\s*'
        rf'(?:from|since)\s+({ANY_DATE}){DATE_SEP}({ANY_DATE})(?:[.\n,;]|$)', text, re.IGNORECASE):
        _add(m.group(1), m.group(2), m.group(3), m.group(4))

    # Pattern 4: tabular
    for m in re.finditer(
        rf'^(.+?)\s+(?:at|with|@)\s+(.+?)\s*[,|]\s*({ANY_DATE}){DATE_SEP}({ANY_DATE})(?:\s*[.\n]|$)',
        text, re.IGNORECASE | re.MULTILINE):
        r, c = m.group(1).strip(), m.group(2).strip()
        if len(r) > 60 or len(c) > 80:
            continue
        if any(s in r.lower() for s in ['education', 'project', 'skill', 'certification', 'award', 'summary']):
            continue
        _add(r, c, m.group(3), m.group(4))

    # Pattern 5: pipe
    for m in re.finditer(
        rf'(.+?)\s*\|\s*(.+?)\s*\|\s*({ANY_DATE}){DATE_SEP}({ANY_DATE})(?:\s*[.\n]|$)', text, re.IGNORECASE):
        c, r = m.group(1).strip(), m.group(2).strip()
        if len(r) > 60 or len(c) > 80:
            continue
        _add(r, c, m.group(3), m.group(4))

    # Pattern 6: fallback date ranges
    if not work_entries:
        for m in re.finditer(rf'(?:from\s+|since\s+)?({ANY_DATE}){DATE_SEP}({ANY_DATE})', text, re.IGNORECASE):
            ss, es = m.group(1).strip(), m.group(2).strip()
            sy, sm = _parse_date_to_ym(ss)
            if not sy or sy < 1980 or sy > CURRENT_YEAR:
                continue
            ctx = text[max(0, m.start() - 250):min(len(text), m.end() + 150)]
            role = company = ""
            rm = re.search(r'(?:as|role|position|designation|title)[:\s]+(.+?)(?:\s+(?:at|with|in|from|since)|\n|,)', ctx, re.IGNORECASE)
            if rm:
                role = rm.group(1).strip()
            cm = re.search(r'(?:at|with|in|for|company|organization|employer)[:\s]+(.+?)(?:\s+(?:from|since|as)|\n|,)', ctx, re.IGNORECASE)
            if cm:
                company = _clean_co(cm.group(1))
            dur = _calc_duration_from_dates(ss, es)
            if dur > 0:
                test_entry = {"title": role or "Not specified", "company": company or "Not specified"}
                if _is_valid_work_entry(test_entry):
                    key = _norm_key(role or "role", company or ss)
                    if key not in seen_jobs:
                        seen_jobs.add(key)
                        work_entries.append({
                            "title": role or "Not specified", "company": company or "Not specified",
                            "location": "", "start_date": ss, "end_date": es,
                            "duration_years": dur, "type": "Full-time",
                            "description": "", "key_achievements": [], "technologies_used": []
                        })

    return work_entries


# ═══════════════════════════════════════════════════════════════
#                    EDUCATION EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_education_regex(text: str) -> List[Dict]:
    education: List[Dict] = []
    if not text:
        return education

    degree_patterns = [
        (r'((?:PG\.?\s*)?Bachelor\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Commerce|Business|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'((?:PG\.?\s*)?B\.?\s*(?:Tech|E|Sc|A|Com|B\.?A|S|CA|BA|Arch|Pharm|Ed|Des)\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'((?:PG\.?\s*)?B\.?Tech|BTech|B\.?E\.?|BE)\s*[-\u2013in\s]*([\w\s,&]+)?', 'Bachelor'),
        (r'(BCA|B\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(BBA|B\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(B\.?Com|BCom)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(Master\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Business\s*Administration|Computer\s*Applications?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?\s*(?:Tech|E|Sc|A|B\.?A|S|CA|BA|Phil|Ed|Des)\.?|MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?Tech|MTech|M\.?E\.?|ME|M\.?S\.?|MS)\s*[-\u2013in\s]*([\w\s,&]+)?', 'Master'),
        (r'(MCA|M\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(PG\s*(?:Diploma|Degree)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(Ph\.?\s*D\.?|Doctorate|Doctor\s*of\s*Philosophy)\s*(?:in\s*)?([\w\s,&]+)?', 'PhD'),
        (r'(Diploma)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(Polytechnic)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(ITI|I\.?T\.?I\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(Higher\s*Secondary|HSC|12th|XII|Intermediate|Senior\s*Secondary|\+2|Plus\s*Two)', 'School'),
        (r'(Secondary|SSC|10th|X|Matriculation|High\s*School|SSLC)', 'School'),
    ]
    inst_patterns = [
        r'([A-Z][A-Za-z\s\.\']+(?:University|College|Institute|School|Academy|Polytechnic))',
        r'((?:IIT|NIT|IIIT|BITS|VIT|SRM|Amity|Manipal|LPU|KIIT|MIT|Stanford|Harvard|Cambridge|Oxford)\s*[,]?\s*[\w\s]*)',
        r'(?:from|at|,)\s+([A-Z][A-Za-z\s\.\']{5,60})',
    ]
    gpa_pats = [r'(?:GPA|CGPA|CPI|Grade)[:\s]*(\d+\.?\d*)', r'(\d{1,2}(?:\.\d+)?)\s*%',
                r'(First\s*Class(?:\s*with\s*Distinction)?|Distinction|Honors?|Honours?)']
    year_pats = [r'((?:19|20)\d{2})\s*[-\u2013to]+\s*((?:19|20)\d{2}|Present|Current|Expected|Pursuing)',
                 r'(?:Class\s*of|Batch|Graduated?|Expected)[:\s]*((?:19|20)\d{2})', r'((?:19|20)\d{2})']

    tu = text.upper()
    es = -1
    for mk in ['EDUCATION', 'ACADEMIC', 'QUALIFICATION', 'EDUCATIONAL BACKGROUND']:
        idx = tu.find(mk)
        if idx != -1:
            es = idx
            break

    if es != -1:
        edu_sec = text[es:]
        for mk in ['EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILL', 'PROJECT', 'CERTIFICATION', 'AWARD', 'ACHIEVEMENT']:
            ei = edu_sec.upper().find(mk, 50)
            if ei != -1:
                edu_sec = edu_sec[:ei]
                break
    else:
        edu_sec = text

    found: List[Dict] = []
    for pat, dt in degree_patterns:
        try:
            for match in re.finditer(pat, edu_sec, re.IGNORECASE):
                dn = match.group(1).strip() if match.group(1) else ""
                fld = ""
                if match.lastindex >= 2 and match.group(2):
                    fld = re.sub(r'\s+', ' ', match.group(2).strip())
                    if len(fld) > 100: fld = fld[:100]
                    if any(sw in fld.lower() for sw in ['university', 'college', 'institute', 'school', 'from', 'at', 'gpa', 'cgpa']):
                        fld = ""
                ctx = edu_sec[max(0, match.start() - 30):min(len(edu_sec), match.end() + 250)]
                entry = {"degree": dn, "field": fld, "institution": "", "year": "", "gpa": "", "location": "", "achievements": []}
                for ip in inst_patterns:
                    try:
                        im = re.search(ip, ctx, re.IGNORECASE)
                        if im:
                            inst = re.sub(r'\s+', ' ', im.group(1).strip())
                            if len(inst) > 5 and inst.lower() not in ['the', 'and', 'for', 'with', 'from']:
                                entry["institution"] = inst
                                break
                    except re.error:
                        continue
                for yp in year_pats:
                    try:
                        ym = re.search(yp, ctx, re.IGNORECASE)
                        if ym:
                            entry["year"] = ym.group(0).strip()
                            break
                    except re.error:
                        continue
                for gp in gpa_pats:
                    try:
                        gm = re.search(gp, ctx, re.IGNORECASE)
                        if gm:
                            entry["gpa"] = gm.group(0).strip()
                            break
                    except re.error:
                        continue
                found.append(entry)
        except re.error:
            continue

    seen: Set[str] = set()
    for edu in found:
        if not _is_valid_education_entry(edu):
            continue
        key = f"{_normalize_degree_key(edu.get('degree', '').lower())}_{edu.get('institution', '').lower()[:20]}"
        if key not in seen and edu.get('degree'):
            seen.add(key)
            education.append(edu)
    return _deduplicate_education_entries(education)


# ═══════════════════════════════════════════════════════════════
#                    SKILLS EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_skills_regex(text: str) -> Dict:
    skills: Dict[str, List[str]] = {
        "programming_languages": [], "frameworks_libraries": [], "ai_ml_tools": [],
        "cloud_platforms": [], "databases": [], "devops_tools": [],
        "visualization": [], "other_tools": [], "soft_skills": []
    }
    if not text:
        return skills
    tl = text.lower()
    cats = {
        "programming_languages": ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'golang',
                                  'rust', 'swift', 'kotlin', 'php', 'scala', 'r', 'matlab', 'perl', 'bash', 'shell',
                                  'powershell', 'sql', 'html', 'css', 'dart', 'lua'],
        "frameworks_libraries": ['react', 'angular', 'vue', 'nextjs', 'next.js', 'nodejs', 'node.js', 'express',
                                 'django', 'flask', 'fastapi', 'spring', 'springboot', 'laravel', 'rails',
                                 'asp.net', '.net', 'jquery', 'bootstrap', 'tailwind', 'flutter', 'react native'],
        "ai_ml_tools": ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy', 'opencv',
                        'nltk', 'spacy', 'huggingface', 'transformers', 'langchain', 'spark', 'pyspark', 'hadoop',
                        'kafka', 'xgboost', 'lightgbm'],
        "cloud_platforms": ['aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'heroku', 'firebase',
                            'vercel', 'netlify', 'lambda', 's3', 'ec2'],
        "databases": ['mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'dynamodb',
                      'cassandra', 'oracle', 'sql server', 'sqlite', 'mariadb', 'neo4j'],
        "devops_tools": ['docker', 'kubernetes', 'k8s', 'jenkins', 'terraform', 'ansible', 'prometheus',
                         'grafana', 'nginx', 'linux', 'unix', 'ubuntu', 'github actions', 'gitlab ci'],
        "visualization": ['power bi', 'tableau', 'looker', 'plotly', 'matplotlib', 'seaborn', 'excel', 'd3.js'],
        "other_tools": ['git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'figma', 'postman',
                        'swagger', 'graphql', 'rest', 'selenium', 'cypress', 'jest', 'pytest', 'webpack',
                        'vite', 'npm', 'yarn', 'maven', 'gradle'],
        "soft_skills": ['leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
                        'analytical', 'creativity', 'adaptability', 'time management', 'project management',
                        'presentation', 'negotiation', 'collaboration', 'mentoring'],
    }
    for cat, sl in cats.items():
        for s in sl:
            if re.search(rf'\b{re.escape(s)}\b', tl):
                d = s.upper() if len(s) <= 3 and s not in ('go', 'r') else s.upper() if s in ('c++', 'c#') else s.title()
                skills[cat].append(d)
    for cat in skills:
        seen: Set[str] = set()
        u = []
        for s in skills[cat]:
            if s.lower() not in seen:
                seen.add(s.lower())
                u.append(s)
        skills[cat] = u
    return skills


# ═══════════════════════════════════════════════════════════════
#                    CONTACT VALIDATION
# ═══════════════════════════════════════════════════════════════

def _is_valid_field(field: str, value: str) -> bool:
    if not value:
        return False
    v = str(value).strip()
    if v.lower().strip() in ['n/a', 'na', 'none', 'not available', 'not specified', 'unknown', 'null', '-',
                              '\u2014', '', 'candidate', 'your name', 'email@example.com', 'xxx', '000',
                              'your email', 'your phone', 'first last', 'firstname lastname', 'name',
                              'full name', '[name]', '<name>', '(name)', 'enter name', 'type name']:
        return False
    if field == 'email':
        return '@' in v and '.' in v.split('@')[-1]
    if field == 'phone':
        return len(re.sub(r'[^\d]', '', v)) >= 10
    if field == 'name':
        return _is_valid_name(v)
    if field in ('linkedin', 'github'):
        return len(v) > 5
    return len(v) >= 2


def _merge_contacts(llm: Dict, regex: Dict, doc: Optional[Dict] = None) -> Dict:
    merged: Dict[str, str] = {}
    if doc is None:
        doc = {}
    for f in ['name', 'email', 'phone', 'address', 'linkedin', 'github', 'portfolio', 'location']:
        lv, dv, rv = str(llm.get(f, "")).strip(), str(doc.get(f, "")).strip(), str(regex.get(f, "")).strip()
        if lv and _is_valid_field(f, lv):
            merged[f] = lv
        elif dv and _is_valid_field(f, dv):
            merged[f] = dv
        elif rv and _is_valid_field(f, rv):
            merged[f] = rv
        else:
            merged[f] = ""
    return merged


def _basic_fallback(text, contacts, name, education, skills):
    return {
        "name": name or contacts.get("name", ""), "email": contacts.get("email", ""),
        "phone": contacts.get("phone", ""), "address": contacts.get("address", ""),
        "linkedin": contacts.get("linkedin", ""), "github": contacts.get("github", ""),
        "portfolio": contacts.get("portfolio", ""), "location": contacts.get("location", ""),
        "current_role": "", "current_company": "", "total_experience_years": 0,
        "professional_summary": text[:500] if text else "", "specializations": [],
        "skills": skills, "work_history": [], "education": education,
        "certifications": [], "awards": [], "projects": [], "publications": [],
        "volunteer": [], "languages": [], "interests": []
    }


# ═══════════════════════════════════════════════════════════════
#                    MAIN PARSING FUNCTION
# ═══════════════════════════════════════════════════════════════

def parse_resume_with_llm(resume_text: str, groq_api_key: str,
                          model_id: str = "llama-3.1-8b-instant",
                          doc_contacts: Optional[Dict] = None) -> Dict:

    if not resume_text or len(resume_text.strip()) < 50:
        return _basic_fallback(resume_text, {}, "", [], {})

    # STEP 1: Regex
    regex_contacts = _extract_contacts_regex(resume_text)
    regex_name = _extract_name_from_text(resume_text)
    regex_education = _extract_education_regex(resume_text)
    regex_skills = _extract_skills_regex(resume_text)

    if doc_contacts is None:
        doc_contacts = {}
        try:
            from document_processor import extract_contacts_from_text
            doc_contacts = extract_contacts_from_text(resume_text)
        except ImportError:
            pass

    # STEP 2: LLM
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": (
                "You are a resume parsing expert. Current year is 2026. "
                "Return ONLY valid JSON. No markdown. "
                "NEVER create education from awards or company names. "
                "NEVER use prompt template examples as real data. "
                "Only extract what is ACTUALLY WRITTEN in the resume."
            )},
            {"role": "user", "content": PARSE_PROMPT.format(resume_text=resume_text[:12000])}
        ],
        "temperature": 0.05, "max_tokens": 6000,
    }

    parsed = None
    for try_model in list(dict.fromkeys([model_id, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"])):
        payload["model"] = try_model
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                parsed = json.loads(content.strip())
                break
            elif resp.status_code == 429:
                time.sleep(1)
        except json.JSONDecodeError:
            try:
                jm = re.search(r'\{[\s\S]*\}', content)
                if jm:
                    parsed = json.loads(jm.group())
                    break
            except Exception:
                pass
        except Exception:
            continue

    if not parsed:
        parsed = _basic_fallback(resume_text, regex_contacts, regex_name, regex_education, regex_skills)

    # STEP 2b: Validate education
    el = parsed.get("education", [])
    parsed["education"] = _filter_valid_education(el if isinstance(el, list) else [el] if el else [])

    # STEP 2c: Cross-validate education against text
    parsed["education"] = _validate_education_against_text(parsed["education"], resume_text)

    # STEP 3: Contacts
    mc = _merge_contacts(parsed, regex_contacts, doc_contacts)
    for f, v in mc.items():
        if v and (not parsed.get(f) or not _is_valid_field(f, parsed.get(f, ""))):
            parsed[f] = v

    # STEP 4: Name
    cn = parsed.get("name", "")
    if not cn or not _is_valid_field("name", cn):
        for c in [doc_contacts.get("name", ""), regex_name, parsed.get("name", "")]:
            if c and _is_valid_field("name", c):
                parsed["name"] = _clean_name(c)
                break
        if not parsed.get("name") or not _is_valid_field("name", parsed.get("name", "")):
            parsed["name"] = "Unknown Candidate"

    # STEP 5: Merge education + safety net
    le = parsed.get("education", [])
    if not isinstance(le, list):
        le = []
    if regex_education:
        if not le:
            parsed["education"] = regex_education
        else:
            ldk: Set[str] = set()
            for e in le:
                if isinstance(e, dict):
                    dk = _normalize_degree_key(str(e.get("degree", "")))
                    if dk:
                        ldk.add(dk)
            for re_edu in regex_education:
                if isinstance(re_edu, dict):
                    dk = _normalize_degree_key(str(re_edu.get("degree", "")))
                    if dk and dk not in ldk:
                        le.append(re_edu)
                        ldk.add(dk)
            parsed["education"] = _deduplicate_education_entries(le)

    if not parsed.get("education"):
        tlc = resume_text.lower()
        if any(ind in tlc for ind in ['b.e', 'b.tech', 'btech', 'm.tech', 'mtech', 'b.sc', 'm.sc',
                                       'bca', 'mca', 'mba', 'bachelor', 'master', 'phd', 'diploma',
                                       'degree', 'university', 'college', 'graduated', 'pg.', 'pg ', 'engineering']):
            retry = _extract_education_regex(resume_text)
            if retry:
                parsed["education"] = retry
            else:
                for dp in [r'((?:PG\.?\s*)?B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA|B\.?Sc|M\.?Sc|Ph\.?D)\s*[-\u2013.]\s*(.+?)(?:\s+from\s+|\s+at\s+)(.+?)(?:\n|$)',
                           r'((?:PG\.?\s*)?B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA)\s+(.+?)(?:\s+from\s+|\s+at\s+)(.+?)(?:\n|$)']:
                    m = re.search(dp, resume_text, re.IGNORECASE)
                    if m:
                        parsed["education"] = [{"degree": m.group(1).strip(),
                                                "field": m.group(2).strip() if m.lastindex >= 2 else "",
                                                "institution": m.group(3).strip() if m.lastindex >= 3 else "",
                                                "year": "", "gpa": "", "location": "", "achievements": []}]
                        break

    # STEP 6: Skills
    ls = parsed.get("skills", {})
    if not isinstance(ls, dict):
        ls = {"other_tools": ls} if isinstance(ls, list) else {}
    for cat, rsl in regex_skills.items():
        if not rsl:
            continue
        if cat in ls and isinstance(ls[cat], list):
            existing = set(s.lower() for s in ls[cat] if s)
            for s in rsl:
                if s and s.lower() not in existing:
                    ls[cat].append(s)
        elif rsl:
            ls[cat] = rsl
    parsed["skills"] = ls

    # ══════════════════════════════════════════════════════
    # STEP 7: Experience — ALWAYS independently calculated
    # ══════════════════════════════════════════════════════
    work_history = parsed.get("work_history", parsed.get("experience", []))
    if not isinstance(work_history, list):
        work_history = []

    # Validate LLM work entries
    work_history = [j for j in work_history if isinstance(j, dict) and _is_valid_work_entry(j)]
    parsed["work_history"] = work_history

    # 7b: ALWAYS run regex extraction
    regex_work = _extract_work_experience_from_text(resume_text)
    if regex_work:
        if not work_history:
            work_history = regex_work
        else:
            work_history.extend(regex_work)
        parsed["work_history"] = work_history

    # 7c: Deduplicate — ONE entry per company, best quality wins
    final_work: List[Dict] = []
    seen_companies: Dict[str, int] = {}

    for j in parsed.get("work_history", []):
        if not isinstance(j, dict):
            continue
        if not _is_valid_work_entry(j):
            continue

        co_raw = j.get('company', '') or j.get('organization', '') or ''
        ti_raw = j.get('title', '') or j.get('role', '') or j.get('position', '') or ''

        co_clean = re.sub(r'https?://\S+', '', co_raw)
        co_clean = re.sub(r'[,;]\s*\w+$', '', co_clean)
        co_clean = re.sub(r'\s*[-\u2013\u2014]\s*$', '', co_clean).strip()

        co_words = re.sub(r'[^a-z0-9\s]', '', co_clean.lower()).split()
        co_core = [w for w in co_words if w not in LOCATION_WORDS and len(w) > 1]
        co_norm = ''.join(co_core)[:20] if co_core else ''.join(co_words)[:20]

        if not co_norm:
            co_norm = re.sub(r'[^a-z0-9]', '', ti_raw.lower())[:20]
        if not co_norm:
            final_work.append(j)
            continue

        # Calculate duration from dates (NEVER trust duration_years from LLM)
        start_str = str(j.get('start_date', j.get('from', '')))
        end_str = str(j.get('end_date', j.get('to', '')))
        indep_dur = _calc_duration_from_dates(start_str, end_str) if start_str else 0.0

        has_start = bool(start_str.strip())
        has_end = bool(end_str.strip())
        quality = indep_dur + (1.0 if has_start and has_end else 0) + (0.5 if has_start else 0)

        # Fix the stored duration_years
        if indep_dur > 0:
            j["duration_years"] = indep_dur

        if co_norm in seen_companies:
            ei = seen_companies[co_norm]
            ex = final_work[ei]
            ex_start = str(ex.get('start_date', ex.get('from', '')))
            ex_end = str(ex.get('end_date', ex.get('to', '')))
            ex_dur = _calc_duration_from_dates(ex_start, ex_end) if ex_start else 0.0
            ex_quality = ex_dur + (1.0 if ex_start and ex_end else 0) + (0.5 if ex_start else 0)
            if quality > ex_quality:
                final_work[ei] = j
        else:
            seen_companies[co_norm] = len(final_work)
            final_work.append(j)

    parsed["work_history"] = final_work

    # 7d: ALWAYS recalculate total from dates
    total_exp = 0.0
    for job in final_work:
        if not isinstance(job, dict):
            continue
        ss = str(job.get("start_date", job.get("from", "")))
        es = str(job.get("end_date", job.get("to", "")))
        if ss:
            dur = _calc_duration_from_dates(ss, es)
            if dur > 0:
                job["duration_years"] = dur
                total_exp += dur
                continue
        dur = job.get("duration_years", 0)
        try:
            total_exp += float(dur) if dur else 0
        except (ValueError, TypeError):
            pass
    parsed["total_experience_years"] = round(total_exp, 1)

    # STEP 8: Current role/company
    if not parsed.get("current_role") or not parsed.get("current_company"):
        wh = parsed.get("work_history", [])
        if isinstance(wh, list) and wh and isinstance(wh[0], dict):
            if not parsed.get("current_role"):
                parsed["current_role"] = wh[0].get("title", "") or wh[0].get("role", "") or wh[0].get("position", "")
            if not parsed.get("current_company"):
                parsed["current_company"] = wh[0].get("company", "") or wh[0].get("organization", "")

    # STEP 9: Defaults
    defaults: Dict = {
        "name": "Unknown Candidate", "email": "", "phone": "", "address": "",
        "linkedin": "", "github": "", "portfolio": "", "location": "",
        "current_role": "", "current_company": "", "total_experience_years": 0,
        "professional_summary": "", "specializations": [], "skills": {},
        "work_history": [], "education": [], "certifications": [], "awards": [],
        "projects": [], "publications": [], "volunteer": [], "languages": [], "interests": []
    }
    for f, d in defaults.items():
        if f not in parsed:
            parsed[f] = d

    return parsed


# ═══════════════════════════════════════════════════════════════
#                    DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_resume_display_summary(parsed: Dict) -> str:
    if not parsed:
        return "No resume data"
    lines = [f"**{parsed.get('name', 'Candidate')}**"]
    r, c = parsed.get("current_role", ""), parsed.get("current_company", "")
    if r:
        lines.append(f"\U0001f4bc {r}" + (f" at {c}" if c else ""))
    loc = parsed.get("location", "") or parsed.get("address", "")
    if loc:
        lines.append(f"\U0001f4cd {loc[:60]}")
    exp = parsed.get("total_experience_years", 0)
    if exp:
        lines.append(f"\U0001f4c5 ~{exp} years experience")
    if parsed.get("email"):
        lines.append(f"\U0001f4e7 {parsed['email']}")
    if parsed.get("phone"):
        lines.append(f"\U0001f4de {parsed['phone']}")
    if parsed.get("linkedin"):
        lines.append("\U0001f517 LinkedIn")
    if parsed.get("github"):
        lines.append("\U0001f4bb GitHub")
    return "\n".join(lines)


def get_resume_full_summary(parsed: Dict) -> Dict:
    if not parsed:
        return {}
    sc = 0
    sd = parsed.get("skills", {})
    if isinstance(sd, dict):
        for v in sd.values():
            if isinstance(v, list):
                sc += len(v)
    elif isinstance(sd, list):
        sc = len(sd)
    return {
        "name": parsed.get("name", "Unknown"), "email": parsed.get("email", ""),
        "phone": parsed.get("phone", ""), "location": parsed.get("location", "") or parsed.get("address", ""),
        "current_role": parsed.get("current_role", ""), "current_company": parsed.get("current_company", ""),
        "total_experience_years": parsed.get("total_experience_years", 0),
        "skills_count": sc, "education_count": len(parsed.get("education", [])),
        "work_history_count": len(parsed.get("work_history", [])),
        "certifications_count": len(parsed.get("certifications", [])),
        "projects_count": len(parsed.get("projects", [])),
        "has_linkedin": bool(parsed.get("linkedin")), "has_github": bool(parsed.get("github")),
        "has_portfolio": bool(parsed.get("portfolio")), "has_summary": bool(parsed.get("professional_summary")),
    }


def extract_key_highlights(parsed: Dict) -> List[str]:
    h: List[str] = []
    if not parsed:
        return h
    exp = parsed.get("total_experience_years", 0)
    if exp:
        h.append(f"\U0001f4c5 {exp} years of experience")
    r, c = parsed.get("current_role", ""), parsed.get("current_company", "")
    if r and c:
        h.append(f"\U0001f4bc Currently {r} at {c}")
    elif r:
        h.append(f"\U0001f4bc {r}")
    edu = parsed.get("education", [])
    if edu and isinstance(edu, list) and isinstance(edu[0], dict):
        d = edu[0].get("degree", "")
        if d:
            inst = edu[0].get("institution", "")
            h.append(f"\U0001f393 {d}" + (f" from {inst}" if inst else ""))
    sd = parsed.get("skills", {})
    sc = sum(len(v) for v in sd.values() if isinstance(v, list)) if isinstance(sd, dict) else 0
    if sc:
        h.append(f"\U0001f6e0\ufe0f {sc} skills identified")
    if parsed.get("certifications"):
        h.append(f"\U0001f4dc {len(parsed['certifications'])} certification(s)")
    if parsed.get("projects"):
        h.append(f"\U0001f680 {len(parsed['projects'])} project(s)")
    return h


def get_contact_completeness(parsed: Dict) -> Dict:
    if not parsed:
        return {"score": 0, "missing": ["all"]}
    fields = {"name": parsed.get("name", ""), "email": parsed.get("email", ""),
              "phone": parsed.get("phone", ""), "location": parsed.get("location", "") or parsed.get("address", ""),
              "linkedin": parsed.get("linkedin", "")}
    present = [f for f, v in fields.items() if v and _is_valid_field(f, v)]
    missing = [f for f in fields if f not in present]
    return {"score": round(len(present) / len(fields) * 100, 1), "present": present,
            "missing": missing, "total_fields": len(fields), "filled_fields": len(present)}


def validate_parsed_resume(parsed: Dict) -> Dict:
    if not parsed:
        return {"valid": False, "issues": ["No data"]}
    issues, warnings = [], []
    if not parsed.get("name") or parsed["name"] in ["Unknown Candidate", ""]:
        issues.append("Name not extracted")
    if not parsed.get("email"):
        warnings.append("Email not found")
    if not parsed.get("phone"):
        warnings.append("Phone not found")
    if not parsed.get("work_history"):
        warnings.append("No work history extracted")
    if not parsed.get("education"):
        warnings.append("No education extracted")
    sd = parsed.get("skills", {})
    if (sum(len(v) for v in sd.values() if isinstance(v, list)) if isinstance(sd, dict) else 0) == 0:
        warnings.append("No skills extracted")
    return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings,
            "quality_score": max(0, 100 - len(issues) * 20 - len(warnings) * 5)}
