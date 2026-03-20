"""
Enhanced Resume Parser V9
- Triple extraction: Document Processor + Regex + LLM
- Enhanced name extraction: 8 strategies including Regards/Declaration sections
- Enhanced education validation + cross-validation against resume text
- FIXED: Structured resume format parsing (multi-line role/company/date blocks)
- FIXED: Section-aware extraction — only extracts from WORK EXPERIENCE section
- FIXED: Prevents date ranges from education/certs/awards leaking into work history
- Independent experience calculation — NEVER trusts LLM totals
- Work entry validation — rejects garbage role/company names
- Spaced-out text detection
- PIN-prefix email cleaning
- Accurate experience calculation using CURRENT_YEAR = 2026
"""

import json
import re
import time
import requests
from typing import Dict, List, Tuple, Optional, Set

CURRENT_YEAR = 2026
CURRENT_MONTH = 3

# ═══════════════════════════════════════════════════════════════
#          BLACKLISTS & CONSTANTS
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
    "about", "more", "regarding", "section", "introduction",
    "career", "table", "contents", "index", "appendix",
    "voice", "message", "routing", "optimization",
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

_GARBAGE_WORK_WORDS: Set[str] = {
    "in", "a", "an", "the", "at", "of", "for", "to", "and", "or",
    "with", "from", "by", "on", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "not", "no", "yes", "position", "role", "company",
    "voice", "message", "routing", "optimization", "client",
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

# Section headers that signal END of work experience section
_NON_WORK_SECTION_HEADERS: Set[str] = {
    'education', 'skills', 'certifications', 'certification',
    'awards', 'scholarships', 'awards & scholarships',
    'projects', 'publications', 'volunteer', 'languages',
    'interests', 'hobbies', 'references', 'declaration',
    'training', 'courses', 'accomplishments',
    'technical skills', 'professional skills',
    'academic details', 'academic qualifications',
    'personal details', 'personal information',
}

# Section headers that signal START of work experience section
_WORK_SECTION_HEADERS: Set[str] = {
    'work experience', 'experience', 'professional experience',
    'employment history', 'employment', 'work history',
    'career history', 'professional background',
    'relevant experience', 'industry experience',
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
            "start_date": "Month Year or MM/YYYY",
            "end_date": "Month Year or MM/YYYY or Present",
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
9. WORK HISTORY: Only include entries from the WORK EXPERIENCE section.
   DO NOT include education, certifications, or awards as work entries.

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

    ds_clean = re.sub(r'[^a-z\s]', '', ds).strip()
    if ds_clean in PRESENT_WORDS or any(pw in ds for pw in PRESENT_WORDS):
        return CURRENT_YEAR, CURRENT_MONTH

    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*['\u2019]\s*(\d{2,4})", ds)
    if m:
        yr = int(m.group(2))
        if yr < 100:
            yr += 2000 if yr < 50 else 1900
        return yr, month_map.get(m.group(1)[:3], 6)

    m = re.search(
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
        r'aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[.,]?\s*(\d{4})', ds)
    if m:
        return int(m.group(2)), month_map.get(m.group(1)[:3], 6)

    m = re.search(r'(\d{1,2})[/\-\.](\d{4})', ds)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))

    m = re.search(r'(\d{4})[/\-\.](\d{1,2})', ds)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))

    m = re.search(r'((?:19|20)\d{2})', ds)
    if m:
        return int(m.group(1)), 6
    return None, None


def _calc_duration_from_dates(start_str: str, end_str: str) -> float:
    sy, sm = _parse_date_to_ym(start_str)
    if not sy:
        return 0.0
    if not end_str or end_str.strip().lower() in PRESENT_WORDS or end_str.strip() in ['-', '\u2013', '']:
        ey, em = CURRENT_YEAR, CURRENT_MONTH
    else:
        ey, em = _parse_date_to_ym(end_str)
        if not ey:
            ey, em = CURRENT_YEAR, CURRENT_MONTH
    return round(max(0, ((ey - sy) * 12 + ((em or 6) - (sm or 6))) / 12.0), 1)


# ═══════════════════════════════════════════════════════════════
#               WORK ENTRY VALIDATION
# ═══════════════════════════════════════════════════════════════

def _is_valid_work_entry(job: Dict) -> bool:
    if not isinstance(job, dict):
        return False
    title = str(job.get("title", "") or job.get("role", "") or job.get("position", "")).strip()
    company = str(job.get("company", "") or job.get("organization", "")).strip()

    if not title and not company:
        return False

    def _check_field(val):
        if not val:
            return True  # empty is OK if the other field is valid
        if len(val) < 2:
            return False
        words = val.lower().split()
        meaningful = [w for w in words if w not in _GARBAGE_WORK_WORDS and len(w) > 1]
        if not meaningful:
            return False
        if val.lower().startswith(('in ', 'a ', 'an ', 'the ', 'at ', 'of ')):
            return False
        preps = sum(1 for w in words if w in {'in', 'a', 'an', 'the', 'at', 'of', 'for', 'to', 'with'})
        if preps >= len(words) * 0.5 and len(words) > 2:
            return False
        return True

    return _check_field(title) and _check_field(company)


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
        w1, w2 = words[i].lower().strip(".,;:()"), words[i + 1].lower().strip(".,;:()")
        if w1 == w2 and len(w1) > 1:
            return True
    counts: Dict[str, int] = {}
    for w in words:
        k = w.lower().strip(".,;:()")
        if len(k) > 1:
            counts[k] = counts.get(k, 0) + 1
    return any(c >= 3 for c in counts.values())


def _has_garbage_pattern(text: str) -> bool:
    if not text:
        return False
    if text.count('. .') >= 2 or text.count(' . ') >= 2:
        return True
    s = text.strip()
    if s and s[0].islower():
        fw = s.split()[0] if s.split() else ""
        if fw and fw not in {"in", "at", "of", "for", "the", "and", "or", "with", "to", "from"} and len(fw) < 5:
            return True
    for hw in ["education", "experience", "skills", "summary", "objective", "profile"]:
        if text.lower().count(hw) >= 2:
            return True
    return False


def _is_valid_education_entry(edu: Dict) -> bool:
    if not isinstance(edu, dict):
        return False
    degree = str(edu.get("degree", "")).strip()
    fos = str(edu.get("field", "") or edu.get("major", "") or edu.get("branch", "") or edu.get("specialization", "")).strip()
    inst = str(edu.get("institution", "") or edu.get("university", "") or edu.get("college", "")).strip()

    if not degree or len(degree) < 2:
        return False
    if sum(1 for c in degree if c.isalpha()) < len(degree) * 0.5:
        return False

    combined = f"{degree} {fos} {inst}".lower()
    for kw in ["award", "performer", "performance", "star performer", "received", "recognition",
                "appreciation", "good performances", "best employee", "employee of", "delivered",
                "achievements", "achieved", "client project", "client ems", "project delivery",
                "team lead", "team member", "responsible for", "worked on", "developed", "implemented"]:
        if kw in combined:
            return False

    if _has_repetition_pattern(inst) or _has_repetition_pattern(fos) or _has_repetition_pattern(degree):
        return False
    if _has_garbage_pattern(inst) or _has_garbage_pattern(fos):
        return False

    il = inst.lower().strip()
    if il.count("education") >= 1 and len(il) < 30:
        return False
    if il in {"education", "qualifications", "academic details", "academic qualifications",
              "educational details", "educational qualifications", "academic background", "educational background"}:
        return False

    ashort = {"it", "cs", "ai", "ml", "ee", "ec", "me", "ce"}
    if fos and len(fos) == 1:
        return False
    if fos and len(fos) < 3 and fos.lower() not in ashort:
        return False
    if inst and len(re.sub(r'[^a-zA-Z]', '', inst)) < 3:
        return False
    if inst and inst.split() and inst.split()[0][0].islower() and len(inst.split()[0]) < 6:
        return False
    if fos and fos[0].islower() and len(fos) < 5 and fos.lower() not in ashort:
        return False

    if degree.lower().strip('.') in {"ma", "me", "ms", "ba", "be", "bs"}:
        skw = ["university", "college", "institute", "school", "academy", "polytechnic",
               "iit", "nit", "iiit", "bits", "vit", "mit", "anna", "delhi", "mumbai", "vidyalaya", "vidyapeeth"]
        if inst:
            if not any(k in il for k in skw):
                return False
        else:
            return False

    ci = ["mahindra", "infosys", "wipro", "tcs", "cognizant", "accenture", "capgemini", "hcl",
          "tech mahindra", "client", "pvt", "ltd", "inc", "llc", "corp", "solutions",
          "technologies", "services", "consulting", "private limited", "limited", "software"]
    if inst:
        skw2 = ["university", "college", "institute", "school", "academy", "polytechnic"]
        if any(k in il for k in ci) and not any(k in il for k in skw2):
            return False
    if inst and len(inst) > 150:
        return False
    return True


def _normalize_degree_key(degree: str) -> str:
    if not degree:
        return ""
    d = re.sub(r'[^a-z0-9\s]', '', degree.lower().strip())
    d = re.sub(r'\s+', '', d)
    dnp = re.sub(r'^pg\.?\s*', '', degree.lower().strip())
    dnp = re.sub(r'[^a-z0-9]', '', dnp)

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
    if dnp != d:
        for p, r in norms.items():
            if re.match(p, dnp):
                return r
    return d


def _deduplicate_education_entries(el: List[Dict]) -> List[Dict]:
    if not el:
        return []
    groups: Dict[str, List[Dict]] = {}
    for e in el:
        if not isinstance(e, dict):
            continue
        dg = str(e.get("degree", "")).strip()
        if not dg:
            continue
        k = _normalize_degree_key(dg) or re.sub(r'[^a-z0-9]', '', dg.lower())
        groups.setdefault(k, []).append(e)

    result: List[Dict] = []
    for entries in groups.values():
        if len(entries) == 1:
            result.append(entries[0])
        else:
            best, bs = entries[0], -999
            for e in entries:
                s = 0
                i2 = str(e.get("institution", "") or e.get("university", "") or e.get("college", "")).strip()
                f2 = str(e.get("field", "") or e.get("major", "") or e.get("branch", "")).strip()
                if i2 and len(i2) > 5: s += 3
                if f2 and len(f2) > 2: s += 2
                if e.get("year") or e.get("end_year") or e.get("graduation_year"): s += 2
                if e.get("gpa") or e.get("cgpa") or e.get("grade"): s += 1
                if e.get("location"): s += 1
                if _has_garbage_pattern(i2): s -= 5
                if _has_repetition_pattern(i2): s -= 5
                if s > bs:
                    bs = s
                    best = e
            result.append(best)
    return result


def _filter_valid_education(el: list) -> list:
    if not el or not isinstance(el, list):
        return []
    return _deduplicate_education_entries([e for e in el if isinstance(e, dict) and _is_valid_education_entry(e)])


def _validate_education_against_text(el: List[Dict], resume_text: str) -> List[Dict]:
    if not el or not resume_text:
        return el
    tl = resume_text.lower()
    ta = re.sub(r'[^a-z0-9\s]', '', tl)
    validated = []
    ignore = {'the', 'and', 'for', 'from', 'of', 'in', 'at', 'to', 'with'}
    generic = {'university', 'college', 'institute', 'school', 'academy', 'degree', 'bachelor', 'master', 'doctor', 'diploma', 'science', 'arts', 'technology', 'engineering'}
    dabbrs = {
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

    for edu in el:
        if not isinstance(edu, dict):
            continue
        degree = str(edu.get("degree", "")).strip()
        fv = str(edu.get("field", "") or edu.get("major", "") or edu.get("branch", "") or edu.get("specialization", "")).strip()
        inst = str(edu.get("institution", "") or edu.get("university", "") or edu.get("college", "")).strip()
        found = False

        if inst and len(inst) > 3:
            iw = [w for w in inst.lower().split() if w not in ignore and w not in generic]
            if iw:
                if any(w in tl for w in iw if len(w) > 2):
                    found = True
                if not found:
                    iwa = [re.sub(r'[^a-z0-9]', '', w) for w in iw]
                    if any(w in ta for w in iwa if len(w) > 2):
                        found = True
            else:
                ia = re.sub(r'[^a-z0-9\s]', '', inst.lower()).strip()
                if ia and ia in ta:
                    found = True

        if not found and degree and len(degree) > 1:
            dl = degree.lower()
            da = re.sub(r'[^a-z0-9\s]', '', dl)
            if dl in tl or da in ta:
                found = True
            if not found:
                for ff, abbrs in dabbrs.items():
                    if ff in dl or dl in ff:
                        if any(a in tl for a in abbrs):
                            found = True
                            break
            if not found:
                dw = [w for w in dl.split() if len(w) > 1 and w not in ignore and w not in generic]
                if dw and sum(1 for w in dw if w in tl) >= 1:
                    found = True
            if not found:
                am = re.search(r'(?:PG\.?\s*)?(B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA|B\.?Sc|M\.?Sc|Ph\.?D)', degree, re.IGNORECASE)
                if am:
                    ab = am.group(0).lower()
                    if ab in tl or ab.replace('.', '') in ta:
                        found = True

        if not found and fv and len(fv) > 3:
            if fv.lower() in tl:
                found = True
            else:
                fw = [w for w in fv.lower().split() if len(w) > 3 and w not in ignore and w not in generic]
                if fw and any(w in tl for w in fw):
                    found = True
        if found:
            validated.append(edu)

    if not validated and el:
        inds = ['b.e', 'b.tech', 'btech', 'm.tech', 'mtech', 'b.sc', 'm.sc', 'bca', 'mca', 'mba', 'phd',
                'bachelor', 'master', 'diploma', 'degree', 'university', 'college', 'education', 'graduated', 'pg.', 'pg ', 'engineering']
        if any(i in tl for i in inds):
            return el
    return validated


# ═══════════════════════════════════════════════════════════════
#                    CONTACT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_contacts_regex(text: str) -> Dict:
    contacts: Dict[str, str] = {
        "email": "", "phone": "", "address": "",
        "linkedin": "", "github": "", "portfolio": "", "location": ""
    }
    if not text:
        return contacts

    # ═══════ EMAIL ═══════
    email_patterns = [
        r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*@\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*\[\s*at\s*\]\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*\(\s*at\s*\)\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'(?:email|e-mail|mail)[\s.:]*[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}',
    ]
    for p in email_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            email = m.group(0).strip()
            email = re.sub(r'^(?:email|e-mail|mail)[\s.:]*', '', email, flags=re.IGNORECASE)
            email = re.sub(r'\s+', '', email)
            email = re.sub(r'\[\s*at\s*\]|\(\s*at\s*\)', '@', email, flags=re.IGNORECASE)

            if '@' in email:
                local_part = email.split('@')[0]
                pin_match = re.match(r'^(\d{4,7})[.\-_]', local_part)
                if pin_match:
                    email = email[len(pin_match.group(1)) + 1:]

                if not pin_match:
                    local_part = email.split('@')[0]
                    digit_prefix = re.match(r'^(\d{5,7})([a-zA-Z])', local_part)
                    if digit_prefix:
                        email = local_part[len(digit_prefix.group(1)):] + '@' + email.split('@')[1]

            if '@' in email and '.' in email.split('@')[-1]:
                final_local = email.split('@')[0]
                if any(c.isalpha() for c in final_local):
                    if len(final_local) >= 2:
                        contacts["email"] = email
                        break

    # ═══════ PHONE ═══════
    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        r'\+\d{1,3}[-.\s]?\d{4,5}[-.\s]?\d{5,6}',
        r'\+\d{10,15}',
        r'\+91[-.\s]?\d{5}[-.\s]?\d{5}',
        r'\+91[-.\s]?\d{10}',
        r'(?<!\d)91[-.\s]?\d{10}(?!\d)',
        r'(?<!\d)0\d{2,4}[-.\s]?\d{6,8}(?!\d)',
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        r'(?<!\d)\d{3}[-.\s]\d{3}[-.\s]\d{4}(?!\d)',
        r'(?<!\d)1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)',
        r'\+44[-.\s]?\d{4}[-.\s]?\d{6}',
        r'(?<!\d)0\d{4}[-.\s]?\d{6}(?!\d)',
        r'(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*[\+]?[\d][\d\s\-().]{8,18}',
        r'(?<!\d)\d{3}[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)',
        r'(?<!\d)\d{5}[-.\s]?\d{5}(?!\d)',
        r'(?<!\d)\d{10}(?!\d)',
    ]
    for p in phone_patterns:
        try:
            for match in re.findall(p, text, re.IGNORECASE):
                cleaned = re.sub(
                    r'^(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*',
                    '', match, flags=re.IGNORECASE
                ).strip()
                digits = re.sub(r'[^\d]', '', cleaned)
                if 10 <= len(digits) <= 15:
                    contacts["phone"] = cleaned
                    break
            if contacts["phone"]:
                break
        except re.error:
            continue

    # ═══════ LINKEDIN ═══════
    for p in [
        r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?',
        r'linkedin\.com/in/[\w-]+',
    ]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            contacts["linkedin"] = m.group(0).strip()
            break

    # ═══════ GITHUB ═══════
    for p in [
        r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?',
        r'github\.com/[\w-]+',
    ]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            contacts["github"] = m.group(0).strip()
            break

    # ═══════ PORTFOLIO ═══════
    portfolio_patterns = [
        r'(?:portfolio|website|web|site|blog)[\s.:]+(?:https?://)?[\w.-]+\.[a-z]{2,}[\w/.-]*',
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:dev|io|me|tech|design|codes?|site|online|app)/?[\w/.-]*',
    ]
    for p in portfolio_patterns:
        for match in re.findall(p, text, re.IGNORECASE):
            url = re.sub(
                r'^(?:portfolio|website|web|site|blog)[\s.:]+',
                '', match, flags=re.IGNORECASE
            ).strip()
            url_lower = url.lower()
            if ('linkedin' not in url_lower
                    and 'github' not in url_lower
                    and '@' not in url
                    and 'ibm.com' not in url_lower):
                contacts["portfolio"] = url
                break
        if contacts["portfolio"]:
            break

    # ═══════ ADDRESS ═══════
    address_patterns = [
        r'(?:Address|Location|Residence|Home|Addr)[\s.:]+([^\n]{15,150})',
        r'(?:No[.:]?\s*)?[\d]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar|Colony|'
        r'Layout|Block|Lane|Apt|Apartment|Floor|Fl|Building|Bldg|Society|Housing)[\w\s,.-]+(?:\d{5,6})',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}',
        r'[\'"\u2018\u2019]?[\w\s]+(?:Society|Housing|Apartment|Complex|Residency)[\w\s,.-]+\d{5,6}',
    ]
    for p in address_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            addr = (m.group(1) if m.lastindex else m.group(0)).strip()
            addr = re.sub(
                r'^(?:Address|Location|Residence|Home|Addr)[\s.:]+',
                '', addr, flags=re.IGNORECASE
            ).strip()
            addr = addr.lstrip("'\"\u2018\u2019")
            if 15 < len(addr) < 200:
                contacts["address"] = addr
                break

    # ═══════ LOCATION ═══════
    location_patterns = [
        r'(?:Location|Based in|Located at|City|Current Location)[\s.:]+([A-Za-z][A-Za-z\s,]+)',
        r'\b(Bangalore|Bengaluru|Mumbai|Delhi|NCR|Chennai|Hyderabad|Pune|Kolkata|'
        r'Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Lucknow|Chandigarh|Indore|Bhopal|'
        r'Kochi|Coimbatore|Trivandrum|Mysore|Nagpur|Surat|Vadodara|Bhubaneswar|'
        r'Patna|Ranchi|Guwahati|Visakhapatnam|Vijayawada|Kolhapur|Pondicherry)\b',
        r'\b(New York|NYC|San Francisco|SF|Los Angeles|LA|Chicago|Seattle|Boston|'
        r'Austin|Denver|Atlanta|Dallas|Houston|Phoenix|San Diego|San Jose|Portland|'
        r'Miami|Washington DC|Philadelphia)\b',
        r'\b(London|Berlin|Amsterdam|Dublin|Singapore|Tokyo|Sydney|Toronto|'
        r'Vancouver|Melbourne|Paris|Munich|Barcelona|Stockholm|Copenhagen|'
        r'Zurich|Dubai|Hong Kong)\b',
        r'\b(India|USA|US|United States|UK|United Kingdom|Canada|Australia|'
        r'Germany|Netherlands|Singapore|UAE)\b',
    ]
    for p in location_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            loc = (m.group(1) if m.lastindex else m.group(0)).strip()
            loc = re.sub(
                r'^(?:Location|Based in|Located at|City|Current Location)[\s.:]+',
                '', loc, flags=re.IGNORECASE
            ).strip()
            if 2 <= len(loc) <= 100:
                contacts["location"] = loc
                break

    return contacts


# ═══════════════════════════════════════════════════════════════
#                    NAME EXTRACTION — 8 STRATEGIES
# ═══════════════════════════════════════════════════════════════

def _is_spaced_out_text(line: str) -> bool:
    if not line or len(line) < 5:
        return False
    parts = [p for p in line.split(' ') if p]
    if len(parts) < 3:
        return False
    return sum(1 for p in parts if len(p) == 1) / len(parts) > 0.5


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
            'signature', 'company', 'corporation', 'limited', 'pvt', 'ltd',
            'more', 'voice', 'message', 'routing', 'optimization']

    # Strategy 1: Labeled name
    for p in [r'(?:Name|Full Name|Candidate Name|Applicant Name)[\s.:]+([A-Z][a-zA-Z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-zA-Z]+){1,2})',
              r'(?:Name|Full Name)[\s.:]+([A-Z][A-Z\s]+)']:
        m = re.search(p, text, re.IGNORECASE)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    # Strategy 1.5: ALL-CAPS multi-word name in first 20 lines
    for line in lines[:20]:
        line = line.strip()
        if not line or len(line) < 5 or len(line) > 50:
            continue
        if any(kw in line.lower() for kw in skip):
            continue
        if '@' in line or 'http' in line.lower():
            continue
        if re.match(r'^\d', line) or len(re.findall(r'\d', line)) >= 3:
            continue
        if _is_spaced_out_text(line):
            continue
        if line.isupper() or (line == line.upper() and ' ' in line):
            words = line.split()
            if 2 <= len(words) <= 4 and all(len(w) >= 2 and w.isalpha() for w in words):
                tn = line.title()
                if _is_valid_name(tn):
                    return _clean_name(tn)

    # Strategy 2: First line that looks like a name
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
        if _is_spaced_out_text(line):
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

    # Strategy 3: First line mixed content
    if lines:
        m = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})', lines[0].strip())
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    # Strategy 4: "I am" / "My name is"
    for p in [r"(?:I am|I'm|My name is|This is|Myself)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})"]:
        m = re.search(p, text, re.IGNORECASE)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    # Strategy 5: From email
    em = re.search(r'([\w.]+)@', text)
    if em:
        parts = re.split(r'[._]', em.group(1))
        if len(parts) >= 2:
            pn = ' '.join(p.capitalize() for p in parts if p.isalpha() and len(p) > 1)
            if len(pn.split()) >= 2:
                return pn

    # Strategy 6: ALL CAPS line (first 10 lines)
    headers = {'RESUME', 'CURRICULUM', 'VITAE', 'CV', 'OBJECTIVE', 'SUMMARY', 'EXPERIENCE', 'EDUCATION',
               'SKILLS', 'CONTACT', 'PROFILE', 'ABOUT', 'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS',
               'AWARDS', 'REFERENCES', 'DECLARATION', 'PERSONAL', 'PROFESSIONAL', 'TECHNICAL', 'WORK',
               'HISTORY', 'QUALIFICATIONS', 'RESPONSIBILITIES', 'DETAILS', 'INFORMATION', 'OVERVIEW',
               'MORE ABOUT ME', 'AWARDS & ACHIEVEMENT', 'WORK EXPERIENCE', 'TECHNICAL SKILLS'}
    for line in lines[:10]:
        line = line.strip()
        if line and line.isupper() and 5 <= len(line) <= 40:
            words = line.split()
            if 2 <= len(words) <= 4 and not any(h in line for h in headers):
                tn = line.title()
                if _is_valid_name(tn):
                    return tn

    # Strategy 7: Name after "Regards" / "Sincerely" at end of resume
    for p in [r'(?:Regards|Sincerely|Yours\s+(?:truly|faithfully|sincerely)|'
              r'Thank\s*(?:you|s)|Best\s+regards|Kind\s+regards|Warm\s+regards|'
              r'Respectfully|Cordially)\s*[,.]?\s*\n\s*'
              r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})']:
        m = re.search(p, text, re.IGNORECASE)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    # Strategy 8: Name after "Declaration" section
    for p in [r'(?:Declaration|I\s+hereby\s+declare).+?(?:Date|Location|Place|Regards)\s*[:\s]*\w*\s*\n\s*'
              r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\s*$']:
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    return ""


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    nc = name.strip()
    nl = nc.lower()
    if nl in {"", "n/a", "na", "none", "unknown", "candidate", "your name", "first last",
              "firstname lastname", "name", "full name", "[name]", "<name>", "(name)",
              "enter name", "type name", "not available", "curriculum vitae", "resume", "cv",
              "more about me", "about me"}:
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
#        SECTION EXTRACTION HELPER
# ═══════════════════════════════════════════════════════════════

def _extract_work_section(text: str) -> str:
    """Extract ONLY the work experience section from resume text.
    
    This prevents dates from education, certifications, and awards
    from being incorrectly parsed as work experience.
    """
    if not text:
        return ""

    lines = text.split('\n')
    work_start = -1
    work_end = len(lines)

    # Find work experience section start
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        # Remove common formatting characters
        cleaned = re.sub(r'[^a-z\s]', '', stripped).strip()
        if cleaned in _WORK_SECTION_HEADERS or any(cleaned.startswith(h) for h in _WORK_SECTION_HEADERS):
            work_start = i + 1  # Start after the header line
            break

    if work_start == -1:
        # No explicit work section header found — return empty
        # (will rely on LLM or narrative patterns applied to full text)
        return ""

    # Find where work section ends (next non-work section header)
    for i in range(work_start, len(lines)):
        stripped = lines[i].strip().lower()
        cleaned = re.sub(r'[^a-z\s&]', '', stripped).strip()
        if cleaned in _NON_WORK_SECTION_HEADERS or any(cleaned.startswith(h) for h in _NON_WORK_SECTION_HEADERS):
            work_end = i
            break

    work_text = '\n'.join(lines[work_start:work_end])
    return work_text.strip()


# ═══════════════════════════════════════════════════════════════
#        WORK EXPERIENCE EXTRACTION (FIXED)
# ═══════════════════════════════════════════════════════════════

def _extract_work_experience_from_text(text: str) -> List[Dict]:
    """Extract work experience entries from resume text.
    
    FIXED in V9:
    - First extracts the WORK EXPERIENCE section to avoid date leaks
    - Handles structured multi-line format:
        Role • Type
        Company • Location • MM/YYYY - MM/YYYY
      OR:
        Role • Type
        Company • Location
        MM/YYYY - MM/YYYY
    - Narrative patterns only applied to full text as last resort
    - Fallback pattern 6 is section-aware and much more conservative
    """
    if not text:
        return []

    work_entries: List[Dict] = []
    seen_jobs: Set[str] = set()

    # ── Date building blocks ──
    MY = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
          r'Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s*\d{4}')
    ND = r'\d{1,2}[/\-\.]\d{4}'
    NDR = r'\d{4}[/\-\.]\d{1,2}'
    MA = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)['\u2019]\s*\d{2,4}"
    PK = r'(?:[Pp]resent|[Cc]urrent(?:ly)?|[Nn]ow|[Oo]ngoing|[Tt]ill\s*[Dd]ate|[Tt]ill\s*[Nn]ow|[Tt]o\s*[Dd]ate|[Tt]oday|[Cc]ontinuing)'
    JY = r'(?:19|20)\d{2}'
    AD = f'(?:{MY}|{ND}|{NDR}|{MA}|{PK}|{JY})'
    DS = r'\s*(?:to|till|until|[-\u2013\u2014])\s*'
    DATE_RANGE = rf'({AD})\s*(?:to|till|until|[-\u2013\u2014])\s*({AD})'

    def _cco(c):
        c = re.sub(r'https?://\S+', '', c).strip()
        c = re.sub(r'[,;]\s*\w+$', '', c)
        return c.rstrip('-,. ;:')

    def _nk(r, c):
        return f"{re.sub(r'[^a-z0-9]', '', r.lower())[:25]}_{re.sub(r'[^a-z0-9]', '', c.lower())[:25]}"

    def _add(role, company, start, end, loc="", emp_type="Full-time", description=""):
        role = role.strip().rstrip(',. ')
        company = _cco(company)
        end = re.sub(r'https?://\S+', '', end).strip().rstrip('.,;: ')
        if not _is_valid_work_entry({"title": role, "company": company}):
            return
        key = _nk(role, company)
        if key in seen_jobs:
            return
        seen_jobs.add(key)
        dur = _calc_duration_from_dates(start, end)
        ce = end.strip() if end.strip() and end.strip().lower() not in ['', '-', '\u2013', '\u2014'] else "Present"
        work_entries.append({
            "title": role, "company": company, "location": loc,
            "start_date": start.strip(), "end_date": ce, "duration_years": dur,
            "type": emp_type, "description": description,
            "key_achievements": [], "technologies_used": []
        })

    # ══════════════════════════════════════════════════════
    # PHASE 1: Structured format parsing (section-aware)
    # ══════════════════════════════════════════════════════
    work_section = _extract_work_section(text)

    if work_section:
        # ── Pattern A: Multi-line structured blocks ──
        # Handles formats like:
        #   Role Title • Full-time
        #   Company Name • Location • MM/YYYY - MM/YYYY
        # OR:
        #   Role Title • Full-time
        #   Company Name • Location
        #   MM/YYYY - Present
        
        ws_lines = work_section.split('\n')
        i = 0
        while i < len(ws_lines):
            line = ws_lines[i].strip()
            if not line:
                i += 1
                continue

            # Skip bullet points and description lines
            if line.startswith(('•', '-', '*', '\u2022', '\u25e6', '\u2023')):
                i += 1
                continue

            # Try to detect a role line (contains job title keywords or bullet separator)
            # A role line typically: "Role Title • Type" or "Role Title"
            role_match = re.match(
                r'^(.+?(?:Developer|Engineer|Analyst|Scientist|Manager|Designer|Architect|'
                r'Consultant|Lead|Specialist|Administrator|Coordinator|Executive|Officer|'
                r'Intern|Trainee|Associate|Director|Head|VP|President|Summarizer|'
                r'Record\s+Analyst|Data\s+Science).*?)$',
                line, re.IGNORECASE
            )

            if not role_match:
                # Also try: line has bullet separator and looks like "Title • Type"
                if re.match(r'^[^•\n]+\s*[\u2022•]\s*(?:Full-time|Part-time|Internship|Contract|Freelance|Temporary)',
                            line, re.IGNORECASE):
                    role_match = re.match(r'^(.+)$', line)

            if not role_match:
                i += 1
                continue

            role_line = role_match.group(1).strip()

            # Extract employment type from role line
            emp_type = "Full-time"
            type_match = re.search(
                r'[\u2022•\-|,]\s*(Full-time|Part-time|Internship|Contract|Freelance|Temporary)\s*$',
                role_line, re.IGNORECASE
            )
            if type_match:
                emp_type = type_match.group(1).strip()
                role_line = role_line[:type_match.start()].strip().rstrip('•\u2022-|, ')

            # Check if this line itself contains a date range (single-line format)
            inline_date = re.search(DATE_RANGE, role_line)
            if inline_date:
                # Remove date from role
                role_line = role_line[:inline_date.start()].strip().rstrip('•\u2022-|, ')

            # Now look ahead for company line and/or date line
            company = ""
            location = ""
            start_date = ""
            end_date = ""

            if inline_date:
                start_date = inline_date.group(1)
                end_date = inline_date.group(2)

            # Look at next lines for company and dates
            lookahead_limit = min(i + 4, len(ws_lines))
            j = i + 1
            while j < lookahead_limit:
                next_line = ws_lines[j].strip()
                if not next_line:
                    j += 1
                    continue

                # Skip bullet points
                if next_line.startswith(('•', '-', '*', '\u2022', '\u25e6', '\u2023')):
                    break  # Hit description section, stop looking

                # Check if this line is a date range line
                date_match = re.match(rf'^\s*({AD})\s*(?:to|till|until|[-\u2013\u2014])\s*({AD})\s*$', next_line, re.IGNORECASE)
                if date_match:
                    if not start_date:
                        start_date = date_match.group(1)
                        end_date = date_match.group(2)
                    j += 1
                    continue

                # Check if this line is a company line (may contain dates too)
                # Company lines look like: "Company Name • Location • MM/YYYY - MM/YYYY"
                # or: "Company Name • Location"
                company_with_date = re.match(
                    rf'^(.+?)\s*[\u2022•]\s*({AD})\s*(?:to|till|until|[-\u2013\u2014])\s*({AD})\s*$',
                    next_line, re.IGNORECASE
                )
                if company_with_date:
                    company_loc = company_with_date.group(1).strip()
                    if not start_date:
                        start_date = company_with_date.group(2)
                        end_date = company_with_date.group(3)
                    # Split company • location
                    cl_parts = re.split(r'\s*[\u2022•]\s*', company_loc)
                    company = cl_parts[0].strip() if cl_parts else company_loc
                    if len(cl_parts) > 1:
                        location = cl_parts[-1].strip()
                    j += 1
                    continue

                # Check if line has date range embedded
                line_date = re.search(DATE_RANGE, next_line)
                if line_date:
                    # Extract company/location from before the date
                    pre_date = next_line[:line_date.start()].strip().rstrip('•\u2022-|, ')
                    if pre_date and not company:
                        cl_parts = re.split(r'\s*[\u2022•]\s*', pre_date)
                        company = cl_parts[0].strip() if cl_parts else pre_date
                        if len(cl_parts) > 1:
                            location = cl_parts[-1].strip()
                    if not start_date:
                        start_date = line_date.group(1)
                        end_date = line_date.group(2)
                    j += 1
                    continue

                # Otherwise it's a company line without dates
                if not company:
                    # It might be "Company • Location" or just "Company"
                    cl_parts = re.split(r'\s*[\u2022•]\s*', next_line)
                    # Filter out employment type tokens
                    cl_parts = [p.strip() for p in cl_parts
                                if p.strip().lower() not in ('full-time', 'part-time', 'internship',
                                                              'contract', 'freelance', 'temporary')]
                    if cl_parts:
                        company = cl_parts[0].strip()
                        if len(cl_parts) > 1:
                            # Last part might be location or date
                            last_part = cl_parts[-1].strip()
                            # Check if it's a date
                            if re.match(rf'^{AD}$', last_part, re.IGNORECASE):
                                pass  # skip, it's a partial date
                            else:
                                location = last_part
                    j += 1
                    continue

                # If we already have company, this might be another role block
                break

            # If we found a valid entry, add it
            if role_line and (company or start_date):
                # Clean up company — remove location words at end if location not set
                if company and not location:
                    # Check for "Company • Location" pattern already split
                    loc_match = re.search(
                        r'[,\u2022•]\s*(Pondicherry|Chennai|Hyderabad|Bangalore|Bengaluru|Mumbai|'
                        r'Delhi|Pune|Kolkata|Noida|Gurgaon|Gurugram|India)\s*$',
                        company, re.IGNORECASE
                    )
                    if loc_match:
                        location = loc_match.group(1).strip()
                        company = company[:loc_match.start()].strip().rstrip(',•\u2022 ')

                _add(role_line, company, start_date, end_date, location, emp_type)
                i = j  # Jump past the lines we consumed
            else:
                i += 1

    # ══════════════════════════════════════════════════════
    # PHASE 2: Narrative patterns (full text) — only if structured parsing found nothing
    # ══════════════════════════════════════════════════════
    if not work_entries:
        # Pattern 1: "currently working/employed as X at Y since/from DATE to DATE"
        for m in re.finditer(
            r'(?:currently\s+)?(?:working|employed|serving)\s+(?:as\s+)?(.+?)\s+(?:with|at|in|for)\s+(.+?)\s*'
            rf'(?:since|from)\s+({AD}){DS}({AD})(?:\s*[.\n]|$)', text, re.IGNORECASE):
            _add(m.group(1), m.group(2), m.group(3), m.group(4))

        # Pattern 2: "worked/employed/served as X at Y from DATE to DATE"
        for m in re.finditer(
            r'(?:worked|employed|served|joined|was)\s+(?:as\s+)?(.+?)\s+(?:at|with|in|for)\s+(.+?)\s*'
            rf'(?:from|since)\s+({AD}){DS}({AD})(?:[.\n,;]|$)', text, re.IGNORECASE):
            _add(m.group(1), m.group(2), m.group(3), m.group(4))

        # Pattern 3: tabular "Role at Company, DATE - DATE"
        for m in re.finditer(
            rf'^(.+?)\s+(?:at|with|@)\s+(.+?)\s*[,|]\s*({AD}){DS}({AD})(?:\s*[.\n]|$)',
            text, re.IGNORECASE | re.MULTILINE):
            r, c = m.group(1).strip(), m.group(2).strip()
            if len(r) > 60 or len(c) > 80:
                continue
            if any(s in r.lower() for s in ['education', 'project', 'skill', 'certification', 'award', 'summary']):
                continue
            _add(r, c, m.group(3), m.group(4))

        # Pattern 4: pipe-separated "Company | Role | DATE - DATE"
        for m in re.finditer(
            rf'(.+?)\s*\|\s*(.+?)\s*\|\s*({AD}){DS}({AD})(?:\s*[.\n]|$)', text, re.IGNORECASE):
            c, r = m.group(1).strip(), m.group(2).strip()
            if len(r) > 60 or len(c) > 80:
                continue
            _add(r, c, m.group(3), m.group(4))

    # ══════════════════════════════════════════════════════
    # PHASE 3: Conservative fallback — ONLY within work section
    # ══════════════════════════════════════════════════════
    # OLD BUG: Pattern 6 scanned the ENTIRE text for date ranges,
    # picking up education dates, certification dates, award dates, etc.
    # FIX: Only scan within the work section, and require meaningful context.
    if not work_entries and work_section:
        for m in re.finditer(DATE_RANGE, work_section, re.IGNORECASE):
            ss, es = m.group(1).strip(), m.group(2).strip()
            sy, sm = _parse_date_to_ym(ss)
            if not sy or sy < 1980 or sy > CURRENT_YEAR:
                continue
            ctx = work_section[max(0, m.start() - 250):min(len(work_section), m.end() + 150)]
            role = company = ""
            rm = re.search(
                r'(?:as|role|position|designation|title)[:\s]+(.+?)(?:\s+(?:at|with|in|from|since)|\n|,)',
                ctx, re.IGNORECASE
            )
            if rm:
                role = rm.group(1).strip()
            cm = re.search(
                r'(?:at|with|in|for|company|organization|employer)[:\s]+(.+?)(?:\s+(?:from|since|as)|\n|,)',
                ctx, re.IGNORECASE
            )
            if cm:
                company = _cco(cm.group(1))
            dur = _calc_duration_from_dates(ss, es)
            if dur > 0 and (role or company):
                te = {"title": role or "Not specified", "company": company or "Not specified"}
                if _is_valid_work_entry(te):
                    key = _nk(role or "role", company or ss)
                    if key not in seen_jobs:
                        seen_jobs.add(key)
                        work_entries.append({
                            "title": te["title"], "company": te["company"], "location": "",
                            "start_date": ss, "end_date": es, "duration_years": dur,
                            "type": "Full-time", "description": "",
                            "key_achievements": [], "technologies_used": []
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
        (r'((?:PG\.?\s*)?Bachelor\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Commerce|Business|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'((?:PG\.?\s*)?B\.?\s*(?:Tech|E|Sc|A|Com|B\.?A|S|CA|BA|Arch|Pharm|Ed|Des)\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'((?:PG\.?\s*)?B\.?Tech|BTech|B\.?E\.?|BE)\s*[-\u2013in\s]*([\w\s,&]+)?', 'B'),
        (r'(BCA|B\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'(BBA|B\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'(B\.?Com|BCom)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'(Master\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Business\s*Administration|Computer\s*Applications?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(M\.?\s*(?:Tech|E|Sc|A|B\.?A|S|CA|BA|Phil|Ed|Des)\.?|MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(M\.?Tech|MTech|M\.?E\.?|ME|M\.?S\.?|MS)\s*[-\u2013in\s]*([\w\s,&]+)?', 'M'),
        (r'(MCA|M\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(PG\s*(?:Diploma|Degree)?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(Ph\.?\s*D\.?|Doctorate|Doctor\s*of\s*Philosophy)\s*(?:in\s*)?([\w\s,&]+)?', 'P'),
        (r'(Diploma)\s*(?:in\s*)?([\w\s,&]+)?', 'D'),
        (r'(Polytechnic)\s*(?:in\s*)?([\w\s,&]+)?', 'D'),
        (r'(ITI|I\.?T\.?I\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'D'),
        (r'(Higher\s*Secondary|HSC|12th|XII|Intermediate|Senior\s*Secondary|\+2|Plus\s*Two)', 'S'),
        (r'(Secondary|SSC|SSLC|10th|X|Matriculation|High\s*School)', 'S'),
    ]
    inst_pats = [
        r'([A-Z][A-Za-z\s\.\']+(?:University|College|Institute|School|Academy|Polytechnic))',
        r'((?:IIT|NIT|IIIT|BITS|VIT|SRM|Amity|Manipal|LPU|KIIT|MIT|Stanford|Harvard|Cambridge|Oxford)\s*[,]?\s*[\w\s]*)',
        r'(?:from|at|,)\s+([A-Z][A-Za-z\s\.\']{5,60})',
    ]
    gpa_pats = [r'(?:GPA|CGPA|CPI|Grade)[:\s]*(\d+\.?\d*)', r'(\d{1,2}(?:\.\d+)?)\s*%',
                r'(First\s*Class(?:\s*with\s*Distinction)?|Distinction|Honors?|Honours?)']
    yr_pats = [r'((?:19|20)\d{2})\s*[-\u2013to]+\s*((?:19|20)\d{2}|Present|Current|Expected|Pursuing)',
               r'(?:Class\s*of|Batch|Graduated?|Expected)[:\s]*((?:19|20)\d{2})', r'((?:19|20)\d{2})']

    tu = text.upper()
    es = -1
    for mk in ['EDUCATION', 'ACADEMIC', 'QUALIFICATION', 'EDUCATIONAL BACKGROUND']:
        idx = tu.find(mk)
        if idx != -1:
            es = idx
            break
    if es != -1:
        esec = text[es:]
        for mk in ['EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILL', 'PROJECT', 'CERTIFICATION', 'AWARD', 'ACHIEVEMENT']:
            ei = esec.upper().find(mk, 50)
            if ei != -1:
                esec = esec[:ei]
                break
    else:
        esec = text

    found: List[Dict] = []
    for pat, dt in degree_patterns:
        try:
            for match in re.finditer(pat, esec, re.IGNORECASE):
                dn = match.group(1).strip() if match.group(1) else ""
                fld = ""
                if match.lastindex >= 2 and match.group(2):
                    fld = re.sub(r'\s+', ' ', match.group(2).strip())
                    if len(fld) > 100: fld = fld[:100]
                    if any(sw in fld.lower() for sw in ['university', 'college', 'institute', 'school', 'from', 'at', 'gpa', 'cgpa']):
                        fld = ""
                ctx = esec[max(0, match.start() - 30):min(len(esec), match.end() + 250)]
                entry = {"degree": dn, "field": fld, "institution": "", "year": "", "gpa": "", "location": "", "achievements": []}
                for ip in inst_pats:
                    try:
                        im = re.search(ip, ctx, re.IGNORECASE)
                        if im:
                            inst = re.sub(r'\s+', ' ', im.group(1).strip())
                            if len(inst) > 5 and inst.lower() not in ['the', 'and', 'for', 'with', 'from']:
                                entry["institution"] = inst
                                break
                    except re.error:
                        continue
                for yp in yr_pats:
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

    # STEP 1: Regex extraction
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

    # STEP 2: LLM parsing
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": (
                "You are a resume parsing expert. Current year is 2026. "
                "Return ONLY valid JSON. No markdown. "
                "NEVER create education from awards or company names. "
                "NEVER use prompt template examples as real data. "
                "Only extract what is ACTUALLY WRITTEN in the resume. "
                "Only include WORK entries from the WORK EXPERIENCE section."
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
    parsed["education"] = _validate_education_against_text(parsed["education"], resume_text)

    # STEP 3: Merge contacts
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

    # STEP 5: Education merge
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
            for re_e in regex_education:
                if isinstance(re_e, dict):
                    dk = _normalize_degree_key(str(re_e.get("degree", "")))
                    if dk and dk not in ldk:
                        le.append(re_e)
                        ldk.add(dk)
            parsed["education"] = _deduplicate_education_entries(le)

    if not parsed.get("education"):
        tlc = resume_text.lower()
        if any(i in tlc for i in ['b.e', 'b.tech', 'btech', 'm.tech', 'mtech', 'b.sc', 'm.sc', 'bca', 'mca', 'mba',
                                   'bachelor', 'master', 'phd', 'diploma', 'degree', 'university', 'college',
                                   'graduated', 'pg.', 'pg ', 'engineering']):
            retry = _extract_education_regex(resume_text)
            if retry:
                parsed["education"] = retry
            else:
                for dp in [r'((?:PG\.?\s*)?B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA|B\.?Sc|M\.?Sc|Ph\.?D)\s*[-\u2013.:]\s*(.+?)(?:\s+from\s+|\s+at\s+)(.+?)(?:\n|$)',
                           r'((?:PG\.?\s*)?B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA)\s+(.+?)(?:\s+from\s+|\s+at\s+)(.+?)(?:\n|$)']:
                    m = re.search(dp, resume_text, re.IGNORECASE)
                    if m:
                        parsed["education"] = [{"degree": m.group(1).strip(),
                                                "field": m.group(2).strip() if m.lastindex >= 2 else "",
                                                "institution": m.group(3).strip() if m.lastindex >= 3 else "",
                                                "year": "", "gpa": "", "location": "", "achievements": []}]
                        break

    # STEP 6: Skills merge
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
    # STEP 7: Work Experience (FIXED)
    # ══════════════════════════════════════════════════════
    work_history = parsed.get("work_history", parsed.get("experience", []))
    if not isinstance(work_history, list):
        work_history = []
    work_history = [j for j in work_history if isinstance(j, dict) and _is_valid_work_entry(j)]

    # 7b: Run regex extraction
    regex_work = _extract_work_experience_from_text(resume_text)

    # 7b-FIXED: Intelligent merge — prefer higher-quality entries per company
    # Instead of blindly extending, merge by company identity
    if regex_work:
        if not work_history:
            work_history = regex_work
        else:
            # Build a set of company keys already in LLM results
            llm_company_keys: Set[str] = set()
            for j in work_history:
                co = str(j.get('company', '') or j.get('organization', '') or '').strip()
                co_clean = re.sub(r'[^a-z0-9]', '', co.lower())[:20]
                if co_clean:
                    llm_company_keys.add(co_clean)

            # Only add regex entries that are NOT already represented by LLM
            for rj in regex_work:
                co = str(rj.get('company', '') or '').strip()
                co_clean = re.sub(r'[^a-z0-9]', '', co.lower())[:20]
                if co_clean and co_clean not in llm_company_keys:
                    work_history.append(rj)
                    llm_company_keys.add(co_clean)
                elif not co_clean:
                    # No company — skip garbage entries
                    pass

    parsed["work_history"] = work_history

    # 7c: Dedup — keep best entry per company
    final_work: List[Dict] = []
    seen_co: Dict[str, int] = {}
    for j in parsed.get("work_history", []):
        if not isinstance(j, dict) or not _is_valid_work_entry(j):
            continue
        co_raw = j.get('company', '') or j.get('organization', '') or ''
        ti_raw = j.get('title', '') or j.get('role', '') or j.get('position', '') or ''
        cc = re.sub(r'https?://\S+', '', co_raw)
        cc = re.sub(r'[,;]\s*\w+$', '', cc)
        cc = re.sub(r'\s*[-\u2013\u2014]\s*$', '', cc).strip()
        cw = re.sub(r'[^a-z0-9\s]', '', cc.lower()).split()
        core = [w for w in cw if w not in LOCATION_WORDS and len(w) > 1]
        cn = ''.join(core)[:20] if core else ''.join(cw)[:20]
        if not cn:
            cn = re.sub(r'[^a-z0-9]', '', ti_raw.lower())[:20]
        if not cn:
            final_work.append(j)
            continue
        ss = str(j.get('start_date', j.get('from', '')))
        es = str(j.get('end_date', j.get('to', '')))
        idur = _calc_duration_from_dates(ss, es) if ss else 0.0
        q = idur + (1.0 if ss and es else 0) + (0.5 if ss else 0)
        # Bonus quality for having real role and company names
        if ti_raw and ti_raw != "Not specified":
            q += 2.0
        if co_raw and co_raw != "Not specified":
            q += 2.0
        if idur > 0:
            j["duration_years"] = idur
        if cn in seen_co:
            ei = seen_co[cn]
            ex = final_work[ei]
            exs = str(ex.get('start_date', ex.get('from', '')))
            exe = str(ex.get('end_date', ex.get('to', '')))
            exd = _calc_duration_from_dates(exs, exe) if exs else 0.0
            ex_ti = ex.get('title', '') or ex.get('role', '') or ''
            ex_co = ex.get('company', '') or ex.get('organization', '') or ''
            exq = exd + (1.0 if exs and exe else 0) + (0.5 if exs else 0)
            if ex_ti and ex_ti != "Not specified":
                exq += 2.0
            if ex_co and ex_co != "Not specified":
                exq += 2.0
            if q > exq:
                final_work[ei] = j
        else:
            seen_co[cn] = len(final_work)
            final_work.append(j)
    parsed["work_history"] = final_work

    # 7d: Recalculate total experience from dates — NEVER trust LLM total
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

    # 7e: Extract experience from profile text if still 0
    if parsed.get("total_experience_years", 0) == 0:
        for check_text in [parsed.get("professional_summary", "") or parsed.get("summary", "") or "", resume_text[:3000]]:
            if not check_text:
                continue
            cl = check_text.lower()
            for ep in [r'(\d+)\s*\+\s*years?\s*(?:of\s*)?(?:experience|exp)',
                       r'(\d+)\s*(?:\+\s*)?years?\s*(?:of\s*)?(?:experience|exp)',
                       r'over\s+(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)',
                       r'more\s+than\s+(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)',
                       r'(\d+)\s*(?:\+\s*)?years?\s*(?:of\s*)?(?:industry|work|professional)',
                       r'experience\s*(?:of\s*)?(\d+)\s*\+?\s*years?',
                       r'nearly\s+(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)']:
                em = re.search(ep, cl)
                if em:
                    yrs = int(em.group(1))
                    if 1 <= yrs <= 40:
                        parsed["total_experience_years"] = float(yrs)
                        break
            if parsed.get("total_experience_years", 0) > 0:
                break

    # 7f: Create work entry from header/profile if still empty
    if not parsed.get("work_history"):
        header_lines = resume_text.strip().split('\n')[:10]
        for line in header_lines:
            line = line.strip()
            if not line or len(line) < 5 or _is_spaced_out_text(line):
                continue
            rcp = (r'^(.+?(?:Developer|Engineer|Analyst|Scientist|Manager|Designer|Architect|'
                   r'Consultant|Lead|Specialist|Administrator|Coordinator|Executive|Officer))'
                   r'\s*[,\-\u2013\u2014|]+\s*(.+?)(?:\s*[,\-\u2013\u2014|]+\s*.+)?$')
            rm = re.match(rcp, line, re.IGNORECASE)
            if rm:
                role = rm.group(1).strip()
                co_raw = rm.group(2).strip()
                company = re.sub(r'https?://\S+', '', co_raw).strip()
                company = re.sub(r'[,\-\u2013\u2014|]\s*\w+$', '', company).strip().rstrip('-,. ;:')
                if company and len(company) >= 2 and _is_valid_work_entry({"title": role, "company": company}):
                    ey = parsed.get("total_experience_years", 0)
                    try:
                        ey = float(ey)
                    except (ValueError, TypeError):
                        ey = 0
                    parsed["work_history"] = [{"title": role, "company": company, "location": "",
                                               "start_date": "", "end_date": "Present", "duration_years": ey if ey > 0 else 0,
                                               "type": "Full-time", "description": "", "key_achievements": [], "technologies_used": []}]
                    if not parsed.get("current_role"):
                        parsed["current_role"] = role
                    if not parsed.get("current_company"):
                        parsed["current_company"] = company
                    break

        # Strategy B: "currently working" in text body
        if not parsed.get("work_history"):
            for cp in [r'currently\s+working\s+(?:as\s+)?(?:a\s+)?[\'"]?(.+?)[\'"]?\s+'
                       r'(?:for|with|at|in)\s+(?:client\s+)?(.+?)(?:\.|,|\n|$)']:
                cm = re.search(cp, resume_text, re.IGNORECASE)
                if cm:
                    role = cm.group(1).strip().rstrip('\'\"')
                    company = cm.group(2).strip().rstrip('.,;')
                    company = re.sub(r'https?://\S+', '', company).strip()
                    if len(company.split()) > 4 or any(w in company.lower() for w in
                            ['voice', 'message', 'routing', 'optimization', 'project', 'system']):
                        company = parsed.get("current_company", "")
                        if not company:
                            for hl in header_lines[:5]:
                                hl = hl.strip()
                                if not hl or _is_spaced_out_text(hl):
                                    continue
                                hm = re.search(r'(?:Developer|Engineer|Analyst|Scientist|Manager)\s*[,\-\u2013\u2014|]+\s*(\w+)', hl, re.IGNORECASE)
                                if hm:
                                    company = hm.group(1).strip()
                                    break
                    if company and _is_valid_work_entry({"title": role, "company": company}):
                        ey = parsed.get("total_experience_years", 0)
                        try:
                            ey = float(ey)
                        except (ValueError, TypeError):
                            ey = 0
                        parsed["work_history"] = [{"title": role, "company": company, "location": "",
                                                   "start_date": "", "end_date": "Present", "duration_years": ey if ey > 0 else 0,
                                                   "type": "Full-time", "description": "", "key_achievements": [], "technologies_used": []}]
                        if not parsed.get("current_role"):
                            parsed["current_role"] = role
                        if not parsed.get("current_company"):
                            parsed["current_company"] = company
                        break

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
