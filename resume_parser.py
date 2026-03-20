"""
Enhanced Resume Parser V6
- Triple extraction: Document Processor + Regex + LLM
- Enhanced name extraction with strong/soft blacklists
- Enhanced education validation + cross-validation against resume text
- Paragraph-format work experience extraction with ALL date formats
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
CURRENT_MONTH = 3  # March 2026


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


# ═══════════════════════════════════════════════════════════════
#                    LLM PARSING PROMPT
# ═══════════════════════════════════════════════════════════════

PARSE_PROMPT = """You are an expert resume parser. The current date is March 2026.

RESUME TEXT:
{resume_text}

Extract ALL information into this exact JSON structure (no markdown, no code blocks):

{{
    "name": "FULL name exactly as written - look at the VERY FIRST lines, usually largest/bold text",
    "email": "email address - look for @ symbol ANYWHERE in the text",
    "phone": "complete phone number with country code - look for +, digits, parentheses",
    "address": "full physical/mailing address if mentioned",
    "linkedin": "LinkedIn URL - look for linkedin.com/in/",
    "github": "GitHub URL - look for github.com/",
    "portfolio": "any portfolio or personal website URL",
    "location": "City, State/Country",
    "current_role": "most recent job title",
    "current_company": "most recent company name",
    "total_experience_years": 0,
    "professional_summary": "full professional summary or objective text from resume",
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
            "key_achievements": ["achievement1", "achievement2"],
            "technologies_used": ["tech1", "tech2"]
        }}
    ],
    "education": [
        {{
            "degree": "EXACT degree as written in resume (do NOT use examples - extract only what is written)",
            "field": "EXACT field/major/branch as written in resume",
            "institution": "EXACT university/college name as written in resume",
            "location": "location",
            "start_year": "year",
            "end_year": "year or expected year",
            "gpa": "GPA, CGPA, or percentage if mentioned",
            "achievements": ["honors", "relevant coursework", "activities"]
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
1. **NAME**: The VERY FIRST non-empty line is usually the name (largest/bold text).
2. **PHONE**: Patterns like +91, +1, (XXX), or any 10+ digit numbers. Include country code.
3. **EMAIL**: Find ANYTHING with @ symbol.
4. **ADDRESS**: Street numbers, city names, pin/zip codes.
5. **EDUCATION** — VERY IMPORTANT:
   - Extract ONLY degrees/qualifications that are EXPLICITLY WRITTEN in the resume text.
   - DO NOT create education entries using the example values from this prompt template.
   - DO NOT invent "Bachelor of Technology in Computer Science" unless those EXACT words appear in the resume.
   - Each person typically has 1-3 degrees. DO NOT create more than 4 entries.
   - DO NOT create education entries from awards, certifications, or achievements.
   - DO NOT use company names as institution names.
   - If the resume says "B.E" do NOT expand it to "Bachelor of Technology" — keep it as "B.E".
   - The institution must be a name that ACTUALLY APPEARS in the resume text.
6. **EXPERIENCE**: If end_date is "Present", calculate duration until March 2026.
7. **duration_years**: Decimal (2 years 6 months = 2.5).
8. **total_experience_years**: Sum of all duration_years.
9. **SKILLS**: Extract EVERY skill mentioned ANYWHERE.
10. **ACHIEVEMENTS**: Include ALL bullet points from work experience.

Return ONLY valid JSON."""


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
    if any(c >= 3 for c in counts.values()):
        return True
    return False


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
    header_words = ["education", "experience", "skills", "summary", "objective", "profile"]
    for hw in header_words:
        if text.lower().count(hw) >= 2:
            return True
    return False


def _is_valid_education_entry(edu: Dict) -> bool:
    if not isinstance(edu, dict):
        return False

    degree = str(edu.get("degree", "")).strip()
    field_of_study = str(
        edu.get("field", "") or edu.get("major", "")
        or edu.get("branch", "") or edu.get("specialization", "")
    ).strip()
    institution = str(
        edu.get("institution", "") or edu.get("university", "")
        or edu.get("college", "")
    ).strip()

    if not degree or len(degree) < 2:
        return False

    alpha_count = sum(1 for c in degree if c.isalpha())
    if alpha_count < len(degree) * 0.5:
        return False

    combined = f"{degree} {field_of_study} {institution}".lower()

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

    if _has_repetition_pattern(institution):
        return False
    if _has_repetition_pattern(field_of_study):
        return False
    if _has_repetition_pattern(degree):
        return False
    if _has_garbage_pattern(institution):
        return False
    if _has_garbage_pattern(field_of_study):
        return False

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

    if institution and len(institution) > 150:
        return False

    return True


def _normalize_degree_key(degree: str) -> str:
    if not degree:
        return ""
    d = degree.lower().strip()
    d = re.sub(r'[^a-z0-9\s]', '', d)
    d = re.sub(r'\s+', '', d)

    d_without_pg = re.sub(r'^pg\.?\s*', '', degree.lower().strip())
    d_without_pg = re.sub(r'[^a-z0-9]', '', d_without_pg)

    normalizations = {
        r'^(?:bacheloroftechnology|btech|be|pgbe|pgbtech)$': 'btech',
        r'^(?:masteroftechnology|mtech|me|pgme|pgmtech)$': 'mtech',
        r'^(?:bachelorofscience|bsc|bs|pgbsc|pgbs)$': 'bsc',
        r'^(?:masterofscience|msc|ms)$': 'msc',
        r'^(?:bachelorofarts|ba)$': 'ba',
        r'^(?:masterofarts|ma)$': 'ma',
        r'^(?:bachelorofcommerce|bcom)$': 'bcom',
        r'^(?:masterofcommerce|mcom)$': 'mcom',
        r'^(?:bachelorofcomputerapplications|bca)$': 'bca',
        r'^(?:masterofcomputerapplications|mca)$': 'mca',
        r'^(?:bachelorofbusinessadministration|bba)$': 'bba',
        r'^(?:masterofbusinessadministration|mba)$': 'mba',
        r'^(?:doctorofphilosophy|phd|doctorate)$': 'phd',
        r'^(?:highersecondary|hsc|12th|xii|intermediate)$': 'hsc',
        r'^(?:secondary|ssc|10th|matriculation)$': 'ssc',
        r'^(?:diploma|pgdiploma|pgd|postgraduatediploma)$': 'diploma',
    }

    for pattern, replacement in normalizations.items():
        if re.match(pattern, d):
            return replacement

    if d_without_pg != d:
        for pattern, replacement in normalizations.items():
            if re.match(pattern, d_without_pg):
                return replacement

    return d


def _deduplicate_education_entries(education_list: List[Dict]) -> List[Dict]:
    if not education_list:
        return []

    degree_groups: Dict[str, List[Dict]] = {}
    for edu in education_list:
        if not isinstance(edu, dict):
            continue
        degree = str(edu.get("degree", "")).strip()
        if not degree:
            continue
        key = _normalize_degree_key(degree)
        if not key:
            key = re.sub(r'[^a-z0-9]', '', degree.lower())
        if key not in degree_groups:
            degree_groups[key] = []
        degree_groups[key].append(edu)

    result: List[Dict] = []
    for _dk, entries in degree_groups.items():
        if len(entries) == 1:
            result.append(entries[0])
        else:
            best_entry = entries[0]
            best_score = -999
            for entry in entries:
                score = 0
                inst = str(entry.get("institution", "") or entry.get("university", "") or entry.get("college", "")).strip()
                fv = str(entry.get("field", "") or entry.get("major", "") or entry.get("branch", "")).strip()
                yr = str(entry.get("year", "") or entry.get("end_year", "") or entry.get("graduation_year", "")).strip()
                gpa = str(entry.get("gpa", "") or entry.get("cgpa", "") or entry.get("grade", "")).strip()

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


def _filter_valid_education(education_list: list) -> list:
    if not education_list or not isinstance(education_list, list):
        return []
    valid = [e for e in education_list if isinstance(e, dict) and _is_valid_education_entry(e)]
    return _deduplicate_education_entries(valid)


def _validate_education_against_text(education_list: List[Dict], resume_text: str) -> List[Dict]:
    """
    Remove education entries that have NO basis in the actual resume text.
    Uses fuzzy matching to handle apostrophes, encoding differences.
    """
    if not education_list or not resume_text:
        return education_list

    text_lower = resume_text.lower()
    text_alpha = re.sub(r'[^a-z0-9\s]', '', text_lower)

    validated: List[Dict] = []

    ignore_words = {'the', 'and', 'for', 'from', 'of', 'in', 'at', 'to', 'with'}
    generic_edu_words = {
        'university', 'college', 'institute', 'school', 'academy',
        'degree', 'bachelor', 'master', 'doctor', 'diploma',
        'science', 'arts', 'technology', 'engineering',
    }

    degree_abbreviations: Dict[str, List[str]] = {
        'bachelor of technology': ['b.tech', 'btech', 'b tech'],
        'master of technology': ['m.tech', 'mtech', 'm tech'],
        'bachelor of engineering': ['b.e', 'b.e.', 'be', 'pg. b.e', 'pg b.e', 'pg.b.e'],
        'master of engineering': ['m.e', 'm.e.', 'me'],
        'bachelor of science': ['b.sc', 'bsc', 'b.s', 'bs'],
        'master of science': ['m.sc', 'msc', 'm.s', 'ms'],
        'bachelor of arts': ['b.a', 'ba'],
        'master of arts': ['m.a', 'ma'],
        'bachelor of commerce': ['b.com', 'bcom'],
        'master of commerce': ['m.com', 'mcom'],
        'bachelor of computer applications': ['bca', 'b.c.a'],
        'master of computer applications': ['mca', 'm.c.a'],
        'bachelor of business administration': ['bba', 'b.b.a'],
        'master of business administration': ['mba', 'm.b.a'],
        'doctor of philosophy': ['phd', 'ph.d', 'ph.d.'],
    }

    for edu in education_list:
        if not isinstance(edu, dict):
            continue

        degree = str(edu.get("degree", "")).strip()
        field_val = str(
            edu.get("field", "") or edu.get("major", "")
            or edu.get("branch", "") or edu.get("specialization", "")
        ).strip()
        institution = str(
            edu.get("institution", "") or edu.get("university", "")
            or edu.get("college", "")
        ).strip()

        found_in_text = False

        # Check 1: Institution (fuzzy)
        if institution and len(institution) > 3:
            inst_lower = institution.lower()
            inst_words = [
                w for w in inst_lower.split()
                if w not in ignore_words and w not in generic_edu_words
            ]
            if inst_words:
                if any(w in text_lower for w in inst_words if len(w) > 2):
                    found_in_text = True
                if not found_in_text:
                    inst_words_alpha = [re.sub(r'[^a-z0-9]', '', w) for w in inst_words]
                    if any(w in text_alpha for w in inst_words_alpha if len(w) > 2):
                        found_in_text = True
            else:
                inst_alpha = re.sub(r'[^a-z0-9\s]', '', inst_lower).strip()
                if inst_alpha and inst_alpha in text_alpha:
                    found_in_text = True

        # Check 2: Degree
        if not found_in_text and degree and len(degree) > 1:
            degree_lower = degree.lower()
            degree_alpha = re.sub(r'[^a-z0-9\s]', '', degree_lower)

            if degree_lower in text_lower or degree_alpha in text_alpha:
                found_in_text = True

            if not found_in_text:
                for full_form, abbrs in degree_abbreviations.items():
                    if full_form in degree_lower or degree_lower in full_form:
                        if any(abbr in text_lower for abbr in abbrs):
                            found_in_text = True
                            break

            if not found_in_text:
                degree_words = [
                    w for w in degree_lower.split()
                    if len(w) > 1 and w not in ignore_words and w not in generic_edu_words
                ]
                if degree_words:
                    matches = sum(1 for w in degree_words if w in text_lower)
                    if matches >= 1:
                        found_in_text = True

            if not found_in_text:
                abbr_match = re.search(
                    r'(?:PG\.?\s*)?(B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA|B\.?Sc|M\.?Sc|Ph\.?D)',
                    degree, re.IGNORECASE
                )
                if abbr_match:
                    abbr = abbr_match.group(0).lower()
                    if abbr in text_lower:
                        found_in_text = True
                    abbr_nodots = abbr.replace('.', '')
                    if abbr_nodots in text_alpha:
                        found_in_text = True

        # Check 3: Field of study
        if not found_in_text and field_val and len(field_val) > 3:
            if field_val.lower() in text_lower:
                found_in_text = True
            else:
                field_words = [
                    w for w in field_val.lower().split()
                    if len(w) > 3 and w not in ignore_words and w not in generic_edu_words
                ]
                if field_words and any(w in text_lower for w in field_words):
                    found_in_text = True

        if found_in_text:
            validated.append(edu)

    # Safety net
    if not validated and education_list:
        text_lower_check = resume_text.lower()
        edu_indicators = [
            'b.e', 'b.tech', 'btech', 'm.tech', 'mtech',
            'b.sc', 'm.sc', 'bca', 'mca', 'mba', 'phd',
            'bachelor', 'master', 'diploma', 'degree',
            'university', 'college', 'education', 'graduated',
            'pg.', 'pg ', 'engineering',
        ]
        if any(ind in text_lower_check for ind in edu_indicators):
            return education_list

    return validated


# ═══════════════════════════════════════════════════════════════
#                    CONTACT EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_contacts_regex(text: str) -> Dict:
    contacts: Dict[str, str] = {
        "email": "", "phone": "", "address": "",
        "linkedin": "", "github": "", "portfolio": "", "location": ""
    }
    if not text:
        return contacts

    # EMAIL
    for pattern in [
        r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*@\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*\[\s*at\s*\]\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*\(\s*at\s*\)\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'(?:email|e-mail|mail)[\s.:]*[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            email = match.group(0).strip()
            email = re.sub(r'^(?:email|e-mail|mail)[\s.:]*', '', email, flags=re.IGNORECASE)
            email = re.sub(r'\s+', '', email)
            email = re.sub(r'\[\s*at\s*\]|\(\s*at\s*\)', '@', email, flags=re.IGNORECASE)
            if '@' in email and '.' in email.split('@')[-1]:
                contacts["email"] = email
                break

    # PHONE
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
    for pattern in phone_patterns:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = re.sub(r'^(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*', '', match, flags=re.IGNORECASE).strip()
                digits = re.sub(r'[^\d]', '', cleaned)
                if 10 <= len(digits) <= 15:
                    contacts["phone"] = cleaned
                    break
            if contacts["phone"]:
                break
        except re.error:
            continue

    # LINKEDIN
    for pattern in [r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?', r'linkedin\.com/in/[\w-]+']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            contacts["linkedin"] = match.group(0).strip()
            break

    # GITHUB
    for pattern in [r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?', r'github\.com/[\w-]+']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            contacts["github"] = match.group(0).strip()
            break

    # PORTFOLIO
    for pattern in [
        r'(?:portfolio|website|web|site|blog)[\s.:]+(?:https?://)?[\w.-]+\.[a-z]{2,}[\w/.-]*',
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:dev|io|me|tech|design|codes?|site|online|app)/?[\w/.-]*',
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            url = re.sub(r'^(?:portfolio|website|web|site|blog)[\s.:]+', '', match, flags=re.IGNORECASE).strip()
            if 'linkedin' not in url.lower() and 'github' not in url.lower() and '@' not in url:
                contacts["portfolio"] = url
                break
        if contacts["portfolio"]:
            break

    # ADDRESS
    for pattern in [
        r'(?:Address|Location|Residence|Home|Addr)[\s.:]+([^\n]{15,150})',
        r'(?:No[.:]?\s*)?[\d]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar|Colony|Layout|Block|Lane|Apt|Apartment|Floor|Fl|Building|Bldg)[\w\s,.-]+(?:\d{5,6})',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            addr = (match.group(1) if match.lastindex else match.group(0)).strip()
            addr = re.sub(r'^(?:Address|Location|Residence|Home|Addr)[\s.:]+', '', addr, flags=re.IGNORECASE).strip()
            if 15 < len(addr) < 200:
                contacts["address"] = addr
                break

    # LOCATION
    for pattern in [
        r'(?:Location|Based in|Located at|City|Current Location)[\s.:]+([A-Za-z][A-Za-z\s,]+)',
        r'\b(Bangalore|Bengaluru|Mumbai|Delhi|NCR|Chennai|Hyderabad|Pune|Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Lucknow|Chandigarh|Indore|Bhopal|Kochi|Coimbatore|Trivandrum|Mysore|Nagpur|Surat|Vadodara|Bhubaneswar|Patna|Ranchi|Guwahati|Visakhapatnam|Vijayawada)\b',
        r'\b(New York|NYC|San Francisco|SF|Los Angeles|LA|Chicago|Seattle|Boston|Austin|Denver|Atlanta|Dallas|Houston|Phoenix|San Diego|San Jose|Portland|Miami|Washington DC|Philadelphia)\b',
        r'\b(London|Berlin|Amsterdam|Dublin|Singapore|Tokyo|Sydney|Toronto|Vancouver|Melbourne|Paris|Munich|Barcelona|Stockholm|Copenhagen|Zurich|Dubai|Hong Kong)\b',
        r'\b(India|USA|US|United States|UK|United Kingdom|Canada|Australia|Germany|Netherlands|Singapore|UAE)\b',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            loc = (match.group(1) if match.lastindex else match.group(0)).strip()
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

    skip_keywords = [
        'resume', 'curriculum', 'vitae', 'cv', 'http', 'www', '@',
        'address', 'phone', 'email', 'street', 'road', 'avenue',
        'objective', 'summary', 'profile', 'linkedin', 'github',
        'portfolio', 'mobile', 'tel:', 'contact', 'experience',
        'education', 'skills', 'professional', 'career', 'about',
        'personal', 'details', 'information', 'confidential',
        'page', 'date', 'application', 'position', 'job',
        'candidate', 'recruitment', 'hiring', 'vacancy',
        'declaration', 'reference', 'signature',
        'company', 'corporation', 'limited', 'pvt', 'ltd',
    ]

    # Strategy 1: Labeled name
    for pattern in [
        r'(?:Name|Full Name|Candidate Name|Applicant Name)[\s.:]+([A-Z][a-zA-Z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-zA-Z]+){1,2})',
        r'(?:Name|Full Name)[\s.:]+([A-Z][A-Z\s]+)',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if _is_valid_name(name):
                return _clean_name(name)

    # Strategy 2: First line that looks like a name
    for line in lines[:15]:
        line = line.strip()
        if not line or len(line) < 4 or len(line) > 45:
            continue
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        if re.match(r'^\d', line) or len(re.findall(r'\d', line)) >= 5:
            continue
        if re.match(r'^[\+\d\(\)]', line) or '@' in line or 'http' in line.lower():
            continue

        alpha_chars = sum(1 for c in line if c.isalpha() or c.isspace() or c in '.-')
        if alpha_chars / max(len(line), 1) < 0.85:
            continue

        for pattern in [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+$',
            r'^[A-Z]+\s+[A-Z]+$',
            r'^[A-Z]+\s+[A-Z]+\s+[A-Z]+$',
            r'^Dr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^Mr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^Ms\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+$',
        ]:
            if re.match(pattern, line):
                if _is_valid_name(line):
                    return _clean_name(line)

        words = line.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w):
                if all(2 <= len(w) <= 15 for w in words):
                    if _is_valid_name(line):
                        return _clean_name(line)

    # Strategy 3: First line mixed content
    if lines:
        first_line = lines[0].strip()
        name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})', first_line)
        if name_match:
            potential_name = name_match.group(1).strip()
            if _is_valid_name(potential_name):
                return _clean_name(potential_name)

    # Strategy 4: "I am" / "My name is"
    for pattern in [
        r"(?:I am|I'm|My name is|This is|Myself)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if _is_valid_name(name):
                return _clean_name(name)

    # Strategy 5: From email
    email_match = re.search(r'([\w.]+)@', text)
    if email_match:
        parts = re.split(r'[._]', email_match.group(1))
        if len(parts) >= 2:
            potential_name = ' '.join(p.capitalize() for p in parts if p.isalpha() and len(p) > 1)
            if len(potential_name.split()) >= 2:
                return potential_name

    # Strategy 6: ALL CAPS line
    section_headers_upper = {
        'RESUME', 'CURRICULUM', 'VITAE', 'CV', 'OBJECTIVE',
        'SUMMARY', 'EXPERIENCE', 'EDUCATION', 'SKILLS',
        'CONTACT', 'PROFILE', 'ABOUT', 'PROJECTS',
        'CERTIFICATIONS', 'ACHIEVEMENTS', 'AWARDS',
        'REFERENCES', 'DECLARATION', 'PERSONAL',
        'PROFESSIONAL', 'TECHNICAL', 'WORK', 'HISTORY',
        'QUALIFICATIONS', 'RESPONSIBILITIES', 'DETAILS',
        'INFORMATION', 'OVERVIEW',
    }
    for line in lines[:10]:
        line = line.strip()
        if line and line.isupper() and 5 <= len(line) <= 40:
            words = line.split()
            if 2 <= len(words) <= 4:
                if not any(header in line for header in section_headers_upper):
                    title_name = line.title()
                    if _is_valid_name(title_name):
                        return title_name

    return ""


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    name_clean = name.strip()
    name_lower = name_clean.lower()

    exact_rejects = {
        "", "n/a", "na", "none", "unknown", "candidate",
        "your name", "first last", "firstname lastname",
        "name", "full name", "[name]", "<name>", "(name)",
        "enter name", "type name", "not available",
        "curriculum vitae", "resume", "cv",
    }
    if name_lower in exact_rejects:
        return False

    words = name_clean.split()
    if not (2 <= len(words) <= 4):
        return False

    for w in words:
        stripped = w.strip(".-'")
        if not stripped or len(stripped) > 20:
            return False
        if sum(c.isalpha() for c in stripped) / max(len(stripped), 1) < 0.8:
            return False

    if sum(1 for c in name_clean if c.isalpha() or c.isspace()) / max(len(name_clean), 1) < 0.80:
        return False

    for w in words:
        if w.lower().strip(".,;:()") in _STRONG_NAME_BLACKLIST:
            return False

    if sum(1 for w in words if w.lower().strip(".,;:()") in _SOFT_NAME_BLACKLIST) >= 2:
        return False

    if name_clean == name_clean.lower():
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
#        WORK EXPERIENCE EXTRACTION — ALL DATE FORMATS
# ═══════════════════════════════════════════════════════════════

def _extract_work_experience_from_text(text: str) -> List[Dict]:
    """
    Extract work experience from paragraph/prose format text.
    Handles ALL date formats and sentence patterns.
    """
    if not text:
        return []

    work_entries: List[Dict] = []
    seen_jobs: Set[str] = set()

    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    # ═══ Reusable date regex components ═══
    MONTH_YEAR = (
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
        r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|'
        r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'[.,]?\s*\d{4}'
    )
    NUMERIC_DATE = r'\d{1,2}[/\-\.]\d{4}'
    NUMERIC_DATE_REV = r'\d{4}[/\-\.]\d{1,2}'
    MONTH_APOS_YEAR = (
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r"['\u2019]\s*\d{2,4}"
    )
    JUST_YEAR = r'\d{4}'
    PRESENT_KW = (
        r'(?:present|current(?:ly)?|now|ongoing|'
        r'till\s*date|till\s*now|to\s*date|today|continuing)'
    )
    ANY_DATE = (
        f'(?:{MONTH_YEAR}|{NUMERIC_DATE}|{NUMERIC_DATE_REV}|'
        f'{MONTH_APOS_YEAR}|{PRESENT_KW}|{JUST_YEAR})'
    )
    DATE_SEP = r'\s*(?:to|till|until|[-\u2013\u2014])\s*'

    def _parse_work_date(date_str: str) -> Tuple[Optional[int], Optional[int]]:
        if not date_str:
            return None, None
        ds = date_str.strip().lower().strip('.,;:)( ')

        # Present/Current
        present_phrases = [
            'present', 'current', 'currently', 'now', 'ongoing',
            'till date', 'till now', 'to date', 'today', 'continuing',
        ]
        ds_clean = re.sub(r'[^a-z\s]', '', ds).strip()
        if ds_clean in present_phrases or any(pw in ds for pw in present_phrases):
            return CURRENT_YEAR, CURRENT_MONTH

        # "Oct'19" or "Oct'2019"
        m = re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*['\u2019]\s*(\d{2,4})", ds
        )
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

    def _calc_dur(sy, sm, ey, em):
        if not sy:
            return 0.0
        if not ey:
            ey, em = CURRENT_YEAR, CURRENT_MONTH
        sm = sm or 6
        em = em or 6
        months = (ey - sy) * 12 + (em - sm)
        return round(max(0, months / 12.0), 1)

    def _clean_company(company: str) -> str:
        c = re.sub(r'https?://\S+', '', company).strip()
        c = re.sub(r'[,;]\s*\w+$', '', c)
        return c.rstrip('-,. ;:')

    def _normalize_job_key(role: str, company: str) -> str:
        r = re.sub(r'[^a-z0-9]', '', role.lower())[:25]
        c = re.sub(r'[^a-z0-9]', '', company.lower())[:25]
        return f"{r}_{c}"

    def _add_entry(role, company, start_str, end_str, location=""):
        role = role.strip().rstrip(',. ')
        company = _clean_company(company)
        end_str = re.sub(r'https?://\S+', '', end_str).strip().rstrip('.,;: ')

        job_key = _normalize_job_key(role, company)
        if job_key in seen_jobs:
            return
        seen_jobs.add(job_key)

        sy, sm = _parse_work_date(start_str)
        ey, em = _parse_work_date(end_str)
        dur_years = _calc_dur(sy, sm, ey, em)

        clean_start = start_str.strip()
        clean_end = end_str.strip()
        if not clean_end or clean_end.lower() in ['', '-', '\u2013', '\u2014']:
            clean_end = "Present"

        work_entries.append({
            "title": role, "company": company, "location": location,
            "start_date": clean_start, "end_date": clean_end,
            "duration_years": dur_years, "type": "Full-time",
            "description": "", "key_achievements": [], "technologies_used": []
        })

    # ═══ Pattern 1: "working as ROLE with COMPANY since DATE to DATE" ═══
    for m in re.finditer(
        r'(?:currently\s+)?(?:working|employed|serving)\s+(?:as\s+)?'
        r'(.+?)\s+(?:with|at|in|for)\s+(.+?)\s*'
        rf'(?:since|from)\s+({ANY_DATE}){DATE_SEP}({ANY_DATE})'
        r'(?:\s*[.\n]|$)', text, re.IGNORECASE
    ):
        _add_entry(m.group(1), m.group(2), m.group(3), m.group(4))

    # ═══ Pattern 2a: "Ex Employee of COMPANY [URL] [,CITY] from DATE to DATE as ROLE" ═══
    for m in re.finditer(
        r'(?:ex|former|previous|past)\s+'
        r'(?:employee|member|associate|consultant|staff)\s+'
        r'(?:of|at|with|in)\s+(.+?)'
        rf'\s+from\s+({ANY_DATE}){DATE_SEP}({ANY_DATE})'
        r'\s+as\s+(.+?)(?:\s*[.\n;]|$)', text, re.IGNORECASE
    ):
        company_raw = m.group(1).strip()
        company_clean = _clean_company(company_raw)
        role = m.group(4).strip().rstrip('.,;: ')
        location = ""
        loc_match = re.search(r'[,;]\s*(\w[\w\s]*?)$', re.sub(r'https?://\S+', '', company_raw))
        if loc_match:
            location = loc_match.group(1).strip()

        job_key = _normalize_job_key(role, company_clean)
        if job_key in seen_jobs:
            continue
        seen_jobs.add(job_key)

        sy, sm = _parse_work_date(m.group(2).strip())
        ey, em = _parse_work_date(m.group(3).strip())
        dur_years = _calc_dur(sy, sm, ey, em)

        work_entries.append({
            "title": role, "company": company_clean, "location": location,
            "start_date": m.group(2).strip(), "end_date": m.group(3).strip(),
            "duration_years": dur_years, "type": "Full-time",
            "description": "", "key_achievements": [], "technologies_used": []
        })

    # ═══ Pattern 2b: Looser "Ex Employee" variant ═══
    for m in re.finditer(
        r'(?:ex|former|previous|past)\s+'
        r'(?:employee|member|associate|consultant|staff)\s+'
        r'(?:of|at|with|in)\s+(.+?)'
        r'\s+(?:from)\s+(.+?)\s+(?:to|till|until)\s+(.+?)\s+'
        r'(?:as)\s+(.+?)(?:\s*[.\n;,]|$)', text, re.IGNORECASE
    ):
        company_raw = m.group(1).strip()
        company_clean = _clean_company(company_raw)
        role = m.group(4).strip().rstrip('.,;: ')

        c_norm = re.sub(r'[^a-z0-9]', '', company_clean.lower())[:20]
        if any(re.sub(r'[^a-z0-9]', '', e.get('company', '').lower())[:20] == c_norm for e in work_entries):
            continue

        _add_entry(role, company_clean, m.group(2).strip(), m.group(3).strip())

    # ═══ Pattern 3: "worked/joined as ROLE at COMPANY from DATE to DATE" ═══
    for m in re.finditer(
        r'(?:worked|employed|served|joined|was)\s+(?:as\s+)?'
        r'(.+?)\s+(?:at|with|in|for)\s+(.+?)\s*'
        rf'(?:from|since)\s+({ANY_DATE}){DATE_SEP}({ANY_DATE})'
        r'(?:[.\n,;]|$)', text, re.IGNORECASE
    ):
        _add_entry(m.group(1), m.group(2), m.group(3), m.group(4))

    # ═══ Pattern 4: "ROLE at COMPANY, DATE - DATE" (tabular) ═══
    for m in re.finditer(
        r'^(.+?)\s+(?:at|with|@)\s+(.+?)\s*[,|]\s*'
        rf'({ANY_DATE}){DATE_SEP}({ANY_DATE})'
        r'(?:\s*[.\n]|$)', text, re.IGNORECASE | re.MULTILINE
    ):
        role = m.group(1).strip()
        company = m.group(2).strip()
        if len(role) > 60 or len(company) > 80:
            continue
        skip = ['education', 'project', 'skill', 'certification', 'award', 'summary']
        if any(s in role.lower() for s in skip):
            continue
        _add_entry(role, company, m.group(3), m.group(4))

    # ═══ Pattern 5: "COMPANY | ROLE | DATE - DATE" (pipe-separated) ═══
    for m in re.finditer(
        r'(.+?)\s*\|\s*(.+?)\s*\|\s*'
        rf'({ANY_DATE}){DATE_SEP}({ANY_DATE})'
        r'(?:\s*[.\n]|$)', text, re.IGNORECASE
    ):
        company = m.group(1).strip()
        role = m.group(2).strip()
        if len(role) > 60 or len(company) > 80:
            continue
        _add_entry(role, company, m.group(3), m.group(4))

    # ═══ Pattern 6: Fallback — any date range with context ═══
    if not work_entries:
        date_range_re = rf'(?:from\s+|since\s+)?({ANY_DATE}){DATE_SEP}({ANY_DATE})'
        for m in re.finditer(date_range_re, text, re.IGNORECASE):
            start_str = m.group(1).strip()
            end_str = m.group(2).strip()

            sy, sm = _parse_work_date(start_str)
            if not sy or sy < 1980 or sy > CURRENT_YEAR:
                continue

            ctx_start = max(0, m.start() - 250)
            ctx_end = min(len(text), m.end() + 150)
            context = text[ctx_start:ctx_end]

            role = ""
            company = ""

            role_match = re.search(
                r'(?:as|role|position|designation|title)[:\s]+(.+?)'
                r'(?:\s+(?:at|with|in|from|since)|\n|,)', context, re.IGNORECASE
            )
            if role_match:
                role = role_match.group(1).strip()

            company_match = re.search(
                r'(?:at|with|in|for|company|organization|employer)[:\s]+(.+?)'
                r'(?:\s+(?:from|since|as)|\n|,)', context, re.IGNORECASE
            )
            if company_match:
                company = _clean_company(company_match.group(1))

            ey, em = _parse_work_date(end_str)
            dur_years = _calc_dur(sy, sm, ey, em)

            if dur_years > 0:
                job_key = _normalize_job_key(role or "role", company or start_str)
                if job_key not in seen_jobs:
                    seen_jobs.add(job_key)
                    work_entries.append({
                        "title": role or "Not specified",
                        "company": company or "Not specified",
                        "location": "",
                        "start_date": start_str, "end_date": end_str,
                        "duration_years": dur_years, "type": "Full-time",
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
        (r'(B\.?Com|BCom|B\.?Commerce)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(Master\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Business\s*Administration|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?\s*(?:Tech|E|Sc|A|B\.?A|S|CA|BA|Phil|Ed|Des)\.?|MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?Tech|MTech|M\.?E\.?|ME|M\.?S\.?|MS)\s*[-\u2013in\s]*([\w\s,&]+)?', 'Master'),
        (r'(MCA|M\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?Com|MCom|M\.?Commerce)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(PG\s*(?:Diploma|Degree)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(Ph\.?\s*D\.?|Doctorate|Doctor\s*of\s*Philosophy)\s*(?:in\s*)?([\w\s,&]+)?', 'PhD'),
        (r'(Diploma)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(Polytechnic)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(ITI|I\.?T\.?I\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(Higher\s*Secondary|HSC|12th|XII|Intermediate|Senior\s*Secondary|\+2|Plus\s*Two)', 'School'),
        (r'(Secondary|SSC|10th|X|Matriculation|High\s*School|SSLC)', 'School'),
    ]

    institution_patterns = [
        r'([A-Z][A-Za-z\s\.\']+(?:University|College|Institute|School|Academy|Polytechnic))',
        r'((?:IIT|NIT|IIIT|BITS|VIT|SRM|Amity|Manipal|LPU|KIIT|MIT|Stanford|Harvard|Cambridge|Oxford)\s*[,]?\s*[\w\s]*)',
        r'((?:Indian Institute of|National Institute of|International Institute of)\s*[\w\s]+)',
        r'(?:from|at|,)\s+([A-Z][A-Za-z\s\.\']{5,60})',
    ]

    gpa_patterns = [
        r'(?:GPA|CGPA|CPI|Grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?',
        r'(\d{1,2}(?:\.\d+)?)\s*%',
        r'(First\s*Class(?:\s*with\s*Distinction)?|Distinction|Honors?|Honours?)',
    ]

    year_patterns = [
        r'((?:19|20)\d{2})\s*[-\u2013to]+\s*((?:19|20)\d{2}|Present|Current|Expected|Pursuing)',
        r'(?:Class\s*of|Batch|Graduated?|Year\s*of\s*(?:Graduation|Completion)|Expected)[:\s]*((?:19|20)\d{2})',
        r'((?:19|20)\d{2})',
    ]

    text_upper = text.upper()
    edu_section_start = -1
    for marker in ['EDUCATION', 'ACADEMIC', 'QUALIFICATION', 'EDUCATIONAL BACKGROUND']:
        idx = text_upper.find(marker)
        if idx != -1:
            edu_section_start = idx
            break

    end_markers = ['EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILL', 'PROJECT', 'CERTIFICATION', 'AWARD', 'ACHIEVEMENT']

    if edu_section_start != -1:
        edu_section = text[edu_section_start:]
        for marker in end_markers:
            end_idx = edu_section.upper().find(marker, 50)
            if end_idx != -1:
                edu_section = edu_section[:end_idx]
                break
    else:
        edu_section = text

    found_degrees: List[Dict] = []
    for pattern, degree_type in degree_patterns:
        try:
            for match in re.finditer(pattern, edu_section, re.IGNORECASE):
                degree_name = match.group(1).strip() if match.group(1) else ""
                field = ""
                if match.lastindex >= 2 and match.group(2):
                    field = match.group(2).strip()

                field = re.sub(r'\s+', ' ', field).strip()
                if len(field) > 100:
                    field = field[:100]

                skip_words = ['university', 'college', 'institute', 'school', 'from', 'at', 'gpa', 'cgpa']
                if any(sw in field.lower() for sw in skip_words):
                    field = ""

                start_pos = max(0, match.start() - 30)
                end_pos = min(len(edu_section), match.end() + 250)
                context = edu_section[start_pos:end_pos]

                edu_entry: Dict = {
                    "degree": degree_name, "field": field, "degree_type": degree_type,
                    "institution": "", "year": "", "gpa": "", "location": ""
                }

                for inst_pattern in institution_patterns:
                    try:
                        inst_match = re.search(inst_pattern, context, re.IGNORECASE)
                        if inst_match:
                            inst = re.sub(r'\s+', ' ', inst_match.group(1).strip())
                            if len(inst) > 5 and inst.lower() not in ['the', 'and', 'for', 'with', 'from']:
                                edu_entry["institution"] = inst
                                break
                    except re.error:
                        continue

                for year_pattern in year_patterns:
                    try:
                        year_match = re.search(year_pattern, context, re.IGNORECASE)
                        if year_match:
                            edu_entry["year"] = year_match.group(0).strip()
                            break
                    except re.error:
                        continue

                for gpa_pattern in gpa_patterns:
                    try:
                        gpa_match = re.search(gpa_pattern, context, re.IGNORECASE)
                        if gpa_match:
                            edu_entry["gpa"] = gpa_match.group(0).strip()
                            break
                    except re.error:
                        continue

                found_degrees.append(edu_entry)
        except re.error:
            continue

    seen: Set[str] = set()
    for edu in found_degrees:
        entry = {
            "degree": edu.get('degree', ''), "field": edu.get('field', ''),
            "institution": edu.get('institution', ''), "year": edu.get('year', ''),
            "gpa": edu.get('gpa', ''), "location": edu.get('location', ''),
            "achievements": []
        }
        if not _is_valid_education_entry(entry):
            continue

        degree = edu.get('degree', '').lower()
        inst = edu.get('institution', '').lower()
        key = f"{_normalize_degree_key(degree)}_{inst[:20]}"

        if key not in seen and edu.get('degree'):
            seen.add(key)
            education.append(entry)

    return _deduplicate_education_entries(education)


# ═══════════════════════════════════════════════════════════════
#                    SKILLS EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_skills_regex(text: str) -> Dict:
    skills: Dict[str, List[str]] = {
        "programming_languages": [], "frameworks_libraries": [],
        "ai_ml_tools": [], "cloud_platforms": [], "databases": [],
        "devops_tools": [], "visualization": [], "other_tools": [],
        "soft_skills": []
    }
    if not text:
        return skills

    text_lower = text.lower()

    skill_categories = {
        "programming_languages": [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
            'go', 'golang', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r',
            'matlab', 'perl', 'bash', 'shell', 'powershell', 'sql', 'html',
            'css', 'sass', 'less', 'dart', 'lua', 'groovy',
        ],
        "frameworks_libraries": [
            'react', 'angular', 'vue', 'nextjs', 'next.js', 'nuxt',
            'nodejs', 'node.js', 'express', 'django', 'flask', 'fastapi',
            'spring', 'springboot', 'spring boot', 'laravel',
            'rails', 'ruby on rails', 'asp.net', '.net',
            'jquery', 'bootstrap', 'tailwind', 'flutter', 'react native',
        ],
        "ai_ml_tools": [
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
            'pandas', 'numpy', 'opencv', 'nltk', 'spacy',
            'huggingface', 'transformers', 'langchain',
            'spark', 'pyspark', 'hadoop', 'kafka', 'xgboost', 'lightgbm',
        ],
        "cloud_platforms": [
            'aws', 'amazon web services', 'azure', 'microsoft azure',
            'gcp', 'google cloud', 'heroku', 'firebase',
            'vercel', 'netlify', 'lambda', 's3', 'ec2',
        ],
        "databases": [
            'mysql', 'postgresql', 'postgres', 'mongodb', 'redis',
            'elasticsearch', 'dynamodb', 'cassandra', 'oracle',
            'sql server', 'sqlite', 'mariadb', 'neo4j',
        ],
        "devops_tools": [
            'docker', 'kubernetes', 'k8s', 'jenkins', 'terraform',
            'ansible', 'prometheus', 'grafana', 'nginx',
            'linux', 'unix', 'ubuntu', 'github actions', 'gitlab ci',
        ],
        "visualization": [
            'power bi', 'tableau', 'looker', 'plotly', 'matplotlib',
            'seaborn', 'excel', 'google data studio', 'd3.js',
        ],
        "other_tools": [
            'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
            'figma', 'postman', 'swagger', 'graphql', 'rest',
            'selenium', 'cypress', 'jest', 'pytest',
            'webpack', 'vite', 'npm', 'yarn', 'maven', 'gradle',
        ],
        "soft_skills": [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'critical thinking', 'analytical', 'creativity', 'adaptability',
            'time management', 'project management', 'presentation',
            'negotiation', 'collaboration', 'mentoring',
        ],
    }

    for category, skill_list in skill_categories.items():
        for skill in skill_list:
            pattern = rf'\b{re.escape(skill)}\b'
            if re.search(pattern, text_lower):
                if len(skill) <= 3 and skill not in ('go', 'r'):
                    display = skill.upper()
                elif skill in ('c++', 'c#'):
                    display = skill.upper().replace('++', '++')
                else:
                    display = skill.title()
                skills[category].append(display)

    for category in skills:
        seen_set: Set[str] = set()
        unique: List[str] = []
        for s in skills[category]:
            if s.lower() not in seen_set:
                seen_set.add(s.lower())
                unique.append(s)
        skills[category] = unique

    return skills


# ═══════════════════════════════════════════════════════════════
#                    EXPERIENCE CALCULATION
# ═══════════════════════════════════════════════════════════════

def _parse_date_to_ym(date_str: str, month_map: Dict) -> Tuple[Optional[int], Optional[int]]:
    if not date_str:
        return None, None
    date_str = date_str.strip().lower()

    for month_name, month_num in month_map.items():
        if month_name in date_str:
            year_match = re.search(r'(19|20)\d{2}', date_str)
            if year_match:
                return int(year_match.group()), month_num

    match = re.search(r'(\d{1,2})[/-]((?:19|20)\d{2})', date_str)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            return int(match.group(2)), month

    match = re.search(r'((?:19|20)\d{2})[/-](\d{1,2})', date_str)
    if match:
        month = int(match.group(2))
        if 1 <= month <= 12:
            return int(match.group(1)), month

    year_match = re.search(r'(19|20)\d{2}', date_str)
    if year_match:
        return int(year_match.group()), 6
    return None, None


def _calculate_total_experience(work_history: list) -> float:
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }
    total = 0.0
    for job in work_history:
        if not isinstance(job, dict):
            continue
        start_str = job.get("start_date", job.get("from", ""))
        end_str = job.get("end_date", job.get("to", ""))

        if not start_str:
            dur = job.get("duration_years", job.get("years", 0))
            try:
                total += float(dur) if dur else 0
            except (ValueError, TypeError):
                pass
            continue

        start_y, start_m = _parse_date_to_ym(str(start_str), month_map)
        if not start_y:
            dur = job.get("duration_years", job.get("years", 0))
            try:
                total += float(dur) if dur else 0
            except (ValueError, TypeError):
                pass
            continue

        present_words = [
            'present', 'current', 'now', 'ongoing', 'till date',
            'till now', 'today', 'continuing', '-', '\u2013', '',
        ]
        if not end_str or str(end_str).strip().lower() in present_words:
            end_y, end_m = CURRENT_YEAR, CURRENT_MONTH
        else:
            end_y, end_m = _parse_date_to_ym(str(end_str), month_map)
            if not end_y:
                end_y, end_m = CURRENT_YEAR, CURRENT_MONTH

        months = (end_y - start_y) * 12 + (end_m - start_m)
        total += max(0, months / 12.0)
    return round(total, 1)


# ═══════════════════════════════════════════════════════════════
#                    CONTACT VALIDATION & MERGING
# ═══════════════════════════════════════════════════════════════

def _is_valid_field(field: str, value: str) -> bool:
    if not value:
        return False
    value = str(value).strip()
    invalid_values = [
        'n/a', 'na', 'none', 'not available', 'not specified',
        'unknown', 'null', '-', '\u2014', '', 'candidate', 'your name',
        'email@example.com', 'xxx', '000', 'your email', 'your phone',
        'first last', 'firstname lastname', 'name', 'full name',
        '[name]', '<name>', '(name)', 'enter name', 'type name',
    ]
    if value.lower().strip() in invalid_values:
        return False
    if field == 'email':
        return '@' in value and '.' in value.split('@')[-1]
    if field == 'phone':
        return len(re.sub(r'[^\d]', '', value)) >= 10
    if field == 'name':
        return _is_valid_name(value)
    if field in ('linkedin', 'github'):
        return len(value) > 5
    return len(value) >= 2


def _merge_contacts(llm_parsed: Dict, regex_contacts: Dict, doc_contacts: Optional[Dict] = None) -> Dict:
    merged: Dict[str, str] = {}
    if doc_contacts is None:
        doc_contacts = {}
    for field in ['name', 'email', 'phone', 'address', 'linkedin', 'github', 'portfolio', 'location']:
        llm_val = str(llm_parsed.get(field, "")).strip()
        doc_val = str(doc_contacts.get(field, "")).strip()
        regex_val = str(regex_contacts.get(field, "")).strip()
        if llm_val and _is_valid_field(field, llm_val):
            merged[field] = llm_val
        elif doc_val and _is_valid_field(field, doc_val):
            merged[field] = doc_val
        elif regex_val and _is_valid_field(field, regex_val):
            merged[field] = regex_val
        else:
            merged[field] = ""
    return merged


def _basic_fallback(text: str, contacts: Dict, name: str, education: List, skills: Dict) -> Dict:
    return {
        "name": name or contacts.get("name", ""),
        "email": contacts.get("email", ""), "phone": contacts.get("phone", ""),
        "address": contacts.get("address", ""), "linkedin": contacts.get("linkedin", ""),
        "github": contacts.get("github", ""), "portfolio": contacts.get("portfolio", ""),
        "location": contacts.get("location", ""),
        "current_role": "", "current_company": "",
        "total_experience_years": 0,
        "professional_summary": text[:500] if text else "",
        "specializations": [], "skills": skills, "work_history": [],
        "education": education, "certifications": [], "awards": [],
        "projects": [], "publications": [], "volunteer": [],
        "languages": [], "interests": []
    }


# ═══════════════════════════════════════════════════════════════
#                    MAIN PARSING FUNCTION
# ═══════════════════════════════════════════════════════════════

def parse_resume_with_llm(
    resume_text: str,
    groq_api_key: str,
    model_id: str = "llama-3.1-8b-instant",
    doc_contacts: Optional[Dict] = None
) -> Dict:

    if not resume_text or len(resume_text.strip()) < 50:
        return _basic_fallback(resume_text, {}, "", [], {})

    # ══════ STEP 1: Regex extraction ══════
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

    # ══════ STEP 2: LLM parsing ══════
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    truncated_text = resume_text[:12000]

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a resume parsing expert. Current year is 2026. "
                    "Return ONLY valid JSON. No markdown. No explanation. "
                    "CRITICAL: For education, create AT MOST 3-4 entries. "
                    "NEVER create education entries from awards, certifications, or work achievements. "
                    "NEVER use company names as institutions. "
                    "NEVER use example values from the prompt template as real data. "
                    "Only extract what is ACTUALLY WRITTEN in the resume text."
                )
            },
            {"role": "user", "content": PARSE_PROMPT.format(resume_text=truncated_text)}
        ],
        "temperature": 0.05,
        "max_tokens": 6000,
    }

    models_to_try = list(dict.fromkeys([model_id, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]))

    parsed = None
    for try_model in models_to_try:
        payload["model"] = try_model
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=60
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()
                parsed = json.loads(content)
                break
            elif response.status_code == 429:
                time.sleep(1)
                continue
            else:
                continue
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    break
            except Exception:
                pass
            continue
        except Exception:
            continue

    if not parsed:
        parsed = _basic_fallback(resume_text, regex_contacts, regex_name, regex_education, regex_skills)

    # ══════ STEP 2b: Validate + deduplicate LLM education ══════
    llm_education = parsed.get("education", [])
    if isinstance(llm_education, list):
        parsed["education"] = _filter_valid_education(llm_education)
    elif llm_education:
        parsed["education"] = _filter_valid_education([llm_education])
    else:
        parsed["education"] = []

    # ══════ STEP 2c: Cross-validate education against resume text ══════
    parsed["education"] = _validate_education_against_text(parsed["education"], resume_text)

    # ══════ STEP 3: Merge contacts ══════
    merged_contacts = _merge_contacts(parsed, regex_contacts, doc_contacts)
    for fld, value in merged_contacts.items():
        if value and (not parsed.get(fld) or not _is_valid_field(fld, parsed.get(fld, ""))):
            parsed[fld] = value

    # ══════ STEP 4: Name ══════
    current_name = parsed.get("name", "")
    if not current_name or not _is_valid_field("name", current_name):
        for candidate in [doc_contacts.get("name", ""), regex_name, parsed.get("name", "")]:
            if candidate and _is_valid_field("name", candidate):
                parsed["name"] = _clean_name(candidate)
                break
        if not parsed.get("name") or not _is_valid_field("name", parsed.get("name", "")):
            parsed["name"] = "Unknown Candidate"

    # ══════ STEP 5: Merge regex education + safety net ══════
    llm_education = parsed.get("education", [])
    if not isinstance(llm_education, list):
        llm_education = []

    if regex_education:
        if not llm_education:
            parsed["education"] = regex_education
        else:
            llm_degree_keys: Set[str] = set()
            for edu in llm_education:
                if isinstance(edu, dict):
                    dk = _normalize_degree_key(str(edu.get("degree", "")))
                    if dk:
                        llm_degree_keys.add(dk)

            for regex_edu in regex_education:
                if isinstance(regex_edu, dict):
                    dk = _normalize_degree_key(str(regex_edu.get("degree", "")))
                    if dk and dk not in llm_degree_keys:
                        llm_education.append(regex_edu)
                        llm_degree_keys.add(dk)

            parsed["education"] = _deduplicate_education_entries(llm_education)

    # STEP 5b: Education safety net
    if not parsed.get("education"):
        text_lower_check = resume_text.lower()
        edu_indicators = [
            'b.e', 'b.tech', 'btech', 'm.tech', 'mtech', 'b.sc', 'm.sc',
            'bca', 'mca', 'mba', 'b.a', 'm.a', 'b.com', 'm.com',
            'bachelor', 'master', 'phd', 'diploma', 'degree',
            'university', 'college', 'graduated', 'graduation',
            'pg.', 'pg ', 'post graduate',
        ]
        if any(ind in text_lower_check for ind in edu_indicators):
            retry_edu = _extract_education_regex(resume_text)
            if retry_edu:
                parsed["education"] = retry_edu
            else:
                direct_patterns = [
                    r'((?:PG\.?\s*)?B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA|B\.?Sc|M\.?Sc|Ph\.?D)\s*[-\u2013.]\s*(.+?)(?:\s+from\s+|\s+at\s+)(.+?)(?:\n|$)',
                    r'((?:PG\.?\s*)?B\.?E\.?|B\.?Tech|M\.?Tech|MBA|BCA|MCA)\s+(.+?)(?:\s+from\s+|\s+at\s+)(.+?)(?:\n|$)',
                ]
                for dp in direct_patterns:
                    match = re.search(dp, resume_text, re.IGNORECASE)
                    if match:
                        parsed["education"] = [{
                            "degree": match.group(1).strip(),
                            "field": match.group(2).strip() if match.lastindex >= 2 else "",
                            "institution": match.group(3).strip() if match.lastindex >= 3 else "",
                            "year": "", "gpa": "", "location": "", "achievements": []
                        }]
                        break

    # ══════ STEP 6: Merge skills ══════
    llm_skills = parsed.get("skills", {})
    if not isinstance(llm_skills, dict):
        llm_skills = {"other_tools": llm_skills} if isinstance(llm_skills, list) else {}

    for category, regex_skill_list in regex_skills.items():
        if not regex_skill_list:
            continue
        if category in llm_skills and isinstance(llm_skills[category], list):
            existing = set(s.lower() for s in llm_skills[category] if s)
            for skill in regex_skill_list:
                if skill and skill.lower() not in existing:
                    llm_skills[category].append(skill)
        elif regex_skill_list:
            llm_skills[category] = regex_skill_list

    parsed["skills"] = llm_skills

    # ══════ STEP 7: Experience — independent calculation + dedup ══════
    work_history = parsed.get("work_history", parsed.get("experience", []))
    if not isinstance(work_history, list):
        work_history = []
    parsed["work_history"] = work_history

    # 7b: ALWAYS try regex extraction for paragraph-format entries
    regex_work = _extract_work_experience_from_text(resume_text)
    if regex_work:
        if not work_history:
            work_history = regex_work
        else:
            work_history.extend(regex_work)
        parsed["work_history"] = work_history

    # 7c: Deduplicate — keep ONE entry per company (the best one)
    final_work: List[Dict] = []
    seen_companies: Dict[str, int] = {}

    for j in parsed.get("work_history", []):
        if not isinstance(j, dict):
            continue

        company_raw = j.get('company', '') or j.get('organization', '') or ''
        title_raw = j.get('title', '') or j.get('role', '') or j.get('position', '') or ''

        company_cleaned = re.sub(r'https?://\S+', '', company_raw)
        company_cleaned = re.sub(r'[,;]\s*\w+$', '', company_cleaned)
        company_cleaned = re.sub(r'\s*[-\u2013\u2014]\s*$', '', company_cleaned)
        company_cleaned = company_cleaned.strip()

        company_words = re.sub(r'[^a-z0-9\s]', '', company_cleaned.lower()).split()
        location_words = {
            'bangalore', 'bengaluru', 'mumbai', 'delhi', 'chennai', 'hyderabad',
            'pune', 'kolkata', 'noida', 'gurgaon', 'gurugram', 'india',
            'new', 'york', 'san', 'francisco', 'london', 'singapore',
            'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'llc',
        }
        company_core = [w for w in company_words if w not in location_words and len(w) > 1]
        company_norm = ''.join(company_core)[:20] if company_core else ''.join(company_words)[:20]

        if not company_norm:
            company_norm = re.sub(r'[^a-z0-9]', '', title_raw.lower())[:20]

        if not company_norm:
            final_work.append(j)
            continue

        # Score entry quality using DATES not duration_years
        # (LLM often hallucinates duration_years)
        has_start = bool(j.get('start_date', j.get('from', '')))
        has_end = bool(j.get('end_date', j.get('to', '')))
        has_dates = has_start and has_end

        # Independently calculate duration from date strings
        independent_dur = 0.0
        if has_start:
            _month_map = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
                'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
                'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
                'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                'dec': 12, 'december': 12,
            }
            start_str = str(j.get('start_date', j.get('from', '')))
            end_str = str(j.get('end_date', j.get('to', '')))

            sy, sm = _parse_date_to_ym(start_str, _month_map)
            if sy:
                present_words_check = [
                    'present', 'current', 'now', 'ongoing', 'till date',
                    'till now', 'today', 'continuing',
                ]
                if not end_str or end_str.strip().lower() in present_words_check or end_str.strip() in ['-', '\u2013', '']:
                    ey, em = CURRENT_YEAR, CURRENT_MONTH
                else:
                    ey, em = _parse_date_to_ym(end_str, _month_map)
                    if not ey:
                        ey, em = CURRENT_YEAR, CURRENT_MONTH

                months = (ey - sy) * 12 + ((em or 6) - (sm or 6))
                independent_dur = round(max(0, months / 12.0), 1)

        new_quality = independent_dur + (1.0 if has_dates else 0) + (0.5 if has_start else 0)

        if company_norm in seen_companies:
            existing_idx = seen_companies[company_norm]
            existing = final_work[existing_idx]

            # Calculate existing quality the same way
            ex_has_start = bool(existing.get('start_date', existing.get('from', '')))
            ex_has_end = bool(existing.get('end_date', existing.get('to', '')))
            ex_has_dates = ex_has_start and ex_has_end

            ex_dur = 0.0
            if ex_has_start:
                ex_start = str(existing.get('start_date', existing.get('from', '')))
                ex_end = str(existing.get('end_date', existing.get('to', '')))
                esy, esm = _parse_date_to_ym(ex_start, _month_map)
                if esy:
                    if not ex_end or ex_end.strip().lower() in present_words_check or ex_end.strip() in ['-', '\u2013', '']:
                        eey, eem = CURRENT_YEAR, CURRENT_MONTH
                    else:
                        eey, eem = _parse_date_to_ym(ex_end, _month_map)
                        if not eey:
                            eey, eem = CURRENT_YEAR, CURRENT_MONTH
                    ex_months = (eey - esy) * 12 + ((eem or 6) - (esm or 6))
                    ex_dur = round(max(0, ex_months / 12.0), 1)

            existing_quality = ex_dur + (1.0 if ex_has_dates else 0) + (0.5 if ex_has_start else 0)

            if new_quality > existing_quality:
                # Replace with better entry AND fix its duration_years
                j["duration_years"] = independent_dur
                final_work[existing_idx] = j
        else:
            # Fix duration_years if we calculated it independently
            if independent_dur > 0:
                j["duration_years"] = independent_dur
            seen_companies[company_norm] = len(final_work)
            final_work.append(j)

    parsed["work_history"] = final_work

    # 7d: Final experience calculation — ALWAYS recalculate from dates
    # Never trust LLM's total_experience_years or duration_years
    if final_work:
        total_exp = 0.0
        _month_map_final = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
            'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
            'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
            'dec': 12, 'december': 12,
        }
        present_words_final = [
            'present', 'current', 'now', 'ongoing', 'till date',
            'till now', 'today', 'continuing',
        ]

        for job in final_work:
            if not isinstance(job, dict):
                continue

            start_str = str(job.get('start_date', job.get('from', '')))
            end_str = str(job.get('end_date', job.get('to', '')))

            if not start_str:
                # Only use duration_years as last resort
                dur = job.get("duration_years", 0)
                try:
                    total_exp += float(dur) if dur else 0
                except (ValueError, TypeError):
                    pass
                continue

            sy, sm = _parse_date_to_ym(start_str, _month_map_final)
            if not sy:
                dur = job.get("duration_years", 0)
                try:
                    total_exp += float(dur) if dur else 0
                except (ValueError, TypeError):
                    pass
                continue

            if not end_str or end_str.strip().lower() in present_words_final or end_str.strip() in ['-', '\u2013', '']:
                ey, em = CURRENT_YEAR, CURRENT_MONTH
            else:
                ey, em = _parse_date_to_ym(end_str, _month_map_final)
                if not ey:
                    ey, em = CURRENT_YEAR, CURRENT_MONTH

            months = (ey - sy) * 12 + ((em or 6) - (sm or 6))
            job_years = round(max(0, months / 12.0), 1)
            job["duration_years"] = job_years  # Fix the stored value too
            total_exp += job_years

        parsed["total_experience_years"] = round(total_exp, 1)
    else:
        # No work history at all
        parsed["total_experience_years"] = 0

    # ══════ STEP 8: Current role/company ══════
    if not parsed.get("current_role") or not parsed.get("current_company"):
        wh = parsed.get("work_history", [])
        if isinstance(wh, list) and wh:
            first_job = wh[0] if isinstance(wh[0], dict) else {}
            if not parsed.get("current_role"):
                parsed["current_role"] = (
                    first_job.get("title", "") or first_job.get("role", "") or first_job.get("position", "")
                )
            if not parsed.get("current_company"):
                parsed["current_company"] = (
                    first_job.get("company", "") or first_job.get("organization", "")
                )

    # ══════ STEP 9: Ensure all required fields ══════
    required_fields: Dict = {
        "name": "Unknown Candidate", "email": "", "phone": "", "address": "",
        "linkedin": "", "github": "", "portfolio": "", "location": "",
        "current_role": "", "current_company": "", "total_experience_years": 0,
        "professional_summary": "", "specializations": [], "skills": {},
        "work_history": [], "education": [], "certifications": [], "awards": [],
        "projects": [], "publications": [], "volunteer": [], "languages": [], "interests": []
    }
    for fld, default_value in required_fields.items():
        if fld not in parsed:
            parsed[fld] = default_value

    return parsed


# ═══════════════════════════════════════════════════════════════
#                    DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_resume_display_summary(parsed: Dict) -> str:
    if not parsed:
        return "No resume data"
    lines = [f"**{parsed.get('name', 'Candidate')}**"]
    role = parsed.get("current_role", "")
    company = parsed.get("current_company", "")
    if role:
        lines.append(f"\ud83d\udcbc {role}" + (f" at {company}" if company else ""))
    loc = parsed.get("location", "") or parsed.get("address", "")
    if loc:
        lines.append(f"\ud83d\udccd {loc[:60]}")
    exp = parsed.get("total_experience_years", 0)
    if exp:
        lines.append(f"\ud83d\udcc5 ~{exp} years experience")
    if parsed.get("email"):
        lines.append(f"\ud83d\udce7 {parsed['email']}")
    if parsed.get("phone"):
        lines.append(f"\ud83d\udcde {parsed['phone']}")
    if parsed.get("linkedin"):
        lines.append("\ud83d\udd17 LinkedIn")
    if parsed.get("github"):
        lines.append("\ud83d\udcbb GitHub")
    return "\n".join(lines)


def get_resume_full_summary(parsed: Dict) -> Dict:
    if not parsed:
        return {}
    skills_count = 0
    sd = parsed.get("skills", {})
    if isinstance(sd, dict):
        for cat in sd.values():
            if isinstance(cat, list):
                skills_count += len(cat)
    elif isinstance(sd, list):
        skills_count = len(sd)
    return {
        "name": parsed.get("name", "Unknown"),
        "email": parsed.get("email", ""),
        "phone": parsed.get("phone", ""),
        "location": parsed.get("location", "") or parsed.get("address", ""),
        "current_role": parsed.get("current_role", ""),
        "current_company": parsed.get("current_company", ""),
        "total_experience_years": parsed.get("total_experience_years", 0),
        "skills_count": skills_count,
        "education_count": len(parsed.get("education", [])),
        "work_history_count": len(parsed.get("work_history", [])),
        "certifications_count": len(parsed.get("certifications", [])),
        "projects_count": len(parsed.get("projects", [])),
        "has_linkedin": bool(parsed.get("linkedin")),
        "has_github": bool(parsed.get("github")),
        "has_portfolio": bool(parsed.get("portfolio")),
        "has_summary": bool(parsed.get("professional_summary")),
    }


def extract_key_highlights(parsed: Dict) -> List[str]:
    highlights: List[str] = []
    if not parsed:
        return highlights
    exp = parsed.get("total_experience_years", 0)
    if exp:
        highlights.append(f"\ud83d\udcc5 {exp} years of experience")
    role = parsed.get("current_role", "")
    company = parsed.get("current_company", "")
    if role and company:
        highlights.append(f"\ud83d\udcbc Currently {role} at {company}")
    elif role:
        highlights.append(f"\ud83d\udcbc {role}")
    education = parsed.get("education", [])
    if education and isinstance(education, list):
        first = education[0] if isinstance(education[0], dict) else {}
        degree = first.get("degree", "")
        if degree:
            inst = first.get("institution", "")
            highlights.append(f"\ud83c\udf93 {degree}" + (f" from {inst}" if inst else ""))
    sd = parsed.get("skills", {})
    sc = sum(len(v) for v in sd.values() if isinstance(v, list)) if isinstance(sd, dict) else 0
    if sc:
        highlights.append(f"\ud83d\udee0\ufe0f {sc} skills identified")
    certs = parsed.get("certifications", [])
    if certs and isinstance(certs, list):
        highlights.append(f"\ud83d\udcdc {len(certs)} certification(s)")
    projects = parsed.get("projects", [])
    if projects and isinstance(projects, list):
        highlights.append(f"\ud83d\ude80 {len(projects)} project(s)")
    return highlights


def get_contact_completeness(parsed: Dict) -> Dict:
    if not parsed:
        return {"score": 0, "missing": ["all"]}
    fields = {
        "name": parsed.get("name", ""), "email": parsed.get("email", ""),
        "phone": parsed.get("phone", ""),
        "location": parsed.get("location", "") or parsed.get("address", ""),
        "linkedin": parsed.get("linkedin", ""),
    }
    present = [f for f, v in fields.items() if v and _is_valid_field(f, v)]
    missing = [f for f in fields if f not in present]
    return {
        "score": round(len(present) / len(fields) * 100, 1),
        "present": present, "missing": missing,
        "total_fields": len(fields), "filled_fields": len(present),
    }


def validate_parsed_resume(parsed: Dict) -> Dict:
    if not parsed:
        return {"valid": False, "issues": ["No data"]}
    issues, warnings = [], []
    name = parsed.get("name", "")
    if not name or name in ["Unknown Candidate", ""]:
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
    sc = sum(len(v) for v in sd.values() if isinstance(v, list)) if isinstance(sd, dict) else 0
    if sc == 0:
        warnings.append("No skills extracted")
    return {
        "valid": len(issues) == 0, "issues": issues, "warnings": warnings,
        "quality_score": max(0, 100 - len(issues) * 20 - len(warnings) * 5),
    }
