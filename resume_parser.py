"""
Enhanced Resume Parser V5
- Triple extraction: Document Processor + Regex + LLM
- Enhanced name extraction with strong/soft blacklists
- Enhanced education validation + post-LLM quality check
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
            "start_date": "Month Year (e.g., April 2022)",
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
            "degree": "Full degree name (e.g., Bachelor of Technology, Master of Science)",
            "field": "field of study / major / branch (e.g., Computer Science, Electronics)",
            "institution": "university or college name",
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
   - Extract ONLY genuine degrees from the EDUCATION section of the resume.
   - Each person typically has 1-3 degrees (e.g., one Bachelor's, one Master's).
   - DO NOT create more than 4 education entries total.
   - DO NOT create education entries from awards, certifications, or achievements.
   - DO NOT use company names (e.g., Tech Mahindra, Infosys) as institution names.
   - DO NOT repeat the same degree multiple times with different fragments.
   - If you find B.E. from Gondawana University, create ONE entry, not five.
   - The "field" must be a real subject (Computer Science, Electronics, etc.), NOT a single letter.
   - The "institution" must be a real school/university, NOT a section header or company.
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
    """Detect repeated words/phrases indicating LLM parsing errors."""
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
    """
    Detect generic garbage patterns in any text field.
    Catches fragments, nonsense sequences, excessive punctuation.
    """
    if not text:
        return False

    # Excessive dots/periods not part of abbreviations
    if text.count('. .') >= 2:
        return True
    if text.count(' . ') >= 2:
        return True

    # Starts with lowercase fragment (truncated word)
    stripped = text.strip()
    if stripped and stripped[0].islower():
        first_word = stripped.split()[0] if stripped.split() else ""
        # Allow known lowercase starters
        allowed = {"in", "at", "of", "for", "the", "and", "or", "with", "to", "from"}
        if first_word and first_word not in allowed and len(first_word) < 5:
            return True

    # Contains section header keywords repeated
    header_words = ["education", "experience", "skills", "summary", "objective", "profile"]
    for hw in header_words:
        if text.lower().count(hw) >= 2:
            return True

    return False


def _is_valid_education_entry(edu: Dict) -> bool:
    """
    Return True only when the education dict genuinely describes a
    degree/diploma.
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

    if field_of_study and len(field_of_study) < 3:
        if field_of_study.lower() not in allowed_short_fields:
            return False

    # Single character field → always reject
    if field_of_study and len(field_of_study) == 1:
        return False

    if institution and len(re.sub(r'[^a-zA-Z]', '', institution)) < 3:
        return False

    # Fragment: starts lowercase + short first word
    if institution and institution.split():
        first_word = institution.split()[0]
        if first_word and first_word[0].islower() and len(first_word) < 6:
            return False

    if field_of_study and field_of_study[0].islower() and len(field_of_study) < 5:
        if field_of_study.lower() not in allowed_short_fields:
            return False

    # 6. Ambiguous 2-letter degrees need a school-like institution
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

    # 8. Reject if institution text is suspiciously long (likely paragraph/description)
    if institution and len(institution) > 150:
        return False

    return True


def _normalize_degree_key(degree: str) -> str:
    """Normalize degree for deduplication purposes."""
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


def _deduplicate_education_entries(education_list: List[Dict]) -> List[Dict]:
    """
    Aggressively deduplicate education entries.
    For each degree level, keep only the BEST entry (most complete).
    """
    if not education_list:
        return []

    # Group by normalized degree
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

    # For each degree, pick the best entry
    result: List[Dict] = []
    for _degree_key, entries in degree_groups.items():
        if len(entries) == 1:
            result.append(entries[0])
        else:
            # Score each entry by completeness
            best_entry = entries[0]
            best_score = 0
            for entry in entries:
                score = 0
                inst = str(
                    entry.get("institution", "")
                    or entry.get("university", "")
                    or entry.get("college", "")
                ).strip()
                field_val = str(
                    entry.get("field", "")
                    or entry.get("major", "")
                    or entry.get("branch", "")
                ).strip()
                year = str(
                    entry.get("year", "")
                    or entry.get("end_year", "")
                    or entry.get("graduation_year", "")
                ).strip()
                gpa = str(
                    entry.get("gpa", "")
                    or entry.get("cgpa", "")
                    or entry.get("grade", "")
                ).strip()
                loc = str(entry.get("location", "")).strip()

                # Score based on completeness and quality
                if inst and len(inst) > 5:
                    score += 3
                if field_val and len(field_val) > 2:
                    score += 2
                if year:
                    score += 2
                if gpa:
                    score += 1
                if loc:
                    score += 1
                # Penalize garbage
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
    """Filter education list to only valid entries, then deduplicate."""
    if not education_list or not isinstance(education_list, list):
        return []
    valid = [e for e in education_list if isinstance(e, dict) and _is_valid_education_entry(e)]
    return _deduplicate_education_entries(valid)


# ═══════════════════════════════════════════════════════════════
#                    CONTACT EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_contacts_regex(text: str) -> Dict:
    """Extract contact details using comprehensive regex patterns"""
    contacts: Dict[str, str] = {
        "email": "", "phone": "", "address": "",
        "linkedin": "", "github": "", "portfolio": "", "location": ""
    }

    if not text:
        return contacts

    # EMAIL
    email_patterns = [
        r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*@\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*\[\s*at\s*\]\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*\(\s*at\s*\)\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'(?:email|e-mail|mail)[\s.:]*[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}',
    ]
    for pattern in email_patterns:
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
    website_patterns = [
        r'(?:portfolio|website|web|site|blog)[\s.:]+(?:https?://)?[\w.-]+\.[a-z]{2,}[\w/.-]*',
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:dev|io|me|tech|design|codes?|site|online|app)/?[\w/.-]*',
    ]
    for pattern in website_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            url = re.sub(r'^(?:portfolio|website|web|site|blog)[\s.:]+', '', match, flags=re.IGNORECASE).strip()
            if 'linkedin' not in url.lower() and 'github' not in url.lower() and '@' not in url:
                contacts["portfolio"] = url
                break
        if contacts["portfolio"]:
            break

    # ADDRESS
    address_patterns = [
        r'(?:Address|Location|Residence|Home|Addr)[\s.:]+([^\n]{15,150})',
        r'(?:No[.:]?\s*)?[\d]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar|Colony|Layout|Block|Lane|Apt|Apartment|Floor|Fl|Building|Bldg)[\w\s,.-]+(?:\d{5,6})',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}',
    ]
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            addr = (match.group(1) if match.lastindex else match.group(0)).strip()
            addr = re.sub(r'^(?:Address|Location|Residence|Home|Addr)[\s.:]+', '', addr, flags=re.IGNORECASE).strip()
            if 15 < len(addr) < 200:
                contacts["address"] = addr
                break

    # LOCATION
    location_patterns = [
        r'(?:Location|Based in|Located at|City|Current Location)[\s.:]+([A-Za-z][A-Za-z\s,]+)',
        r'\b(Bangalore|Bengaluru|Mumbai|Delhi|NCR|Chennai|Hyderabad|Pune|Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Lucknow|Chandigarh|Indore|Bhopal|Kochi|Coimbatore|Trivandrum|Mysore|Nagpur|Surat|Vadodara|Bhubaneswar|Patna|Ranchi|Guwahati|Visakhapatnam|Vijayawada)\b',
        r'\b(New York|NYC|San Francisco|SF|Los Angeles|LA|Chicago|Seattle|Boston|Austin|Denver|Atlanta|Dallas|Houston|Phoenix|San Diego|San Jose|Portland|Miami|Washington DC|Philadelphia)\b',
        r'\b(London|Berlin|Amsterdam|Dublin|Singapore|Tokyo|Sydney|Toronto|Vancouver|Melbourne|Paris|Munich|Barcelona|Stockholm|Copenhagen|Zurich|Dubai|Hong Kong)\b',
        r'\b(India|USA|US|United States|UK|United Kingdom|Canada|Australia|Germany|Netherlands|Singapore|UAE)\b',
    ]
    for pattern in location_patterns:
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
    """Extract candidate name using multiple strategies"""
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
    name_label_patterns = [
        r'(?:Name|Full Name|Candidate Name|Applicant Name)[\s.:]+([A-Z][a-zA-Z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-zA-Z]+){1,2})',
        r'(?:Name|Full Name)[\s.:]+([A-Z][A-Z\s]+)',
    ]
    for pattern in name_label_patterns:
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

        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+$',
            r'^[A-Z]+\s+[A-Z]+$',
            r'^[A-Z]+\s+[A-Z]+\s+[A-Z]+$',
            r'^Dr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^Mr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^Ms\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+$',
        ]
        for pattern in name_patterns:
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
    intro_patterns = [
        r"(?:I am|I'm|My name is|This is|Myself)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})",
    ]
    for pattern in intro_patterns:
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
    """Validate with two-tier blacklist."""
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
#                    EDUCATION EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_education_regex(text: str) -> List[Dict]:
    """Extract education details using regex patterns as fallback"""
    education: List[Dict] = []
    if not text:
        return education

    degree_patterns = [
        (r'(Bachelor\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Commerce|Business|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(B\.?\s*(?:Tech|E|Sc|A|Com|B\.?A|S|CA|BA|Arch|Pharm|Ed|Des)\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(B\.?Tech|BTech|B\.?E\.?|BE)\s*[-–in\s]*([\w\s,&]+)?', 'Bachelor'),
        (r'(BCA|B\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(BBA|B\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(B\.?Com|BCom|B\.?Commerce)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(Master\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Business\s*Administration|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?\s*(?:Tech|E|Sc|A|B\.?A|S|CA|BA|Phil|Ed|Des)\.?|MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?Tech|MTech|M\.?E\.?|ME|M\.?S\.?|MS)\s*[-–in\s]*([\w\s,&]+)?', 'Master'),
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
        r'((?:19|20)\d{2})\s*[-–to]+\s*((?:19|20)\d{2}|Present|Current|Expected|Pursuing)',
        r'(?:Class\s*of|Batch|Graduated?|Year\s*of\s*(?:Graduation|Completion)|Expected)[:\s]*((?:19|20)\d{2})',
        r'((?:19|20)\d{2})',
    ]

    # Find education section
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

    # Validate + deduplicate
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
    """Extract skills from text using regex patterns"""
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
            'spark', 'pyspark', 'hadoop', 'kafka',
            'xgboost', 'lightgbm',
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

    # Deduplicate
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

        present_words = ['present', 'current', 'now', 'ongoing', 'till date', 'till now', 'today', 'continuing', '-', '–', '']
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
        'unknown', 'null', '-', '—', '', 'candidate', 'your name',
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
    """
    Parse resume using LLM + regex + document extraction.
    Validates and deduplicates education entries aggressively.
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return _basic_fallback(resume_text, {}, "", [], {})

    # STEP 1: Regex extraction
    regex_contacts = _extract_contacts_regex(resume_text)
    regex_name = _extract_name_from_text(resume_text)
    regex_education = _extract_education_regex(resume_text)
    regex_skills = _extract_skills_regex(resume_text)

    # STEP 1b: Document contacts
    if doc_contacts is None:
        doc_contacts = {}
        try:
            from document_processor import extract_contacts_from_text
            doc_contacts = extract_contacts_from_text(resume_text)
        except ImportError:
            pass

    # STEP 2: LLM parsing
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
                    "Each person usually has 1-3 degrees. "
                    "NEVER create education entries from awards, certifications, or work achievements. "
                    "NEVER use company names as institutions."
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

    # STEP 2b: Validate + deduplicate LLM education
    llm_education = parsed.get("education", [])
    if isinstance(llm_education, list):
        parsed["education"] = _filter_valid_education(llm_education)
    elif llm_education:
        parsed["education"] = _filter_valid_education([llm_education])
    else:
        parsed["education"] = []

    # STEP 3: Merge contacts
    merged_contacts = _merge_contacts(parsed, regex_contacts, doc_contacts)
    for field, value in merged_contacts.items():
        if value and (not parsed.get(field) or not _is_valid_field(field, parsed.get(field, ""))):
            parsed[field] = value

    # STEP 4: Name
    current_name = parsed.get("name", "")
    if not current_name or not _is_valid_field("name", current_name):
        for candidate in [doc_contacts.get("name", ""), regex_name, parsed.get("name", "")]:
            if candidate and _is_valid_field("name", candidate):
                parsed["name"] = _clean_name(candidate)
                break
        if not parsed.get("name") or not _is_valid_field("name", parsed.get("name", "")):
            parsed["name"] = "Unknown Candidate"

    # STEP 5: Merge regex education (already validated)
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
                    llm_degree_keys.add(dk)

            for regex_edu in regex_education:
                if isinstance(regex_edu, dict):
                    dk = _normalize_degree_key(str(regex_edu.get("degree", "")))
                    if dk and dk not in llm_degree_keys:
                        llm_education.append(regex_edu)
                        llm_degree_keys.add(dk)

            parsed["education"] = _deduplicate_education_entries(llm_education)

    # STEP 6: Merge skills
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

    # STEP 7: Experience
    work_history = parsed.get("work_history", parsed.get("experience", []))
    if isinstance(work_history, list) and work_history:
        calculated = _calculate_total_experience(work_history)
        if calculated > 0:
            parsed["total_experience_years"] = calculated
        parsed["work_history"] = work_history

    # STEP 8: Current role/company
    if not parsed.get("current_role") or not parsed.get("current_company"):
        wh = parsed.get("work_history", [])
        if isinstance(wh, list) and wh:
            first_job = wh[0] if isinstance(wh[0], dict) else {}
            if not parsed.get("current_role"):
                parsed["current_role"] = first_job.get("title", "") or first_job.get("role", "") or first_job.get("position", "")
            if not parsed.get("current_company"):
                parsed["current_company"] = first_job.get("company", "") or first_job.get("organization", "")

    # STEP 9: Ensure all fields
    required_fields: Dict = {
        "name": "Unknown Candidate", "email": "", "phone": "", "address": "",
        "linkedin": "", "github": "", "portfolio": "", "location": "",
        "current_role": "", "current_company": "", "total_experience_years": 0,
        "professional_summary": "", "specializations": [], "skills": {},
        "work_history": [], "education": [], "certifications": [], "awards": [],
        "projects": [], "publications": [], "volunteer": [], "languages": [], "interests": []
    }
    for field, default_value in required_fields.items():
        if field not in parsed:
            parsed[field] = default_value

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
        lines.append(f"💼 {role}" + (f" at {company}" if company else ""))
    loc = parsed.get("location", "") or parsed.get("address", "")
    if loc:
        lines.append(f"📍 {loc[:60]}")
    exp = parsed.get("total_experience_years", 0)
    if exp:
        lines.append(f"📅 ~{exp} years experience")
    if parsed.get("email"):
        lines.append(f"📧 {parsed['email']}")
    if parsed.get("phone"):
        lines.append(f"📞 {parsed['phone']}")
    if parsed.get("linkedin"):
        lines.append("🔗 LinkedIn")
    if parsed.get("github"):
        lines.append("💻 GitHub")
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
        highlights.append(f"📅 {exp} years of experience")
    role = parsed.get("current_role", "")
    company = parsed.get("current_company", "")
    if role and company:
        highlights.append(f"💼 Currently {role} at {company}")
    elif role:
        highlights.append(f"💼 {role}")
    education = parsed.get("education", [])
    if education and isinstance(education, list):
        first = education[0] if isinstance(education[0], dict) else {}
        degree = first.get("degree", "")
        if degree:
            inst = first.get("institution", "")
            highlights.append(f"🎓 {degree}" + (f" from {inst}" if inst else ""))
    sd = parsed.get("skills", {})
    sc = sum(len(v) for v in sd.values() if isinstance(v, list)) if isinstance(sd, dict) else 0
    if sc:
        highlights.append(f"🛠️ {sc} skills identified")
    certs = parsed.get("certifications", [])
    if certs and isinstance(certs, list):
        highlights.append(f"📜 {len(certs)} certification(s)")
    projects = parsed.get("projects", [])
    if projects and isinstance(projects, list):
        highlights.append(f"🚀 {len(projects)} project(s)")
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
