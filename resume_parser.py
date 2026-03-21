"""
Enhanced Resume Parser V11
- Triple extraction: Document Processor + Regex + LLM
- Enhanced name extraction: 8 strategies including Regards/Declaration sections
- Enhanced education validation + cross-validation against resume text
- Robust structured resume parsing — rejects description lines
- Strict work entry validation — rejects sentences, descriptions, bullet text
- Section-aware extraction — prevents date/text leaks from other sections
- LLM garbage entries filtered before merge
- Centralized email PIN-prefix cleaning applied to ALL sources
- Portfolio URL validation — rejects employer domains and mangled URLs
- Labeled format parsing (Role:, Company:, Duration:)
- Concatenated garbage in role titles cleaned
- Positive location verification against 400+ known cities/states/countries
- Location cleaning — strips sign-offs, names, description fragments
- Independent experience calculation — NEVER trusts LLM totals
- Spaced-out text detection
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

_DESCRIPTION_INDICATORS: Set[str] = {
    "experience in", "responsible for", "worked on", "working with",
    "good experience", "strong experience", "hands-on experience",
    "proficient in", "expertise in", "knowledge of", "familiar with",
    "understanding of", "exposure to", "involved in", "contributed to",
    "developed", "designed", "implemented", "managed", "created",
    "built", "maintained", "deployed", "configured", "optimized",
    "utilized", "leveraged", "performed", "conducted", "executed",
    "analyzed", "supported", "assisted", "collaborated", "coordinated",
    "having good", "having experience", "having knowledge",
    "it supports", "it provides", "it enables", "it allows",
    "supports vpn", "over ip", "service provider",
    "quality of service", "strong understanding",
    "detail project", "project overview", "workflow",
    "sequence 1", "sequence 2", "phase 1", "phase 2",
    "opex", "capex", "ip/mpls", "netconf", "yang",
    "key responsibilities", "key achievements", "responsibilities include",
    "duties include", "tasks include",
}

_MAX_ROLE_LENGTH = 80
_MAX_COMPANY_LENGTH = 100
_MAX_ROLE_WORDS = 10
_MAX_COMPANY_WORDS = 12

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
    'pondicherry', 'puducherry',
}

PRESENT_WORDS: Set[str] = {
    'present', 'current', 'currently', 'now', 'ongoing',
    'till date', 'till now', 'to date', 'today', 'continuing',
}

_NON_WORK_SECTION_HEADERS: Set[str] = {
    'education', 'skills', 'certifications', 'certification',
    'awards', 'scholarships', 'awards & scholarships',
    'awards and scholarships',
    'projects', 'publications', 'volunteer', 'languages',
    'interests', 'hobbies', 'references', 'declaration',
    'training', 'courses', 'accomplishments',
    'technical skills', 'professional skills',
    'academic details', 'academic qualifications',
    'personal details', 'personal information',
    'achievements', 'key skills', 'core competencies',
    'tools and technologies', 'tools & technologies',
}

_WORK_SECTION_HEADERS: Set[str] = {
    'work experience', 'experience', 'professional experience',
    'employment history', 'employment', 'work history',
    'career history', 'professional background',
    'relevant experience', 'industry experience',
    'professional summary and experience',
}

_JOB_TITLE_PATTERNS = [
    r'(?:Senior|Junior|Lead|Principal|Staff|Chief|Head|VP|Associate|Assistant)?\s*'
    r'(?:Software|Full[\s-]?Stack|Front[\s-]?End|Back[\s-]?End|Data|ML|AI|Cloud|DevOps|QA|Test|Mobile|Web|'
    r'System|Network|Security|Database|Platform|Infrastructure|Site\s*Reliability|Machine\s*Learning|'
    r'Deep\s*Learning|NLP|Computer\s*Vision|Big\s*Data|Business\s*Intelligence|BI|ETL|Solutions?|'
    r'Technical|Product|Program|Project|Operations?|IT|Application|Research|Analytics?|'
    r'Medical\s*Record|Quality|Customer\s*(?:Success|Support)|Sales|Marketing|HR|Finance|'
    r'Embedded|Firmware|Hardware|VLSI|ASIC|FPGA|Robotics|Automation|RPA|SAP|Oracle|Salesforce|'
    r'Python|Java|React|Angular|Node|Django|iOS|Android|Unity|Blockchain|Cyber\s*Security)?\s*'
    r'(?:Engineer(?:ing)?|Developer|Architect|Analyst|Scientist|Manager|Designer|Consultant|'
    r'Administrator|Coordinator|Specialist|Executive|Officer|Director|Intern|Trainee|'
    r'Programmer|Tester|Lead|Summarizer|Reviewer|Transcriptionist|Coder)',
]

_EMPLOYER_DOMAINS: Set[str] = {
    'ibm.com', 'google.com', 'microsoft.com', 'amazon.com', 'apple.com',
    'facebook.com', 'meta.com', 'twitter.com', 'linkedin.com', 'github.com',
    'infosys.com', 'wipro.com', 'tcs.com', 'cognizant.com', 'accenture.com',
    'capgemini.com', 'hcl.com', 'datamatics.com', 'oracle.com', 'salesforce.com',
    'netflix.com', 'uber.com', 'airbnb.com', 'spotify.com', 'stripe.com',
    'adobe.com', 'intel.com', 'cisco.com', 'dell.com', 'hp.com', 'sap.com',
    'deloitte.com', 'pwc.com', 'kpmg.com', 'ey.com', 'mckinsey.com',
    'paypal.com', 'vmware.com', 'redhat.com', 'nvidia.com', 'qualcomm.com',
}

_KNOWN_LOCATIONS: Set[str] = {
    'bangalore', 'bengaluru', 'mumbai', 'delhi', 'new delhi', 'ncr',
    'chennai', 'hyderabad', 'pune', 'kolkata', 'noida', 'gurgaon',
    'gurugram', 'ahmedabad', 'jaipur', 'lucknow', 'chandigarh',
    'indore', 'bhopal', 'kochi', 'coimbatore', 'trivandrum',
    'thiruvananthapuram', 'mysore', 'mysuru', 'nagpur', 'surat',
    'vadodara', 'bhubaneswar', 'patna', 'ranchi', 'guwahati',
    'visakhapatnam', 'vijayawada', 'kolhapur', 'pondicherry',
    'puducherry', 'mangalore', 'mangaluru', 'dehradun', 'shimla',
    'jammu', 'srinagar', 'amritsar', 'ludhiana', 'kanpur',
    'varanasi', 'agra', 'meerut', 'nashik', 'aurangabad',
    'rajkot', 'jodhpur', 'madurai', 'tiruchirappalli', 'trichy',
    'salem', 'tiruppur', 'erode', 'vellore', 'guntur', 'warangal',
    'nellore', 'hubli', 'belgaum', 'belagavi', 'gulbarga',
    'kalaburagi', 'thrissur', 'kozhikode', 'calicut',
    'raipur', 'bilaspur', 'gwalior', 'jabalpur', 'ujjain',
    'dhanbad', 'jamshedpur', 'bokaro', 'siliguri', 'durgapur',
    'asansol', 'howrah', 'cuttack', 'rourkela', 'sambalpur',
    'moradabad', 'bareilly', 'aligarh', 'gorakhpur',
    'faridabad', 'ghaziabad', 'greater noida',
    'ambala', 'karnal', 'panipat', 'sonipat', 'rohtak',
    'hisar', 'bhiwani', 'panchkula', 'mohali', 'zirakpur',
    'navi mumbai', 'thane', 'kalyan', 'dombivli', 'vasai',
    'virar', 'mira bhayandar', 'bhiwandi', 'panvel',
    'secunderabad', 'kompally', 'gachibowli', 'madhapur',
    'hitec city', 'whitefield', 'electronic city', 'marathahalli',
    'koramangala', 'indiranagar', 'hsr layout', 'jp nagar',
    'btm layout', 'silk board', 'mahadevapura', 'sarjapur',
    'ambegaon', 'katraj', 'kothrud', 'hinjewadi', 'wakad',
    'baner', 'aundh', 'viman nagar', 'kalyani nagar', 'hadapsar',
    'maharashtra', 'karnataka', 'tamil nadu', 'telangana',
    'andhra pradesh', 'kerala', 'west bengal', 'uttar pradesh',
    'rajasthan', 'gujarat', 'madhya pradesh', 'punjab',
    'haryana', 'bihar', 'odisha', 'jharkhand', 'chhattisgarh',
    'uttarakhand', 'himachal pradesh', 'goa', 'assam',
    'new york', 'nyc', 'san francisco', 'sf', 'los angeles', 'la',
    'chicago', 'seattle', 'boston', 'austin', 'denver', 'atlanta',
    'dallas', 'houston', 'phoenix', 'san diego', 'san jose',
    'portland', 'miami', 'washington', 'washington dc', 'philadelphia',
    'detroit', 'minneapolis', 'tampa', 'orlando', 'charlotte',
    'nashville', 'raleigh', 'durham', 'salt lake city', 'pittsburgh',
    'columbus', 'indianapolis', 'kansas city', 'milwaukee',
    'las vegas', 'sacramento', 'richmond', 'jacksonville',
    'baltimore', 'st louis', 'san antonio', 'fort worth',
    'california', 'texas', 'florida', 'illinois',
    'pennsylvania', 'ohio', 'georgia', 'north carolina', 'michigan',
    'new jersey', 'virginia', 'arizona', 'massachusetts',
    'tennessee', 'indiana', 'missouri', 'maryland',
    'wisconsin', 'colorado', 'minnesota', 'oregon',
    'connecticut', 'utah', 'nevada',
    'london', 'berlin', 'amsterdam', 'dublin', 'singapore',
    'tokyo', 'sydney', 'toronto', 'vancouver', 'melbourne',
    'paris', 'munich', 'barcelona', 'stockholm', 'copenhagen',
    'zurich', 'dubai', 'hong kong', 'shanghai', 'beijing',
    'seoul', 'taipei', 'bangkok', 'kuala lumpur', 'jakarta',
    'manila', 'ho chi minh', 'cairo', 'lagos', 'nairobi',
    'cape town', 'johannesburg', 'sao paulo', 'buenos aires',
    'mexico city', 'montreal', 'ottawa', 'calgary', 'edmonton',
    'brisbane', 'perth', 'auckland', 'wellington',
    'frankfurt', 'hamburg', 'vienna', 'prague', 'warsaw',
    'budapest', 'bucharest', 'athens', 'istanbul', 'lisbon',
    'madrid', 'rome', 'milan', 'geneva', 'brussels',
    'helsinki', 'oslo', 'tel aviv', 'riyadh', 'doha',
    'abu dhabi', 'muscat', 'kuwait city',
    'india', 'usa', 'us', 'united states',
    'uk', 'united kingdom', 'canada', 'australia', 'germany',
    'netherlands', 'uae', 'united arab emirates',
    'japan', 'china', 'south korea', 'france', 'italy', 'spain',
    'brazil', 'mexico', 'ireland', 'switzerland', 'sweden',
    'norway', 'denmark', 'finland', 'belgium', 'austria',
    'poland', 'portugal', 'greece', 'turkey',
    'israel', 'saudi arabia', 'qatar', 'new zealand',
    'south africa', 'nigeria', 'kenya', 'egypt', 'thailand',
    'malaysia', 'indonesia', 'philippines', 'vietnam',
    'taiwan', 'russia', 'ukraine',
}

_LOCATION_BAD_TERMS: Set[str] = {
    'developer', 'engineer', 'manager', 'analyst', 'consultant',
    'experience', 'skills', 'education', 'project', 'resume',
    'currently', 'working', 'employed', 'company', 'role',
    'position', 'responsibilities', 'efficient', 'space',
    'algorithm', 'data', 'system', 'application', 'software',
    'hardware', 'network', 'server', 'database', 'cloud',
    'api', 'framework', 'library', 'module', 'function',
    'variable', 'class', 'object', 'method', 'interface',
    'implementation', 'optimization', 'performance', 'scalable',
    'distributed', 'architecture', 'design', 'pattern',
    'testing', 'deployment', 'monitoring', 'logging',
    'processing', 'pipeline', 'workflow', 'automation',
    'machine', 'learning', 'model', 'training', 'inference',
    'certificate', 'certification', 'award', 'achievement',
    'proficient', 'expertise', 'knowledge', 'understanding',
    'strong', 'good', 'excellent', 'hands-on',
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
    "email": "email address without any leading PIN/ZIP code digits",
    "phone": "complete phone number with country code",
    "address": "full physical/mailing address if mentioned",
    "linkedin": "LinkedIn URL",
    "github": "GitHub URL",
    "portfolio": "personal portfolio or website URL only - NOT employer company URLs",
    "location": "ONLY the city name or city, state, country - nothing else",
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
            "title": "exact SHORT job title like 'Software Engineer' or 'Data Analyst'",
            "company": "company name ONLY - no descriptions",
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
3. EMAIL: Find ANYTHING with @ symbol. Remove any leading PIN/ZIP digits.
4. EDUCATION: Extract ONLY degrees EXPLICITLY WRITTEN. NEVER create from awards.
5. EXPERIENCE: If end_date is "Present", calculate duration until March 2026.
6. duration_years: Calculate as decimal. 2 years 6 months = 2.5.
7. total_experience_years: Sum of all duration_years.
8. SKILLS: Extract EVERY skill mentioned ANYWHERE.
9. WORK HISTORY:
   - "title" must be SHORT (e.g. "Python Developer") NOT a sentence.
   - "company" must be a real company name NOT a description.
   - DO NOT extract bullet points or technical paragraphs as work entries.
10. LOCATION: Return ONLY city/state/country. NO descriptions, NO sign-offs, NO names.
11. PORTFOLIO: Only personal websites. NOT employer URLs like ibm.com, google.com.

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
#               EMAIL CLEANING
# ═══════════════════════════════════════════════════════════════

def _clean_email(email: str) -> str:
    if not email or '@' not in email:
        return email
    email = email.strip()
    email = re.sub(r'\s+', '', email)
    email = re.sub(r'\[\s*at\s*\]|\(\s*at\s*\)', '@', email, flags=re.IGNORECASE)
    if '@' not in email:
        return email
    local_part, domain = email.split('@', 1)
    pin_match = re.match(r'^(\d{4,7})[.\-_](.+)$', local_part)
    if pin_match:
        remaining = pin_match.group(2)
        if any(c.isalpha() for c in remaining) and len(remaining) >= 2:
            local_part = remaining
    if not pin_match:
        digit_prefix = re.match(r'^(\d{5,7})([a-zA-Z].*)$', local_part)
        if digit_prefix:
            remaining = digit_prefix.group(2)
            if any(c.isalpha() for c in remaining) and len(remaining) >= 2:
                local_part = remaining
    digit_suffix = re.match(r'^([a-zA-Z][\w.]*?)(\d{5,7})$', local_part)
    if digit_suffix:
        remaining = digit_suffix.group(1).rstrip('._')
        if len(remaining) >= 3:
            local_part = remaining
    cleaned = f"{local_part}@{domain}"
    if any(c.isalpha() for c in local_part) and len(local_part) >= 2:
        return cleaned
    return email


# ═══════════════════════════════════════════════════════════════
#               LOCATION VALIDATION & CLEANING
# ═══════════════════════════════════════════════════════════════

def _is_valid_location(location: str) -> bool:
    if not location:
        return False
    loc = location.strip()
    if len(loc) < 2 or len(loc) > 100:
        return False
    ll = loc.lower()
    sign_off_words = {'regards', 'sincerely', 'thankyou', 'yours', 'respectfully',
                      'cordially', 'declaration', 'hereby', 'signature'}
    for word in ll.split():
        if word.strip('.,;:') in sign_off_words:
            return False
    if re.search(r',\s*[A-Z][a-z]+\s+[A-Z][a-z]+', loc):
        parts = loc.split(',', 1)
        if len(parts) > 1:
            after_comma = parts[1].strip()
            name_words = re.findall(r'[A-Z][a-z]+', after_comma)
            if len(name_words) >= 2:
                return False
    if '@' in loc or re.search(r'\d{7,}', loc):
        return False
    words = loc.split()
    if len(words) > 8:
        return False
    loc_words_lower = [w.lower().strip('.,;:()') for w in words]
    if any(w in _LOCATION_BAD_TERMS for w in loc_words_lower):
        return False
    ll_clean = re.sub(r'[^a-z\s]', '', ll).strip()
    if ll_clean in _KNOWN_LOCATIONS:
        return True
    for known_loc in _KNOWN_LOCATIONS:
        if known_loc in ll_clean:
            return True
    for w in loc_words_lower:
        w_clean = re.sub(r'[^a-z]', '', w)
        if w_clean and w_clean in _KNOWN_LOCATIONS:
            return True
    for idx in range(len(loc_words_lower) - 1):
        bigram = f"{loc_words_lower[idx]} {loc_words_lower[idx + 1]}"
        bigram_clean = re.sub(r'[^a-z\s]', '', bigram)
        if bigram_clean in _KNOWN_LOCATIONS:
            return True
    return False


def _clean_location(location: str) -> str:
    if not location:
        return ""
    loc = location.strip()
    sign_off_patterns = [
        r'\s*[,.]?\s*(?:Regards|Sincerely|Thank\s*you|Yours|Best|Kind|Warm|Respectfully|Cordially).*$',
        r'\s*[,.]?\s*(?:Declaration|I\s+hereby).*$',
    ]
    for pat in sign_off_patterns:
        loc = re.sub(pat, '', loc, flags=re.IGNORECASE).strip()
    name_trail = re.search(r'^(.+?)\s*[,]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$', loc)
    if name_trail:
        candidate_loc = name_trail.group(1).strip()
        candidate_name = name_trail.group(2).strip()
        name_words = candidate_name.split()
        if len(name_words) >= 2 and all(w[0].isupper() for w in name_words):
            loc = candidate_loc
    loc = loc.rstrip('.,;: ')
    ll = re.sub(r'[^a-z\s]', '', loc.lower()).strip()
    best_match = ""
    best_len = 0
    for known_loc in _KNOWN_LOCATIONS:
        if known_loc in ll and len(known_loc) > best_len:
            best_match = known_loc
            best_len = len(known_loc)
    if best_match:
        match = re.search(re.escape(best_match), loc, re.IGNORECASE)
        if match:
            result_parts = []
            for part in loc.split(','):
                part_clean = re.sub(r'[^a-z\s]', '', part.lower()).strip()
                if part_clean in _KNOWN_LOCATIONS or any(kl in part_clean for kl in _KNOWN_LOCATIONS if len(kl) > 2):
                    result_parts.append(part.strip())
            if result_parts:
                return ', '.join(result_parts)
            return match.group(0).strip()
    if len(loc) < 2:
        return ""
    return loc


def _is_valid_address(address: str) -> bool:
    if not address:
        return False
    addr = address.strip()
    if len(addr) < 10 or len(addr) > 250:
        return False
    ll = addr.lower()
    sign_offs = ['regards', 'sincerely', 'thank you', 'yours truly', 'declaration', 'i hereby']
    for so in sign_offs:
        if so in ll:
            return False
    return True


# ═══════════════════════════════════════════════════════════════
#         WORK ENTRY VALIDATION
# ═══════════════════════════════════════════════════════════════

def _looks_like_sentence_or_description(text: str) -> bool:
    if not text:
        return False
    tl = text.lower().strip()
    for indicator in _DESCRIPTION_INDICATORS:
        if indicator in tl:
            return True
    words = tl.split()
    if len(words) > _MAX_ROLE_WORDS:
        return True
    if text.count(',') >= 2:
        return True
    if text.count('(') >= 2 or text.count(')') >= 2:
        return True
    desc_starters = {
        'good', 'strong', 'excellent', 'having', 'worked', 'working',
        'responsible', 'involved', 'contributed', 'developed', 'designed',
        'implemented', 'managed', 'created', 'built', 'maintained',
        'deployed', 'configured', 'optimized', 'utilized', 'leveraged',
        'performed', 'conducted', 'executed', 'analyzed', 'supported',
        'assisted', 'collaborated', 'it', 'the', 'this', 'these',
        'various', 'multiple', 'several', 'detail', 'overview',
    }
    if words and words[0] in desc_starters:
        return True
    desc_patterns = [
        r'like\s+\w+', r'such\s+as', r'including\s+',
        r'using\s+\w+', r'with\s+\w+\s+and\s+',
        r'over\s+ip', r'services?\s+over',
        r'\bvpn\b', r'\bmpls\b', r'\bnetconf\b',
        r'quality\s+of\s+service',
    ]
    for pat in desc_patterns:
        if re.search(pat, tl):
            return True
    filler_words = {'in', 'a', 'an', 'the', 'of', 'for', 'to', 'and', 'or', 'with',
                    'from', 'by', 'on', 'is', 'was', 'are', 'were', 'has', 'have',
                    'like', 'using', 'over', 'its', 'it', 'that', 'which', 'this'}
    if len(words) > 3:
        filler_count = sum(1 for w in words if w in filler_words)
        if filler_count / len(words) > 0.4:
            return True
    return False


def _is_valid_role_title(title: str) -> bool:
    if not title or not title.strip():
        return False
    t = title.strip()
    if len(t) > _MAX_ROLE_LENGTH or len(t) < 3:
        return False
    if len(t.split()) > _MAX_ROLE_WORDS:
        return False
    if _looks_like_sentence_or_description(t):
        return False
    if sum(1 for c in t if c.isalpha()) / max(len(t), 1) < 0.6:
        return False
    return True


def _is_valid_company_name(company: str) -> bool:
    if not company or not company.strip():
        return False
    c = company.strip()
    if len(c) > _MAX_COMPANY_LENGTH or len(c) < 2:
        return False
    if len(c.split()) > _MAX_COMPANY_WORDS:
        return False
    if _looks_like_sentence_or_description(c):
        return False
    if sum(1 for ch in c if ch.isalpha()) / max(len(c), 1) < 0.5:
        return False
    return True


def _is_valid_work_entry(job: Dict) -> bool:
    if not isinstance(job, dict):
        return False
    title = str(job.get("title", "") or job.get("role", "") or job.get("position", "")).strip()
    company = str(job.get("company", "") or job.get("organization", "")).strip()
    if not title and not company:
        return False
    if title and not _is_valid_role_title(title):
        return False
    if company and not _is_valid_company_name(company):
        return False

    def _check_field(val):
        if not val:
            return True
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


def _clean_role_title(raw: str) -> str:
    if not raw:
        return ""
    cleaned = re.sub(
        r'\s*[\u2022•\-|,]\s*(?:Full[-\s]?time|Part[-\s]?time|Internship|Contract|Freelance|Temporary)\s*$',
        '', raw, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip('•\u2022-|,.: ')
    cleaned = re.sub(r'^(?:Role|Position|Title|Designation)\s*[:]\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'(?:PROJECT|PROJECTS?)\s*(?:Sequence|Details?|Overview|Description|Summary|:).*$',
                     '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'(?:EXPERIENCE|EDUCATION|SKILLS|PROJECTS?|CERTIFICATIONS?|AWARDS?)\s*[:.]?\s*$',
                     '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _clean_company_name(raw: str) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r'https?://\S+', '', raw).strip()
    cleaned = re.sub(r'[,;]\s*\w+$', '', cleaned)
    cleaned = cleaned.rstrip('-,. ;:•\u2022|')
    return cleaned


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
        r'^(?:secondary|ssc|10th|matriculation|sslc)$': 'ssc',
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
                    bs, best = s, e
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
                            found = True; break
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
    contacts: Dict[str, str] = {"email": "", "phone": "", "address": "", "linkedin": "", "github": "", "portfolio": "", "location": ""}
    if not text:
        return contacts
    # EMAIL
    for p in [r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}', r'[\w._%+-]+\s*@\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
              r'[\w._%+-]+\s*\[\s*at\s*\]\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
              r'[\w._%+-]+\s*\(\s*at\s*\)\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
              r'(?:email|e-mail|mail)[\s.:]*[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            email = re.sub(r'^(?:email|e-mail|mail)[\s.:]*', '', m.group(0).strip(), flags=re.IGNORECASE)
            email = _clean_email(email)
            if email and '@' in email and '.' in email.split('@')[-1]:
                fl = email.split('@')[0]
                if any(c.isalpha() for c in fl) and len(fl) >= 2:
                    contacts["email"] = email; break
    # PHONE
    for p in [r'\+\d{1,3}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', r'\+\d{1,3}[-.\s]?\d{4,5}[-.\s]?\d{5,6}',
              r'\+\d{10,15}', r'\+91[-.\s]?\d{5}[-.\s]?\d{5}', r'\+91[-.\s]?\d{10}',
              r'(?<!\d)91[-.\s]?\d{10}(?!\d)', r'(?<!\d)0\d{2,4}[-.\s]?\d{6,8}(?!\d)',
              r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}', r'(?<!\d)\d{3}[-.\s]\d{3}[-.\s]\d{4}(?!\d)',
              r'(?<!\d)1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)',
              r'(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*[\+]?[\d][\d\s\-().]{8,18}',
              r'(?<!\d)\d{5}[-.\s]?\d{5}(?!\d)', r'(?<!\d)\d{10}(?!\d)']:
        try:
            for match in re.findall(p, text, re.IGNORECASE):
                cleaned = re.sub(r'^(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*', '', match, flags=re.IGNORECASE).strip()
                digits = re.sub(r'[^\d]', '', cleaned)
                if 10 <= len(digits) <= 15:
                    contacts["phone"] = cleaned; break
            if contacts["phone"]: break
        except re.error:
            continue
    # LINKEDIN
    for p in [r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?', r'linkedin\.com/in/[\w-]+']:
        m = re.search(p, text, re.IGNORECASE)
        if m: contacts["linkedin"] = m.group(0).strip(); break
    # GITHUB
    for p in [r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?', r'github\.com/[\w-]+']:
        m = re.search(p, text, re.IGNORECASE)
        if m: contacts["github"] = m.group(0).strip(); break
    # PORTFOLIO
    for p in [r'(?:portfolio|website|web|site|blog)[\s.:]+(?:https?://)?[\w.-]+\.[a-z]{2,}[\w/.-]*',
              r'(?:https?://)?(?:www\.)?[\w-]+\.(?:dev|io|me|tech|design|codes?|site|online|app)/?[\w/.-]*']:
        for match in re.findall(p, text, re.IGNORECASE):
            url = re.sub(r'^(?:portfolio|website|web|site|blog)[\s.:]+', '', match, flags=re.IGNORECASE).strip()
            ul = url.lower()
            if 'linkedin' in ul or 'github' in ul or '@' in url:
                continue
            if any(d in ul for d in _EMPLOYER_DOMAINS):
                continue
            url_path = url.split('/', 3)[-1] if '/' in url else ''
            if url_path and (re.search(r'[A-Z]{3,}', url_path) or re.search(r'-[A-Z][a-z]+[A-Z]', url_path)):
                continue
            if len(url) > 10:
                contacts["portfolio"] = url; break
        if contacts["portfolio"]: break
    # ADDRESS
    for p in [r'(?:Address|Location|Residence|Home|Addr)[\s.:]+([^\n]{15,150})',
              r'(?:No[.:]?\s*)?[\d]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar|Colony|Layout|Block|Lane|Apt|Apartment|Floor|Fl|Building|Bldg|Society|Housing)[\w\s,.-]+(?:\d{5,6})',
              r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            addr = (m.group(1) if m.lastindex else m.group(0)).strip()
            addr = re.sub(r'^(?:Address|Location|Residence|Home|Addr)[\s.:]+', '', addr, flags=re.IGNORECASE).strip().lstrip("'\"\u2018\u2019")
            if 15 < len(addr) < 200:
                contacts["address"] = addr; break
    # LOCATION
    for p in [r'(?:Location|Based in|Located at|City|Current Location)[\s.:]+([A-Za-z][A-Za-z\s,]+)',
              r'\b(Bangalore|Bengaluru|Mumbai|Delhi|NCR|Chennai|Hyderabad|Pune|Kolkata|Noida|Gurgaon|Gurugram|'
              r'Ahmedabad|Jaipur|Lucknow|Chandigarh|Indore|Bhopal|Kochi|Coimbatore|Trivandrum|Mysore|Nagpur|'
              r'Surat|Vadodara|Bhubaneswar|Patna|Ranchi|Guwahati|Visakhapatnam|Vijayawada|Kolhapur|'
              r'Pondicherry|Puducherry|Ambegaon|Katraj|Kothrud|Hinjewadi|Wakad|Baner|Hadapsar)\b',
              r'\b(New York|NYC|San Francisco|SF|Los Angeles|LA|Chicago|Seattle|Boston|Austin|Denver|Atlanta|'
              r'Dallas|Houston|Phoenix|San Diego|San Jose|Portland|Miami|Washington DC|Philadelphia)\b',
              r'\b(London|Berlin|Amsterdam|Dublin|Singapore|Tokyo|Sydney|Toronto|Vancouver|Melbourne|'
              r'Paris|Munich|Barcelona|Stockholm|Copenhagen|Zurich|Dubai|Hong Kong)\b',
              r'\b(India|USA|US|United States|UK|United Kingdom|Canada|Australia|Germany|Netherlands|Singapore|UAE)\b']:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            loc = (m.group(1) if m.lastindex else m.group(0)).strip()
            loc = re.sub(r'^(?:Location|Based in|Located at|City|Current Location)[\s.:]+', '', loc, flags=re.IGNORECASE).strip()
            loc = _clean_location(loc)
            if loc and _is_valid_location(loc):
                contacts["location"] = loc; break
    return contacts


# ═══════════════════════════════════════════════════════════════
#                    NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _is_spaced_out_text(line: str) -> bool:
    if not line or len(line) < 5:
        return False
    parts = [p for p in line.split(' ') if p]
    if len(parts) < 3:
        return False
    return sum(1 for p in parts if len(p) == 1) / len(parts) > 0.5


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

    # Handle "M.SREEKANTH" or "K.Ravi" format — initial.name counts as 2 words
    if len(words) == 1:
        # Check for Initial.Name pattern: "M.SREEKANTH", "K.Ravi", "Dr.Smith"
        dot_split = re.split(r'\.', nc)
        dot_split = [p.strip() for p in dot_split if p.strip()]
        if len(dot_split) >= 2:
            # Treat as multi-word name: ["M", "SREEKANTH"]
            words = dot_split
        else:
            return False

    if not (2 <= len(words) <= 4):
        # Also allow single initial + name: len could be 2 after dot split
        if not (len(words) == 2):
            return False

    for w in words:
        s = w.strip(".-'")
        if not s or len(s) > 20:
            return False
        # Allow single-letter initials
        if len(s) == 1:
            if not s.isalpha():
                return False
            continue
        if sum(c.isalpha() for c in s) / max(len(s), 1) < 0.8:
            return False
    if sum(1 for c in nc if c.isalpha() or c.isspace() or c in '.-') / max(len(nc), 1) < 0.80:
        return False
    # Check blacklists — use the base words (strip periods)
    check_words = [w.strip(".-'") for w in words]
    if any(w.lower() in _STRONG_NAME_BLACKLIST for w in check_words if len(w) > 1):
        return False
    if sum(1 for w in check_words if w.lower() in _SOFT_NAME_BLACKLIST and len(w) > 1) >= 2:
        return False
    # Must have at least some uppercase
    if nc == nc.lower():
        return False
    return True

def _clean_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Handle ALL CAPS: "M.SREEKANTH" → "M.Sreekanth", "JOHN DOE" → "John Doe"
    if name.isupper():
        # Preserve initials: "M.SREEKANTH" → split by dot, title-case non-initials
        parts = name.split('.')
        if len(parts) >= 2 and len(parts[0].strip()) <= 2:
            # Initial.Name format
            result_parts = []
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                if len(part) <= 2:
                    # It's an initial — keep uppercase
                    result_parts.append(part.upper())
                else:
                    result_parts.append(part.title())
            name = '.'.join(result_parts)
        else:
            name = name.title()
    
    name = re.sub(r'\s*\.\s*', '.', name)  # "M. Sreekanth" → "M.Sreekanth"
    # But add space after dot if followed by a long name part: "M.Sreekanth" → "M. Sreekanth"
    name = re.sub(r'\.([A-Z][a-z]{2,})', r'. \1', name)
    
    return re.sub(r'\s+', ' ', name).strip()

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

    # Strategy 0: Extract name from lines that have "Name    Mobile/Email" format
    # e.g., "M.SREEKANTH                   Mobile  :   8309833844"
    for line in lines[:10]:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
        # Check if line contains contact labels with lots of spacing
        contact_split = re.split(r'\s{3,}', stripped)  # Split on 3+ spaces
        if len(contact_split) >= 2:
            first_part = contact_split[0].strip()
            rest = ' '.join(contact_split[1:]).lower()
            # If the rest contains contact-related words, first part might be name
            if any(kw in rest for kw in ['mobile', 'phone', 'email', 'tel', 'contact', 'cell']):
                if first_part and len(first_part) >= 3 and len(first_part) <= 40:
                    # Clean the candidate name
                    candidate = first_part.strip()
                    if candidate.isupper():
                        candidate = candidate.title()
                    if _is_valid_name(candidate):
                        return _clean_name(candidate)

    # Strategy 1: Labeled name
    for p in [r'(?:Name|Full Name|Candidate Name|Applicant Name)[\s.:]+([A-Z][a-zA-Z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-zA-Z]+){1,2})',
              r'(?:Name|Full Name)[\s.:]+([A-Z][A-Z\s]+)',
              # Also handle "Name : M.SREEKANTH" format
              r'(?:Name|Full Name|Candidate Name)[\s.:]+([A-Z]\.?\s*[A-Z][a-zA-Z]+)',
              r'(?:Name|Full Name|Candidate Name)[\s.:]+([A-Z][a-zA-Z]*\.[A-Z][a-zA-Z]+)']:
        m = re.search(p, text, re.IGNORECASE)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    # Strategy 1.5: ALL-CAPS multi-word name OR Initial.Name in first 20 lines
    for line in lines[:20]:
        line = line.strip()
        if not line or len(line) < 3 or len(line) > 50:
            continue
        # First split by large gaps to isolate name from contact info
        parts_by_space = re.split(r'\s{3,}', line)
        candidate_line = parts_by_space[0].strip() if parts_by_space else line
        if not candidate_line or len(candidate_line) < 3:
            continue

        if any(kw in candidate_line.lower() for kw in skip):
            continue
        if '@' in candidate_line or 'http' in candidate_line.lower():
            continue
        if re.match(r'^[\+\d\(\)]', candidate_line):
            continue
        if _is_spaced_out_text(candidate_line):
            continue

        # Check for Initial.Name format: "M.SREEKANTH", "K.Ravi Kumar"
        initial_name = re.match(
            r'^([A-Z]\.?\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)$',
            candidate_line
        )
        if initial_name:
            candidate = initial_name.group(1).strip()
            if candidate.isupper():
                candidate = candidate.title()
            # Handle "M.Sreekanth" → keep as-is if valid
            if _is_valid_name(candidate):
                return _clean_name(candidate)

        # Check for ALL-CAPS names
        if candidate_line.isupper() or (candidate_line == candidate_line.upper() and ' ' in candidate_line):
            words = candidate_line.split()
            if 2 <= len(words) <= 4 and all(len(w) >= 2 and w.isalpha() for w in words):
                tn = candidate_line.title()
                if _is_valid_name(tn):
                    return _clean_name(tn)

    # Strategy 2: First line that looks like a name
    for line in lines[:15]:
        line = line.strip()
        # Split by large gaps first
        parts_by_space = re.split(r'\s{3,}', line)
        candidate_line = parts_by_space[0].strip() if parts_by_space else line

        if not candidate_line or len(candidate_line) < 4 or len(candidate_line) > 45:
            continue
        if any(kw in candidate_line.lower() for kw in skip):
            continue
        if re.match(r'^\d', candidate_line) or len(re.findall(r'\d', candidate_line)) >= 5:
            continue
        if re.match(r'^[\+\d\(\)]', candidate_line) or '@' in candidate_line or 'http' in candidate_line.lower():
            continue
        if _is_spaced_out_text(candidate_line):
            continue
        if sum(1 for c in candidate_line if c.isalpha() or c.isspace() or c in '.-') / max(len(candidate_line), 1) < 0.85:
            continue
        for p in [r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',
                  r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+$', r'^[A-Z]+\s+[A-Z]+$',
                  r'^[A-Z]+\s+[A-Z]+\s+[A-Z]+$', r'^Dr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
                  r'^Mr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$', r'^Ms\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',
                  # Initial.Name patterns
                  r'^[A-Z]\.[A-Z][a-zA-Z]+$', r'^[A-Z]\.\s*[A-Z][a-zA-Z]+$',
                  r'^[A-Z]\.[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+$']:
            if re.match(p, candidate_line) and _is_valid_name(candidate_line):
                return _clean_name(candidate_line)
        words = candidate_line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w) and all(2 <= len(w) <= 15 for w in words):
            if _is_valid_name(candidate_line):
                return _clean_name(candidate_line)

    # Strategy 3: First line mixed content
    if lines:
        first = lines[0].strip()
        # Split by large gaps
        parts_by_space = re.split(r'\s{3,}', first)
        first = parts_by_space[0].strip() if parts_by_space else first
        # Try Initial.Name
        m = re.match(r'^([A-Z]\.?\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)', first)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())
        m = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})', first)
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
               'MORE ABOUT ME', 'WORK EXPERIENCE', 'TECHNICAL SKILLS'}
    for line in lines[:10]:
        line = line.strip()
        # Split by large gaps
        parts_by_space = re.split(r'\s{3,}', line)
        candidate_line = parts_by_space[0].strip() if parts_by_space else line
        if candidate_line and candidate_line.isupper() and 3 <= len(candidate_line) <= 40:
            words = candidate_line.split()
            if 1 <= len(words) <= 4 and not any(h in candidate_line for h in headers):
                # Handle single-word Initial.Name
                if len(words) == 1 and '.' in candidate_line:
                    tn = candidate_line.title()
                    if _is_valid_name(tn):
                        return tn
                elif len(words) >= 2:
                    tn = candidate_line.title()
                    if _is_valid_name(tn):
                        return tn

    # Strategy 7: Name after "Regards"
    for p in [r'(?:Regards|Sincerely|Yours\s+(?:truly|faithfully|sincerely)|Thank\s*(?:you|s)|Best\s+regards|Kind\s+regards|Warm\s+regards|Respectfully|Cordially)\s*[,.]?\s*\n\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})']:
        m = re.search(p, text, re.IGNORECASE)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    # Strategy 8: Declaration
    for p in [r'(?:Declaration|I\s+hereby\s+declare).+?(?:Date|Location|Place|Regards)\s*[:\s]*\w*\s*\n\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\s*$']:
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if m and _is_valid_name(m.group(1).strip()):
            return _clean_name(m.group(1).strip())

    return ""


# ═══════════════════════════════════════════════════════════════
#        SECTION EXTRACTION & WORK EXPERIENCE
# ═══════════════════════════════════════════════════════════════

def _extract_work_section(text: str) -> str:
    if not text: return ""
    lines = text.split('\n')
    work_start, work_end = -1, len(lines)
    for i, line in enumerate(lines):
        cleaned = re.sub(r'[^a-z\s]', '', line.strip().lower()).strip()
        if cleaned in _WORK_SECTION_HEADERS or any(cleaned == h for h in _WORK_SECTION_HEADERS):
            work_start = i + 1; break
        for header in _WORK_SECTION_HEADERS:
            if header in cleaned and len(cleaned) < len(header) + 15:
                work_start = i + 1; break
        if work_start != -1: break
    if work_start == -1: return ""
    for i in range(work_start, len(lines)):
        cleaned = re.sub(r'[^a-z\s&]', '', lines[i].strip().lower()).strip()
        if cleaned in _NON_WORK_SECTION_HEADERS:
            work_end = i; break
        for header in _NON_WORK_SECTION_HEADERS:
            if header == cleaned or (header in cleaned and len(cleaned) < len(header) + 10):
                work_end = i; break
        if work_end != len(lines): break
    return '\n'.join(lines[work_start:work_end]).strip()


def _is_role_line(line: str) -> bool:
    if not line or not line.strip(): return False
    stripped = line.strip()
    if stripped.startswith(('•', '-', '*', '\u2022', '\u25e6', '\u2023', '\u2043')): return False
    if _looks_like_sentence_or_description(stripped): return False
    if len(stripped) > _MAX_ROLE_LENGTH: return False
    role_prefix = re.match(r'^(?:Role|Position|Title|Designation)\s*[:]\s*(.+?)(?:\s*[\u2022•|]|$)', stripped, re.IGNORECASE)
    if role_prefix:
        candidate = role_prefix.group(1).strip()
        if candidate and not _looks_like_sentence_or_description(candidate):
            for pattern in _JOB_TITLE_PATTERNS:
                if re.search(pattern, candidate, re.IGNORECASE): return True
    for pattern in _JOB_TITLE_PATTERNS:
        match = re.search(pattern, stripped, re.IGNORECASE)
        if match:
            words_before = stripped[:match.start()].strip()
            if words_before:
                wb = words_before.split()
                if len(wb) > 3: continue
                tp = {'senior', 'junior', 'lead', 'principal', 'staff', 'chief', 'head', 'vp', 'associate', 'assistant',
                      'deputy', 'sr', 'jr', 'sr.', 'jr.', 'python', 'java', 'data', 'ml', 'ai', 'cloud', 'devops',
                      'full', 'stack', 'front', 'back', 'end', 'full-stack', 'frontend', 'backend', 'mobile', 'web',
                      'qa', 'test', 'embedded', 'network', 'system', 'security', 'database', 'platform', 'technical',
                      'product', 'project', 'program', 'operations', 'medical', 'record', 'business', 'intelligence',
                      'machine', 'learning', 'deep', 'nlp', 'computer', 'vision', 'site', 'reliability', '-', '\u2013', '/', '&'}
                if not all(w.lower().strip('.,()') in tp for w in wb): continue
            return True
    if re.match(r'^[^•\n]+\s*[\u2022•]\s*(?:Full[-\s]?time|Part[-\s]?time|Internship|Contract|Freelance)', stripped, re.IGNORECASE):
        pre = re.split(r'\s*[\u2022•]\s*', stripped)[0].strip()
        if pre and not _looks_like_sentence_or_description(pre) and len(pre) < 60: return True
    return False


def _is_company_line(line: str) -> bool:
    if not line or not line.strip(): return False
    stripped = line.strip()
    if stripped.startswith(('•', '-', '*', '\u2022', '\u25e6', '\u2023', '\u2043')): return False
    if _looks_like_sentence_or_description(stripped): return False
    if len(stripped) > _MAX_COMPANY_LENGTH: return False
    if re.search(r'(?:Pvt|Private|Ltd|Limited|Inc|Corp|LLC|LLP|Co\.|Company|Group|Solutions|Technologies|Consulting|Services|Systems|Soft(?:ware)?|Tech|Labs?|Studio|Digital|Media|Networks|Communications|Industries|Enterprises|Associates|Partners|Foundation|Organization|Institute|Division)\b', stripped, re.IGNORECASE):
        return True
    if re.match(r'^[^•]+\s*[\u2022•]\s*[A-Z][a-z]+', stripped): return True
    return False


def _is_date_line(line: str) -> bool:
    if not line or not line.strip(): return False
    stripped = line.strip()
    for pat in [r'\d{1,2}[/\-\.]\d{4}\s*[-\u2013\u2014]\s*(?:\d{1,2}[/\-\.]\d{4}|[Pp]resent|[Cc]urrent)',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*\d{4}\s*[-\u2013\u2014]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*\d{4}|[Pp]resent|[Cc]urrent)',
                r'(?:19|20)\d{2}\s*[-\u2013\u2014]\s*(?:(?:19|20)\d{2}|[Pp]resent|[Cc]urrent)']:
        m = re.search(pat, stripped, re.IGNORECASE)
        if m:
            remainder = stripped[:m.start()].strip() + stripped[m.end():].strip()
            if len(remainder.strip('•\u2022-|,.: ')) < len(stripped) * 0.3: return True
    return False


def _extract_date_range(text: str) -> Tuple[str, str]:
    if not text: return "", ""
    for pat in [r'(\d{1,2}[/\-\.]\d{4})\s*[-\u2013\u2014]\s*(\d{1,2}[/\-\.]\d{4}|[Pp]resent|[Cc]urrent(?:ly)?|[Nn]ow|[Oo]ngoing)',
                r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*\d{4})\s*[-\u2013\u2014]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*\d{4}|[Pp]resent|[Cc]urrent(?:ly)?|[Nn]ow|[Oo]ngoing)',
                r'((?:19|20)\d{2})\s*[-\u2013\u2014]\s*((?:19|20)\d{2}|[Pp]resent|[Cc]urrent(?:ly)?|[Nn]ow|[Oo]ngoing)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1), m.group(2)
    return "", ""


def _extract_work_experience_from_text(text: str) -> List[Dict]:
    if not text: return []
    work_entries: List[Dict] = []
    seen_jobs: Set[str] = set()

    def _nk(r, c):
        return f"{re.sub(r'[^a-z0-9]', '', r.lower())[:25]}_{re.sub(r'[^a-z0-9]', '', c.lower())[:25]}"

    def _add(role, company, start, end, loc="", emp_type="Full-time"):
        role, company = _clean_role_title(role), _clean_company_name(company)
        if not _is_valid_work_entry({"title": role, "company": company}): return False
        key = _nk(role, company)
        if key in seen_jobs: return False
        seen_jobs.add(key)
        dur = _calc_duration_from_dates(start, end)
        ce = end.strip() if end.strip() and end.strip().lower() not in ['', '-', '\u2013', '\u2014'] else "Present"
        work_entries.append({"title": role, "company": company, "location": loc, "start_date": start.strip(),
                             "end_date": ce, "duration_years": dur, "type": emp_type, "description": "",
                             "key_achievements": [], "technologies_used": []})
        return True

    work_section = _extract_work_section(text)
    if work_section:
        ws_lines = work_section.split('\n')
        i = 0
        while i < len(ws_lines):
            line = ws_lines[i].strip()
            if not line or line.startswith(('•', '-', '*', '\u2022', '\u25e6', '\u2023', '\u2043')):
                i += 1; continue
            # Strategy A: Labeled format
            rlm = re.match(r'^(?:Role|Position|Title|Designation)\s*[:]\s*(.+?)(?:\s*[\u2022•|]|$)', line, re.IGNORECASE)
            clm = re.match(r'^(?:Company|Organization|Employer|Client)\s*[:]\s*(.+?)(?:\s*[\u2022•|]|$)', line, re.IGNORECASE)
            if rlm or clm:
                role, company, location, start_date, end_date, emp_type = "", "", "", "", "", "Full-time"
                j = i
                while j < len(ws_lines):
                    cl = ws_lines[j].strip()
                    if not cl: j += 1; continue
                    if cl.startswith(('•', '-', '*', '\u2022', '\u25e6', '\u2023')): break
                    rl = re.match(r'^(?:Role|Position|Title|Designation)\s*[:]\s*(.+)', cl, re.IGNORECASE)
                    cm = re.match(r'^(?:Company|Organization|Employer|Client)\s*[:]\s*(.+)', cl, re.IGNORECASE)
                    dl = re.match(r'^(?:Duration|Period|Tenure|Dates?|From|Timeline)\s*[:]\s*(.+)', cl, re.IGNORECASE)
                    ll = re.match(r'^(?:Location|City|Place)\s*[:]\s*(.+)', cl, re.IGNORECASE)
                    tlm = re.match(r'^(?:Type|Employment\s*Type|Mode)\s*[:]\s*(.+)', cl, re.IGNORECASE)
                    if rl: role = _clean_role_title(rl.group(1).strip())
                    elif cm: company = _clean_company_name(cm.group(1).strip())
                    elif dl:
                        s, e = _extract_date_range(dl.group(1).strip())
                        if s: start_date, end_date = s, e
                    elif ll: location = ll.group(1).strip()
                    elif tlm: emp_type = tlm.group(1).strip()
                    else:
                        if _is_date_line(cl):
                            s, e = _extract_date_range(cl)
                            if s and not start_date: start_date, end_date = s, e
                        elif _is_role_line(cl) and role: break
                        elif _is_company_line(cl) and not company: company = _clean_company_name(cl)
                        elif j > i + 1: break
                    j += 1
                if role or company:
                    _add(role or "Not specified", company or "Not specified", start_date, end_date, location, emp_type)
                i = j; continue
            # Strategy B: Standard multi-line
            if not _is_role_line(line):
                i += 1; continue
            role_line, emp_type = line, "Full-time"
            tm = re.search(r'\s*[\u2022•\-|,]\s*(Full[-\s]?time|Part[-\s]?time|Internship|Contract|Freelance|Temporary)\s*$', role_line, re.IGNORECASE)
            if tm:
                emp_type = tm.group(1).strip()
                role_line = role_line[:tm.start()].strip().rstrip('•\u2022-|, ')
            role_line = _clean_role_title(role_line)
            if not role_line or not _is_valid_role_title(role_line):
                i += 1; continue
            rs, re_ = _extract_date_range(line)
            if rs:
                role_line = re.sub(r'\s*\d{1,2}[/\-\.]\d{4}\s*[-\u2013\u2014].*$', '', role_line).strip().rstrip('•\u2022-|, ')
            company, location, start_date, end_date = "", "", rs, re_
            ll_limit, j, found_co, found_dt = min(i + 5, len(ws_lines)), i + 1, False, bool(rs)
            while j < ll_limit:
                nl = ws_lines[j].strip()
                if not nl: j += 1; continue
                if nl.startswith(('•', '-', '*', '\u2022', '\u25e6', '\u2023', '\u2043')): break
                if _is_role_line(nl) and found_co: break
                if _is_date_line(nl) and not found_dt:
                    s, e = _extract_date_range(nl)
                    if s: start_date, end_date, found_dt = s, e, True
                    j += 1; continue
                if not found_co:
                    ls, le = _extract_date_range(nl)
                    if ls:
                        dm = re.search(r'\d{1,2}[/\-\.]\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', nl, re.IGNORECASE)
                        pd = nl[:dm.start()].strip().rstrip('•\u2022-|, ') if dm else nl
                        if pd and not _looks_like_sentence_or_description(pd):
                            parts = re.split(r'\s*[\u2022•]\s*', pd)
                            parts = [p.strip() for p in parts if p.strip()]
                            if parts:
                                company = parts[0]
                                if len(parts) > 1: location = parts[-1]
                        if not found_dt: start_date, end_date, found_dt = ls, le, True
                        found_co = True
                    elif _is_company_line(nl) or (not _is_role_line(nl) and not _looks_like_sentence_or_description(nl) and len(nl) < _MAX_COMPANY_LENGTH):
                        parts = [p.strip() for p in re.split(r'\s*[\u2022•]\s*', nl) if p.strip()]
                        parts = [p for p in parts if p.lower() not in ('full-time', 'part-time', 'internship', 'contract', 'freelance', 'temporary', 'full time', 'part time')]
                        if parts:
                            company = parts[0]
                            for p in parts[1:]:
                                s, e = _extract_date_range(p)
                                if s and not found_dt: start_date, end_date, found_dt = s, e, True
                                elif any(lw in p.lower() for lw in LOCATION_WORDS) or len(p.split()) <= 2: location = p
                        found_co = True
                j += 1
            if role_line and (company or start_date):
                company = _clean_company_name(company)
                if company and not location:
                    lm = re.search(r'\s*[\u2022•,]\s*(Pondicherry|Puducherry|Chennai|Hyderabad|Bangalore|Bengaluru|Mumbai|Delhi|Pune|Kolkata|Noida|Gurgaon|Gurugram|India)\s*$', company, re.IGNORECASE)
                    if lm:
                        location = lm.group(1).strip()
                        company = company[:lm.start()].strip().rstrip(',•\u2022 ')
                _add(role_line, company, start_date, end_date, location, emp_type)
                i = j
            else:
                i += 1
    # PHASE 2: Narrative patterns
    if not work_entries:
        MY = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s*\d{4}'
        ND = r'\d{1,2}[/\-\.]\d{4}'
        NDR = r'\d{4}[/\-\.]\d{1,2}'
        MA = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)['\u2019]\s*\d{2,4}"
        PK = r'(?:[Pp]resent|[Cc]urrent(?:ly)?|[Nn]ow|[Oo]ngoing|[Tt]ill\s*[Dd]ate)'
        JY = r'(?:19|20)\d{2}'
        AD = f'(?:{MY}|{ND}|{NDR}|{MA}|{PK}|{JY})'
        DS = r'\s*(?:to|till|until|[-\u2013\u2014])\s*'
        for m in re.finditer(r'(?:currently\s+)?(?:working|employed|serving)\s+(?:as\s+)?(.+?)\s+(?:with|at|in|for)\s+(.+?)\s*(?:since|from)\s+(' + AD + r')' + DS + r'(' + AD + r')(?:\s*[.\n]|$)', text, re.IGNORECASE):
            if _is_valid_role_title(_clean_role_title(m.group(1).strip())):
                _add(m.group(1).strip(), m.group(2).strip(), m.group(3), m.group(4))
        for m in re.finditer(r'(?:worked|employed|served|joined|was)\s+(?:as\s+)?(.+?)\s+(?:at|with|in|for)\s+(.+?)\s*(?:from|since)\s+(' + AD + r')' + DS + r'(' + AD + r')(?:[.\n,;]|$)', text, re.IGNORECASE):
            if _is_valid_role_title(_clean_role_title(m.group(1).strip())):
                _add(m.group(1).strip(), m.group(2).strip(), m.group(3), m.group(4))
    return work_entries


# ═══════════════════════════════════════════════════════════════
#                    EDUCATION EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_education_regex(text: str) -> List[Dict]:
    education: List[Dict] = []
    if not text: return education
    degree_patterns = [
        (r'((?:PG\.?\s*)?Bachelor\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Commerce|Business|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'((?:PG\.?\s*)?B\.?\s*(?:Tech|E|Sc|A|Com|B\.?A|S|CA|BA|Arch|Pharm|Ed|Des)\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'((?:PG\.?\s*)?B\.?Tech|BTech|B\.?E\.?|BE)\s*[-\u2013in\s]*([\w\s,&]+)?', 'B'),
        (r'(BCA|B\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'), (r'(BBA|B\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'(B\.?Com|BCom)\s*(?:in\s*)?([\w\s,&]+)?', 'B'),
        (r'(Master\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Business\s*Administration|Computer\s*Applications?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(M\.?\s*(?:Tech|E|Sc|A|B\.?A|S|CA|BA|Phil|Ed|Des)\.?|MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(M\.?Tech|MTech|M\.?E\.?|ME|M\.?S\.?|MS)\s*[-\u2013in\s]*([\w\s,&]+)?', 'M'),
        (r'(MCA|M\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'), (r'(MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(PG\s*(?:Diploma|Degree)?)\s*(?:in\s*)?([\w\s,&]+)?', 'M'),
        (r'(Ph\.?\s*D\.?|Doctorate|Doctor\s*of\s*Philosophy)\s*(?:in\s*)?([\w\s,&]+)?', 'P'),
        (r'(Diploma)\s*(?:in\s*)?([\w\s,&]+)?', 'D'), (r'(Polytechnic)\s*(?:in\s*)?([\w\s,&]+)?', 'D'),
        (r'(ITI|I\.?T\.?I\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'D'),
        (r'(Higher\s*Secondary|HSC|12th|XII|Intermediate|Senior\s*Secondary|\+2|Plus\s*Two)', 'S'),
        (r'(Secondary|SSC|SSLC|10th|X|Matriculation|High\s*School)', 'S'),
    ]
    inst_pats = [r'([A-Z][A-Za-z\s\.\']+(?:University|College|Institute|School|Academy|Polytechnic))',
                 r'((?:IIT|NIT|IIIT|BITS|VIT|SRM|Amity|Manipal|LPU|KIIT|MIT|Stanford|Harvard|Cambridge|Oxford)\s*[,]?\s*[\w\s]*)',
                 r'(?:from|at|,)\s+([A-Z][A-Za-z\s\.\']{5,60})']
    gpa_pats = [r'(?:GPA|CGPA|CPI|Grade)[:\s]*(\d+\.?\d*)', r'(\d{1,2}(?:\.\d+)?)\s*%',
                r'(First\s*Class(?:\s*with\s*Distinction)?|Distinction|Honors?|Honours?)']
    yr_pats = [r'((?:19|20)\d{2})\s*[-\u2013to]+\s*((?:19|20)\d{2}|Present|Current|Expected|Pursuing)',
               r'(?:Class\s*of|Batch|Graduated?|Expected)[:\s]*((?:19|20)\d{2})', r'((?:19|20)\d{2})']
    tu = text.upper()
    es = -1
    for mk in ['EDUCATION', 'ACADEMIC', 'QUALIFICATION', 'EDUCATIONAL BACKGROUND']:
        idx = tu.find(mk)
        if idx != -1: es = idx; break
    esec = text[es:] if es != -1 else text
    if es != -1:
        for mk in ['EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILL', 'PROJECT', 'CERTIFICATION', 'AWARD', 'ACHIEVEMENT']:
            ei = esec.upper().find(mk, 50)
            if ei != -1: esec = esec[:ei]; break
    found: List[Dict] = []
    for pat, dt in degree_patterns:
        try:
            for match in re.finditer(pat, esec, re.IGNORECASE):
                dn = match.group(1).strip() if match.group(1) else ""
                fld = ""
                if match.lastindex >= 2 and match.group(2):
                    fld = re.sub(r'\s+', ' ', match.group(2).strip())
                    if len(fld) > 100: fld = fld[:100]
                    if any(sw in fld.lower() for sw in ['university', 'college', 'institute', 'school', 'from', 'at', 'gpa', 'cgpa']): fld = ""
                ctx = esec[max(0, match.start() - 30):min(len(esec), match.end() + 250)]
                entry = {"degree": dn, "field": fld, "institution": "", "year": "", "gpa": "", "location": "", "achievements": []}
                for ip in inst_pats:
                    try:
                        im = re.search(ip, ctx, re.IGNORECASE)
                        if im:
                            inst = re.sub(r'\s+', ' ', im.group(1).strip())
                            if len(inst) > 5 and inst.lower() not in ['the', 'and', 'for', 'with', 'from']:
                                entry["institution"] = inst; break
                    except re.error: continue
                for yp in yr_pats:
                    try:
                        ym = re.search(yp, ctx, re.IGNORECASE)
                        if ym: entry["year"] = ym.group(0).strip(); break
                    except re.error: continue
                for gp in gpa_pats:
                    try:
                        gm = re.search(gp, ctx, re.IGNORECASE)
                        if gm: entry["gpa"] = gm.group(0).strip(); break
                    except re.error: continue
                found.append(entry)
        except re.error: continue
    seen: Set[str] = set()
    for edu in found:
        if not _is_valid_education_entry(edu): continue
        key = f"{_normalize_degree_key(edu.get('degree', '').lower())}_{edu.get('institution', '').lower()[:20]}"
        if key not in seen and edu.get('degree'):
            seen.add(key); education.append(edu)
    return _deduplicate_education_entries(education)


# ═══════════════════════════════════════════════════════════════
#                    SKILLS EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_skills_regex(text: str) -> Dict:
    skills: Dict[str, List[str]] = {"programming_languages": [], "frameworks_libraries": [], "ai_ml_tools": [],
                                     "cloud_platforms": [], "databases": [], "devops_tools": [],
                                     "visualization": [], "other_tools": [], "soft_skills": []}
    if not text: return skills
    tl = text.lower()
    cats = {
        "programming_languages": ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'golang', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r', 'matlab', 'perl', 'bash', 'shell', 'powershell', 'sql', 'html', 'css', 'dart', 'lua'],
        "frameworks_libraries": ['react', 'angular', 'vue', 'nextjs', 'next.js', 'nodejs', 'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'springboot', 'laravel', 'rails', 'asp.net', '.net', 'jquery', 'bootstrap', 'tailwind', 'flutter', 'react native'],
        "ai_ml_tools": ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy', 'opencv', 'nltk', 'spacy', 'huggingface', 'transformers', 'langchain', 'spark', 'pyspark', 'hadoop', 'kafka', 'xgboost', 'lightgbm'],
        "cloud_platforms": ['aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'heroku', 'firebase', 'vercel', 'netlify', 'lambda', 's3', 'ec2'],
        "databases": ['mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'dynamodb', 'cassandra', 'oracle', 'sql server', 'sqlite', 'mariadb', 'neo4j'],
        "devops_tools": ['docker', 'kubernetes', 'k8s', 'jenkins', 'terraform', 'ansible', 'prometheus', 'grafana', 'nginx', 'linux', 'unix', 'ubuntu', 'github actions', 'gitlab ci'],
        "visualization": ['power bi', 'tableau', 'looker', 'plotly', 'matplotlib', 'seaborn', 'excel', 'd3.js'],
        "other_tools": ['git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'figma', 'postman', 'swagger', 'graphql', 'rest', 'selenium', 'cypress', 'jest', 'pytest', 'webpack', 'vite', 'npm', 'yarn', 'maven', 'gradle'],
        "soft_skills": ['leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking', 'analytical', 'creativity', 'adaptability', 'time management', 'project management', 'presentation', 'negotiation', 'collaboration', 'mentoring'],
    }
    for cat, sl in cats.items():
        for s in sl:
            if re.search(rf'\b{re.escape(s)}\b', tl):
                d = s.upper() if len(s) <= 3 and s not in ('go', 'r') else s.upper() if s in ('c++', 'c#') else s.title()
                skills[cat].append(d)
    for cat in skills:
        seen: Set[str] = set()
        skills[cat] = [s for s in skills[cat] if s.lower() not in seen and not seen.add(s.lower())]
    return skills


# ═══════════════════════════════════════════════════════════════
#                    CONTACT VALIDATION & MERGE
# ═══════════════════════════════════════════════════════════════

def _is_valid_field(field: str, value: str) -> bool:
    if not value: return False
    v = str(value).strip()
    if v.lower().strip() in ['n/a', 'na', 'none', 'not available', 'not specified', 'unknown', 'null', '-', '\u2014', '',
                              'candidate', 'your name', 'email@example.com', 'xxx', '000', 'your email', 'your phone',
                              'first last', 'firstname lastname', 'name', 'full name', '[name]', '<name>', '(name)',
                              'enter name', 'type name']:
        return False
    if field == 'email':
        if '@' not in v or '.' not in v.split('@')[-1]: return False
        lp = v.split('@')[0]
        return not lp.isdigit() and any(c.isalpha() for c in lp)
    if field == 'phone': return len(re.sub(r'[^\d]', '', v)) >= 10
    if field == 'name': return _is_valid_name(v)
    if field in ('linkedin', 'github'): return len(v) > 5
    if field == 'portfolio':
        vl = v.lower()
        if any(d in vl for d in _EMPLOYER_DOMAINS): return False
        if re.search(r'[A-Z]{3,}', v.split('/')[-1] if '/' in v else ''): return False
        return len(v) >= 5
    if field == 'location': return _is_valid_location(v)
    if field == 'address': return _is_valid_address(v)
    return len(v) >= 2


def _merge_contacts(llm: Dict, regex: Dict, doc: Optional[Dict] = None) -> Dict:
    merged: Dict[str, str] = {}
    if doc is None: doc = {}
    for f in ['name', 'email', 'phone', 'address', 'linkedin', 'github', 'portfolio', 'location']:
        lv, dv, rv = str(llm.get(f, "")).strip(), str(doc.get(f, "")).strip(), str(regex.get(f, "")).strip()
        if f == 'email':
            lv, dv, rv = _clean_email(lv), _clean_email(dv), _clean_email(rv)
        if f in ('location', 'address'):
            lv, dv, rv = _clean_location(lv), _clean_location(dv), _clean_location(rv)
        if lv and _is_valid_field(f, lv): merged[f] = lv
        elif dv and _is_valid_field(f, dv): merged[f] = dv
        elif rv and _is_valid_field(f, rv): merged[f] = rv
        else: merged[f] = ""
    return merged


def _basic_fallback(text, contacts, name, education, skills):
    return {"name": name or contacts.get("name", ""), "email": contacts.get("email", ""),
            "phone": contacts.get("phone", ""), "address": contacts.get("address", ""),
            "linkedin": contacts.get("linkedin", ""), "github": contacts.get("github", ""),
            "portfolio": contacts.get("portfolio", ""), "location": contacts.get("location", ""),
            "current_role": "", "current_company": "", "total_experience_years": 0,
            "professional_summary": text[:500] if text else "", "specializations": [],
            "skills": skills, "work_history": [], "education": education,
            "certifications": [], "awards": [], "projects": [], "publications": [],
            "volunteer": [], "languages": [], "interests": []}


# ═══════════════════════════════════════════════════════════════
#                    MAIN PARSING FUNCTION
# ═══════════════════════════════════════════════════════════════

def parse_resume_with_llm(resume_text: str, groq_api_key: str,
                          model_id: str = "llama-3.1-8b-instant",
                          doc_contacts: Optional[Dict] = None) -> Dict:
    if not resume_text or len(resume_text.strip()) < 50:
        return _basic_fallback(resume_text, {}, "", [], {})

    regex_contacts = _extract_contacts_regex(resume_text)
    regex_name = _extract_name_from_text(resume_text)
    regex_education = _extract_education_regex(resume_text)
    regex_skills = _extract_skills_regex(resume_text)

    if doc_contacts is None:
        doc_contacts = {}
        try:
            from document_processor import extract_contacts_from_text
            doc_contacts = extract_contacts_from_text(resume_text)
        except ImportError: pass

    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": [
        {"role": "system", "content": "You are a resume parsing expert. Current year is 2026. Return ONLY valid JSON. No markdown. "
         "NEVER create education from awards. Only extract what is ACTUALLY WRITTEN. "
         "title must be SHORT job title. NEVER put sentences as title or company. "
         "For location return ONLY city/state/country. For portfolio only personal websites NOT employer URLs."},
        {"role": "user", "content": PARSE_PROMPT.format(resume_text=resume_text[:12000])}
    ], "temperature": 0.05, "max_tokens": 6000}

    parsed = None
    for try_model in list(dict.fromkeys([model_id, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"])):
        payload["model"] = try_model
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"): content = content.split("\n", 1)[-1]
                if content.endswith("```"): content = content.rsplit("```", 1)[0]
                parsed = json.loads(content.strip()); break
            elif resp.status_code == 429: time.sleep(1)
        except json.JSONDecodeError:
            try:
                jm = re.search(r'\{[\s\S]*\}', content)
                if jm: parsed = json.loads(jm.group()); break
            except Exception: pass
        except Exception: continue
    if not parsed:
        parsed = _basic_fallback(resume_text, regex_contacts, regex_name, regex_education, regex_skills)

    # Education validation
    el = parsed.get("education", [])
    parsed["education"] = _filter_valid_education(el if isinstance(el, list) else [el] if el else [])
    parsed["education"] = _validate_education_against_text(parsed["education"], resume_text)

    # Contact merge + cleaning
    mc = _merge_contacts(parsed, regex_contacts, doc_contacts)
    for f, v in mc.items():
        if v and (not parsed.get(f) or not _is_valid_field(f, parsed.get(f, ""))):
            parsed[f] = v
    if parsed.get("email"): parsed["email"] = _clean_email(parsed["email"])
    if parsed.get("portfolio") and not _is_valid_field("portfolio", parsed["portfolio"]): parsed["portfolio"] = ""
    if parsed.get("location"):
        parsed["location"] = _clean_location(parsed["location"])
        if not _is_valid_location(parsed["location"]): parsed["location"] = ""
    if parsed.get("address"):
        ca = _clean_location(parsed["address"])
        if ca and _is_valid_address(ca): parsed["address"] = ca
        elif not _is_valid_address(parsed["address"]): parsed["address"] = ""

    # Name
    cn = parsed.get("name", "")
    if not cn or not _is_valid_field("name", cn):
        for c in [doc_contacts.get("name", ""), regex_name, parsed.get("name", "")]:
            if c and _is_valid_field("name", c): parsed["name"] = _clean_name(c); break
        if not parsed.get("name") or not _is_valid_field("name", parsed.get("name", "")): parsed["name"] = "Unknown Candidate"

    # Education merge
    le = parsed.get("education", [])
    if not isinstance(le, list): le = []
    if regex_education:
        if not le: parsed["education"] = regex_education
        else:
            ldk: Set[str] = {_normalize_degree_key(str(e.get("degree", ""))) for e in le if isinstance(e, dict)} - {""}
            for re_e in regex_education:
                if isinstance(re_e, dict):
                    dk = _normalize_degree_key(str(re_e.get("degree", "")))
                    if dk and dk not in ldk: le.append(re_e); ldk.add(dk)
            parsed["education"] = _deduplicate_education_entries(le)
    if not parsed.get("education"):
        tlc = resume_text.lower()
        if any(i in tlc for i in ['b.e', 'b.tech', 'btech', 'm.tech', 'mtech', 'b.sc', 'm.sc', 'bca', 'mca', 'mba',
                                   'bachelor', 'master', 'phd', 'diploma', 'degree', 'university', 'college',
                                   'graduated', 'pg.', 'pg ', 'engineering']):
            retry = _extract_education_regex(resume_text)
            if retry: parsed["education"] = retry

    # Skills merge
    ls = parsed.get("skills", {})
    if not isinstance(ls, dict): ls = {"other_tools": ls} if isinstance(ls, list) else {}
    for cat, rsl in regex_skills.items():
        if not rsl: continue
        if cat in ls and isinstance(ls[cat], list):
            existing = set(s.lower() for s in ls[cat] if s)
            for s in rsl:
                if s and s.lower() not in existing: ls[cat].append(s)
        elif rsl: ls[cat] = rsl
    parsed["skills"] = ls

    # Work experience — validate LLM entries
    work_history = parsed.get("work_history", parsed.get("experience", []))
    if not isinstance(work_history, list): work_history = []
    validated = []
    for j in work_history:
        if not isinstance(j, dict): continue
        title = _clean_role_title(str(j.get("title", "") or j.get("role", "") or j.get("position", "")).strip())
        company = _clean_company_name(str(j.get("company", "") or j.get("organization", "")).strip())
        if not _is_valid_role_title(title): continue
        if company and not _is_valid_company_name(company): continue
        if not title and not company: continue
        j["title"], j["company"] = title, company
        validated.append(j)
    work_history = validated

    # Regex work + intelligent merge
    regex_work = _extract_work_experience_from_text(resume_text)
    if regex_work:
        if not work_history: work_history = regex_work
        else:
            lck: Set[str] = {re.sub(r'[^a-z0-9]', '', str(j.get('company', '')).lower())[:20] for j in work_history} - {""}
            for rj in regex_work:
                co = re.sub(r'[^a-z0-9]', '', str(rj.get('company', '')).lower())[:20]
                if co and co not in lck: work_history.append(rj); lck.add(co)
    parsed["work_history"] = work_history

    # Dedup by company
    final_work: List[Dict] = []
    seen_co: Dict[str, int] = {}
    for j in parsed.get("work_history", []):
        if not isinstance(j, dict) or not _is_valid_work_entry(j): continue
        co = _clean_company_name(j.get('company', '') or j.get('organization', '') or '')
        ti = j.get('title', '') or j.get('role', '') or j.get('position', '') or ''
        cw = [w for w in re.sub(r'[^a-z0-9\s]', '', co.lower()).split() if w not in LOCATION_WORDS and len(w) > 1]
        cn = ''.join(cw)[:20] or re.sub(r'[^a-z0-9]', '', ti.lower())[:20]
        if not cn: final_work.append(j); continue
        ss, es = str(j.get('start_date', j.get('from', ''))), str(j.get('end_date', j.get('to', '')))
        idur = _calc_duration_from_dates(ss, es) if ss else 0.0
        q = idur + (1.0 if ss and es else 0) + (0.5 if ss else 0) + (2.0 if ti and ti != "Not specified" else 0) + (2.0 if co and co != "Not specified" else 0) + (0.5 if j.get('description') else 0) + (0.5 if j.get('key_achievements') else 0)
        if idur > 0: j["duration_years"] = idur
        if cn in seen_co:
            ex = final_work[seen_co[cn]]
            exs, exe = str(ex.get('start_date', ex.get('from', ''))), str(ex.get('end_date', ex.get('to', '')))
            exd = _calc_duration_from_dates(exs, exe) if exs else 0.0
            exq = exd + (1.0 if exs and exe else 0) + (0.5 if exs else 0) + (2.0 if (ex.get('title', '') or ex.get('role', '')) and (ex.get('title', '') or ex.get('role', '')) != "Not specified" else 0) + (2.0 if (ex.get('company', '') or ex.get('organization', '')) and (ex.get('company', '') or ex.get('organization', '')) != "Not specified" else 0) + (0.5 if ex.get('description') else 0) + (0.5 if ex.get('key_achievements') else 0)
            if q > exq: final_work[seen_co[cn]] = j
        else: seen_co[cn] = len(final_work); final_work.append(j)
    parsed["work_history"] = final_work

    # Recalculate total experience
    total_exp = 0.0
    for job in final_work:
        if not isinstance(job, dict): continue
        ss, es = str(job.get("start_date", job.get("from", ""))), str(job.get("end_date", job.get("to", "")))
        if ss:
            dur = _calc_duration_from_dates(ss, es)
            if dur > 0: job["duration_years"] = dur; total_exp += dur; continue
        try: total_exp += float(job.get("duration_years", 0) or 0)
        except (ValueError, TypeError): pass
    parsed["total_experience_years"] = round(total_exp, 1)

    # Extract from summary if still 0
    if parsed.get("total_experience_years", 0) == 0:
        for ct in [parsed.get("professional_summary", "") or parsed.get("summary", "") or "", resume_text[:3000]]:
            if not ct: continue
            for ep in [r'(\d+)\s*\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', r'over\s+(\d+)\s*years?',
                       r'more\s+than\s+(\d+)\s*years?', r'nearly\s+(\d+)\s*years?', r'around\s+(\d+)\s*years?',
                       r'experience\s*(?:of\s*)?(\d+)\s*\+?\s*years?']:
                em = re.search(ep, ct.lower())
                if em:
                    yrs = int(em.group(1))
                    if 1 <= yrs <= 40: parsed["total_experience_years"] = float(yrs); break
            if parsed.get("total_experience_years", 0) > 0: break

    # Fallback work entry from header
    if not parsed.get("work_history"):
        for line in resume_text.strip().split('\n')[:10]:
            line = line.strip()
            if not line or len(line) < 5 or _is_spaced_out_text(line): continue
            rm = re.match(r'^(.+?(?:Developer|Engineer|Analyst|Scientist|Manager|Designer|Architect|Consultant|Lead|Specialist|Administrator|Coordinator|Executive|Officer))\s*[,\-\u2013\u2014|]+\s*(.+?)(?:\s*[,\-\u2013\u2014|]+\s*.+)?$', line, re.IGNORECASE)
            if rm:
                role, company = _clean_role_title(rm.group(1).strip()), _clean_company_name(rm.group(2).strip())
                if company and len(company) >= 2 and _is_valid_role_title(role) and _is_valid_company_name(company) and _is_valid_work_entry({"title": role, "company": company}):
                    ey = float(parsed.get("total_experience_years", 0) or 0)
                    parsed["work_history"] = [{"title": role, "company": company, "location": "", "start_date": "", "end_date": "Present", "duration_years": ey if ey > 0 else 0, "type": "Full-time", "description": "", "key_achievements": [], "technologies_used": []}]
                    if not parsed.get("current_role"): parsed["current_role"] = role
                    if not parsed.get("current_company"): parsed["current_company"] = company
                    break

    # Current role/company
    if not parsed.get("current_role") or not parsed.get("current_company"):
        wh = parsed.get("work_history", [])
        if isinstance(wh, list) and wh and isinstance(wh[0], dict):
            if not parsed.get("current_role"): parsed["current_role"] = wh[0].get("title", "") or wh[0].get("role", "") or wh[0].get("position", "")
            if not parsed.get("current_company"): parsed["current_company"] = wh[0].get("company", "") or wh[0].get("organization", "")
    if parsed.get("current_role") and not _is_valid_role_title(parsed["current_role"]): parsed["current_role"] = ""
    if parsed.get("current_company") and not _is_valid_company_name(parsed["current_company"]): parsed["current_company"] = ""

    # Defaults
    for f, d in {"name": "Unknown Candidate", "email": "", "phone": "", "address": "", "linkedin": "", "github": "",
                 "portfolio": "", "location": "", "current_role": "", "current_company": "", "total_experience_years": 0,
                 "professional_summary": "", "specializations": [], "skills": {}, "work_history": [], "education": [],
                 "certifications": [], "awards": [], "projects": [], "publications": [], "volunteer": [],
                 "languages": [], "interests": []}.items():
        if f not in parsed: parsed[f] = d
    return parsed


# ═══════════════════════════════════════════════════════════════
#                    DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_resume_display_summary(parsed: Dict) -> str:
    if not parsed: return "No resume data"
    lines = [f"**{parsed.get('name', 'Candidate')}**"]
    r, c = parsed.get("current_role", ""), parsed.get("current_company", "")
    if r: lines.append(f"\U0001f4bc {r}" + (f" at {c}" if c else ""))
    loc = parsed.get("location", "") or parsed.get("address", "")
    if loc: lines.append(f"\U0001f4cd {loc[:60]}")
    exp = parsed.get("total_experience_years", 0)
    if exp: lines.append(f"\U0001f4c5 ~{exp} years experience")
    if parsed.get("email"): lines.append(f"\U0001f4e7 {parsed['email']}")
    if parsed.get("phone"): lines.append(f"\U0001f4de {parsed['phone']}")
    if parsed.get("linkedin"): lines.append("\U0001f517 LinkedIn")
    if parsed.get("github"): lines.append("\U0001f4bb GitHub")
    return "\n".join(lines)


def get_resume_full_summary(parsed: Dict) -> Dict:
    if not parsed: return {}
    sc = sum(len(v) for v in parsed.get("skills", {}).values() if isinstance(v, list)) if isinstance(parsed.get("skills", {}), dict) else len(parsed.get("skills", [])) if isinstance(parsed.get("skills"), list) else 0
    return {"name": parsed.get("name", "Unknown"), "email": parsed.get("email", ""), "phone": parsed.get("phone", ""),
            "location": parsed.get("location", "") or parsed.get("address", ""),
            "current_role": parsed.get("current_role", ""), "current_company": parsed.get("current_company", ""),
            "total_experience_years": parsed.get("total_experience_years", 0), "skills_count": sc,
            "education_count": len(parsed.get("education", [])), "work_history_count": len(parsed.get("work_history", [])),
            "certifications_count": len(parsed.get("certifications", [])), "projects_count": len(parsed.get("projects", [])),
            "has_linkedin": bool(parsed.get("linkedin")), "has_github": bool(parsed.get("github")),
            "has_portfolio": bool(parsed.get("portfolio")), "has_summary": bool(parsed.get("professional_summary"))}


def extract_key_highlights(parsed: Dict) -> List[str]:
    h: List[str] = []
    if not parsed: return h
    exp = parsed.get("total_experience_years", 0)
    if exp: h.append(f"\U0001f4c5 {exp} years of experience")
    r, c = parsed.get("current_role", ""), parsed.get("current_company", "")
    if r and c: h.append(f"\U0001f4bc Currently {r} at {c}")
    elif r: h.append(f"\U0001f4bc {r}")
    edu = parsed.get("education", [])
    if edu and isinstance(edu, list) and isinstance(edu[0], dict):
        d = edu[0].get("degree", "")
        if d: h.append(f"\U0001f393 {d}" + (f" from {edu[0].get('institution', '')}" if edu[0].get("institution") else ""))
    sc = sum(len(v) for v in parsed.get("skills", {}).values() if isinstance(v, list)) if isinstance(parsed.get("skills", {}), dict) else 0
    if sc: h.append(f"\U0001f6e0\ufe0f {sc} skills identified")
    if parsed.get("certifications"): h.append(f"\U0001f4dc {len(parsed['certifications'])} certification(s)")
    if parsed.get("projects"): h.append(f"\U0001f680 {len(parsed['projects'])} project(s)")
    return h


def get_contact_completeness(parsed: Dict) -> Dict:
    if not parsed: return {"score": 0, "missing": ["all"]}
    fields = {"name": parsed.get("name", ""), "email": parsed.get("email", ""), "phone": parsed.get("phone", ""),
              "location": parsed.get("location", "") or parsed.get("address", ""), "linkedin": parsed.get("linkedin", "")}
    present = [f for f, v in fields.items() if v and _is_valid_field(f, v)]
    missing = [f for f in fields if f not in present]
    return {"score": round(len(present) / len(fields) * 100, 1), "present": present, "missing": missing,
            "total_fields": len(fields), "filled_fields": len(present)}


def validate_parsed_resume(parsed: Dict) -> Dict:
    if not parsed: return {"valid": False, "issues": ["No data"]}
    issues, warnings = [], []
    if not parsed.get("name") or parsed["name"] in ["Unknown Candidate", ""]: issues.append("Name not extracted")
    if not parsed.get("email"): warnings.append("Email not found")
    if not parsed.get("phone"): warnings.append("Phone not found")
    if not parsed.get("work_history"): warnings.append("No work history extracted")
    if not parsed.get("education"): warnings.append("No education extracted")
    sc = sum(len(v) for v in parsed.get("skills", {}).values() if isinstance(v, list)) if isinstance(parsed.get("skills", {}), dict) else 0
    if sc == 0: warnings.append("No skills extracted")
    return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings,
            "quality_score": max(0, 100 - len(issues) * 20 - len(warnings) * 5)}
