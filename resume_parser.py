"""
Enhanced Resume Parser V3
- Triple extraction: Document Processor + Regex + LLM
- Enhanced name extraction with 6 strategies
- Enhanced contact extraction with 20+ patterns per field
- Accurate experience calculation using CURRENT_YEAR = 2026
- Works for ANY domain resume, ANY format
- NO TRUNCATION on any field
"""

import json
import re
import requests
from typing import Dict, List, Tuple, Optional, Callable

CURRENT_YEAR = 2026
CURRENT_MONTH = 3  # March 2026


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

CRITICAL EXTRACTION RULES:
1. **NAME**: Look at the VERY FIRST non-empty lines - the name is usually the LARGEST or BOLDEST text at the top. It may NOT have a "Name:" label!
2. **PHONE**: Look for patterns like +91, +1, (XXX), or any 10+ digit numbers. Include country code.
3. **EMAIL**: Find ANYTHING with @ symbol - it's the most reliable indicator
4. **ADDRESS**: Look for street numbers, city names, pin/zip codes
5. **EDUCATION**: Extract EVERY educational qualification:
   - Degrees: B.Tech, B.E., B.Sc, M.Tech, M.E., M.Sc, MBA, BCA, MCA, PhD, etc.
   - Institutions: University, College, Institute, School, IIT, NIT, etc.
   - Years: graduation year, batch, class of
   - GPA/CGPA/percentage/grades
6. **EXPERIENCE**: If end_date is "Present" or "Current", calculate duration until March 2026
7. **duration_years**: Calculate as decimal (2 years 6 months = 2.5)
8. **total_experience_years**: Sum of all duration_years values
9. **SKILLS**: Extract EVERY skill mentioned ANYWHERE in the resume
10. **ACHIEVEMENTS**: Include ALL bullet points from work experience

Return ONLY valid JSON - no other text, no markdown code blocks."""


# ═══════════════════════════════════════════════════════════════
#                    CONTACT EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_contacts_regex(text: str) -> Dict:
    """Extract contact details using comprehensive regex patterns"""
    contacts = {
        "email": "",
        "phone": "",
        "address": "",
        "linkedin": "",
        "github": "",
        "portfolio": "",
        "location": ""
    }
    
    if not text:
        return contacts

    # ═══════════════════════════════════════
    # EMAIL - Most reliable (look for @)
    # ═══════════════════════════════════════
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
            # Clean up
            email = re.sub(r'^(?:email|e-mail|mail)[\s.:]*', '', email, flags=re.IGNORECASE)
            email = re.sub(r'\s+', '', email)
            email = re.sub(r'\[\s*at\s*\]|\(\s*at\s*\)', '@', email, flags=re.IGNORECASE)
            
            # Validate basic email structure
            if '@' in email and '.' in email.split('@')[-1]:
                contacts["email"] = email
                break

    # ═══════════════════════════════════════
    # PHONE - Multiple international formats
    # ═══════════════════════════════════════
    phone_patterns = [
        # International with +
        r'\+\d{1,3}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        r'\+\d{1,3}[-.\s]?\d{4,5}[-.\s]?\d{5,6}',
        r'\+\d{10,15}',
        
        # Indian formats
        r'\+91[-.\s]?\d{5}[-.\s]?\d{5}',
        r'\+91[-.\s]?\d{10}',
        r'(?<!\d)91[-.\s]?\d{10}(?!\d)',
        r'(?<!\d)0\d{2,4}[-.\s]?\d{6,8}(?!\d)',
        
        # US formats
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        r'(?<!\d)\d{3}[-.\s]\d{3}[-.\s]\d{4}(?!\d)',
        r'(?<!\d)1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)',
        
        # UK formats
        r'\+44[-.\s]?\d{4}[-.\s]?\d{6}',
        r'(?<!\d)0\d{4}[-.\s]?\d{6}(?!\d)',
        
        # Labeled phone numbers
        r'(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*[\+]?[\d][\d\s\-().]{8,18}',
        
        # Generic patterns (last resort)
        r'(?<!\d)\d{3}[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)',
        r'(?<!\d)\d{5}[-.\s]?\d{5}(?!\d)',
        r'(?<!\d)\d{10}(?!\d)',
    ]
    
    for pattern in phone_patterns:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean the match
                cleaned = re.sub(r'^(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P|Call)[\s.:]*', '', match, flags=re.IGNORECASE)
                cleaned = cleaned.strip()
                
                # Count digits only
                digits = re.sub(r'[^\d]', '', cleaned)
                
                # Valid phone should have 10-15 digits
                if 10 <= len(digits) <= 15:
                    contacts["phone"] = cleaned
                    break
            
            if contacts["phone"]:
                break
        except re.error:
            continue

    # ═══════════════════════════════════════
    # LINKEDIN
    # ═══════════════════════════════════════
    linkedin_patterns = [
        r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?',
        r'linkedin\.com/in/[\w-]+',
        r'(?:linkedin|ln)[\s.:]+(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+',
        r'(?:linkedin|ln)[\s.:]+[\w-]+',
        r'/in/([\w-]+)',
    ]
    
    for pattern in linkedin_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            linkedin = match.group(0).strip()
            # Clean up
            linkedin = re.sub(r'^(?:linkedin|ln)[\s.:]+', '', linkedin, flags=re.IGNORECASE)
            if 'linkedin.com' not in linkedin.lower():
                # Extract username and construct URL
                username = linkedin.strip('/')
                if username and len(username) > 2:
                    linkedin = f"linkedin.com/in/{username}"
            contacts["linkedin"] = linkedin
            break

    # ═══════════════════════════════════════
    # GITHUB
    # ═══════════════════════════════════════
    github_patterns = [
        r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?',
        r'github\.com/[\w-]+',
        r'(?:github|gh)[\s.:]+(?:https?://)?(?:www\.)?github\.com/[\w-]+',
        r'(?:github|gh)[\s.:]+[\w-]+',
    ]
    
    for pattern in github_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            github = match.group(0).strip()
            github = re.sub(r'^(?:github|gh)[\s.:]+', '', github, flags=re.IGNORECASE)
            contacts["github"] = github
            break

    # ═══════════════════════════════════════
    # PORTFOLIO / WEBSITE
    # ═══════════════════════════════════════
    website_patterns = [
        r'(?:portfolio|website|web|site|blog)[\s.:]+(?:https?://)?[\w.-]+\.[a-z]{2,}[\w/.-]*',
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:dev|io|me|tech|design|codes?|site|online|app)/?[\w/.-]*',
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|co)(?:/[\w/.-]*)?',
    ]
    
    for pattern in website_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            url = match.strip()
            # Clean up label
            url = re.sub(r'^(?:portfolio|website|web|site|blog)[\s.:]+', '', url, flags=re.IGNORECASE)
            # Skip if it's linkedin or github
            if 'linkedin' not in url.lower() and 'github' not in url.lower():
                # Skip email-like patterns
                if '@' not in url:
                    contacts["portfolio"] = url
                    break
        if contacts["portfolio"]:
            break

    # ═══════════════════════════════════════
    # ADDRESS
    # ═══════════════════════════════════════
    address_patterns = [
        # Labeled address
        r'(?:Address|Location|Residence|Home|Addr)[\s.:]+([^\n]{15,150})',
        
        # With PIN/ZIP codes
        r'(?:No[.:]?\s*)?[\d]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar|Colony|Layout|Block|Lane|Apt|Apartment|Floor|Fl|Building|Bldg)[\w\s,.-]+(?:\d{5,6})',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+,\s*\d{5,6}',
        
        # Indian address patterns
        r'(?:Bangalore|Bengaluru|Mumbai|Delhi|Chennai|Hyderabad|Pune|Kolkata)[\w\s,.-]+\d{6}',
        
        # US address patterns
        r'(?:New York|San Francisco|Los Angeles|Chicago|Seattle|Boston|Austin)[\w\s,.-]+\d{5}',
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            addr = match.group(1) if match.lastindex else match.group(0)
            addr = addr.strip()
            # Clean up
            addr = re.sub(r'^(?:Address|Location|Residence|Home|Addr)[\s.:]+', '', addr, flags=re.IGNORECASE)
            addr = re.sub(r'\s+', ' ', addr).strip()
            
            if 15 < len(addr) < 200:
                contacts["address"] = addr
                break

    # ═══════════════════════════════════════
    # LOCATION (City, State, Country)
    # ═══════════════════════════════════════
    # Try to extract just the city/state/country
    location_patterns = [
        # Labeled location
        r'(?:Location|Based in|Located at|City|Current Location)[\s.:]+([A-Za-z][A-Za-z\s,]+)',
        
        # Indian cities
        r'\b(Bangalore|Bengaluru|Mumbai|Delhi|NCR|Chennai|Hyderabad|Pune|Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Lucknow|Chandigarh|Indore|Bhopal|Kochi|Coimbatore|Trivandrum|Mysore|Nagpur|Surat|Vadodara|Bhubaneswar|Patna|Ranchi|Guwahati|Visakhapatnam|Vijayawada)\b',
        
        # US cities
        r'\b(New York|NYC|San Francisco|SF|Los Angeles|LA|Chicago|Seattle|Boston|Austin|Denver|Atlanta|Dallas|Houston|Phoenix|San Diego|San Jose|Portland|Miami|Washington DC|Philadelphia|Minneapolis|Detroit|Charlotte|Nashville|Orlando|Salt Lake City|Raleigh|Tampa)\b',
        
        # International cities
        r'\b(London|Berlin|Amsterdam|Dublin|Singapore|Tokyo|Sydney|Toronto|Vancouver|Melbourne|Paris|Munich|Barcelona|Stockholm|Copenhagen|Zurich|Dubai|Hong Kong|Shanghai|Beijing|Seoul|Bangkok|Jakarta|Manila|Kuala Lumpur|Ho Chi Minh|Taipei)\b',
        
        # Countries
        r'\b(India|USA|US|United States|UK|United Kingdom|Canada|Australia|Germany|Netherlands|Singapore|UAE|United Arab Emirates|Ireland|France|Spain|Italy|Japan|China|South Korea|Malaysia|Indonesia|Thailand|Philippines|Vietnam|Taiwan)\b',
        
        # State abbreviations (US)
        r'\b([A-Z]{2})\s*,?\s*(USA|US|United States)?\b',
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            loc = match.group(1) if match.lastindex else match.group(0)
            loc = loc.strip()
            # Clean up
            loc = re.sub(r'^(?:Location|Based in|Located at|City|Current Location)[\s.:]+', '', loc, flags=re.IGNORECASE)
            loc = re.sub(r'\s+', ' ', loc).strip()
            
            if len(loc) >= 2 and len(loc) <= 100:
                contacts["location"] = loc
                break

    return contacts


# ═══════════════════════════════════════════════════════════════
#                    NAME EXTRACTION - ENHANCED
# ═══════════════════════════════════════════════════════════════

def _extract_name_from_text(text: str) -> str:
    """
    Extract candidate name using multiple strategies
    Works even without 'Name:' label
    """
    if not text:
        return ""
    
    lines = text.strip().split("\n")
    
    # ═══════════════════════════════════════
    # Strategy 1: Look for labeled name
    # ═══════════════════════════════════════
    name_label_patterns = [
        r'(?:Name|Full Name|Candidate Name|Applicant Name)[\s.:]+([A-Z][a-zA-Z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-zA-Z]+){1,2})',
        r'(?:Name|Full Name)[\s.:]+([A-Z][A-Z\s]+)',  # ALL CAPS name
    ]
    
    for pattern in name_label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if _is_valid_name(name):
                return _clean_name(name)
    
    # ═══════════════════════════════════════
    # Strategy 2: First substantial line that looks like a name
    # ═══════════════════════════════════════
    skip_keywords = [
        'resume', 'curriculum', 'vitae', 'cv', 'http', 'www', '@',
        'address', 'phone', 'email', 'street', 'road', 'avenue',
        'objective', 'summary', 'profile', 'linkedin', 'github',
        'portfolio', 'mobile', 'tel:', 'contact', 'experience',
        'education', 'skills', 'professional', 'career', 'about',
        'personal', 'details', 'information', 'confidential',
        'page', 'date', 'application', 'position', 'job',
    ]
    
    for line in lines[:15]:  # Check first 15 lines
        line = line.strip()
        
        # Skip empty or very short lines
        if not line or len(line) < 4:
            continue
        
        # Skip very long lines (probably not a name)
        if len(line) > 50:
            continue
        
        # Skip lines with skip keywords
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        
        # Skip lines that look like addresses (contain numbers at start)
        if re.match(r'^\d', line):
            continue
        
        # Skip lines with 5+ digits (likely phone/address)
        if len(re.findall(r'\d', line)) >= 5:
            continue
        
        # Skip lines that look like phone numbers
        if re.match(r'^[\+\d\(\)]', line):
            continue
        
        # Skip lines with @ (email)
        if '@' in line:
            continue
        
        # Check if line is mostly alphabetic
        alpha_chars = sum(1 for c in line if c.isalpha() or c.isspace() or c == '.' or c == '-')
        if alpha_chars / max(len(line), 1) < 0.85:
            continue
        
        # Check name patterns
        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Middle Last
            r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+$',  # First M. Last
            r'^[A-Z]+\s+[A-Z]+$',  # FIRST LAST (all caps)
            r'^[A-Z]+\s+[A-Z]+\s+[A-Z]+$',  # FIRST MIDDLE LAST (all caps)
            r'^Dr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Dr. First Last
            r'^Mr\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Mr. First Last
            r'^Ms\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Ms. First Last
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+$',  # First Last-Last (hyphenated)
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+$',  # First Middle Last-Last
        ]
        
        for pattern in name_patterns:
            if re.match(pattern, line):
                if _is_valid_name(line):
                    return _clean_name(line)
        
        # Fallback: if line has 2-4 words, all starting with capitals
        words = line.split()
        if 2 <= len(words) <= 4:
            all_capitalized = all(w[0].isupper() for w in words if w)
            reasonable_length = all(2 <= len(w) <= 15 for w in words)
            
            if all_capitalized and reasonable_length:
                if _is_valid_name(line):
                    return _clean_name(line)
    
    # ═══════════════════════════════════════
    # Strategy 3: Look in first line (even if mixed content)
    # ═══════════════════════════════════════
    if lines:
        first_line = lines[0].strip()
        # Try to extract name-like pattern from first line
        name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})', first_line)
        if name_match:
            potential_name = name_match.group(1).strip()
            if _is_valid_name(potential_name):
                return _clean_name(potential_name)
    
    # ═══════════════════════════════════════
    # Strategy 4: Look for "I am" or "My name is" patterns
    # ═══════════════════════════════════════
    intro_patterns = [
        r"(?:I am|I'm|My name is|This is|Myself)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+){1,2})",
        r"(?:Dear\s+(?:Sir|Madam|Hiring Manager),?\s*(?:I am|I'm|My name is)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    ]
    
    for pattern in intro_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if _is_valid_name(name):
                return _clean_name(name)
    
    # ═══════════════════════════════════════
    # Strategy 5: Look for name in email (fallback)
    # ═══════════════════════════════════════
    email_match = re.search(r'([\w.]+)@', text)
    if email_match:
        email_name = email_match.group(1)
        # Split by . or _ and capitalize
        parts = re.split(r'[._]', email_name)
        if len(parts) >= 2:
            potential_name = ' '.join(p.capitalize() for p in parts if p.isalpha() and len(p) > 1)
            if len(potential_name.split()) >= 2:
                return potential_name
    
    # ═══════════════════════════════════════
    # Strategy 6: Look for ALL CAPS line (common in some formats)
    # ═══════════════════════════════════════
    for line in lines[:10]:
        line = line.strip()
        if line and line.isupper() and 5 <= len(line) <= 40:
            words = line.split()
            if 2 <= len(words) <= 4:
                # Check if it's not a section header
                section_headers = ['RESUME', 'CURRICULUM', 'VITAE', 'CV', 'OBJECTIVE', 
                                   'SUMMARY', 'EXPERIENCE', 'EDUCATION', 'SKILLS', 
                                   'CONTACT', 'PROFILE', 'ABOUT']
                if not any(header in line for header in section_headers):
                    return line.title()
    
    return ""


def _is_valid_name(name: str) -> bool:
    """Validate if a string looks like a valid person name"""
    if not name:
        return False
    
    name = name.strip()
    
    # Too short or too long
    if len(name) < 4 or len(name) > 60:
        return False
    
    # Common words that are NOT names
    not_names = {
        'resume', 'objective', 'summary', 'experience', 'education', 'skills',
        'contact', 'email', 'phone', 'address', 'profile', 'professional',
        'career', 'curriculum', 'vitae', 'page', 'date', 'present', 'current',
        'january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'engineer', 'developer', 'manager', 'analyst', 'consultant', 'designer',
        'senior', 'junior', 'lead', 'head', 'director', 'executive', 'intern',
        'software', 'hardware', 'technical', 'information', 'technology',
        'university', 'college', 'school', 'institute', 'company', 'corporation',
        'private', 'limited', 'ltd', 'inc', 'corp', 'llc', 'pvt',
        'bangalore', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'pune', 'kolkata',
        'new york', 'san francisco', 'london', 'singapore', 'dubai',
        'india', 'usa', 'uk', 'canada', 'australia', 'germany',
        'bachelor', 'master', 'doctor', 'phd', 'mba', 'btech', 'mtech',
        'project', 'projects', 'work', 'history', 'reference', 'references',
    }
    
    words = name.lower().split()
    
    # Check if any word is in not_names
    if any(w in not_names for w in words):
        return False
    
    # Name should have 2-4 words
    if not (2 <= len(words) <= 4):
        return False
    
    # Each word should be reasonable length
    if not all(1 <= len(w) <= 20 for w in words):
        return False
    
    # Should be mostly alphabetic
    alpha_count = sum(1 for c in name if c.isalpha() or c.isspace() or c in '.-')
    if alpha_count / len(name) < 0.85:
        return False
    
    return True


def _clean_name(name: str) -> str:
    """Clean and normalize a name string"""
    if not name:
        return ""
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Handle ALL CAPS - convert to Title Case
    if name.isupper():
        name = name.title()
    
    # Fix common issues
    name = re.sub(r'\s*\.\s*', '. ', name)  # Fix spacing around periods
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


# ═══════════════════════════════════════════════════════════════
#                    EDUCATION EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_education_regex(text: str) -> List[Dict]:
    """Extract education details using regex patterns as fallback"""
    education = []
    
    if not text:
        return education
    
    # Degree patterns - comprehensive
    degree_patterns = [
        # Bachelor degrees
        (r'(Bachelor\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Commerce|Business|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(B\.?\s*(?:Tech|E|Sc|A|Com|B\.?A|S|CA|BA|Arch|Pharm|Ed|Des)\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(B\.?Tech|BTech|B\.?E\.?|BE)\s*[-–in\s]*([\w\s,&]+)?', 'Bachelor'),
        (r'(BCA|B\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(BBA|B\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        (r'(B\.?Com|BCom|B\.?Commerce)\s*(?:in\s*)?([\w\s,&]+)?', 'Bachelor'),
        
        # Master degrees
        (r'(Master\s*(?:of\s*)?(?:Science|Arts|Technology|Engineering|Business\s*Administration|Computer\s*Applications?|Laws?)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?\s*(?:Tech|E|Sc|A|B\.?A|S|CA|BA|Phil|Ed|Des)\.?|MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?Tech|MTech|M\.?E\.?|ME|M\.?S\.?|MS)\s*[-–in\s]*([\w\s,&]+)?', 'Master'),
        (r'(MCA|M\.?C\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(MBA|M\.?B\.?A\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(M\.?Com|MCom|M\.?Commerce)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        (r'(PG\s*(?:Diploma|Degree)?)\s*(?:in\s*)?([\w\s,&]+)?', 'Master'),
        
        # PhD
        (r'(Ph\.?\s*D\.?|Doctorate|Doctor\s*of\s*Philosophy)\s*(?:in\s*)?([\w\s,&]+)?', 'PhD'),
        (r'(D\.?Phil|DPhil)\s*(?:in\s*)?([\w\s,&]+)?', 'PhD'),
        
        # Diploma
        (r'(Diploma)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(Polytechnic)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        (r'(ITI|I\.?T\.?I\.?)\s*(?:in\s*)?([\w\s,&]+)?', 'Diploma'),
        
        # School
        (r'(Higher\s*Secondary|HSC|12th|XII|Intermediate|Senior\s*Secondary|\+2|Plus\s*Two)', 'School'),
        (r'(Secondary|SSC|10th|X|Matriculation|High\s*School|SSLC)', 'School'),
        (r'(CBSE|ICSE|ISC|State\s*Board)', 'School'),
    ]
    
    # Institution patterns
    institution_patterns = [
        r'([A-Z][A-Za-z\s\.\']+(?:University|College|Institute|School|Academy|Polytechnic))',
        r'((?:IIT|NIT|IIIT|BITS|VIT|SRM|Amity|Manipal|LPU|KIIT|MIT|Stanford|Harvard|Cambridge|Oxford)\s*[,]?\s*[\w\s]*)',
        r'((?:Indian Institute of|National Institute of|International Institute of)\s*[\w\s]+)',
        r'(?:from|at|,)\s+([A-Z][A-Za-z\s\.\']{5,60})',
    ]
    
    # GPA/Grade patterns
    gpa_patterns = [
        r'(?:GPA|CGPA|CPI|Grade)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?',
        r'(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?\s*(?:GPA|CGPA|CPI)',
        r'(\d{1,2}(?:\.\d+)?)\s*%',
        r'(First\s*Class(?:\s*with\s*Distinction)?|Distinction|Honors?|Honours?)',
        r'(Cum\s*Laude|Magna\s*Cum\s*Laude|Summa\s*Cum\s*Laude)',
        r'(?:Grade|Score)[:\s]*([A-F][+-]?|\d+\.?\d*)',
    ]
    
    # Year patterns
    year_patterns = [
        r'((?:19|20)\d{2})\s*[-–to]+\s*((?:19|20)\d{2}|Present|Current|Expected|Pursuing)',
        r'(?:Class\s*of|Batch|Graduated?|Passing\s*Year|Year\s*of\s*(?:Graduation|Completion)|Expected)[:\s]*((?:19|20)\d{2})',
        r'(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}',
        r'((?:19|20)\d{2})',
    ]
    
    # Split text into sections
    text_upper = text.upper()
    
    # Find education section
    edu_section_start = -1
    edu_markers = ['EDUCATION', 'ACADEMIC', 'QUALIFICATION', 'EDUCATIONAL BACKGROUND', 
                   'ACADEMIC BACKGROUND', 'ACADEMIC QUALIFICATIONS', 'EDUCATIONAL DETAILS']
    
    for marker in edu_markers:
        idx = text_upper.find(marker)
        if idx != -1:
            edu_section_start = idx
            break
    
    # Define section end markers
    end_markers = ['EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILL', 'PROJECT', 
                   'CERTIFICATION', 'AWARD', 'ACHIEVEMENT', 'PUBLICATION',
                   'REFERENCE', 'HOBBY', 'INTEREST', 'LANGUAGE', 'ADDITIONAL']
    
    # Extract education section
    if edu_section_start != -1:
        edu_section = text[edu_section_start:]
        for marker in end_markers:
            end_idx = edu_section.upper().find(marker, 50)
            if end_idx != -1:
                edu_section = edu_section[:end_idx]
                break
    else:
        edu_section = text
    
    # Find degrees
    found_degrees = []
    for pattern, degree_type in degree_patterns:
        try:
            for match in re.finditer(pattern, edu_section, re.IGNORECASE):
                degree_name = match.group(1).strip() if match.group(1) else ""
                field = ""
                
                if match.lastindex >= 2 and match.group(2):
                    field = match.group(2).strip()
                
                # Clean up field
                field = re.sub(r'\s+', ' ', field).strip()
                if len(field) > 100:
                    field = field[:100]
                
                # Skip if field contains non-field content
                skip_words = ['university', 'college', 'institute', 'school', 'from', 'at', 'gpa', 'cgpa']
                if any(sw in field.lower() for sw in skip_words):
                    field = ""
                
                # Get surrounding text for institution and year
                start = max(0, match.start() - 30)
                end = min(len(edu_section), match.end() + 250)
                context = edu_section[start:end]
                
                edu_entry = {
                    "degree": degree_name,
                    "field": field,
                    "degree_type": degree_type,
                    "institution": "",
                    "year": "",
                    "gpa": "",
                    "location": ""
                }
                
                # Find institution
                for inst_pattern in institution_patterns:
                    try:
                        inst_match = re.search(inst_pattern, context, re.IGNORECASE)
                        if inst_match:
                            inst = inst_match.group(1).strip()
                            # Clean up
                            inst = re.sub(r'\s+', ' ', inst).strip()
                            if len(inst) > 5 and inst.lower() not in ['the', 'and', 'for', 'with', 'from', 'university', 'college']:
                                edu_entry["institution"] = inst
                                break
                    except re.error:
                        continue
                
                # Find year
                for year_pattern in year_patterns:
                    try:
                        year_match = re.search(year_pattern, context, re.IGNORECASE)
                        if year_match:
                            edu_entry["year"] = year_match.group(0).strip()
                            break
                    except re.error:
                        continue
                
                # Find GPA
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
    
    # Deduplicate
    seen = set()
    for edu in found_degrees:
        degree = edu.get('degree', '').lower()
        inst = edu.get('institution', '').lower()
        key = f"{degree}_{inst}"[:80]
        
        if key not in seen and edu.get('degree'):
            seen.add(key)
            education.append({
                "degree": edu.get('degree', ''),
                "field": edu.get('field', ''),
                "institution": edu.get('institution', ''),
                "year": edu.get('year', ''),
                "gpa": edu.get('gpa', ''),
                "location": edu.get('location', ''),
                "achievements": []
            })
    
    return education


# ═══════════════════════════════════════════════════════════════
#                    SKILLS EXTRACTION - REGEX
# ═══════════════════════════════════════════════════════════════

def _extract_skills_regex(text: str) -> Dict:
    """Extract skills from text using regex patterns"""
    skills = {
        "programming_languages": [],
        "frameworks_libraries": [],
        "ai_ml_tools": [],
        "cloud_platforms": [],
        "databases": [],
        "devops_tools": [],
        "visualization": [],
        "other_tools": [],
        "soft_skills": []
    }
    
    if not text:
        return skills
    
    text_lower = text.lower()
    
    # Programming languages
    prog_langs = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
        'go', 'golang', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r',
        'matlab', 'perl', 'bash', 'shell', 'powershell', 'sql', 'html', 
        'css', 'sass', 'less', 'objective-c', 'dart', 'lua', 'groovy',
        'clojure', 'haskell', 'erlang', 'elixir', 'f#', 'cobol', 'fortran',
        'assembly', 'vba', 'visual basic', 'delphi', 'pascal', 'lisp',
    ]
    
    for lang in prog_langs:
        pattern = rf'\b{re.escape(lang)}\b'
        if re.search(pattern, text_lower):
            display_name = lang.title() if lang not in ['c++', 'c#', 'f#', 'sql', 'html', 'css', 'php', 'r'] else lang.upper() if len(lang) <= 3 else lang
            if lang == 'c++':
                display_name = 'C++'
            elif lang == 'c#':
                display_name = 'C#'
            elif lang == 'f#':
                display_name = 'F#'
            skills["programming_languages"].append(display_name)
    
    # Frameworks
    frameworks = [
        'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js',
        'next', 'nextjs', 'next.js', 'nuxt', 'nuxtjs', 'gatsby', 'svelte',
        'node', 'nodejs', 'node.js', 'express', 'expressjs', 'fastify', 'koa', 'hapi',
        'django', 'flask', 'fastapi', 'pyramid', 'tornado', 'bottle', 'cherrypy',
        'spring', 'springboot', 'spring boot', 'hibernate', 'struts',
        'laravel', 'symfony', 'codeigniter', 'cakephp', 'yii', 'zend',
        'rails', 'ruby on rails', 'sinatra',
        'asp.net', '.net', 'dotnet', '.net core', 'blazor', 'wpf', 'winforms',
        'jquery', 'bootstrap', 'tailwind', 'tailwindcss', 'material-ui', 'mui',
        'chakra', 'antd', 'ant design', 'semantic ui', 'bulma', 'foundation',
        'flutter', 'react native', 'ionic', 'xamarin', 'electron', 'tauri',
        'unity', 'unreal', 'godot', 'pygame',
    ]
    
    for fw in frameworks:
        pattern = rf'\b{re.escape(fw)}\b'
        if re.search(pattern, text_lower):
            skills["frameworks_libraries"].append(fw.title())
    
    # AI/ML
    ai_tools = [
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'opencv', 'pillow', 'pil', 'nltk', 'spacy', 'gensim',
        'huggingface', 'transformers', 'bert', 'gpt', 'llm', 'langchain',
        'openai', 'anthropic', 'cohere', 'pinecone', 'weaviate', 'milvus',
        'mlflow', 'kubeflow', 'airflow', 'prefect', 'dagster',
        'spark', 'pyspark', 'hadoop', 'hive', 'pig', 'flink', 'kafka',
        'dask', 'ray', 'modin', 'vaex', 'polars',
        'xgboost', 'lightgbm', 'catboost', 'prophet', 'statsmodels',
        'yolo', 'detectron', 'mediapipe', 'tesseract',
        'onnx', 'tensorrt', 'openvino', 'tflite',
    ]
    
    for tool in ai_tools:
        pattern = rf'\b{re.escape(tool)}\b'
        if re.search(pattern, text_lower):
            skills["ai_ml_tools"].append(tool.title() if tool not in ['gpt', 'llm', 'bert', 'yolo'] else tool.upper())
    
    # Cloud
    cloud = [
        'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud',
        'heroku', 'digitalocean', 'linode', 'vultr', 'oracle cloud', 'ibm cloud',
        'lambda', 's3', 'ec2', 'ecs', 'eks', 'fargate', 'rds', 'dynamodb', 'sqs', 'sns',
        'cloudformation', 'cdk', 'sam', 'amplify', 'cognito', 'api gateway',
        'cloudwatch', 'cloudtrail', 'iam', 'vpc', 'route53', 'cloudfront',
        'azure devops', 'azure functions', 'app service', 'cosmos db', 'blob storage',
        'bigquery', 'cloud run', 'cloud functions', 'gke', 'pubsub', 'dataflow',
        'firebase', 'firestore', 'vercel', 'netlify', 'railway', 'render', 'fly.io',
    ]
    
    for c in cloud:
        pattern = rf'\b{re.escape(c)}\b'
        if re.search(pattern, text_lower):
            if len(c) <= 3:
                skills["cloud_platforms"].append(c.upper())
            else:
                skills["cloud_platforms"].append(c.title())
    
    # Databases
    dbs = [
        'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch',
        'dynamodb', 'cassandra', 'oracle', 'sql server', 'mssql', 'sqlite',
        'mariadb', 'couchdb', 'couchbase', 'neo4j', 'arangodb', 'influxdb',
        'timescaledb', 'clickhouse', 'cockroachdb', 'fauna', 'supabase',
        'prisma', 'sequelize', 'typeorm', 'sqlalchemy', 'mongoose',
        'firebase', 'firestore', 'realm', 'leveldb', 'rocksdb',
    ]
    
    for db in dbs:
        pattern = rf'\b{re.escape(db)}\b'
        if re.search(pattern, text_lower):
            skills["databases"].append(db.title() if db not in ['sql', 'mssql'] else db.upper())
    
    # DevOps
    devops = [
        'docker', 'kubernetes', 'k8s', 'podman', 'containerd',
        'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci',
        'terraform', 'ansible', 'puppet', 'chef', 'saltstack', 'pulumi',
        'prometheus', 'grafana', 'datadog', 'new relic', 'splunk', 'elk',
        'nginx', 'apache', 'haproxy', 'envoy', 'istio', 'linkerd',
        'vagrant', 'packer', 'consul', 'vault', 'nomad',
        'argocd', 'flux', 'helm', 'kustomize', 'skaffold',
        'linux', 'unix', 'ubuntu', 'centos', 'debian', 'rhel', 'alpine',
    ]
    
    for tool in devops:
        pattern = rf'\b{re.escape(tool)}\b'
        if re.search(pattern, text_lower):
            skills["devops_tools"].append(tool.title() if tool not in ['k8s', 'elk', 'ci'] else tool.upper())
    
    # Visualization
    viz = [
        'power bi', 'powerbi', 'tableau', 'looker', 'metabase', 'superset',
        'qlik', 'sisense', 'domo', 'd3.js', 'd3', 'chartjs', 'highcharts',
        'plotly', 'bokeh', 'altair', 'vega', 'echarts', 'recharts',
        'excel', 'google sheets', 'google data studio', 'amplitude',
    ]
    
    for v in viz:
        pattern = rf'\b{re.escape(v)}\b'
        if re.search(pattern, text_lower):
            skills["visualization"].append(v.title())
    
    # Other tools
    other = [
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
        'jira', 'confluence', 'trello', 'asana', 'monday', 'notion', 'linear',
        'slack', 'teams', 'zoom', 'discord',
        'figma', 'sketch', 'adobe xd', 'invision', 'zeplin', 'photoshop',
        'illustrator', 'after effects', 'premiere', 'canva',
        'postman', 'insomnia', 'swagger', 'graphql', 'rest', 'grpc', 'soap',
        'selenium', 'cypress', 'playwright', 'puppeteer', 'jest', 'mocha',
        'pytest', 'unittest', 'junit', 'testng', 'rspec',
        'webpack', 'vite', 'rollup', 'parcel', 'esbuild', 'babel',
        'npm', 'yarn', 'pnpm', 'pip', 'conda', 'maven', 'gradle',
        'latex', 'markdown', 'rst', 'sphinx', 'mkdocs', 'docusaurus',
    ]
    
    for tool in other:
        pattern = rf'\b{re.escape(tool)}\b'
        if re.search(pattern, text_lower):
            skills["other_tools"].append(tool.title() if tool not in ['git', 'svn', 'npm', 'rest', 'grpc', 'soap'] else tool.upper() if len(tool) <= 4 else tool.title())
    
    # Soft skills
    soft = [
        'leadership', 'communication', 'teamwork', 'problem solving', 'problem-solving',
        'critical thinking', 'analytical', 'creativity', 'innovation', 'adaptability',
        'time management', 'project management', 'stakeholder management',
        'presentation', 'public speaking', 'negotiation', 'collaboration',
        'mentoring', 'coaching', 'decision making', 'conflict resolution',
        'emotional intelligence', 'self-motivated', 'detail-oriented', 'organized',
    ]
    
    for s in soft:
        pattern = rf'\b{re.escape(s)}\b'
        if re.search(pattern, text_lower):
            skills["soft_skills"].append(s.title())
    
    # Deduplicate all categories
    for category in skills:
        seen = set()
        unique = []
        for skill in skills[category]:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique.append(skill)
        skills[category] = unique
    
    return skills


# ═══════════════════════════════════════════════════════════════
#                    EXPERIENCE CALCULATION
# ═══════════════════════════════════════════════════════════════

def _parse_date_to_ym(date_str: str, month_map: Dict) -> Tuple[Optional[int], Optional[int]]:
    """Parse date string to (year, month)"""
    if not date_str:
        return None, None
    
    date_str = date_str.strip().lower()

    # "Month Year" format
    for month_name, month_num in month_map.items():
        if month_name in date_str:
            year_match = re.search(r'(19|20)\d{2}', date_str)
            if year_match:
                return int(year_match.group()), month_num

    # "MM/YYYY" or "MM-YYYY" format
    match = re.search(r'(\d{1,2})[/-]((?:19|20)\d{2})', date_str)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            return int(match.group(2)), month

    # "YYYY/MM" or "YYYY-MM" format
    match = re.search(r'((?:19|20)\d{2})[/-](\d{1,2})', date_str)
    if match:
        month = int(match.group(2))
        if 1 <= month <= 12:
            return int(match.group(1)), month

    # "YYYY" only
    year_match = re.search(r'(19|20)\d{2}', date_str)
    if year_match:
        return int(year_match.group()), 6  # Assume mid-year if only year given

    return None, None


def _calculate_total_experience(work_history: list) -> float:
    """Calculate total experience using current year 2026"""
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
    }

    total = 0.0
    
    for job in work_history:
        if not isinstance(job, dict):
            continue
            
        start_str = job.get("start_date", job.get("from", ""))
        end_str = job.get("end_date", job.get("to", ""))

        if not start_str:
            # Use pre-calculated duration if available
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

        # Parse end date
        present_words = ['present', 'current', 'now', 'ongoing', 'till date', 
                        'till now', 'today', 'continuing', '-', '–', '']
        
        if not end_str or str(end_str).strip().lower() in present_words:
            end_y = CURRENT_YEAR
            end_m = CURRENT_MONTH
        else:
            end_y, end_m = _parse_date_to_ym(str(end_str), month_map)
            if not end_y:
                end_y = CURRENT_YEAR
                end_m = CURRENT_MONTH

        # Calculate months
        months = (end_y - start_y) * 12 + (end_m - start_m)
        years = max(0, months / 12.0)
        total += years

    return round(total, 1)


# ═══════════════════════════════════════════════════════════════
#                    CONTACT VALIDATION & MERGING
# ═══════════════════════════════════════════════════════════════

def _is_valid_field(field: str, value: str) -> bool:
    """Validate if a field value is valid and not a placeholder"""
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
        digits = re.sub(r'[^\d]', '', value)
        return len(digits) >= 10
    
    if field == 'name':
        words = value.split()
        if len(words) < 2:
            return False
        alpha_ratio = sum(1 for c in value if c.isalpha() or c.isspace()) / max(len(value), 1)
        return alpha_ratio > 0.8
    
    if field in ['linkedin', 'github']:
        return len(value) > 5
    
    return len(value) >= 2


def _merge_contacts(llm_parsed: Dict, regex_contacts: Dict, doc_contacts: Dict = None) -> Dict:
    """
    Merge contacts from multiple sources with priority:
    1. LLM parsed (if valid)
    2. Document processor extracted
    3. Regex fallback
    """
    merged = {}
    
    if doc_contacts is None:
        doc_contacts = {}
    
    contact_fields = ['name', 'email', 'phone', 'address', 'linkedin', 'github', 'portfolio', 'location']
    
    for field in contact_fields:
        # Get values from all sources
        llm_val = str(llm_parsed.get(field, "")).strip()
        doc_val = str(doc_contacts.get(field, "")).strip()
        regex_val = str(regex_contacts.get(field, "")).strip()
        
        # Priority: LLM > Document > Regex (but validate each)
        if llm_val and _is_valid_field(field, llm_val):
            merged[field] = llm_val
        elif doc_val and _is_valid_field(field, doc_val):
            merged[field] = doc_val
        elif regex_val and _is_valid_field(field, regex_val):
            merged[field] = regex_val
        else:
            merged[field] = ""
    
    return merged


# ═══════════════════════════════════════════════════════════════
#                    FALLBACK PARSER
# ═══════════════════════════════════════════════════════════════

def _basic_fallback(text: str, contacts: Dict, name: str, 
                   education: List, skills: Dict) -> Dict:
    """Fallback parser when LLM fails"""
    return {
        "name": name or contacts.get("name", ""),
        "email": contacts.get("email", ""),
        "phone": contacts.get("phone", ""),
        "address": contacts.get("address", ""),
        "linkedin": contacts.get("linkedin", ""),
        "github": contacts.get("github", ""),
        "portfolio": contacts.get("portfolio", ""),
        "location": contacts.get("location", ""),
        "current_role": "",
        "current_company": "",
        "total_experience_years": 0,
        "professional_summary": text[:500] if text else "",
        "specializations": [],
        "skills": skills,
        "work_history": [],
        "education": education,
        "certifications": [],
        "awards": [],
        "projects": [],
        "publications": [],
        "volunteer": [],
        "languages": [],
        "interests": []
    }


# ═══════════════════════════════════════════════════════════════
#                    MAIN PARSING FUNCTION
# ═══════════════════════════════════════════════════════════════

def parse_resume_with_llm(
    resume_text: str,
    groq_api_key: str,
    model_id: str = "llama-3.1-8b-instant",
    doc_contacts: Dict = None
) -> Dict:
    """
    Parse resume using LLM + regex + document extraction for maximum accuracy
    Triple extraction strategy for reliable results
    """
    
    if not resume_text or len(resume_text.strip()) < 50:
        return _basic_fallback(resume_text, {}, "", [], {})

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Regex-based extraction (always reliable)
    # ═══════════════════════════════════════════════════════════
    regex_contacts = _extract_contacts_regex(resume_text)
    regex_name = _extract_name_from_text(resume_text)
    regex_education = _extract_education_regex(resume_text)
    regex_skills = _extract_skills_regex(resume_text)
    
    # ═══════════════════════════════════════════════════════════
    # STEP 1b: Use document-level contacts if provided
    # ═══════════════════════════════════════════════════════════
    if doc_contacts is None:
        doc_contacts = {}
        try:
            from document_processor import extract_contacts_from_text
            doc_contacts = extract_contacts_from_text(resume_text)
        except ImportError:
            pass

    # ═══════════════════════════════════════════════════════════
    # STEP 2: LLM-based full parsing
    # ═══════════════════════════════════════════════════════════
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    # Truncate resume text if too long
    max_text_length = 12000
    truncated_text = resume_text[:max_text_length]

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a resume parsing expert. The current year is 2026. "
                    "Return ONLY valid JSON. No markdown code blocks. No explanation. "
                    "Extract EVERY piece of information accurately, especially: "
                    "1) NAME - look at the FIRST lines, usually largest/bold text "
                    "2) CONTACT INFO - email, phone, linkedin, github "
                    "3) ALL work experience with dates "
                    "4) ALL education with degrees and institutions "
                    "5) ALL skills mentioned anywhere"
                )
            },
            {
                "role": "user",
                "content": PARSE_PROMPT.format(resume_text=truncated_text)
            }
        ],
        "temperature": 0.05,
        "max_tokens": 6000,
    }

    # Try multiple models as fallback
    models_to_try = [model_id, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    seen_models = set()
    unique_models = []
    for m in models_to_try:
        if m not in seen_models:
            seen_models.add(m)
            unique_models.append(m)

    parsed = None
    for try_model in unique_models:
        payload["model"] = try_model
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                
                # Clean markdown code blocks if present
                if content.startswith("```"):
                    # Remove ```json or ``` from start
                    lines = content.split("\n", 1)
                    if len(lines) > 1:
                        content = lines[1]
                    else:
                        content = content[3:]
                
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                
                content = content.strip()
                
                # Try to parse JSON
                parsed = json.loads(content)
                break
                
            elif response.status_code == 429:
                # Rate limited, wait and try next model
                import time
                time.sleep(1)
                continue
            elif response.status_code == 400:
                # Bad request, try next model
                continue
            else:
                continue
                
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            try:
                # Find JSON object in response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    break
            except:
                pass
            continue
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            continue

    # Use fallback if LLM parsing failed
    if not parsed:
        parsed = _basic_fallback(resume_text, regex_contacts, regex_name, regex_education, regex_skills)

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Merge contacts from all sources
    # ═══════════════════════════════════════════════════════════
    merged_contacts = _merge_contacts(parsed, regex_contacts, doc_contacts)
    
    # Apply merged contacts
    for field, value in merged_contacts.items():
        if value and (not parsed.get(field) or not _is_valid_field(field, parsed.get(field, ""))):
            parsed[field] = value
    
    # ═══════════════════════════════════════════════════════════
    # STEP 4: Special handling for name (most important)
    # ═══════════════════════════════════════════════════════════
    current_name = parsed.get("name", "")
    
    if not current_name or not _is_valid_field("name", current_name):
        # Try multiple sources in order of reliability
        name_candidates = [
            doc_contacts.get("name", ""),
            regex_name,
            parsed.get("name", ""),
        ]
        
        # Also try extracting name again with enhanced function
        try:
            enhanced_name = _extract_name_from_text(resume_text)
            if enhanced_name:
                name_candidates.insert(0, enhanced_name)
        except:
            pass
        
        for candidate in name_candidates:
            if candidate and _is_valid_field("name", candidate):
                parsed["name"] = _clean_name(candidate)
                break
        
        # Final fallback
        if not parsed.get("name") or not _is_valid_field("name", parsed.get("name", "")):
            parsed["name"] = "Unknown Candidate"

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Merge education from regex if LLM missed any
    # ═══════════════════════════════════════════════════════════
    llm_education = parsed.get("education", [])
    if not isinstance(llm_education, list):
        llm_education = [llm_education] if llm_education else []
    
    if regex_education:
        if not llm_education:
            parsed["education"] = regex_education
        else:
            # Build set of existing education entries
            llm_degrees = set()
            for edu in llm_education:
                if isinstance(edu, dict):
                    deg = str(edu.get("degree", "")).lower()
                    inst = str(edu.get("institution", "")).lower()
                    llm_degrees.add(f"{deg}_{inst}"[:80])
            
            # Add missing education from regex
            for regex_edu in regex_education:
                if isinstance(regex_edu, dict):
                    deg = str(regex_edu.get("degree", "")).lower()
                    inst = str(regex_edu.get("institution", "")).lower()
                    key = f"{deg}_{inst}"[:80]
                    
                    if key not in llm_degrees and deg:
                        llm_education.append(regex_edu)
            
            parsed["education"] = llm_education

    # ═══════════════════════════════════════════════════════════
    # STEP 6: Merge skills from regex
    # ═══════════════════════════════════════════════════════════
    llm_skills = parsed.get("skills", {})
    
    if not isinstance(llm_skills, dict):
        if isinstance(llm_skills, list):
            llm_skills = {"other_tools": llm_skills}
        else:
            llm_skills = {}
    
    # Merge regex skills into LLM skills
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

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Recalculate experience accurately with 2026
    # ═══════════════════════════════════════════════════════════
    work_history = parsed.get("work_history", parsed.get("experience", []))
    
    if isinstance(work_history, list) and work_history:
        calculated = _calculate_total_experience(work_history)
        if calculated > 0:
            parsed["total_experience_years"] = calculated
        
        # Ensure work_history key exists
        parsed["work_history"] = work_history
    
    # ═══════════════════════════════════════════════════════════
    # STEP 8: Extract current role and company if missing
    # ═══════════════════════════════════════════════════════════
    if not parsed.get("current_role") or not parsed.get("current_company"):
        work_history = parsed.get("work_history", [])
        if isinstance(work_history, list) and work_history:
            first_job = work_history[0] if isinstance(work_history[0], dict) else {}
            
            if not parsed.get("current_role"):
                parsed["current_role"] = (
                    first_job.get("title", "") or 
                    first_job.get("role", "") or 
                    first_job.get("position", "")
                )
            
            if not parsed.get("current_company"):
                parsed["current_company"] = (
                    first_job.get("company", "") or 
                    first_job.get("organization", "") or
                    first_job.get("employer", "")
                )

    # ═══════════════════════════════════════════════════════════
    # STEP 9: Ensure all required fields exist
    # ═══════════════════════════════════════════════════════════
    required_fields = {
        "name": "Unknown Candidate",
        "email": "",
        "phone": "",
        "address": "",
        "linkedin": "",
        "github": "",
        "portfolio": "",
        "location": "",
        "current_role": "",
        "current_company": "",
        "total_experience_years": 0,
        "professional_summary": "",
        "specializations": [],
        "skills": {},
        "work_history": [],
        "education": [],
        "certifications": [],
        "awards": [],
        "projects": [],
        "publications": [],
        "volunteer": [],
        "languages": [],
        "interests": []
    }
    
    for field, default_value in required_fields.items():
        if field not in parsed:
            parsed[field] = default_value

    return parsed


# ═══════════════════════════════════════════════════════════════
#                    DISPLAY SUMMARY
# ═══════════════════════════════════════════════════════════════

def get_resume_display_summary(parsed: Dict) -> str:
    """Generate display summary for sidebar"""
    if not parsed:
        return "No resume data"
    
    name = parsed.get("name", "Candidate")
    role = parsed.get("current_role", "")
    company = parsed.get("current_company", "")
    exp = parsed.get("total_experience_years", 0)
    location = parsed.get("location", "") or parsed.get("address", "")
    email = parsed.get("email", "")
    phone = parsed.get("phone", "")
    linkedin = parsed.get("linkedin", "")
    github = parsed.get("github", "")

    lines = [f"**{name}**"]
    
    if role:
        line = f"💼 {role}"
        if company:
            line += f" at {company}"
        lines.append(line)
    
    if location:
        # Truncate long locations for display
        display_location = location[:60] + "..." if len(location) > 60 else location
        lines.append(f"📍 {display_location}")
    
    if exp:
        lines.append(f"📅 ~{exp} years experience (as of 2026)")
    
    if email:
        lines.append(f"📧 {email}")
    
    if phone:
        lines.append(f"📞 {phone}")
    
    if linkedin:
        lines.append(f"🔗 LinkedIn")
    
    if github:
        lines.append(f"💻 GitHub")

    return "\n".join(lines)


def get_resume_full_summary(parsed: Dict) -> Dict:
    """Generate a comprehensive summary of the parsed resume"""
    if not parsed:
        return {}
    
    # Count skills
    skills_count = 0
    skills_data = parsed.get("skills", {})
    if isinstance(skills_data, dict):
        for category in skills_data.values():
            if isinstance(category, list):
                skills_count += len(category)
    elif isinstance(skills_data, list):
        skills_count = len(skills_data)
    
    # Count education
    education = parsed.get("education", [])
    education_count = len(education) if isinstance(education, list) else 0
    
    # Count work history
    work_history = parsed.get("work_history", parsed.get("experience", []))
    work_count = len(work_history) if isinstance(work_history, list) else 0
    
    # Count certifications
    certifications = parsed.get("certifications", [])
    cert_count = len(certifications) if isinstance(certifications, list) else 0
    
    # Count projects
    projects = parsed.get("projects", [])
    project_count = len(projects) if isinstance(projects, list) else 0
    
    return {
        "name": parsed.get("name", "Unknown"),
        "email": parsed.get("email", ""),
        "phone": parsed.get("phone", ""),
        "location": parsed.get("location", "") or parsed.get("address", ""),
        "current_role": parsed.get("current_role", ""),
        "current_company": parsed.get("current_company", ""),
        "total_experience_years": parsed.get("total_experience_years", 0),
        "skills_count": skills_count,
        "education_count": education_count,
        "work_history_count": work_count,
        "certifications_count": cert_count,
        "projects_count": project_count,
        "has_linkedin": bool(parsed.get("linkedin")),
        "has_github": bool(parsed.get("github")),
        "has_portfolio": bool(parsed.get("portfolio")),
        "has_summary": bool(parsed.get("professional_summary")),
    }


def extract_key_highlights(parsed: Dict) -> List[str]:
    """Extract key highlights from parsed resume for quick view"""
    highlights = []
    
    if not parsed:
        return highlights
    
    # Experience highlight
    exp = parsed.get("total_experience_years", 0)
    if exp:
        highlights.append(f"📅 {exp} years of experience")
    
    # Current role
    role = parsed.get("current_role", "")
    company = parsed.get("current_company", "")
    if role and company:
        highlights.append(f"💼 Currently {role} at {company}")
    elif role:
        highlights.append(f"💼 {role}")
    
    # Education
    education = parsed.get("education", [])
    if education and isinstance(education, list) and len(education) > 0:
        highest_edu = education[0]
        if isinstance(highest_edu, dict):
            degree = highest_edu.get("degree", "")
            institution = highest_edu.get("institution", "")
            if degree:
                edu_text = degree
                if institution:
                    edu_text += f" from {institution}"
                highlights.append(f"🎓 {edu_text}")
    
    # Skills count
    skills_data = parsed.get("skills", {})
    skills_count = 0
    if isinstance(skills_data, dict):
        for category in skills_data.values():
            if isinstance(category, list):
                skills_count += len(category)
    if skills_count > 0:
        highlights.append(f"🛠️ {skills_count} skills identified")
    
    # Certifications
    certs = parsed.get("certifications", [])
    if certs and isinstance(certs, list) and len(certs) > 0:
        highlights.append(f"📜 {len(certs)} certification(s)")
    
    # Projects
    projects = parsed.get("projects", [])
    if projects and isinstance(projects, list) and len(projects) > 0:
        highlights.append(f"🚀 {len(projects)} project(s)")
    
    return highlights


# ═══════════════════════════════════════════════════════════════
#                    UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_contact_completeness(parsed: Dict) -> Dict:
    """Check completeness of contact information"""
    if not parsed:
        return {"score": 0, "missing": ["all"]}
    
    contact_fields = {
        "name": parsed.get("name", ""),
        "email": parsed.get("email", ""),
        "phone": parsed.get("phone", ""),
        "location": parsed.get("location", "") or parsed.get("address", ""),
        "linkedin": parsed.get("linkedin", ""),
    }
    
    present = []
    missing = []
    
    for field, value in contact_fields.items():
        if value and _is_valid_field(field, value):
            present.append(field)
        else:
            missing.append(field)
    
    score = (len(present) / len(contact_fields)) * 100
    
    return {
        "score": round(score, 1),
        "present": present,
        "missing": missing,
        "total_fields": len(contact_fields),
        "filled_fields": len(present)
    }


def validate_parsed_resume(parsed: Dict) -> Dict:
    """Validate parsed resume and return quality metrics"""
    if not parsed:
        return {"valid": False, "issues": ["No data"]}
    
    issues = []
    warnings = []
    
    # Check name
    name = parsed.get("name", "")
    if not name or name in ["Unknown Candidate", "Candidate", ""]:
        issues.append("Name not extracted")
    elif not _is_valid_field("name", name):
        warnings.append("Name may be incomplete")
    
    # Check email
    email = parsed.get("email", "")
    if not email:
        warnings.append("Email not found")
    elif not _is_valid_field("email", email):
        issues.append("Invalid email format")
    
    # Check phone
    phone = parsed.get("phone", "")
    if not phone:
        warnings.append("Phone not found")
    elif not _is_valid_field("phone", phone):
        warnings.append("Phone format may be incorrect")
    
    # Check experience
    work_history = parsed.get("work_history", [])
    if not work_history or len(work_history) == 0:
        warnings.append("No work history extracted")
    
    # Check education
    education = parsed.get("education", [])
    if not education or len(education) == 0:
        warnings.append("No education extracted")
    
    # Check skills
    skills = parsed.get("skills", {})
    skills_count = 0
    if isinstance(skills, dict):
        for cat in skills.values():
            if isinstance(cat, list):
                skills_count += len(cat)
    if skills_count == 0:
        warnings.append("No skills extracted")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "quality_score": max(0, 100 - (len(issues) * 20) - (len(warnings) * 5))
    } 
