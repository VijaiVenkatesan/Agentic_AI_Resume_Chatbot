"""
Enhanced Resume Parser
- Dual extraction: Regex (reliable for contacts) + LLM (intelligent parsing)
- Accurate experience calculation using CURRENT_YEAR = 2026
- Works for ANY domain resume
"""

import json
import re
import requests
from typing import Dict

CURRENT_YEAR = 2026
CURRENT_MONTH = 3


PARSE_PROMPT = """You are an expert resume parser. The current date is July 2026.

RESUME TEXT:
{resume_text}

Extract ALL information into this exact JSON structure (no markdown, no code blocks):

{{
    "name": "FULL name exactly as written on resume - check the very first lines",
    "email": "email address - look for @ symbol anywhere in text",
    "phone": "complete phone number with country code if present",
    "address": "full physical/mailing address if mentioned",
    "linkedin": "LinkedIn URL if mentioned",
    "github": "GitHub URL if mentioned",
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
            "degree": "degree name",
            "field": "field of study",
            "institution": "school name",
            "location": "location",
            "start_year": "year",
            "end_year": "year",
            "gpa": "GPA or percentage",
            "achievements": ["honors", "coursework"]
        }}
    ],
    "certifications": [
        {{
            "name": "cert name",
            "provider": "issuer",
            "date": "date",
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
1. NAME: Look at the very first non-empty line of the resume - that is usually the name
2. PHONE: Look for +91, +1, (XXX), or any 10+ digit number patterns
3. EMAIL: Find anything with @ symbol
4. ADDRESS: Look for street numbers, city names, pin/zip codes
5. EXPERIENCE: If end_date is "Present" or "Current", calculate duration until July 2026
6. duration_years: Calculate as decimal (2 years 6 months = 2.5)
7. total_experience_years: Sum of all duration_years values
8. Extract EVERY skill mentioned ANYWHERE in the resume
9. Include ALL bullet points from work experience
10. Return ONLY valid JSON - no other text"""


def _extract_contacts_regex(text: str) -> Dict:
    """Extract contact details using regex patterns - very reliable"""
    contacts = {
        "email": "", "phone": "", "address": "",
        "linkedin": "", "github": "", "portfolio": ""
    }

    # Email - most reliable
    email_match = re.search(r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}', text)
    if email_match:
        contacts["email"] = email_match.group(0).strip()

    # Phone - multiple international formats
    phone_patterns = [
        r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        r'\+\d{1,3}\s?\d{4,5}\s?\d{5,6}',
        r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
        r'\+\d{10,13}',
        r'\d{10,13}',
    ]
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            cleaned = re.sub(r'[^\d+\-() ]', '', m).strip()
            digits_only = re.sub(r'[^\d]', '', cleaned)
            if 10 <= len(digits_only) <= 15:
                contacts["phone"] = cleaned
                break
        if contacts["phone"]:
            break

    # LinkedIn
    linkedin = re.search(
        r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?', text, re.IGNORECASE
    )
    if linkedin:
        contacts["linkedin"] = linkedin.group(0).strip()

    # GitHub
    github = re.search(
        r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?', text, re.IGNORECASE
    )
    if github:
        contacts["github"] = github.group(0).strip()

    # Portfolio / Website
    website = re.search(
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|io|dev|me|org|net)/?\S*',
        text, re.IGNORECASE
    )
    if website:
        url = website.group(0).strip()
        if 'linkedin' not in url.lower() and 'github' not in url.lower():
            contacts["portfolio"] = url

    # Address - look for street/city/pin patterns
    address_patterns = [
        r'(?:No[.:]?\s*)?[\d]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar|Colony|Layout|Block|Lane)[\w\s,.-]+(?:\d{5,6})',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}',
        r'[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar)\s*,[\w\s,.-]+\d{5,6}',
    ]
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            addr = match.group(0).strip()
            if len(addr) > 15 and len(addr) < 200:
                contacts["address"] = addr
                break

    return contacts


def _extract_name_from_text(text: str) -> str:
    """Extract candidate name from first few lines"""
    lines = text.strip().split("\n")
    for line in lines[:8]:
        line = line.strip()
        if not line or len(line) < 2:
            continue
        # Skip non-name lines
        if '@' in line:
            continue
        if re.search(r'\d{5,}', line):
            continue
        if re.match(r'^[\+\d\(\)]', line):
            continue
        skip_keywords = [
            'resume', 'curriculum', 'cv', 'http', 'www',
            'address', 'phone', 'email', 'street', 'road',
            'objective', 'summary', 'profile', 'linkedin',
            'github', 'portfolio', 'mobile', 'tel:'
        ]
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        # Name should be mostly alphabetic
        alpha_count = sum(1 for c in line if c.isalpha() or c.isspace() or c == '.')
        if alpha_count / max(len(line), 1) > 0.75 and len(line) < 50:
            # Clean up
            name = re.sub(r'\s+', ' ', line).strip()
            if len(name) >= 3:
                return name

    return "Candidate"


def _parse_date_to_ym(date_str: str, month_map: Dict) -> tuple:
    """Parse date string to (year, month)"""
    date_str = date_str.strip().lower()

    # "Month Year" format
    for month_name, month_num in month_map.items():
        if month_name in date_str:
            year_match = re.search(r'(19|20)\d{2}', date_str)
            if year_match:
                return int(year_match.group()), month_num

    # "MM/YYYY" or "MM-YYYY"
    match = re.search(r'(\d{1,2})[/-]((?:19|20)\d{2})', date_str)
    if match:
        return int(match.group(2)), int(match.group(1))

    # "YYYY" only
    year_match = re.search(r'(19|20)\d{2}', date_str)
    if year_match:
        return int(year_match.group()), 1

    return None, None


def _calculate_total_experience(work_history: list) -> float:
    """Calculate total experience using current year 2026"""
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
        'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
        'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
    }

    total = 0.0
    for job in work_history:
        start_str = job.get("start_date", "")
        end_str = job.get("end_date", "")

        if not start_str:
            # Use pre-calculated duration if available
            dur = job.get("duration_years", 0)
            try:
                total += float(dur) if dur else 0
            except (ValueError, TypeError):
                pass
            continue

        start_y, start_m = _parse_date_to_ym(start_str, month_map)
        if not start_y:
            dur = job.get("duration_years", 0)
            try:
                total += float(dur) if dur else 0
            except (ValueError, TypeError):
                pass
            continue

        # Parse end date
        present_words = ['present', 'current', 'now', 'ongoing', 'till date', 'till now', 'today']
        if not end_str or end_str.strip().lower() in present_words:
            end_y = CURRENT_YEAR
            end_m = CURRENT_MONTH
        else:
            end_y, end_m = _parse_date_to_ym(end_str, month_map)
            if not end_y:
                end_y = CURRENT_YEAR
                end_m = CURRENT_MONTH

        months = (end_y - start_y) * 12 + (end_m - start_m)
        years = max(0, months / 12.0)
        total += years

    return round(total, 1)


def parse_resume_with_llm(
    resume_text: str,
    groq_api_key: str,
    model_id: str = "llama-3.1-8b-instant"
) -> Dict:
    """Parse resume using LLM + regex supplementation for reliability"""

    # Step 1: Regex-based contact extraction (always reliable)
    regex_contacts = _extract_contacts_regex(resume_text)
    regex_name = _extract_name_from_text(resume_text)

    # Step 2: LLM-based full parsing
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a resume parsing expert. The current year is 2026. "
                    "Return ONLY valid JSON. No markdown code blocks. No explanation."
                )
            },
            {
                "role": "user",
                "content": PARSE_PROMPT.format(resume_text=resume_text[:12000])
            }
        ],
        "temperature": 0.05,
        "max_tokens": 6000,
    }

    models = [model_id, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    seen = set()
    unique = [m for m in models if m not in seen and not seen.add(m)]

    parsed = None
    for try_model in unique:
        payload["model"] = try_model
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=45
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                # Clean markdown code blocks
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                parsed = json.loads(content.strip())
                break
            elif response.status_code == 400:
                continue
        except json.JSONDecodeError:
            continue
        except Exception:
            continue

    if not parsed:
        parsed = _basic_fallback(resume_text)

    # Step 3: Fill gaps with regex results (regex is more reliable for contacts)
    if not parsed.get("name") or parsed["name"] in ["", "Unknown", "N/A", "Not found", "Candidate"]:
        parsed["name"] = regex_name
    if not parsed.get("email") and regex_contacts["email"]:
        parsed["email"] = regex_contacts["email"]
    if not parsed.get("phone") and regex_contacts["phone"]:
        parsed["phone"] = regex_contacts["phone"]
    if not parsed.get("address") and regex_contacts.get("address"):
        parsed["address"] = regex_contacts["address"]
    if not parsed.get("linkedin") and regex_contacts["linkedin"]:
        parsed["linkedin"] = regex_contacts["linkedin"]
    if not parsed.get("github") and regex_contacts["github"]:
        parsed["github"] = regex_contacts["github"]
    if not parsed.get("portfolio") and regex_contacts.get("portfolio"):
        parsed["portfolio"] = regex_contacts["portfolio"]

    # Step 4: Recalculate experience accurately with 2026
    work_history = parsed.get("work_history", [])
    if work_history:
        calculated = _calculate_total_experience(work_history)
        if calculated > 0:
            parsed["total_experience_years"] = calculated

    return parsed


def _basic_fallback(text: str) -> Dict:
    """Fallback parser when LLM fails"""
    contacts = _extract_contacts_regex(text)
    name = _extract_name_from_text(text)
    return {
        "name": name,
        "email": contacts.get("email", ""),
        "phone": contacts.get("phone", ""),
        "address": contacts.get("address", ""),
        "linkedin": contacts.get("linkedin", ""),
        "github": contacts.get("github", ""),
        "portfolio": contacts.get("portfolio", ""),
        "location": "",
        "current_role": "",
        "current_company": "",
        "total_experience_years": 0,
        "professional_summary": text[:500],
        "specializations": [],
        "skills": {
            "programming_languages": [], "frameworks_libraries": [],
            "ai_ml_tools": [], "cloud_platforms": [],
            "databases": [], "devops_tools": [],
            "visualization": [], "other_tools": [],
            "soft_skills": []
        },
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


def get_resume_display_summary(parsed: Dict) -> str:
    """Generate display summary for sidebar"""
    name = parsed.get("name", "Candidate")
    role = parsed.get("current_role", "")
    company = parsed.get("current_company", "")
    exp = parsed.get("total_experience_years", 0)
    location = parsed.get("location", "")
    email = parsed.get("email", "")
    phone = parsed.get("phone", "")
    address = parsed.get("address", "")

    lines = [f"**{name}**"]
    if role:
        line = f"💼 {role}"
        if company:
            line += f" at {company}"
        lines.append(line)
    if location:
        lines.append(f"📍 {location}")
    elif address:
        lines.append(f"📍 {address[:50]}")
    if exp:
        lines.append(f"📅 ~{exp} years experience (as of 2026)")
    if email:
        lines.append(f"📧 {email}")
    if phone:
        lines.append(f"📞 {phone}")

    return "\n".join(lines)
