"""
LLM-powered Resume Parser
Extracts structured data from raw resume text using Groq.
Works for ANY domain: IT, Healthcare, Finance, Marketing, etc.
"""

import json
import requests
from typing import Dict, Optional


PARSE_PROMPT = """You are an expert resume parser. Analyze the following resume text and extract structured information.

RESUME TEXT:
{resume_text}

Extract the following information and return as valid JSON (no markdown, no code blocks):

{{
    "name": "Full name of the candidate",
    "email": "Email address or empty string",
    "phone": "Phone number or empty string",
    "linkedin": "LinkedIn URL or empty string",
    "location": "City, State/Country or empty string",
    "current_role": "Current/latest job title",
    "current_company": "Current/latest company name",
    "total_experience_years": 0,
    "professional_summary": "Brief professional summary from resume",
    "specializations": ["list", "of", "key", "specializations"],
    "skills": {{
        "programming": ["languages"],
        "frameworks": ["frameworks and libraries"],
        "tools": ["tools and technologies"],
        "cloud": ["cloud platforms"],
        "databases": ["databases"],
        "other": ["other skills"]
    }},
    "work_history": [
        {{
            "title": "Job title",
            "company": "Company name",
            "location": "Location",
            "duration": "Start - End",
            "years": 0.0,
            "type": "Full-time/Internship/Contract",
            "key_achievements": ["achievement 1", "achievement 2"]
        }}
    ],
    "education": [
        {{
            "degree": "Degree name",
            "institution": "Institution name",
            "year": "Year or duration",
            "gpa": "GPA if mentioned"
        }}
    ],
    "certifications": ["cert 1", "cert 2"],
    "awards": ["award 1", "award 2"],
    "projects": ["project 1", "project 2"],
    "languages": ["English", "other languages if mentioned"]
}}

RULES:
1. Extract ONLY information present in the resume
2. If information is not found, use empty string or empty list
3. Estimate total_experience_years from work history
4. For work_history years, estimate duration in decimal years
5. Return ONLY valid JSON, no other text
6. Handle ANY domain: IT, Healthcare, Finance, Marketing, Engineering, etc.
"""


def parse_resume_with_llm(
    resume_text: str,
    groq_api_key: str,
    model_id: str = "llama-3.1-8b-instant"
) -> Dict:
    """
    Parse resume text into structured data using LLM.
    Works for any domain resume.
    """
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
                    "You are a resume parsing expert. "
                    "Return ONLY valid JSON. No markdown. No code blocks. "
                    "No explanation. Just pure JSON."
                )
            },
            {
                "role": "user",
                "content": PARSE_PROMPT.format(
                    resume_text=resume_text[:8000]
                )
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    # Try multiple models for reliability
    models = [model_id, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    seen = set()
    unique = [m for m in models if m not in seen and not seen.add(m)]

    for try_model in unique:
        payload["model"] = try_model
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Clean up response
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

                parsed = json.loads(content)
                return parsed

            elif response.status_code == 400:
                continue

        except json.JSONDecodeError:
            continue
        except Exception:
            continue

    # Fallback: return basic structure from text
    return _basic_parse(resume_text)


def _basic_parse(text: str) -> Dict:
    """Fallback parser when LLM parsing fails"""
    lines = text.split("\n")
    name = lines[0].strip() if lines else "Unknown"

    # Simple email extraction
    import re
    email_match = re.search(
        r'[\w.-]+@[\w.-]+\.\w+', text
    )
    email = email_match.group(0) if email_match else ""

    # Simple phone extraction
    phone_match = re.search(
        r'[\+]?[\d\s\-\(\)]{10,}', text
    )
    phone = phone_match.group(0).strip() if phone_match else ""

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": "",
        "location": "",
        "current_role": "Not parsed",
        "current_company": "Not parsed",
        "total_experience_years": 0,
        "professional_summary": text[:500],
        "specializations": [],
        "skills": {
            "programming": [],
            "frameworks": [],
            "tools": [],
            "cloud": [],
            "databases": [],
            "other": []
        },
        "work_history": [],
        "education": [],
        "certifications": [],
        "awards": [],
        "projects": [],
        "languages": []
    }


def get_resume_display_summary(parsed: Dict) -> str:
    """Generate a display-friendly summary from parsed resume"""
    name = parsed.get("name", "Unknown")
    role = parsed.get("current_role", "N/A")
    company = parsed.get("current_company", "N/A")
    exp = parsed.get("total_experience_years", 0)
    location = parsed.get("location", "N/A")

    all_skills = []
    skills_dict = parsed.get("skills", {})
    for category in skills_dict.values():
        if isinstance(category, list):
            all_skills.extend(category[:5])

    summary = f"**{name}**\n"
    summary += f"📍 {location} | 💼 {role} at {company}\n"
    summary += f"📅 ~{exp} years experience\n"
    if all_skills:
        summary += f"🛠️ {', '.join(all_skills[:8])}"

    return summary