"""
Bulk Resume Processor - Enterprise HR Feature
Handles multiple resume uploads and batch JD matching
Version: 2.0 - Full values, no truncation
"""

import time
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd

from document_processor import process_uploaded_file
from resume_parser import parse_resume_with_llm

CURRENT_YEAR = 2026


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
    
    # Details - NO LIMITS
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


def extract_jd_requirements(jd_text: str) -> Dict:
    """Extract key requirements from Job Description"""
    jd_lower = jd_text.lower()
    
    requirements = {
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
    
    # Extract experience requirements
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
    
    # Extract skills - comprehensive tech skills
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
    
    skills_found = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, jd_lower)
        skills_found.update([m.strip() for m in matches if m.strip()])
    
    requirements["required_skills"] = list(skills_found)
    
    # Extract education requirements
    edu_patterns = [
        r"(bachelor'?s?|master'?s?|phd|ph\.d|doctorate|mba|m\.b\.a)",
        r'(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?sc|m\.?sc|b\.?ca|m\.?ca|b\.?com|m\.?com)',
        r'(computer science|information technology|software engineering|engineering|mathematics|statistics|physics)',
        r'(degree|graduate|post.?graduate|undergraduate)',
    ]
    
    edu_found = set()
    for pattern in edu_patterns:
        matches = re.findall(pattern, jd_lower)
        edu_found.update([m.strip() for m in matches if m.strip()])
    
    requirements["required_education"] = list(edu_found)
    
    # Extract locations
    location_patterns = [
        r'\b(remote|hybrid|on-?site|work from home|wfh|telecommute)\b',
        r'\b(bangalore|bengaluru|hyderabad|chennai|mumbai|delhi|ncr|pune|noida|gurgaon|gurugram|kolkata)\b',
        r'\b(new york|nyc|san francisco|sf|seattle|austin|boston|chicago|los angeles|la|denver|atlanta)\b',
        r'\b(london|berlin|amsterdam|dublin|singapore|tokyo|sydney|toronto|vancouver)\b',
        r'\b(india|usa|us|uk|united states|united kingdom|canada|australia|germany|netherlands)\b',
    ]
    
    locations_found = set()
    for pattern in location_patterns:
        matches = re.findall(pattern, jd_lower)
        locations_found.update([m.strip() for m in matches if m.strip()])
    
    requirements["preferred_locations"] = list(locations_found)
    
    # Extract important keywords (excluding common stopwords)
    words = re.findall(r'\b[a-z]{4,}\b', jd_lower)
    word_freq = {}
    stopwords = {
        'with', 'have', 'will', 'that', 'this', 'from', 'they', 'been', 'were', 
        'said', 'each', 'which', 'their', 'would', 'there', 'could', 'other',
        'into', 'more', 'some', 'than', 'them', 'these', 'then', 'only', 'over',
        'such', 'about', 'should', 'your', 'work', 'experience', 'years', 'team',
        'ability', 'skills', 'role', 'position', 'looking', 'strong', 'good',
        'excellent', 'required', 'preferred', 'must', 'including', 'working',
        'understanding', 'knowledge', 'using', 'ability', 'responsibilities',
        'requirements', 'qualifications', 'company', 'join', 'opportunity',
        'environment', 'culture', 'benefits', 'salary', 'compensation',
        'what', 'when', 'where', 'while', 'being', 'having', 'doing', 'make',
        'like', 'just', 'also', 'well', 'back', 'after', 'before', 'between',
    }
    
    for word in words:
        if word not in stopwords and len(word) >= 4:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    requirements["keywords"] = [w[0] for w in sorted_words[:50]]
    
    return requirements


def calculate_match_scores(
    parsed_resume: Dict, 
    resume_text: str,
    jd_text: str,
    jd_requirements: Dict
) -> Dict:
    """Calculate detailed match scores between resume and JD"""
    
    scores = {
        "overall_score": 0.0,
        "skills_score": 0.0,
        "experience_score": 0.0,
        "education_score": 0.0,
        "location_score": 0.0,
        "keyword_score": 0.0,
        "matched_skills": [],
        "missing_skills": [],
        "strengths": [],
        "gaps": [],
        "recommendation": ""
    }
    
    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()
    
    # ═══════════════════════════════════════
    # 1. SKILLS MATCH (35% weight)
    # ═══════════════════════════════════════
    all_skills = []
    skills_data = parsed_resume.get("skills", {})
    
    if isinstance(skills_data, dict):
        for category in skills_data.values():
            if isinstance(category, list):
                all_skills.extend([s.lower().strip() for s in category if s])
    elif isinstance(skills_data, list):
        all_skills.extend([s.lower().strip() for s in skills_data if s])
    
    # Add specializations
    specs = parsed_resume.get("specializations", [])
    if isinstance(specs, list):
        all_skills.extend([s.lower().strip() for s in specs if s])
    
    # Add technologies from work history
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
        matched = []
        missing = []
        
        # Skill aliases for better matching
        skill_aliases = {
            "javascript": ["js", "ecmascript"],
            "typescript": ["ts"],
            "python": ["py"],
            "kubernetes": ["k8s"],
            "amazon web services": ["aws"],
            "google cloud": ["gcp"],
            "microsoft azure": ["azure"],
            "machine learning": ["ml"],
            "artificial intelligence": ["ai"],
            "natural language processing": ["nlp"],
            "deep learning": ["dl"],
            "react": ["reactjs", "react.js"],
            "node": ["nodejs", "node.js"],
            "angular": ["angularjs"],
            "vue": ["vuejs", "vue.js"],
            "postgres": ["postgresql"],
            "mongo": ["mongodb"],
        }
        
        for req_skill in required_skills:
            found = False
            
            # Direct match
            for resume_skill in all_skills:
                if req_skill in resume_skill or resume_skill in req_skill:
                    matched.append(req_skill)
                    found = True
                    break
            
            if not found:
                # Check aliases
                aliases = skill_aliases.get(req_skill, [])
                for alias in aliases:
                    for resume_skill in all_skills:
                        if alias in resume_skill or resume_skill in alias:
                            matched.append(req_skill)
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                # Check in full resume text as last resort
                if req_skill in resume_lower:
                    matched.append(req_skill)
                else:
                    missing.append(req_skill)
        
        scores["matched_skills"] = list(set(matched))
        scores["missing_skills"] = list(set(missing))
        
        match_ratio = len(matched) / len(required_skills) if required_skills else 0
        scores["skills_score"] = round(min(100, match_ratio * 110), 1)
    else:
        scores["skills_score"] = 70
    
    # ═══════════════════════════════════════
    # 2. EXPERIENCE MATCH (25% weight)
    # ═══════════════════════════════════════
    candidate_exp = parsed_resume.get("total_experience_years", 0)
    try:
        candidate_exp = float(candidate_exp)
    except (ValueError, TypeError):
        candidate_exp = 0
    
    min_exp = jd_requirements.get("min_experience_years", 0)
    max_exp = jd_requirements.get("max_experience_years", 99)
    
    if min_exp > 0:
        if candidate_exp >= min_exp:
            if candidate_exp <= max_exp:
                scores["experience_score"] = 100
            else:
                over_by = candidate_exp - max_exp
                scores["experience_score"] = max(65, 100 - (over_by * 3))
        else:
            if candidate_exp > 0:
                ratio = candidate_exp / min_exp
                scores["experience_score"] = round(min(95, ratio * 100), 1)
            else:
                scores["experience_score"] = 20
    else:
        scores["experience_score"] = 80 if candidate_exp > 0 else 50
    
    # ═══════════════════════════════════════
    # 3. EDUCATION MATCH (15% weight)
    # ═══════════════════════════════════════
    education = parsed_resume.get("education", [])
    required_edu = jd_requirements.get("required_education", [])
    
    edu_text = ""
    if isinstance(education, list):
        edu_text = json.dumps(education).lower()
    elif isinstance(education, dict):
        edu_text = json.dumps(education).lower()
    
    degree_levels = {
        "phd": 6, "ph.d": 6, "doctorate": 6, "doctoral": 6,
        "master": 5, "mba": 5, "m.tech": 5, "mtech": 5, "m.e": 5, "m.sc": 5, "mca": 5, "m.com": 5,
        "bachelor": 4, "b.tech": 4, "btech": 4, "b.e": 4, "b.sc": 4, "bca": 4, "b.com": 4,
        "graduate": 4, "degree": 3,
        "diploma": 2,
        "certificate": 1,
    }
    
    candidate_level = 0
    for degree, level in degree_levels.items():
        if degree in edu_text:
            candidate_level = max(candidate_level, level)
    
    required_level = 0
    for req in required_edu:
        req_lower = req.lower()
        for degree, level in degree_levels.items():
            if degree in req_lower:
                required_level = max(required_level, level)
    
    if required_level > 0:
        if candidate_level >= required_level:
            scores["education_score"] = 100
        elif candidate_level > 0:
            ratio = candidate_level / required_level
            scores["education_score"] = round(ratio * 85, 1)
        else:
            scores["education_score"] = 35
    else:
        scores["education_score"] = 80 if candidate_level > 0 else 50
    
    # ═══════════════════════════════════════
    # 4. LOCATION MATCH (10% weight)
    # ═══════════════════════════════════════
    candidate_location = (
        parsed_resume.get("location", "") or 
        parsed_resume.get("address", "")
    ).lower()
    
    preferred_locations = [loc.lower() for loc in jd_requirements.get("preferred_locations", [])]
    
    if preferred_locations:
        remote_keywords = ["remote", "work from home", "wfh", "telecommute", "hybrid"]
        if any(kw in preferred_locations for kw in remote_keywords):
            scores["location_score"] = 100
        elif candidate_location:
            location_match = any(loc in candidate_location or candidate_location in loc 
                               for loc in preferred_locations if loc not in remote_keywords)
            if location_match:
                scores["location_score"] = 100
            else:
                scores["location_score"] = 55
        else:
            scores["location_score"] = 50
    else:
        scores["location_score"] = 85
    
    # ═══════════════════════════════════════
    # 5. KEYWORD MATCH (15% weight)
    # ═══════════════════════════════════════
    keywords = jd_requirements.get("keywords", [])
    
    if keywords:
        matched_keywords = sum(1 for kw in keywords if kw in resume_lower)
        keyword_ratio = matched_keywords / len(keywords)
        scores["keyword_score"] = round(min(100, keyword_ratio * 140), 1)
    else:
        scores["keyword_score"] = 70
    
    # ═══════════════════════════════════════
    # CALCULATE OVERALL SCORE (Weighted Average)
    # ═══════════════════════════════════════
    scores["overall_score"] = round(
        scores["skills_score"] * 0.35 +
        scores["experience_score"] * 0.25 +
        scores["education_score"] * 0.15 +
        scores["location_score"] * 0.10 +
        scores["keyword_score"] * 0.15,
        1
    )
    
    scores["overall_score"] = min(100, scores["overall_score"])
    
    # ═══════════════════════════════════════
    # IDENTIFY STRENGTHS & GAPS
    # ═══════════════════════════════════════
    
    if scores["skills_score"] >= 75:
        scores["strengths"].append(f"Strong skill match ({scores['skills_score']:.0f}%)")
    elif scores["skills_score"] >= 60:
        scores["strengths"].append(f"Good skill alignment ({scores['skills_score']:.0f}%)")
    
    if scores["experience_score"] >= 90:
        scores["strengths"].append(f"Experience exceeds requirements ({candidate_exp}y)")
    elif scores["experience_score"] >= 75:
        scores["strengths"].append(f"Experience meets requirements ({candidate_exp}y)")
    
    if scores["education_score"] >= 85:
        scores["strengths"].append("Education qualifications match well")
    
    if scores["location_score"] >= 90:
        scores["strengths"].append("Location compatible")
    
    if scores["keyword_score"] >= 70:
        scores["strengths"].append("Good keyword/context match with JD")
    
    if scores["skills_score"] < 50:
        missing_count = len(scores["missing_skills"])
        scores["gaps"].append(f"Significant skill gaps ({missing_count} missing skills)")
    elif scores["skills_score"] < 65:
        scores["gaps"].append(f"Some skill gaps detected")
    
    if scores["experience_score"] < 60:
        scores["gaps"].append(f"Experience gap ({candidate_exp}y vs {min_exp}y required)")
    
    if scores["education_score"] < 50:
        scores["gaps"].append("Education may not meet requirements")
    
    if scores["location_score"] < 60:
        scores["gaps"].append("Location may not align with job requirements")
    
    if scores["keyword_score"] < 50:
        scores["gaps"].append("Low keyword match - consider tailoring resume")
    
    # ═══════════════════════════════════════
    # RECOMMENDATION
    # ═══════════════════════════════════════
    overall = scores["overall_score"]
    if overall >= 80:
        scores["recommendation"] = "🟢 Excellent Match - Highly Recommended"
    elif overall >= 65:
        scores["recommendation"] = "🟡 Good Match - Worth Considering"
    elif overall >= 50:
        scores["recommendation"] = "🟠 Moderate Match - Review Needed"
    elif overall >= 35:
        scores["recommendation"] = "🔴 Below Average - Significant Gaps"
    else:
        scores["recommendation"] = "⚫ Low Match - May Not Fit"
    
    return scores


def get_highest_education(parsed_resume: Dict) -> str:
    """Extract highest education from parsed resume - FULL VALUE, NO TRUNCATION"""
    education = parsed_resume.get("education", [])
    
    if not education:
        return "Not specified"
    
    degree_priority = {
        "phd": 7, "ph.d": 7, "doctorate": 7, "doctor": 7,
        "master": 6, "mba": 6, "m.tech": 6, "mtech": 6, "m.e": 6, "m.sc": 6, "mca": 6, "m.com": 6,
        "bachelor": 5, "b.tech": 5, "btech": 5, "b.e": 5, "b.sc": 5, "bca": 5, "b.com": 5,
        "associate": 4,
        "diploma": 3,
        "12th": 2, "hsc": 2, "higher secondary": 2, "intermediate": 2,
        "10th": 1, "ssc": 1, "secondary": 1, "matriculation": 1,
    }
    
    highest = ""
    highest_priority = 0
    
    if isinstance(education, list):
        for edu in education:
            if isinstance(edu, dict):
                degree = edu.get("degree", "").lower()
                
                for key, priority in degree_priority.items():
                    if key in degree:
                        if priority > highest_priority:
                            highest_priority = priority
                            
                            # Build FULL education string - NO TRUNCATION
                            parts = []
                            
                            if edu.get("degree"):
                                parts.append(edu.get("degree"))
                            
                            if edu.get("field") or edu.get("major") or edu.get("branch"):
                                field = edu.get("field") or edu.get("major") or edu.get("branch")
                                parts.append(f"in {field}")
                            
                            if edu.get("institution") or edu.get("university") or edu.get("college"):
                                inst = edu.get("institution") or edu.get("university") or edu.get("college")
                                parts.append(f"from {inst}")
                            
                            if edu.get("location"):
                                parts.append(f"({edu.get('location')})")
                            
                            if edu.get("year") or edu.get("end_year") or edu.get("graduation_year"):
                                year = edu.get("year") or edu.get("end_year") or edu.get("graduation_year")
                                parts.append(f"[{year}]")
                            
                            if edu.get("gpa") or edu.get("cgpa") or edu.get("grade") or edu.get("percentage"):
                                gpa = edu.get("gpa") or edu.get("cgpa") or edu.get("grade") or edu.get("percentage")
                                parts.append(f"- GPA/Grade: {gpa}")
                            
                            highest = " ".join(parts)
                        break
    
    return highest if highest else "Not specified"


def get_all_education_details(parsed_resume: Dict) -> str:
    """Get ALL education entries as a formatted string - NO TRUNCATION"""
    education = parsed_resume.get("education", [])
    
    if not education:
        return "Not specified"
    
    edu_entries = []
    
    if isinstance(education, list):
        for edu in education:
            if isinstance(edu, dict):
                parts = []
                
                if edu.get("degree"):
                    parts.append(edu.get("degree"))
                
                if edu.get("field") or edu.get("major") or edu.get("branch"):
                    field = edu.get("field") or edu.get("major") or edu.get("branch")
                    parts.append(f"in {field}")
                
                if edu.get("institution") or edu.get("university") or edu.get("college"):
                    inst = edu.get("institution") or edu.get("university") or edu.get("college")
                    parts.append(f"from {inst}")
                
                if edu.get("location"):
                    parts.append(f"({edu.get('location')})")
                
                if edu.get("year") or edu.get("end_year") or edu.get("graduation_year") or edu.get("start_year"):
                    start = edu.get("start_year", "")
                    end = edu.get("year") or edu.get("end_year") or edu.get("graduation_year", "")
                    if start and end:
                        parts.append(f"[{start} - {end}]")
                    elif end:
                        parts.append(f"[{end}]")
                
                if edu.get("gpa") or edu.get("cgpa") or edu.get("grade") or edu.get("percentage"):
                    gpa = edu.get("gpa") or edu.get("cgpa") or edu.get("grade") or edu.get("percentage")
                    parts.append(f"- GPA/Grade: {gpa}")
                
                if edu.get("achievements"):
                    achievements = edu.get("achievements")
                    if isinstance(achievements, list) and achievements:
                        parts.append(f"- Achievements: {', '.join(achievements)}")
                
                if parts:
                    edu_entries.append(" ".join(parts))
    
    return " || ".join(edu_entries) if edu_entries else "Not specified"


def get_all_work_history(parsed_resume: Dict) -> str:
    """Get ALL work history entries as a formatted string - NO TRUNCATION"""
    work_history = parsed_resume.get("work_history", parsed_resume.get("experience", []))
    
    if not work_history:
        return "Not specified"
    
    work_entries = []
    
    if isinstance(work_history, list):
        for job in work_history:
            if isinstance(job, dict):
                parts = []
                
                title = job.get("title") or job.get("role") or job.get("position", "")
                if title:
                    parts.append(title)
                
                company = job.get("company") or job.get("organization", "")
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


def get_all_skills(parsed_resume: Dict) -> str:
    """Get ALL skills as a formatted string - NO TRUNCATION"""
    all_skills = []
    
    skills_data = parsed_resume.get("skills", {})
    
    if isinstance(skills_data, dict):
        for category, skills_list in skills_data.items():
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
    elif isinstance(skills_data, list):
        all_skills.extend(skills_data)
    
    # Add specializations
    specs = parsed_resume.get("specializations", [])
    if isinstance(specs, list):
        all_skills.extend(specs)
    
    # Add technologies from work history
    work_history = parsed_resume.get("work_history", parsed_resume.get("experience", []))
    if isinstance(work_history, list):
        for job in work_history:
            if isinstance(job, dict):
                techs = job.get("technologies_used", job.get("technologies", []))
                if isinstance(techs, list):
                    all_skills.extend(techs)
    
    # Deduplicate while preserving order
    seen = set()
    unique_skills = []
    for skill in all_skills:
        if skill and skill.lower() not in seen:
            seen.add(skill.lower())
            unique_skills.append(skill)
    
    return ", ".join(unique_skills) if unique_skills else "Not specified"


def get_all_certifications(parsed_resume: Dict) -> str:
    """Get ALL certifications as a formatted string - NO TRUNCATION"""
    certifications = parsed_resume.get("certifications", [])
    
    if not certifications:
        return "Not specified"
    
    cert_entries = []
    
    if isinstance(certifications, list):
        for cert in certifications:
            if isinstance(cert, dict):
                parts = []
                
                name = cert.get("name", "")
                if name:
                    parts.append(name)
                
                provider = cert.get("provider") or cert.get("issuer") or cert.get("organization", "")
                if provider:
                    parts.append(f"by {provider}")
                
                date = cert.get("date") or cert.get("issue_date") or cert.get("year", "")
                if date:
                    parts.append(f"({date})")
                
                credential_id = cert.get("credential_id") or cert.get("id", "")
                if credential_id:
                    parts.append(f"[ID: {credential_id}]")
                
                if parts:
                    cert_entries.append(" ".join(parts))
            
            elif isinstance(cert, str) and cert:
                cert_entries.append(cert)
    
    return " || ".join(cert_entries) if cert_entries else "Not specified"


def process_single_resume_for_bulk(
    file_data: Dict,
    jd_text: str,
    jd_requirements: Dict,
    groq_api_key: str,
    model_id: str
) -> CandidateResult:
    """Process a single resume in bulk mode - NO TRUNCATION"""
    start_time = time.time()
    
    file_name = file_data.get("file_name", "Unknown")
    
    try:
        # Get text from already processed file
        resume_text = file_data.get("text", "")
        
        if not resume_text or len(resume_text.strip()) < 50:
            return CandidateResult(
                file_name=file_name,
                candidate_name="[Error]",
                email="",
                phone="",
                location="",
                current_role="",
                current_company="",
                total_experience=0,
                highest_education="",
                success=False,
                error="Could not extract text from resume",
                processing_time=round(time.time() - start_time, 2)
            )
        
        # Parse resume with LLM
        parsed = parse_resume_with_llm(resume_text, groq_api_key, model_id)
        
        # Calculate match scores
        scores = calculate_match_scores(parsed, resume_text, jd_text, jd_requirements)
        
        # Get candidate details - FULL VALUES
        candidate_name = parsed.get("name", "Unknown")
        if not candidate_name or candidate_name in ["", "N/A", "Unknown", "Candidate"]:
            candidate_name = file_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        
        total_exp = parsed.get("total_experience_years", 0)
        try:
            total_exp = float(total_exp)
        except (ValueError, TypeError):
            total_exp = 0
        
        # Build result - NO TRUNCATION on any field
        result = CandidateResult(
            file_name=file_name,
            candidate_name=candidate_name,
            email=parsed.get("email", ""),
            phone=parsed.get("phone", ""),
            location=parsed.get("location", "") or parsed.get("address", ""),
            current_role=parsed.get("current_role", ""),
            current_company=parsed.get("current_company", ""),
            total_experience=round(total_exp, 1),
            highest_education=get_highest_education(parsed),
            overall_score=scores["overall_score"],
            skills_score=scores["skills_score"],
            experience_score=scores["experience_score"],
            education_score=scores["education_score"],
            location_score=scores["location_score"],
            keyword_score=scores["keyword_score"],
            matched_skills=scores["matched_skills"],  # ALL - no limit
            missing_skills=scores["missing_skills"],  # ALL - no limit
            strengths=scores["strengths"],  # ALL
            gaps=scores["gaps"],  # ALL
            recommendation=scores["recommendation"],
            processing_time=round(time.time() - start_time, 2),
            success=True,
            parsed_resume=parsed,
            resume_text=resume_text
        )
        
        return result
        
    except Exception as e:
        return CandidateResult(
            file_name=file_name,
            candidate_name="[Error]",
            email="",
            phone="",
            location="",
            current_role="",
            current_company="",
            total_experience=0,
            highest_education="",
            success=False,
            error=str(e),
            processing_time=round(time.time() - start_time, 2)
        )


def process_bulk_resumes(
    uploaded_files: List,
    jd_text: str,
    groq_api_key: str,
    model_id: str,
    progress_callback=None
) -> BulkProcessingResult:
    """
    Process multiple resumes and match against JD
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        jd_text: Job description text
        groq_api_key: Groq API key
        model_id: Model ID to use
        progress_callback: Optional callback(current, total, message)
    
    Returns:
        BulkProcessingResult with all candidates ranked
    """
    start_time = time.time()
    total = len(uploaded_files)
    
    # Extract JD requirements first
    jd_requirements = extract_jd_requirements(jd_text)
    
    candidates = []
    successful = 0
    failed = 0
    
    # Process files sequentially (to avoid rate limits)
    for i, uploaded_file in enumerate(uploaded_files):
        if progress_callback:
            progress_callback(i + 1, total, f"Processing {uploaded_file.name}...")
        
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Extract text from file
            file_result = process_uploaded_file(uploaded_file, groq_api_key)
            
            if not file_result.get("success", False):
                candidates.append(CandidateResult(
                    file_name=uploaded_file.name,
                    candidate_name="[Error]",
                    email="",
                    phone="",
                    location="",
                    current_role="",
                    current_company="",
                    total_experience=0,
                    highest_education="",
                    success=False,
                    error=file_result.get("error", "Failed to process file")
                ))
                failed += 1
                continue
            
            # Add filename to result
            file_result["file_name"] = uploaded_file.name
            
            # Process resume and calculate scores
            result = process_single_resume_for_bulk(
                file_result,
                jd_text,
                jd_requirements,
                groq_api_key,
                model_id
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
                email="",
                phone="",
                location="",
                current_role="",
                current_company="",
                total_experience=0,
                highest_education="",
                success=False,
                error=str(e)
            ))
            failed += 1
        
        # Small delay to avoid rate limiting
        if i < total - 1:
            time.sleep(0.3)
    
    # Sort by overall score (highest first)
    candidates.sort(key=lambda x: (x.success, x.overall_score), reverse=True)
    
    # Add rank to successful candidates
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
        jd_summary=jd_requirements
    )


def results_to_dataframe(result: BulkProcessingResult) -> pd.DataFrame:
    """Convert bulk processing results to a pandas DataFrame - FULL VALUES, NO TRUNCATION"""
    data = []
    
    for candidate in result.candidates:
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
            "Current Role": candidate.current_role,  # FULL VALUE
            "Company": candidate.current_company,  # FULL VALUE
            "Email": candidate.email,
            "Phone": candidate.phone,
            "Location": candidate.location,  # FULL VALUE
            "Education": candidate.highest_education,  # FULL VALUE
            "Matched Skills": ", ".join(candidate.matched_skills) if candidate.matched_skills else "",  # ALL SKILLS
            "Missing Skills": ", ".join(candidate.missing_skills) if candidate.missing_skills else "",  # ALL SKILLS
            "Strengths": " | ".join(candidate.strengths) if candidate.strengths else "",  # FULL VALUE
            "Gaps": " | ".join(candidate.gaps) if candidate.gaps else "",  # FULL VALUE
            "Recommendation": candidate.recommendation if candidate.success else f"Error: {candidate.error}",
            "File Name": candidate.file_name,
            "Processing Time (s)": candidate.processing_time if candidate.success else "-",
            "Status": "Success" if candidate.success else "Failed",
        }
        data.append(row)
    
    return pd.DataFrame(data)


def export_results_to_excel(result: BulkProcessingResult) -> bytes:
    """Export bulk processing results to Excel file - FULL VALUES, NO TRUNCATION"""
    import io
    
    df = results_to_dataframe(result)
    
    # Create Excel file in memory
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # ══════════════════════════════════════════════════════
        # SHEET 1: CANDIDATE RANKINGS (Main comparison table)
        # ══════════════════════════════════════════════════════
        df.to_excel(writer, sheet_name='Candidate Rankings', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Candidate Rankings']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4338ca',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'text_wrap': True
        })
        
        wrap_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top'
        })
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-calculate column widths based on content
        for i, col in enumerate(df.columns):
            max_content_length = df[col].astype(str).apply(len).max()
            header_length = len(col)
            width = max(header_length, min(max_content_length, 100)) + 2
            worksheet.set_column(i, i, width, wrap_format)
        
        # Freeze top row and first column
        worksheet.freeze_panes(1, 1)
        
        # ══════════════════════════════════════════════════════
        # SHEET 2: DETAILED CANDIDATE INFO (Extended data)
        # ══════════════════════════════════════════════════════
        detailed_data = []
        for candidate in result.candidates:
            if candidate.success:
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
                    "Professional Summary": candidate.parsed_resume.get("professional_summary", candidate.parsed_resume.get("summary", "")),
                    "Total Experience (Years)": candidate.total_experience,
                    "Current Role": candidate.current_role,
                    "Current Company": candidate.current_company,
                    "All Skills": get_all_skills(candidate.parsed_resume),
                    "Work History (Full)": get_all_work_history(candidate.parsed_resume),
                    "Education (Full)": get_all_education_details(candidate.parsed_resume),
                    "Certifications (Full)": get_all_certifications(candidate.parsed_resume),
                    "Specializations": ", ".join(candidate.parsed_resume.get("specializations", [])),
                    "Awards": ", ".join([
                        a.get("name", str(a)) if isinstance(a, dict) else str(a) 
                        for a in candidate.parsed_resume.get("awards", [])
                    ]),
                    "Languages": ", ".join(candidate.parsed_resume.get("languages", [])),
                    "Matched Skills": ", ".join(candidate.matched_skills),
                    "Missing Skills": ", ".join(candidate.missing_skills),
                    "Strengths": " | ".join(candidate.strengths),
                    "Gaps": " | ".join(candidate.gaps),
                    "Recommendation": candidate.recommendation,
                    "File Name": candidate.file_name,
                    "Processing Time (s)": candidate.processing_time,
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Candidate Info', index=False)
            
            detailed_worksheet = writer.sheets['Detailed Candidate Info']
            
            for col_num, value in enumerate(detailed_df.columns.values):
                detailed_worksheet.write(0, col_num, value, header_format)
            
            # Set column widths for detailed sheet
            for i, col in enumerate(detailed_df.columns):
                max_content_length = detailed_df[col].astype(str).apply(len).max()
                header_length = len(col)
                width = max(header_length, min(max_content_length, 120)) + 2
                detailed_worksheet.set_column(i, i, width, wrap_format)
            
            detailed_worksheet.freeze_panes(1, 1)
        
        # ══════════════════════════════════════════════════════
        # SHEET 3: JD REQUIREMENTS
        # ══════════════════════════════════════════════════════
        jd_data = [
            {"Category": "Required Skills", "Details": ", ".join(result.jd_summary.get("required_skills", [])) or "Not specified"},
            {"Category": "Min Experience (Years)", "Details": str(result.jd_summary.get("min_experience_years", 0))},
            {"Category": "Max Experience (Years)", "Details": str(result.jd_summary.get("max_experience_years", "Not specified")) if result.jd_summary.get("max_experience_years", 99) < 99 else "Not specified"},
            {"Category": "Required Education", "Details": ", ".join(result.jd_summary.get("required_education", [])) or "Not specified"},
            {"Category": "Preferred Locations", "Details": ", ".join(result.jd_summary.get("preferred_locations", [])) or "Not specified"},
            {"Category": "Key Keywords", "Details": ", ".join(result.jd_summary.get("keywords", [])) or "Not specified"},
        ]
        
        jd_df = pd.DataFrame(jd_data)
        jd_df.to_excel(writer, sheet_name='JD Requirements', index=False)
        
        jd_worksheet = writer.sheets['JD Requirements']
        jd_worksheet.set_column(0, 0, 25)
        jd_worksheet.set_column(1, 1, 150, wrap_format)
        
        for col_num, value in enumerate(jd_df.columns.values):
            jd_worksheet.write(0, col_num, value, header_format)
        
        # ══════════════════════════════════════════════════════
        # SHEET 4: SUMMARY
        # ══════════════════════════════════════════════════════
        avg_score = sum(c.overall_score for c in result.candidates if c.success) / max(result.successful, 1)
        top_candidate = result.candidates[0] if result.candidates and result.candidates[0].success else None
        
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
            {"Metric": "Lowest Score (successful)", "Value": f"{min(c.overall_score for c in result.candidates if c.success):.1f}%" if result.successful > 0 else "N/A"},
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
        
        summary_worksheet = writer.sheets['Summary']
        summary_worksheet.set_column(0, 0, 35)
        summary_worksheet.set_column(1, 1, 50)
        
        for col_num, value in enumerate(summary_df.columns.values):
            summary_worksheet.write(0, col_num, value, header_format)
        
        # ══════════════════════════════════════════════════════
        # SHEET 5: FAILED FILES (if any)
        # ══════════════════════════════════════════════════════
        failed_candidates = [c for c in result.candidates if not c.success]
        if failed_candidates:
            failed_data = [
                {
                    "File Name": c.file_name,
                    "Error": c.error or "Unknown error"
                }
                for c in failed_candidates
            ]
            
            failed_df = pd.DataFrame(failed_data)
            failed_df.to_excel(writer, sheet_name='Failed Files', index=False)
            
            failed_worksheet = writer.sheets['Failed Files']
            failed_worksheet.set_column(0, 0, 40)
            failed_worksheet.set_column(1, 1, 100, wrap_format)
            
            for col_num, value in enumerate(failed_df.columns.values):
                failed_worksheet.write(0, col_num, value, header_format)
    
    output.seek(0)
    return output.getvalue()


def export_results_to_csv(result: BulkProcessingResult) -> str:
    """Export bulk processing results to CSV string - FULL VALUES"""
    df = results_to_dataframe(result)
    return df.to_csv(index=False)


def export_detailed_results_to_csv(result: BulkProcessingResult) -> str:
    """Export detailed results to CSV with all fields - FULL VALUES"""
    detailed_data = []
    
    for candidate in result.candidates:
        if candidate.success:
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
                "Matched Skills": ", ".join(candidate.matched_skills),
                "Missing Skills": ", ".join(candidate.missing_skills),
                "Strengths": " | ".join(candidate.strengths),
                "Gaps": " | ".join(candidate.gaps),
                "Recommendation": candidate.recommendation,
                "File Name": candidate.file_name,
            })
    
    df = pd.DataFrame(detailed_data)
    return df.to_csv(index=False)
