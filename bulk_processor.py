"""
Bulk Resume Processor - Enterprise HR Feature
Handles multiple resume uploads and batch JD matching
Version: 1.0
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
    
    # Details
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
    requirements["keywords"] = [w[0] for w in sorted_words[:40]]
    
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
        scores["skills_score"] = round(min(100, match_ratio * 110), 1)  # Slight boost
    else:
        scores["skills_score"] = 70  # Default if no specific skills in JD
    
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
                # Over-qualified - slight penalty
                over_by = candidate_exp - max_exp
                scores["experience_score"] = max(65, 100 - (over_by * 3))
        else:
            # Under-qualified
            if candidate_exp > 0:
                ratio = candidate_exp / min_exp
                scores["experience_score"] = round(min(95, ratio * 100), 1)
            else:
                scores["experience_score"] = 20
    else:
        # No specific requirement
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
    
    # Degree level hierarchy
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
        # Remote-friendly job
        remote_keywords = ["remote", "work from home", "wfh", "telecommute", "hybrid"]
        if any(kw in preferred_locations for kw in remote_keywords):
            scores["location_score"] = 100
        elif candidate_location:
            # Check if candidate location matches any preferred location
            location_match = any(loc in candidate_location or candidate_location in loc 
                               for loc in preferred_locations if loc not in remote_keywords)
            if location_match:
                scores["location_score"] = 100
            else:
                scores["location_score"] = 55  # Has location but doesn't match
        else:
            scores["location_score"] = 50  # No location info
    else:
        scores["location_score"] = 85  # No location requirement specified
    
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
    
    # Cap at 100
    scores["overall_score"] = min(100, scores["overall_score"])
    
    # ═══════════════════════════════════════
    # IDENTIFY STRENGTHS & GAPS
    # ═══════════════════════════════════════
    
    # Strengths
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
    
    # Gaps
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
    """Extract highest education from parsed resume"""
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
                field = edu.get("field", edu.get("major", ""))
                institution = edu.get("institution", edu.get("university", edu.get("college", "")))
                
                for key, priority in degree_priority.items():
                    if key in degree:
                        if priority > highest_priority:
                            highest_priority = priority
                            highest = edu.get("degree", "")
                            if field:
                                highest += f" in {field}"
                            if institution:
                                highest += f" ({institution})"
                        break
    
    return highest[:100] if highest else "Not specified"


def process_single_resume_for_bulk(
    file_data: Dict,
    jd_text: str,
    jd_requirements: Dict,
    groq_api_key: str,
    model_id: str
) -> CandidateResult:
    """Process a single resume in bulk mode"""
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
        
        # Get candidate details
        candidate_name = parsed.get("name", "Unknown")
        if not candidate_name or candidate_name in ["", "N/A", "Unknown", "Candidate"]:
            # Try to extract from filename
            candidate_name = file_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        
        total_exp = parsed.get("total_experience_years", 0)
        try:
            total_exp = float(total_exp)
        except (ValueError, TypeError):
            total_exp = 0
        
        # Build result
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
            matched_skills=scores["matched_skills"][:15],
            missing_skills=scores["missing_skills"][:10],
            strengths=scores["strengths"],
            gaps=scores["gaps"],
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
            error=str(e)[:200],
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
                    error=file_result.get("error", "Failed to process file")[:200]
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
                error=str(e)[:200]
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
    """Convert bulk processing results to a pandas DataFrame"""
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
            "Experience (Years)": candidate.total_experience if candidate.success else "-",
            "Current Role": candidate.current_role[:40] if candidate.current_role else "",
            "Company": candidate.current_company[:30] if candidate.current_company else "",
            "Email": candidate.email,
            "Phone": candidate.phone,
            "Location": candidate.location[:30] if candidate.location else "",
            "Education": candidate.highest_education[:50] if candidate.highest_education else "",
            "Matched Skills": ", ".join(candidate.matched_skills[:8]),
            "Missing Skills": ", ".join(candidate.missing_skills[:5]),
            "Recommendation": candidate.recommendation if candidate.success else f"Error: {candidate.error}",
            "File Name": candidate.file_name,
            "Status": "✅ Success" if candidate.success else "❌ Failed",
        }
        data.append(row)
    
    return pd.DataFrame(data)


def export_results_to_excel(result: BulkProcessingResult) -> bytes:
    """Export bulk processing results to Excel file"""
    import io
    
    df = results_to_dataframe(result)
    
    # Create Excel file in memory
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main results sheet
        df.to_excel(writer, sheet_name='Candidate Rankings', index=False)
        
        # Get workbook and worksheet
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
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths
        column_widths = {
            'Rank': 8,
            'Candidate Name': 25,
            'Overall Score': 14,
            'Skills Match': 14,
            'Experience Match': 16,
            'Education Match': 16,
            'Location Match': 14,
            'Experience (Years)': 16,
            'Current Role': 30,
            'Company': 25,
            'Email': 30,
            'Phone': 18,
            'Location': 25,
            'Education': 40,
            'Matched Skills': 50,
            'Missing Skills': 35,
            'Recommendation': 35,
            'File Name': 30,
            'Status': 12,
        }
        
        for i, col in enumerate(df.columns):
            width = column_widths.get(col, 15)
            worksheet.set_column(i, i, width)
        
        # Freeze top row
        worksheet.freeze_panes(1, 0)
        
        # ── JD Requirements Sheet ──
        jd_data = []
        jd_data.append({"Category": "Required Skills", "Details": ", ".join(result.jd_summary.get("required_skills", []))})
        jd_data.append({"Category": "Min Experience", "Details": f"{result.jd_summary.get('min_experience_years', 0)} years"})
        jd_data.append({"Category": "Required Education", "Details": ", ".join(result.jd_summary.get("required_education", []))})
        jd_data.append({"Category": "Preferred Locations", "Details": ", ".join(result.jd_summary.get("preferred_locations", []))})
        jd_data.append({"Category": "Key Keywords", "Details": ", ".join(result.jd_summary.get("keywords", [])[:20])})
        
        jd_df = pd.DataFrame(jd_data)
        jd_df.to_excel(writer, sheet_name='JD Requirements', index=False)
        
        jd_worksheet = writer.sheets['JD Requirements']
        jd_worksheet.set_column(0, 0, 20)
        jd_worksheet.set_column(1, 1, 80)
        
        # ── Summary Sheet ──
        avg_score = sum(c.overall_score for c in result.candidates if c.success) / max(result.successful, 1)
        top_candidate = result.candidates[0] if result.candidates else None
        
        summary_data = [
            {"Metric": "Total Resumes Processed", "Value": result.total_resumes},
            {"Metric": "Successfully Processed", "Value": result.successful},
            {"Metric": "Failed", "Value": result.failed},
            {"Metric": "Processing Time", "Value": f"{result.processing_time} seconds"},
            {"Metric": "Average Match Score", "Value": f"{avg_score:.1f}%"},
            {"Metric": "Top Candidate", "Value": top_candidate.candidate_name if top_candidate else "N/A"},
            {"Metric": "Top Score", "Value": f"{top_candidate.overall_score}%" if top_candidate and top_candidate.success else "N/A"},
            {"Metric": "Candidates ≥80%", "Value": sum(1 for c in result.candidates if c.success and c.overall_score >= 80)},
            {"Metric": "Candidates 65-79%", "Value": sum(1 for c in result.candidates if c.success and 65 <= c.overall_score < 80)},
            {"Metric": "Candidates 50-64%", "Value": sum(1 for c in result.candidates if c.success and 50 <= c.overall_score < 65)},
            {"Metric": "Candidates <50%", "Value": sum(1 for c in result.candidates if c.success and c.overall_score < 50)},
        ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        summary_worksheet = writer.sheets['Summary']
        summary_worksheet.set_column(0, 0, 30)
        summary_worksheet.set_column(1, 1, 40)
    
    output.seek(0)
    return output.getvalue()


def export_results_to_csv(result: BulkProcessingResult) -> str:
    """Export bulk processing results to CSV string"""
    df = results_to_dataframe(result)
    return df.to_csv(index=False)
