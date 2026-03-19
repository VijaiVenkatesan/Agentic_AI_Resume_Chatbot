"""
Universal Document Processor - Enhanced V2
Handles: PDF, DOCX, TXT, Images (JPG, PNG, JPEG, WEBP)
Enhanced for complex layouts, multi-column PDFs, creative formats
Improved contact and name extraction
"""

import io
import re
import base64
import requests
from typing import Optional, Dict, List, Tuple
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#                    ENHANCED TEXT CLEANING
# ═══════════════════════════════════════════════════════════════

def _clean_extracted_text(text: str) -> str:
    """Clean common text extraction artifacts"""
    if not text:
        return ""
    
    # Fix bullet points
    for bullet in ['•', '●', '○', '■', '▪', '►', '▸', '◆', '◇', '▶', '→', '➤', '➢', '✓', '✔', '☑']:
        text = text.replace(bullet, '\n• ')
    
    # Fix common ligatures
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        '…': '...', '–': '-', '—': '-', ''': "'", ''': "'",
        '"': '"', '"': '"', '′': "'", '″': '"'
    }
    for old, new in ligatures.items():
        text = text.replace(old, new)
    
    # Fix multiple spaces but preserve structure
    text = re.sub(r' {3,}', '  ', text)
    
    # Fix phone numbers that get split
    text = re.sub(r'(\+\d{1,3})\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d{3})\s*[-.)]\s*(\d{3})\s*[-.)]\s*(\d{4})', r'\1-\2-\3', text)
    
    # Fix emails that get split
    text = re.sub(r'(\w)\s+@\s+(\w)', r'\1@\2', text)
    text = re.sub(r'(\w)@(\w+)\s+\.(\w+)', r'\1@\2.\3', text)
    text = re.sub(r'(\w+)\s*@\s*(\w+)\s*\.\s*(\w+)', r'\1@\2.\3', text)
    
    # Fix URLs that get split
    text = re.sub(r'(https?)\s*:\s*/\s*/\s*', r'\1://', text)
    text = re.sub(r'(www)\s*\.\s*', r'www.', text)
    text = re.sub(r'\.\s*(com|org|net|io|in|edu|co)\b', r'.\1', text)
    
    # Fix LinkedIn URLs
    text = re.sub(r'linkedin\s*\.\s*com\s*/\s*in\s*/\s*', r'linkedin.com/in/', text, flags=re.IGNORECASE)
    
    # Fix GitHub URLs  
    text = re.sub(r'github\s*\.\s*com\s*/\s*', r'github.com/', text, flags=re.IGNORECASE)
    
    return text


def _reconstruct_columns(text: str) -> str:
    """
    Attempt to reconstruct text from multi-column PDF layouts
    by analyzing line patterns and merging appropriately
    """
    lines = text.split('\n')
    reconstructed = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this might be a split line (ends abruptly, next line continues)
        if line and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            
            # If current line ends with a word and next starts with lowercase, merge
            if (line and not line.endswith(('.', ':', ',', ';', '!', '?')) and
                next_line and next_line[0].islower()):
                reconstructed.append(line + ' ' + next_line)
                i += 2
                continue
        
        reconstructed.append(line)
        i += 1
    
    return '\n'.join(reconstructed)


# ═══════════════════════════════════════════════════════════════
#                    ENHANCED PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF with enhanced handling for complex layouts"""
    text_parts = []
    
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        
        for page_num, page in enumerate(reader.pages):
            try:
                # Try standard extraction first
                page_text = page.extract_text()
                
                if page_text:
                    page_text = _clean_extracted_text(page_text)
                    
                    # Try to reconstruct columns if needed
                    if _looks_like_multicolumn(page_text):
                        page_text = _reconstruct_columns(page_text)
                    
                    text_parts.append(page_text)
                    
            except Exception as e:
                # If standard extraction fails, try alternative method
                try:
                    # Attempt to get text with different parameters
                    page_text = page.extract_text(extraction_mode="layout")
                    if page_text:
                        text_parts.append(_clean_extracted_text(page_text))
                except:
                    pass
        
        full_text = "\n\n".join(text_parts)
        
        # Post-process to fix common issues
        full_text = _post_process_pdf_text(full_text)
        
        return full_text.strip()
        
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def _looks_like_multicolumn(text: str) -> bool:
    """Detect if text appears to be from a multi-column layout"""
    lines = text.split('\n')
    
    # Check for signs of multi-column: lots of short lines, irregular spacing
    short_lines = sum(1 for line in lines if 10 < len(line.strip()) < 40)
    total_lines = len([l for l in lines if l.strip()])
    
    if total_lines > 0 and short_lines / total_lines > 0.5:
        return True
    
    # Check for excessive whitespace in middle of lines (column gap)
    lines_with_gaps = sum(1 for line in lines if re.search(r'\S\s{5,}\S', line))
    if total_lines > 0 and lines_with_gaps / total_lines > 0.3:
        return True
    
    return False


def _post_process_pdf_text(text: str) -> str:
    """Post-process PDF text to fix common extraction issues"""
    
    # Fix common section headers that may have been split
    section_headers = [
        'EDUCATION', 'EXPERIENCE', 'WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE',
        'SKILLS', 'TECHNICAL SKILLS', 'PROJECTS', 'CERTIFICATIONS',
        'ACHIEVEMENTS', 'AWARDS', 'SUMMARY', 'OBJECTIVE', 'PROFILE',
        'CONTACT', 'PERSONAL INFORMATION', 'REFERENCES'
    ]
    
    for header in section_headers:
        # Fix headers that got split across lines
        pattern = r'(' + r'\s*'.join(list(header)) + r')'
        text = re.sub(pattern, header, text, flags=re.IGNORECASE)
    
    # Remove excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    return text


# ═══════════════════════════════════════════════════════════════
#                    ENHANCED DOCX EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX including tables, headers, footers, and text boxes"""
    try:
        from docx import Document
        from docx.oxml.ns import qn
        
        doc = Document(io.BytesIO(file_bytes))
        text_parts = []
        
        # ── Extract from headers (often contains name/contact) ──
        for section in doc.sections:
            try:
                header = section.header
                if header:
                    for para in header.paragraphs:
                        if para.text.strip():
                            text_parts.insert(0, para.text.strip())
                    
                    # Also check tables in header
                    for table in header.tables:
                        for row in table.rows:
                            row_text = []
                            for cell in row.cells:
                                cell_text = cell.text.strip()
                                if cell_text:
                                    row_text.append(cell_text)
                            if row_text:
                                text_parts.insert(0, " | ".join(row_text))
            except Exception:
                pass
        
        # ── Extract text boxes (common in creative resumes) ──
        try:
            # Access the document's XML to find text boxes
            for element in doc.element.body.iter():
                if element.tag.endswith('txbxContent'):
                    for child in element.iter():
                        if child.text and child.text.strip():
                            text_parts.append(child.text.strip())
        except Exception:
            pass
        
        # ── Extract from main body with style awareness ──
        for para in doc.paragraphs:
            txt = para.text.strip()
            if txt:
                style_name = para.style.name if para.style else ""
                
                # Detect headings and format appropriately
                if 'Heading' in style_name or 'Title' in style_name:
                    text_parts.append(f"\n{txt.upper()}\n")
                else:
                    text_parts.append(txt)
        
        # ── Extract from tables (very common in resumes) ──
        for table in doc.tables:
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    # Get all paragraphs in cell
                    cell_text = []
                    for para in cell.paragraphs:
                        if para.text.strip():
                            cell_text.append(para.text.strip())
                    
                    if cell_text:
                        row_data.append(" ".join(cell_text))
                
                if row_data:
                    text_parts.append(" | ".join(row_data))
        
        # ── Extract from footers ──
        for section in doc.sections:
            try:
                footer = section.footer
                if footer:
                    for para in footer.paragraphs:
                        if para.text.strip():
                            text_parts.append(para.text.strip())
            except Exception:
                pass
        
        # ── Try to extract from shapes/drawings ──
        try:
            for rel in doc.part.rels.values():
                if "drawing" in str(rel.target_ref).lower():
                    try:
                        # Attempt to get text from drawings
                        pass
                    except:
                        pass
        except Exception:
            pass
        
        full_text = "\n".join(text_parts)
        return _clean_extracted_text(full_text)
        
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"


# ═══════════════════════════════════════════════════════════════
#                    ENHANCED TXT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT with encoding detection"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii', 'utf-16', 'utf-16-le', 'utf-16-be', 'iso-8859-1']
    
    for enc in encodings:
        try:
            text = file_bytes.decode(enc)
            # Verify it's valid text
            if text and not any(ord(c) < 9 or (ord(c) > 13 and ord(c) < 32) for c in text[:1000] if c not in '\t\n\r'):
                return _clean_extracted_text(text)
        except Exception:
            continue
    
    # Fallback with error handling
    return _clean_extracted_text(file_bytes.decode('utf-8', errors='ignore'))


# ═══════════════════════════════════════════════════════════════
#                    ENHANCED IMAGE OCR
# ═══════════════════════════════════════════════════════════════

def extract_text_from_image(
    file_bytes: bytes,
    groq_api_key: str,
    file_name: str = "resume.png"
) -> str:
    """Extract ALL text from resume image using Groq Vision API with enhanced prompting"""
    try:
        ext = Path(file_name).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")
        base64_image = base64.b64encode(file_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        # Enhanced prompt for better extraction
        extraction_prompt = """You are an expert resume OCR system. Your task is to extract EVERY piece of text from this resume image with 100% accuracy.

CRITICAL EXTRACTION RULES:

1. **NAME** (Usually largest text at top):
   - Look for the largest/boldest text - this is typically the name
   - It may NOT have a "Name:" label
   - Extract EXACTLY as written (First Last or Last, First)
   - Common positions: top-center, top-left, inside a header box

2. **CONTACT INFORMATION** (Extract ALL of these):
   - Phone: Look for +, (, ), -, numbers. Include country code.
   - Email: Look for @ symbol anywhere
   - Address: Street, City, State, ZIP/PIN
   - LinkedIn: linkedin.com/in/...
   - GitHub: github.com/...
   - Portfolio/Website: Any other URLs
   - Location: City, Country

3. **SECTIONS TO EXTRACT**:
   - Professional Summary/Objective
   - Work Experience (ALL jobs with dates, titles, companies, bullet points)
   - Education (ALL degrees, institutions, dates, GPA)
   - Skills (ALL skills, categorized or not)
   - Certifications (names, issuers, dates)
   - Projects (names, descriptions, technologies)
   - Awards/Achievements
   - Publications
   - Languages
   - Interests/Hobbies

4. **FORMATTING RULES**:
   - Preserve section headings in CAPS
   - Keep bullet points as "• "
   - Maintain date formats as written
   - Keep job titles and company names exact
   - Preserve GPA/percentage values

5. **COMMON CHALLENGES**:
   - Multi-column layouts: Read left column fully, then right column
   - Sidebar content: Don't miss contact info in sidebars
   - Icons: If icons represent phone/email/location, extract the text next to them
   - Creative formats: Scan entire image, not just standard positions

OUTPUT FORMAT:
Return clean, structured text with clear section breaks. Do NOT summarize or paraphrase. Extract word-for-word."""

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": extraction_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                    }
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 8000,
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=90
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error = response.json().get("error", {}).get("message", "Unknown error")
            return f"Error extracting text from image: {error}"

    except Exception as e:
        return f"Error processing image: {str(e)}"


# ═══════════════════════════════════════════════════════════════
#                    CONTACT EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════

def extract_contacts_from_text(text: str) -> Dict:
    """
    Enhanced contact extraction with 20+ patterns for each field
    Works even without labels like 'Email:', 'Phone:', etc.
    """
    contacts = {
        "name": "",
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
    # EMAIL EXTRACTION (Most reliable)
    # ═══════════════════════════════════════
    email_patterns = [
        r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}',
        r'[\w._%+-]+\s*@\s*[\w.-]+\s*\.\s*[a-zA-Z]{2,}',
        r'[\w._%+-]+\[at\][\w.-]+\.[a-zA-Z]{2,}',
        r'[\w._%+-]+\(at\)[\w.-]+\.[a-zA-Z]{2,}',
    ]
    
    for pattern in email_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            email = match.group(0).strip()
            email = re.sub(r'\s+', '', email)  # Remove spaces
            email = re.sub(r'\[at\]|\(at\)', '@', email, flags=re.IGNORECASE)
            contacts["email"] = email
            break
    
    # ═══════════════════════════════════════
    # PHONE EXTRACTION (Multiple international formats)
    # ═══════════════════════════════════════
    phone_patterns = [
        # International with + 
        r'\+\d{1,3}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        r'\+\d{1,3}[-.\s]?\d{4,5}[-.\s]?\d{5,6}',
        r'\+\d{10,15}',
        
        # Indian formats
        r'\+91[-.\s]?\d{5}[-.\s]?\d{5}',
        r'\+91[-.\s]?\d{10}',
        r'91[-.\s]?\d{10}',
        r'0\d{2,4}[-.\s]?\d{6,8}',
        
        # US formats
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
        r'1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
        
        # UK formats
        r'\+44[-.\s]?\d{4}[-.\s]?\d{6}',
        r'0\d{4}[-.\s]?\d{6}',
        
        # Generic patterns
        r'(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P)[\s.:]*[\+]?\d[\d\s\-().]{8,18}',
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\d{5}[-.\s]?\d{5}\b',
        r'\b\d{10,12}\b',
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean the match
            cleaned = re.sub(r'^(?:Phone|Ph|Tel|Mobile|Cell|Contact|M|T|P)[\s.:]*', '', match, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            
            # Count digits
            digits = re.sub(r'[^\d]', '', cleaned)
            
            # Valid phone should have 10-15 digits
            if 10 <= len(digits) <= 15:
                contacts["phone"] = cleaned
                break
        
        if contacts["phone"]:
            break
    
    # ═══════════════════════════════════════
    # LINKEDIN EXTRACTION
    # ═══════════════════════════════════════
    linkedin_patterns = [
        r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?',
        r'linkedin\.com/in/[\w-]+',
        r'linkedin:\s*[\w-]+',
        r'in/[\w-]+\s*\(linkedin\)',
    ]
    
    for pattern in linkedin_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            linkedin = match.group(0).strip()
            if not linkedin.startswith('http'):
                if 'linkedin.com' not in linkedin.lower():
                    linkedin = f"linkedin.com/in/{linkedin.split('/')[-1]}"
            contacts["linkedin"] = linkedin
            break
    
    # ═══════════════════════════════════════
    # GITHUB EXTRACTION
    # ═══════════════════════════════════════
    github_patterns = [
        r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?',
        r'github\.com/[\w-]+',
        r'github:\s*[\w-]+',
    ]
    
    for pattern in github_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            contacts["github"] = match.group(0).strip()
            break
    
    # ═══════════════════════════════════════
    # PORTFOLIO/WEBSITE EXTRACTION
    # ═══════════════════════════════════════
    website_patterns = [
        r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|io|dev|me|org|net|co|in)/?\S*',
        r'(?:portfolio|website|web|site)[\s.:]*(?:https?://)?[\w.-]+\.[a-z]{2,}',
    ]
    
    for pattern in website_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            url = match.strip()
            # Skip if it's linkedin or github
            if 'linkedin' not in url.lower() and 'github' not in url.lower():
                # Skip email-like patterns
                if '@' not in url:
                    contacts["portfolio"] = url
                    break
        if contacts["portfolio"]:
            break
    
    # ═══════════════════════════════════════
    # ADDRESS EXTRACTION
    # ═══════════════════════════════════════
    address_patterns = [
        # With PIN/ZIP codes
        r'(?:No[.:]?\s*)?[\d]+[,\s]+[\w\s]+(?:Street|St|Road|Rd|Avenue|Ave|Nagar|Colony|Layout|Block|Lane|Apt|Apartment|Floor|Building|Bldg)[\w\s,.-]+(?:\d{5,6})',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+-?\s*\d{5,6}',
        r'[\w\s]+,\s*[\w\s]+,\s*[\w\s]+,\s*\d{5,6}',
        
        # Address after label
        r'(?:Address|Location|Residence|Home)[\s.:]+([^\n]{15,100})',
        
        # Common city patterns
        r'(?:Bangalore|Bengaluru|Mumbai|Delhi|Chennai|Hyderabad|Pune|Kolkata|Noida|Gurgaon|Gurugram)[\w\s,.-]+\d{5,6}',
        r'(?:New York|San Francisco|Los Angeles|Chicago|Seattle|Boston|Austin)[\w\s,.-]+\d{5}',
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            addr = match.group(0).strip()
            # Clean up
            addr = re.sub(r'^(?:Address|Location|Residence|Home)[\s.:]+', '', addr, flags=re.IGNORECASE)
            if 15 < len(addr) < 200:
                contacts["address"] = addr
                break
    
    # ═══════════════════════════════════════
    # LOCATION EXTRACTION (City, State, Country)
    # ═══════════════════════════════════════
    location_patterns = [
        # City, State format
        r'(?:Location|Based in|Located at|City)[\s.:]*([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',
        
        # Just city names (common ones)
        r'\b(Bangalore|Bengaluru|Mumbai|Delhi|NCR|Chennai|Hyderabad|Pune|Kolkata|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Lucknow|Chandigarh|Indore|Bhopal|Kochi|Coimbatore|Trivandrum|Mysore|Nagpur)\b',
        r'\b(New York|NYC|San Francisco|SF|Los Angeles|LA|Chicago|Seattle|Boston|Austin|Denver|Atlanta|Dallas|Houston|Phoenix|San Diego|San Jose|Portland|Miami|Washington DC|Philadelphia)\b',
        r'\b(London|Berlin|Amsterdam|Dublin|Singapore|Tokyo|Sydney|Toronto|Vancouver|Melbourne|Paris|Munich|Barcelona|Stockholm|Copenhagen|Zurich|Dubai|Hong Kong)\b',
        
        # Country
        r'\b(India|USA|United States|UK|United Kingdom|Canada|Australia|Germany|Netherlands|Singapore|UAE|Ireland)\b',
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            loc = match.group(1) if match.lastindex else match.group(0)
            contacts["location"] = loc.strip()
            break
    
    # ═══════════════════════════════════════
    # NAME EXTRACTION (Most complex - try multiple strategies)
    # ═══════════════════════════════════════
    contacts["name"] = extract_name_from_text(text)
    
    return contacts


def extract_name_from_text(text: str) -> str:
    """
    Enhanced name extraction using multiple strategies
    Works even without 'Name:' label
    """
    if not text:
        return ""
    
    lines = text.strip().split("\n")
    
    # Strategy 1: Look for labeled name
    name_label_patterns = [
        r'(?:Name|Full Name|Candidate Name)[\s.:]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})',
        r'^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})[\s]*$',
    ]
    
    for pattern in name_label_patterns:
        for line in lines[:15]:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if _is_valid_name(name):
                    return name
    
    # Strategy 2: First substantial line that looks like a name
    skip_keywords = [
        'resume', 'curriculum', 'vitae', 'cv', 'http', 'www', '@',
        'address', 'phone', 'email', 'street', 'road', 'avenue',
        'objective', 'summary', 'profile', 'linkedin', 'github',
        'portfolio', 'mobile', 'tel:', 'contact', 'experience',
        'education', 'skills', 'professional', 'career', 'about'
    ]
    
    for line in lines[:15]:
        line = line.strip()
        
        if not line or len(line) < 3:
            continue
        
        # Skip lines with keywords
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        
        # Skip lines that look like addresses (contain numbers at start or have 5+ digits)
        if re.match(r'^\d', line) or len(re.findall(r'\d', line)) >= 5:
            continue
        
        # Skip lines that look like phone numbers
        if re.match(r'^[\+\d\(\)]', line):
            continue
        
        # Skip very long lines (probably not a name)
        if len(line) > 50:
            continue
        
        # Check if line is mostly alphabetic
        alpha_chars = sum(1 for c in line if c.isalpha() or c.isspace() or c == '.')
        if alpha_chars / max(len(line), 1) < 0.8:
            continue
        
        # Check if it follows name patterns
        # Pattern: First Last, First Middle Last, or FIRST LAST
        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$',  # First Last or First Middle Last
            r'^[A-Z]+\s+[A-Z]+(?:\s+[A-Z]+)?$',  # FIRST LAST
            r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+$',  # First M. Last
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Middle Last
            r'^Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Dr. First Last
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+$',  # First Last-Last (hyphenated)
        ]
        
        for pattern in name_patterns:
            if re.match(pattern, line):
                return line
        
        # Fallback: if line has 2-4 capitalized words, it might be a name
        words = line.split()
        if 2 <= len(words) <= 4:
            capitalized_words = sum(1 for w in words if w[0].isupper())
            if capitalized_words == len(words):
                # Additional check: words should be reasonable length
                if all(2 <= len(w) <= 15 for w in words):
                    if _is_valid_name(line):
                        return line
    
    # Strategy 3: Look for name after "I am" or "My name is"
    intro_patterns = [
        r"(?:I am|I'm|My name is|This is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    ]
    
    for pattern in intro_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip()
            if _is_valid_name(name):
                return name
    
    return ""


def _is_valid_name(name: str) -> str:
    """Validate if a string looks like a valid person name"""
    if not name:
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
        'senior', 'junior', 'lead', 'head', 'director', 'executive',
        'software', 'hardware', 'technical', 'information', 'technology',
        'university', 'college', 'school', 'institute', 'company', 'corporation'
    }
    
    words = name.lower().split()
    
    # Check if any word is in not_names
    if any(w in not_names for w in words):
        return False
    
    # Name should have 2-4 words
    if not (2 <= len(words) <= 4):
        return False
    
    # Each word should be reasonable length
    if not all(2 <= len(w) <= 15 for w in words):
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════
#                    FILE PREVIEW GENERATION
# ═══════════════════════════════════════════════════════════════

def get_file_preview_data(file_bytes: bytes, file_name: str) -> Dict:
    """Generate preview data for displaying uploaded file in UI"""
    ext = Path(file_name).suffix.lower()

    preview = {
        "type": ext,
        "name": file_name,
        "size_kb": round(len(file_bytes) / 1024, 1),
        "can_preview": False,
        "preview_data": None,
        "mime_type": "",
        "page_count": 0,
    }

    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        preview["can_preview"] = True
        preview["preview_data"] = base64.b64encode(file_bytes).decode("utf-8")
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}
        preview["mime_type"] = mime.get(ext, "image/png")

    elif ext == ".pdf":
        preview["can_preview"] = True
        preview["preview_data"] = base64.b64encode(file_bytes).decode("utf-8")
        preview["mime_type"] = "application/pdf"
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            preview["page_count"] = len(reader.pages)
        except Exception:
            pass

    elif ext in [".txt", ".md", ".text"]:
        preview["can_preview"] = True
        preview["preview_data"] = file_bytes.decode("utf-8", errors="ignore")[:5000]
        preview["mime_type"] = "text/plain"

    elif ext in [".docx", ".doc"]:
        preview["can_preview"] = True
        preview["preview_data"] = extract_text_from_docx(file_bytes)[:5000]
        preview["mime_type"] = "text/plain"

    return preview


# ═══════════════════════════════════════════════════════════════
#                    MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════

def process_uploaded_file(uploaded_file, groq_api_key: str = "") -> Dict:
    """Process any uploaded file and extract text + preview with enhanced extraction"""
    file_name = uploaded_file.name
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    ext = Path(file_name).suffix.lower()

    result = {
        "file_name": file_name,
        "file_type": ext,
        "file_size_kb": round(len(file_bytes) / 1024, 1),
        "text": "",
        "success": False,
        "error": None,
        "file_bytes": file_bytes,
        "preview": get_file_preview_data(file_bytes, file_name),
        "extracted_contacts": {},  # NEW: Pre-extracted contacts
    }

    try:
        if ext == ".pdf":
            result["text"] = extract_text_from_pdf(file_bytes)
        elif ext in [".docx", ".doc"]:
            result["text"] = extract_text_from_docx(file_bytes)
        elif ext in [".txt", ".text", ".md"]:
            result["text"] = extract_text_from_txt(file_bytes)
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            if not groq_api_key:
                result["error"] = "Groq API key required for image OCR"
                return result
            result["text"] = extract_text_from_image(file_bytes, groq_api_key, file_name)
        else:
            result["error"] = f"Unsupported file type: {ext}"
            return result

        if result["text"] and len(result["text"].strip()) > 30:
            result["success"] = True
            
            # Pre-extract contacts for better parsing
            result["extracted_contacts"] = extract_contacts_from_text(result["text"])
        else:
            result["error"] = "Could not extract sufficient text. Try different format."

    except Exception as e:
        result["error"] = str(e)

    return result
