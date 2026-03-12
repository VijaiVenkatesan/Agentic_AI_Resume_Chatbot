"""
Universal Document Processor - Enhanced
Handles: PDF, DOCX, TXT, Images (JPG, PNG, JPEG, WEBP)
Includes file preview generation for UI display.
"""

import io
import re
import base64
import requests
from typing import Optional, Dict
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF with enhanced cleaning"""
    text = ""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = _clean_extracted_text(page_text)
                text += page_text + "\n\n"
    except Exception as e:
        text = f"Error reading PDF: {str(e)}"
    return text.strip()


def _clean_extracted_text(text: str) -> str:
    """Clean common text extraction artifacts"""
    # Fix bullet points
    for bullet in ['•', '●', '○', '■', '▪', '►', '▸']:
        text = text.replace(bullet, '\n• ')
    # Fix multiple spaces but preserve structure
    text = re.sub(r' {3,}', '  ', text)
    # Fix phone numbers that get split
    text = re.sub(r'(\+\d{1,3})\s+(\d)', r'\1\2', text)
    # Fix emails that get split
    text = re.sub(r'(\w)\s+@\s+(\w)', r'\1@\2', text)
    text = re.sub(r'(\w)@(\w+)\s+\.(\w+)', r'\1@\2.\3', text)
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX including tables and headers"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        text_parts = []

        # Extract from headers first
        for section in doc.sections:
            try:
                header = section.header
                if header:
                    for para in header.paragraphs:
                        if para.text.strip():
                            text_parts.insert(0, para.text.strip())
            except Exception:
                pass

        # Extract paragraphs with style awareness
        for para in doc.paragraphs:
            txt = para.text.strip()
            if txt:
                # Detect headings
                style_name = para.style.name if para.style else ""
                if 'Heading' in style_name or 'Title' in style_name:
                    text_parts.append(f"\n{txt.upper()}\n")
                else:
                    text_parts.append(txt)

        # Extract from tables (very common in resumes)
        for table in doc.tables:
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_data.append(cell_text)
                if row_data:
                    text_parts.append(" | ".join(row_data))

        # Extract from footers
        for section in doc.sections:
            try:
                footer = section.footer
                if footer:
                    for para in footer.paragraphs:
                        if para.text.strip():
                            text_parts.append(para.text.strip())
            except Exception:
                pass

        return "\n".join(text_parts)
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT with encoding detection"""
    for enc in ['utf-8', 'latin-1', 'cp1252', 'ascii', 'utf-16']:
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    return file_bytes.decode('utf-8', errors='ignore')


def extract_text_from_image(
    file_bytes: bytes,
    groq_api_key: str,
    file_name: str = "resume.png"
) -> str:
    """Extract ALL text from resume image using Groq Vision API"""
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

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert resume OCR system. Extract EVERY "
                            "single piece of text from this resume image. Be extremely thorough.\n\n"
                            "MUST EXTRACT:\n"
                            "1. FULL NAME - exactly as written at the top\n"
                            "2. ALL CONTACT INFO - every phone number, email, physical address, "
                            "LinkedIn URL, GitHub URL, portfolio website, any URLs\n"
                            "3. PROFESSIONAL SUMMARY/OBJECTIVE - complete text\n"
                            "4. ALL WORK EXPERIENCE - job titles, company names, locations, "
                            "exact date ranges (month/year), ALL bullet points and descriptions\n"
                            "5. ALL SKILLS - every single skill mentioned anywhere\n"
                            "6. ALL EDUCATION - degrees, institutions, dates, GPA/grades\n"
                            "7. ALL CERTIFICATIONS - names, issuers, dates, IDs\n"
                            "8. ALL AWARDS/HONORS - names, dates, organizations\n"
                            "9. ALL PROJECTS - names, descriptions, technologies\n"
                            "10. ANYTHING ELSE - publications, volunteer work, languages, interests\n\n"
                            "Return clean structured text preserving section headings. "
                            "Do NOT skip anything. Do NOT summarize. Extract word for word."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 8000,
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=60
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error = response.json().get("error", {}).get("message", "Unknown error")
            return f"Error extracting text from image: {error}"

    except Exception as e:
        return f"Error processing image: {str(e)}"


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
        preview["preview_data"] = file_bytes.decode("utf-8", errors="ignore")[:3000]
        preview["mime_type"] = "text/plain"

    elif ext in [".docx", ".doc"]:
        preview["can_preview"] = True
        preview["preview_data"] = extract_text_from_docx(file_bytes)[:3000]
        preview["mime_type"] = "text/plain"

    return preview


def process_uploaded_file(uploaded_file, groq_api_key: str = "") -> Dict:
    """Process any uploaded file and extract text + preview"""
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
        else:
            result["error"] = "Could not extract sufficient text. Try different format."

    except Exception as e:
        result["error"] = str(e)

    return result
