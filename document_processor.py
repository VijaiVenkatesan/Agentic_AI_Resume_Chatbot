"""
Universal Document Processor
Handles: PDF, DOCX, TXT, Images (JPG, PNG, JPEG, WEBP)
Extracts raw text from any resume format.
"""

import io
import base64
import requests
from typing import Optional
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2"""
    text = ""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    except Exception as e:
        text = f"Error reading PDF: {str(e)}"

    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX files"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))

        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        # Also extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_parts.append(" | ".join(row_text))

        return "\n".join(text_parts)
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT files"""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_image(
    file_bytes: bytes,
    groq_api_key: str,
    file_name: str = "resume.png"
) -> str:
    """
    Extract text from image using Groq Vision API (free).
    Supports: JPG, JPEG, PNG, WEBP
    """
    try:
        # Determine mime type
        ext = Path(file_name).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")

        # Encode image to base64
        base64_image = base64.b64encode(file_bytes).decode("utf-8")

        # Call Groq Vision API
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract ALL text from this resume image. "
                                "Maintain the original structure and formatting. "
                                "Include every detail: name, contact info, summary, "
                                "work experience, skills, education, certifications, "
                                "awards, projects. Return ONLY the extracted text, "
                                "no commentary."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            error = response.json().get("error", {}).get("message", "Unknown")
            return f"Error extracting text from image: {error}"

    except Exception as e:
        return f"Error processing image: {str(e)}"


def process_uploaded_file(
    uploaded_file,
    groq_api_key: str = ""
) -> dict:
    """
    Process any uploaded file and extract text.

    Returns:
        dict with 'text', 'file_type', 'file_name', 'success', 'error'
    """
    file_name = uploaded_file.name
    file_bytes = uploaded_file.read()
    ext = Path(file_name).suffix.lower()

    result = {
        "file_name": file_name,
        "file_type": ext,
        "file_size_kb": round(len(file_bytes) / 1024, 1),
        "text": "",
        "success": False,
        "error": None
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
                result["error"] = (
                    "Groq API key required for image processing"
                )
                return result
            result["text"] = extract_text_from_image(
                file_bytes, groq_api_key, file_name
            )

        else:
            result["error"] = f"Unsupported file type: {ext}"
            return result

        if result["text"] and len(result["text"].strip()) > 50:
            result["success"] = True
        else:
            result["error"] = (
                "Could not extract sufficient text from file. "
                "Try a different format."
            )

    except Exception as e:
        result["error"] = str(e)


    return result
