# 🤖 Universal AI Resume Assistant

An intelligent, agentic AI-powered resume chatbot that can parse, analyze, and answer questions about any resume using Agentic RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) tools.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Groq](https://img.shields.io/badge/Groq-LLM-green.svg)


## ✨ Features

- **📄 Universal Resume Parsing** - Supports PDF, DOCX, TXT, and Images (JPG, PNG, WEBP)
- **🧠 Agentic AI Architecture** - Plan → Execute → Synthesize workflow
- **🔧 7 MCP Tools** - Specialized tools for different resume analysis tasks
- **🔍 RAG-Powered Search** - Semantic search through resume content
- **📊 Skill Gap Analysis** - Compare candidate skills against job requirements
- **🎯 JD Matching** - Score resume fit against job descriptions
- **🎓 Education Extraction** - Detailed education and certification analysis
- **📝 Content Generation** - Cover letters and professional summaries
- **👁️ Full-Screen Preview** - View uploaded resumes with download option
- **🌙 Dark Mode UI** - Modern, responsive interface

## 🔧 MCP Tools

| # | Tool | Description |
|---|------|-------------|
| 1 | 📄 `resume_search` | RAG-powered semantic search through resume content |
| 2 | 📊 `skill_analyzer` | Analyze and match skills against requirements |
| 3 | 💼 `experience_calculator` | Calculate total experience with timeline breakdown |
| 4 | 📝 `cover_letter_generator` | Generate tailored cover letters |
| 5 | 👤 `profile_summary` | Create LinkedIn/portfolio summaries |
| 6 | 🎯 `jd_matcher` | Score resume fit against job descriptions |
| 7 | 🎓 `education_extractor` | Extract education, degrees, GPA, certifications |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Groq API Key (free at [console.groq.com](https://console.groq.com/keys))

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/VijaiVenkatesan/Agentic_AI_Resume_Chatbot.git
   cd Agentic_AI_Resume_Chatbot
2. Install dependencies
   pip install -r requirements.txt
3. Set up environment variables
4. Run the application
   streamlit run streamlit_app.py
5. Open in browser
   http://localhost:8501
   
☁️ Deploy to Streamlit Cloud
1. Push to GitHub:
   - git add .
   - git commit -m "Initial commit"
   - git push origin main
   
3. Deploy on Streamlit Cloud:
   - Go to https://share.streamlit.io/
   - Connect your GitHub repository
   - Add GROQ_API_KEY in Secrets:
   GROQ_API_KEY = "your-groq-api-key"
   
- Deploy!

🎯 Usage Examples

Basic Questions
- "What is the candidate's contact information?"
- "List all technical skills"
- "What is the educational background?"
- "Calculate total years of experience"

Skill Analysis
- "Match skills: Python, AWS, Docker, Kubernetes"
- "What are the key technical competencies?"
- "Identify skill gaps for a Senior Engineer role"

Content Generation
- "Write a cover letter for Software Engineer at Google"
- "Generate a LinkedIn summary"
- "Create an elevator pitch"

JD Matching (with uploaded JD)
- "Compare this resume against the job description"
- "How well does this candidate fit the JD?"
- "What are the strengths and gaps?"

🔑 Supported AI Models

Model	Speed	Quality	Best For
- Llama 3.1 8B	⚡⚡ Fast	⭐⭐⭐⭐	Quick queries
- Llama 3.3 70B	🔄 Medium	⭐⭐⭐⭐⭐	Complex analysis
- Llama 4 Scout	⚡ Fast	⭐⭐⭐⭐	Vision/OCR
- Qwen 3 32B	⚡ Fast	⭐⭐⭐⭐⭐	Detailed parsing

📊 How It Works

1. Document Processing
Upload → Detect Type → Extract Text → Clean & Structure
- PDF: PyPDF2 extraction with text cleaning
- DOCX: python-docx with tables/headers support
- Images: Groq Vision API (Llama 4 Scout) for OCR
- TXT: Direct text with encoding detection

2. Resume Parsing
Raw Text → LLM Parsing → Regex Validation → Merged Result
- LLM extracts structured data (name, skills, experience, education)
- Regex patterns validate and fill gaps (contacts, degrees)
- Results merged for maximum accuracy

3. Agentic Processing
Question → Plan (LLM) → Execute (MCP Tools) → Synthesize (LLM)
- Planning: LLM decides which tools to use
- Execution: Tools retrieve/analyze data
- Synthesis: LLM creates human-readable response

4. RAG Search
Query → Embedding → Vector Search → Relevant Chunks → Context
- Resume chunked into sections (400 chars, 150 overlap)
- ChromaDB stores embeddings
- Semantic search retrieves relevant content

📈 Performance

  Metric	         Value
- Resume parsing	2-5 seconds
- Tool execution	0.5-2 seconds/tool
- Total response	3-10 seconds
- Max file size	100 MB
- Supported formats	PDF, DOCX, TXT, JPG, PNG, WEBP

🐛 Troubleshooting

Common Issues
1. "Resume not loaded" error
- Ensure file is uploaded successfully
- Check file format is supported
- Verify file is not corrupted

2. Education not extracted
- The system uses both LLM and regex extraction
- Check if education section is clearly formatted
- Try different resume format

3. API errors
- Verify GROQ_API_KEY is set correctly
- Check API rate limits
- Try a different model
  
4. Preview not working
- PDF preview uses extracted text (not iframe)
- Images display inline with base64
- Use Download button for original file

🤝 Contributing

1. Fork the repository
2. Create feature branch (git checkout -b feature/amazing)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing)
5. Open a Pull Request

🙏 Acknowledgments

Groq - Lightning-fast LLM inference

Streamlit - Amazing web framework

ChromaDB - Vector database

LangChain - Text splitting utilities

📧 Contact

Author: V Vijai

Email: vijaibt1@gmail.com

GitHub: https://github.com/VijaiVenkatesan

<p align="center"> Built with ❤️ using Agentic AI + MCP + RAG + Groq </p><p align="center"> <a href="https://agentic-ai-resume-chatbot.streamlit.app/">🚀 Live Demo</a> • <a href="#-quick-start">📖 Documentation</a> </p>

