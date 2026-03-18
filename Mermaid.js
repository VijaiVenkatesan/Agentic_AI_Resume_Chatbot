graph TD
    subgraph "Universal AI Resume Assistant - Project Architecture"

        subgraph "Phase 1: Document Ingestion & Vectorization"
            A[User Uploads Resume (PDF/DOCX/TXT)] --> B(Streamlit UI)
            B --> C(Load Document <br> (Python))
            C --> D(Split into Chunks <br> (LangChain))
            D --> E(Generate Embeddings <br> (Ollama))
            E --> F{Vector DB <br> (ChromaDB)}
            F -- Ready for Queries --> G(Ready for Queries)
        end

        subgraph "Phase 2: Conversational RAG Cycle"
            H[User Asks Question] --> I(Streamlit UI)
            I --> J(Create Query Embedding)
            J --> K{Vector DB <br> (ChromaDB)}
            K -- Retrieved Context --> L(Augmented Prompt = <br> - User Question <br> - Retrieved Context)
            H -- Original Question --> L
            L --> M(LLM Inference <br> (via Groq API) - Llama 3)
            M --> N(Generated Answer)
            N --> O(Streamlit UI)
        end

        F -- Search Vector DB <br> for Context --> K
    end

    style F fill:#ADD8E6,stroke:#333,stroke-width:2px
    style K fill:#ADD8E6,stroke:#333,stroke-width:2px
    style M fill:#E0FFFF,stroke:#333,stroke-width:2px
    style L fill:#F0F8FF,stroke:#333,stroke-width:2px
