# 🚀 HackRX RAG API  
**Enhanced Document Question Answering System with Adaptive Retrieval-Augmented Generation**  

[![Hackathon](https://img.shields.io/badge/Bajaj%20Finserv-HackRx-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)]()
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Built for **Bajaj Finserv HackRx Hackathon**, this project is a **FastAPI-based backend** designed for advanced **document Q&A**. It combines **Retrieval-Augmented Generation (RAG)**, **Google Gemini LLM**, and **FAISS** for fast, accurate, and scalable question answering across multiple document formats.  

---

## ✨ Key Features  

- 🔹 **Adaptive RAG** – Dynamically tunes retrieval & generation parameters for optimal answers.  
- 🔹 **Multi-format Support** – Works with PDFs, DOCX, PPTX, images, and more.  
- 🔹 **Gemini LLM Integration** – Uses Google Gemini for high-quality, context-aware responses.  
- 🔹 **FAISS Vector Store** – Efficient, scalable document retrieval.  
- 🔹 **Batch Q&A** – Ask multiple questions per document in a single request.  
- 🔹 **Caching Layer** – Avoids redundant computations and speeds up repeated queries.  
- 🔹 **Secure Authentication** – Endpoints protected with Bearer token.  
- 🔹 **Dockerized Deployment** – Ready for production & hackathon demos.  

---

## 📂 Project Structure  

```
HackRX-RAG-API/
├── App/
│   ├── main.py                # FastAPI entrypoint
│   ├── config.py              # Config & env management
│   ├── RAG/
│   │   └── rag_llm.py         # Core RAG + Gemini logic
│   ├── routers/
│   │   └── merge.py           # API routes
│   └── Utils/                 # Helper utilities
│       ├── cache_utils.py
│       ├── downloader.py
│       ├── metadata_extractor.py
│       ├── mission_handler.py
│       └── parameter_selector.py
├── cache/                     # FAISS index + metadata cache
├── scripts/
│   └── warm_cache.py          # Pre-load docs into cache
├── requirements_f.txt         # Dependencies
├── Dockerfile                 # Container setup
├── docker-compose.yml         # Multi-container orchestration
└── logs/                      # Runtime logs
```

---

## ⚡ Installation & Setup  

### 🔧 1. Using Docker (Recommended)  

**Prerequisites:** [Docker](https://www.docker.com/get-started)  

```bash
# Build the image
docker build -t hackrx-rag-api .

# Run the container
docker run -p 8000:8000 --env-file App/.env hackrx-rag-api
```

Or with **Docker Compose**:  

```bash
docker-compose up --build
```

---

### 🔧 2. Local Development  

**Prerequisites:** Python 3.11+  

```bash
# Clone repo and enter project
git clone https://github.com/your-username/hackrx-rag-api.git
cd hackrx-rag-api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_f.txt

# Setup environment
export GEMINI_API_KEY=your_gemini_api_key
export BEARER_TOKEN=your_bearer_token

# Run API
uvicorn App.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🔑 Configuration  

Set environment variables in `App/.env`:  

- `GEMINI_API_KEY` → Google Gemini API key  
- `BEARER_TOKEN` → Bearer token for authentication  
- `UPLOAD_DIR` → Directory for uploaded documents (default: `/app/uploaded_documents`)  

---

## 📌 Usage  

- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)  
- **Main Endpoint**:  

```
POST /api/v1/hackrx/run
```

**Request Example:**  

```json
{
  "document_url": "https://example.com/sample.pdf",
  "questions": [
    "What is the summary of this document?",
    "List the key financial metrics."
  ]
}
```

**Response Example:**  

```json
{
  "answers": [
    "This document discusses ...",
    "The financial metrics are ..."
  ]
}
```

---

## 🏆 Hackathon Context  

This project was built for the **Bajaj Finserv HackRx Hackathon** to demonstrate how **AI-powered document intelligence** can:

# 🏆 Achieve **67th** rank out of 4800 teams in the hackathon
# 📑 Extract insights from complex documents
# ⚡ Accelerate business decision-making
# 🔒 Ensure security with caching & authentication
# 🚀 Scale with Docker for enterprise-ready deployment

---

## 🙌 Acknowledgements  

- [FastAPI](https://fastapi.tiangolo.com/)  
- [LangChain](https://www.langchain.com/)  
- [Google Gemini](https://deepmind.google/technologies/gemini/)  
- [FAISS](https://faiss.ai/)  

---

## 📜 License  

This project is for **hackathon and educational purposes**. See [LICENSE](LICENSE) for details.  
