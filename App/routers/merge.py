from fastapi import APIRouter, HTTPException, Header
from typing import List
import requests
import tempfile
import os
from docx import Document as DocxDocument
from pydantic import BaseModel
from ..RAG.rag_llm import GeneralLLMDocumentQASystem
import fitz

router = APIRouter(tags=["hackrx"])

class HackRXRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

hackrx_rag_system = None

def get_hackrx_rag_system():
    global hackrx_rag_system
    if hackrx_rag_system is None:
        hackrx_rag_system = GeneralLLMDocumentQASystem("")
    return hackrx_rag_system

class DocumentDownloader:
    @staticmethod
    def download_from_url(url: str) -> tuple[str, str]:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        ct = response.headers.get('content-type', '').lower()
        if url.lower().endswith('.pdf') or 'pdf' in ct:
            ext, typ = '.pdf', 'pdf'
        elif url.lower().endswith(('.docx', '.doc')) or 'word' in ct:
            ext, typ = '.docx', 'docx'
        else:
            ext, typ = '.pdf', 'pdf'
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp.write(response.content)
        temp.close()
        return temp.name, typ

    @staticmethod
    def extract_text_from_pdf(path: str) -> str:
        doc = fitz.open(path)
        text_pages = []
        for page in doc:
            text = page.get_text("text")
            text_pages.append(text)
        return "\n\n".join(text_pages).strip()

    @staticmethod
    def extract_text_from_docx(path: str) -> str:
        doc = DocxDocument(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

@router.post("/hackrx/run", response_model=HackRXResponse)
def upload_document(request: HackRXRequest, authorization: str = Header(...)):
    if authorization != "Bearer a087324753b37209904afffffa5ad45b8aac7912c74f61420e7e054237778a95":
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Both documents URL and questions are required")
    temp_path = None
    try:
        if request.documents.startswith("http://") or request.documents.startswith("https://"):
            downloader = DocumentDownloader()
            temp_path, file_type = downloader.download_from_url(request.documents)
            if file_type == 'pdf':
                document_text = downloader.extract_text_from_pdf(temp_path)
            elif file_type == 'docx':
                document_text = downloader.extract_text_from_docx(temp_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        else:
            document_text = request.documents.strip()
        llm = get_hackrx_rag_system()
        doc_data = [{
            "content": document_text,
            "filename": "hackrx_document.txt",
            "doc_id": "hackrx_doc_1"
        }]
        llm.load_documents_from_content(doc_data)
        answers = llm.process_questions_batch(request.questions)
        return HackRXResponse(answers=answers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
