from fastapi import APIRouter, HTTPException, Header
from typing import List, Dict, Any
import requests
import tempfile
import os
from docx import Document as DocxDocument
import email
from email import policy
from pydantic import BaseModel
from ..RAG.rag_llm import GeneralLLMDocumentQASystem
import logging
import fitz  # This should import PyMuPDF, not a local file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["hackrx"])

class HackRXRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# Global RAG system instance
hackrx_rag_system = None

def get_hackrx_rag_system():
    global hackrx_rag_system
    if hackrx_rag_system is None:
        hackrx_rag_system = GeneralLLMDocumentQASystem("")
    return hackrx_rag_system

class DocumentDownloader:
    @staticmethod
    def download_from_url(url: str) -> tuple[str, str]:
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Quick file type detection
            if url.lower().endswith('.pdf') or 'pdf' in response.headers.get('content-type', '').lower():
                file_extension, file_type = '.pdf', 'pdf'
            elif url.lower().endswith(('.docx', '.doc')) or 'word' in response.headers.get('content-type', '').lower():
                file_extension, file_type = '.docx', 'docx'
            else:
                file_extension, file_type = '.pdf', 'pdf'  # Default to PDF
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name, file_type
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            if not text.strip():
                raise ValueError("Empty PDF")
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {str(e)}")

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            if not text.strip():
                raise ValueError("Empty DOCX")
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract DOCX text: {str(e)}")

@router.post("/hackrx/run", response_model=HackRXResponse)
def upload_document(
    request: HackRXRequest,
    authorization: str = Header(...)
):
    # Fast authorization check
    if authorization != "Bearer a087324753b37209904afffffa5ad45b8aac7912c74f61420e7e054237778a95":
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Both documents URL and questions are required")
    
    temp_file_path = None
    
    try:
        # Fast document processing
        downloader = DocumentDownloader()
        temp_file_path, file_type = downloader.download_from_url(request.documents)
        
        if file_type == 'pdf':
            document_text = downloader.extract_text_from_pdf(temp_file_path)
        elif file_type == 'docx':
            document_text = downloader.extract_text_from_docx(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        
        # Load into optimized RAG system
        llm = get_hackrx_rag_system()
        
        documents_data = [{
            "content": document_text,
            "filename": f"hackrx_document.{file_type}",
            "doc_id": "hackrx_doc_1"
        }]
        
        llm.load_documents_from_content(documents_data)
        
        # Ultra-fast batch processing
        answers = llm.process_questions_batch(request.questions)
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

@router.get("/hackrx/health")
def get_system_info():
    try:
        rag = get_hackrx_rag_system()
        return {
            "status": "healthy",
            "rag_system_ready": rag is not None,
            "message": "HackRX endpoint is operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
