from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from .. import database, models, schemas
import shutil
import os
import fitz  
from uuid import uuid4
from ..RAG.rag_llm import GeneralLLMDocumentQASystem

router = APIRouter(tags=["document-upload"])

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global RAG system instance
rag_system = None

def get_rag_system():
    global rag_system
    if rag_system is None:
        rag_system = GeneralLLMDocumentQASystem("AIzaSyA_4zp5-b3hUE5FzDG1ML6AmbAC7nBaxGA")
    return rag_system

@router.post("/documents/upload", response_model=schemas.Fileout)
def upload_document(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    # Generate filename and path
    filename = file.filename or f"{uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save file to disk
    try:
        file_content = file.file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Extract text with PyMuPDF
    try:
        with fitz.open(file_path) as doc:
            text = "".join(page.get_text() for page in doc)  # type: ignore
            if not text or not text.strip():
                os.remove(file_path)  # Clean up empty file
                raise HTTPException(
                    status_code=400,
                    detail="Extracted text is empty. The document may be image-based or corrupted."
                )
    except Exception as e:
        os.remove(file_path)  # Clean up on failure
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

    # Save to database
    document = models.Document(filename=filename, file_path=file_path, content=text)
    db.add(document)
    db.commit()
    db.refresh(document)

    # Initialize RAG system - UPDATED PART
    try:
        llm = get_rag_system()  # Use persistent instance instead of creating new one
        all_documents = db.query(models.Document).all()  # Get all docs to reload complete corpus
        llm.load_documents_from_db(all_documents)  # Your existing method name
    except Exception as e:
        db.delete(document)
        db.commit()
        os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document in RAG system: {str(e)}"
        )

    return document

# Optional: Add system info endpoint to check RAG status
@router.get("/documents/system-info")
def get_system_info():
    """Check RAG system status"""
    try:
        rag = get_rag_system()
        return rag.system_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")
