from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from .. import database, models, schemas
import shutil
import os
import fitz  
from uuid import uuid4
from ..RAG.rag_llm import GeneralLLMDocumentQASystem , DocumentProcessingError

router = APIRouter(tags=["document-upload"])

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/documents/upload", response_model=schemas.Fileout)
def upload_document(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    filename = file.filename or f"{uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save to disk and debug file size
    file_content = file.file.read()
    print("Uploaded file size (bytes):", len(file_content))
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    # Check if empty
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Extract text with PyMuPDF
    try:
        with fitz.open(file_path) as doc:
            text = "".join(page.get_text() for page in doc) # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

    document = models.Document(filename=filename, file_path=file_path, content=text) # type: ignore
    db.add(document)
    db.commit()
    db.refresh(document)

    return document
