from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from .. import database, models, schemas
from ..RAG.rag_llm import QueryParser
from ..routers.documents import get_rag_system



router = APIRouter(tags=["query"])


@router.post("/query", response_model=schemas.Queryout)
def upload_document(query: schemas.QueryIn, db: Session = Depends(database.get_db)):
    new_query = models.Query(query_text=query.query_text) # type: ignore
    db.add(new_query)
    db.commit()
    db.refresh(new_query)

    try:
        llm = get_rag_system()  # Get the same instance used in documents
        
        if not llm.is_ready():  # Your existing method
            raise HTTPException(status_code=400, detail="No documents loaded. Please upload documents first.")
        
        # Use your existing QueryProcessor class through get_query_processor()
        query_processor = llm.get_query_processor()  # Your existing method
        result = query_processor.process_single_query(query.query_text, f"query_{new_query.id}")  # Your existing method
        
        # Return both query info AND the complete RAG result
        return {
            "query_id": new_query.id,
            "query_text": new_query.query_text,
            "decision": result.decision,
            "justification": result.justification,
            "referenced_clauses": [
                {
                    "clause_id": clause.clause_id,
                    "clause_text": clause.clause_text,
                    "document_source": clause.document_source,
                    "page_number": clause.page_number,
                    "confidence_score": clause.confidence_score
                }
                for clause in result.referenced_clauses
            ],
            "extracted_entities": {
                "data": result.extracted_entities.data,
                "raw_query": result.extracted_entities.raw_query,
                "confidence_score": result.extracted_entities.confidence_score
            },
            "processing_id": result.processing_id,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
