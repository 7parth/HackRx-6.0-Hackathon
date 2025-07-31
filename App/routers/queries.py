from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from .. import database, models, schemas
from ..RAG.rag_llm import QueryParser



router = APIRouter(tags=["query"])


@router.post("/query", response_model=schemas.Queryout)
def upload_document(query: schemas.QueryIn, db: Session = Depends(database.get_db)):
    new_query = models.Query(query_text=query.query_text) # type: ignore
    db.add(new_query)
    db.commit()
    db.refresh(new_query)

    #query_llm = QueryParser()
    #query_llm.parse_query(query.query_text)
    return new_query
