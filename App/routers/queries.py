from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from .. import database, models, schemas



router = APIRouter(tags=["query"])


@router.post("/query", response_model=schemas.Queryout)
def upload_document(query: schemas.QueryIn, db: Session = Depends(database.get_db)):
    new_query = models.Query(query_text=query.query_text) # type: ignore
    db.add(new_query)
    db.commit()
    db.refresh(new_query)
    return new_query
