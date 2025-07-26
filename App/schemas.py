from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Annotated, Optional


class Fileout(BaseModel):
    id: int
    filename: str
    file_path: str
    content: str

    class Config:
        from_attributes = True

class Queryout(BaseModel):
    id: int
    query_text: str

    class Config:
        from_attributes = True

class QueryIn(BaseModel):
    query_text: str
