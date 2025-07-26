from .database import Base
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content = Column(Text, nullable=False)

class Query(Base):
    __tablename__ = "queries"  

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)  
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    
