import os
import re
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document as LCDocument

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    data: Dict[str, Any]
    raw_query: str = ""
    confidence_score: float = 0.0

@dataclass
class ClauseReference:
    clause_id: str
    clause_text: str
    document_source: str
    page_number: Optional[int] = None
    confidence_score: float = 0.0

@dataclass
class ProcessingResult:
    decision: str
    justification: str
    referenced_clauses: List[ClauseReference]
    extracted_entities: ExtractedEntity
    processing_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.processing_id:
            self.processing_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_json(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "justification": self.justification,
            "referenced_clauses": [asdict(c) for c in self.referenced_clauses],
            "extracted_entities": asdict(self.extracted_entities),
            "processing_id": self.processing_id,
            "timestamp": self.timestamp,
        }

class DocumentProcessingError(Exception):
    pass

class DocumentProcessor:
    def smart_chunk(self, documents: List[LCDocument], chunk_size=1200, chunk_overlap=250) -> List[LCDocument]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        output_chunks = []
        section_re = re.compile(
            r'''(^|\n)(SECTION|ARTICLE|PART|CHAPTER|CLAUSE|EXCLUSION|BENEFIT)[\s\dA-Z\.\-\:]+''', re.IGNORECASE
        )
        for doc in documents:
            splits = section_re.split(doc.page_content)
            if len(splits) > 1:
                temp_chunks = []
                current = ""
                for part in splits:
                    part = part.strip()
                    if section_re.match(part) and current:
                        temp_chunks.append(current.strip())
                        current = part + "\n"
                    else:
                        current += "\n" + part
                if current.strip():
                    temp_chunks.append(current.strip())
                for chunk in temp_chunks:
                    if len(chunk) > chunk_size * 1.5:
                        sub_chunks = splitter.split_text(chunk)
                        for sub in sub_chunks:
                            output_chunks.append(LCDocument(page_content=sub, metadata=doc.metadata))
                    else:
                        output_chunks.append(LCDocument(page_content=chunk, metadata=doc.metadata))
            else:
                output_chunks.extend(splitter.split_documents([doc]))
        return output_chunks

class QueryParser:
    def __init__(self, llm, schema_fields: Optional[List[Dict[str, str]]] = None):
        self.llm = llm
        self.schema_fields = schema_fields or [
            {"name": "age", "type": "integer", "desc": "age in years if available"},
            {"name": "gender", "type": "string/null", "desc": "gender if present"},
            {"name": "procedure", "type": "string/null", "desc": "event, procedure, transaction type, etc. if present"},
            {"name": "location", "type": "string/null", "desc": "location/city/country (if any)"},
            {"name": "amount", "type": "float/null", "desc": "amount claimed or relevant value in query"},
        ]
        fstr = ", ".join(f"{x['name']} ({x['type']}): {x['desc']}" for x in self.schema_fields)
        self.prompt = PromptTemplate.from_template(
            "Extract these fields from the user query below as a JSON object: {fields}\nQuery: \"{query}\"\nJSON Response Only:"
        )
        self.fields_str = fstr

    def parse_query(self, query: str) -> ExtractedEntity:
        prompt = self.prompt.format(fields=self.fields_str, query=query)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        match = re.search(r"{.*}", content, re.DOTALL)
        try:
            data = json.loads(match.group()) if match else {}
        except Exception:
            data = {}
        return ExtractedEntity(data=data, raw_query=query, confidence_score=1.0 if data else 0.3)

class DecisionEngine:
    def __init__(self, llm):
        self.llm = llm
        self.decision_prompt = PromptTemplate.from_template(
            """You are an expert assistant analyzing a user query against a set of policy, contract, or legal documents.

- Provided Entities/Query Info: {entities}
- Relevant Document Extracts/Clauses:
{clauses}

Based on the extracted entities and the retrieved clauses, give a JSON object with these keys:
{{
  "decision": "approved"|"rejected"|"pending"|"undecidable",
  "justification": "<concise reason referencing relevant clause text>",
  "referenced_clauses": ["clause_id1", ...]
}}
Only output the JSON.
"""
        )

    def make_decision(self, entities: ExtractedEntity, clauses: List[ClauseReference]) -> Dict[str, Any]:
        ent_text = json.dumps(entities.data)
        clause_text = "\n\n".join(
            f"Clause {c.clause_id}: {c.clause_text[:500].strip()} (from {c.document_source})"
            for c in clauses
        )
        prompt = self.decision_prompt.format(entities=ent_text, clauses=clause_text)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        match = re.search(r"{.*}", content, re.DOTALL)
        try:
            result = json.loads(match.group()) if match else {}
        except Exception:
            result = {}
        return result

class GeneralLLMDocumentQASystem:
    def __init__(
        self,
        google_api_key: str,
        schema_fields: Optional[List[Dict[str, str]]] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 250
    ):
        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.0-pro', temperature=0.1)
        self.processor = DocumentProcessor()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.query_parser = QueryParser(self.llm, schema_fields=schema_fields)
        self.decision_engine = DecisionEngine(self.llm)
        self.vector_store = None
        self.retriever = None
        self.processed_files = []

    def load_documents_from_db(self, db_documents: List[Any]):
        lc_documents = []
        for doc in db_documents:
            if not hasattr(doc, "content") or not doc.content or not doc.content.strip():
                logger.warning(f"Skipping document id={getattr(doc, 'id', 'unknown')}: no content")
                continue
            
            lc_doc = LCDocument(
                page_content=doc.content,
                metadata={
                    "source": f"db_doc_{getattr(doc, 'id', 'unknown')}",
                    "doc_id": getattr(doc, "id", None),
                    "filename": getattr(doc, "filename", "unknown"),
                    "type": "text"
                }
            )
            lc_documents.append(lc_doc)

        if not lc_documents:
            raise DocumentProcessingError("No valid text documents found to load.")

        chunks = self.processor.smart_chunk(lc_documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3}
        )
        self.processed_files = [doc.filename for doc in db_documents if hasattr(doc, "filename")]

    def process_query(self, query: str) -> ProcessingResult:
        if not query.strip():
            raise DocumentProcessingError("Query cannot be empty.")
        if not self.retriever:
            raise DocumentProcessingError("No documents loaded. Please load documents first.")

        entities = self.query_parser.parse_query(query)
        retrieved_docs = self.retriever.invoke(query)

        clause_refs = []
        for i, doc in enumerate(retrieved_docs):
            clause_refs.append(ClauseReference(
                clause_id=f"clause_{i+1}",
                clause_text=doc.page_content,
                document_source=doc.metadata.get("source", "unknown"),
                page_number=doc.metadata.get("page"),
                confidence_score=0.8
            ))

        decision = self.decision_engine.make_decision(entities, clause_refs)
        referenced_clauses = clause_refs
        if "referenced_clauses" in decision:
            referenced_clauses = [
                clause_refs[int(idx.lstrip("clause_")) - 1]
                for idx in decision["referenced_clauses"]
                if idx.lstrip("clause_").isdigit()
            ]

        return ProcessingResult(
            decision=decision.get("decision", "undecidable"),
            justification=decision.get("justification", ""),
            referenced_clauses=referenced_clauses,
            extracted_entities=entities
        )

    def system_info(self) -> Dict[str, Any]:
        return {
            "files": self.processed_files,
            "num_chunks": len(self.vector_store.index_to_docstore_id) if self.vector_store else 0,
            "schema_fields": self.query_parser.schema_fields
        }