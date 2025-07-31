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

from ..config import settings

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

class QueryProcessor:
    def __init__(self, qa_system: 'GeneralLLMDocumentQASystem'):
        self.qa_system = qa_system
        self.query_history = []
    
    def process_single_query(self, query: str, query_id: Optional[str] = None) -> ProcessingResult:
        try:
            logger.info(f"Processing single query: {query[:100]}...")
            
            if not self.qa_system.retriever:
                raise DocumentProcessingError("No documents loaded in the system. Please load documents first.")
            
            # Process the query using the main QA system
            result = self.qa_system.process_query(query)
            
            # Add query ID if provided
            if query_id:
                result.processing_id = f"{query_id}_{result.processing_id}"
            
            # Store in history
            self.query_history.append({
                "query": query,
                "query_id": query_id,
                "result": result,
                "timestamp": result.timestamp
            })
            
            logger.info(f"Query processed successfully. Decision: {result.decision}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise DocumentProcessingError(f"Query processing failed: {e}")
    
    def process_multiple_queries(self, queries: List[Dict[str, str]]) -> List[ProcessingResult]:
        try:
            logger.info(f"Processing {len(queries)} queries...")
            
            results = []
            for i, query_data in enumerate(queries):
                query = query_data.get("query", "")
                query_id = query_data.get("query_id", f"batch_query_{i+1}")
                
                if not query.strip():
                    logger.warning(f"Skipping empty query at index {i}")
                    continue
                
                try:
                    result = self.process_single_query(query, query_id)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing query {i+1}: {e}")
                    # Create error result
                    error_result = ProcessingResult(
                        decision="error",
                        justification=f"Processing error: {str(e)}",
                        referenced_clauses=[],
                        extracted_entities=ExtractedEntity(data={}, raw_query=query)
                    )
                    if query_id:
                        error_result.processing_id = f"{query_id}_{error_result.processing_id}"
                    results.append(error_result)
            
            logger.info(f"Processed {len(results)} queries successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch query processing: {e}")
            raise DocumentProcessingError(f"Batch query processing failed: {e}")
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get history of processed queries"""
        return [
            {
                "query": item["query"],
                "query_id": item["query_id"], 
                "decision": item["result"].decision,
                "timestamp": item["timestamp"]
            }
            for item in self.query_history
        ]
    
    def clear_query_history(self):
        """Clear query history"""
        self.query_history = []
        logger.info("Query history cleared")

class GeneralLLMDocumentQASystem:
    def __init__(
        self,
        google_api_key: str,
        schema_fields: Optional[List[Dict[str, str]]] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 250
    ):
        os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.1)
        self.processor = DocumentProcessor()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.query_parser = QueryParser(self.llm, schema_fields=schema_fields)
        self.decision_engine = DecisionEngine(self.llm)
        self.vector_store = None
        self.retriever = None
        self.processed_files = []
        
        # Initialize query processor
        self.query_processor = QueryProcessor(self)
        
        logger.info("GeneralLLMDocumentQASystem initialized successfully")

    def load_documents_from_db(self, db_documents: List[Any]):
        logger.info(f"Loading {len(db_documents)} documents from database...")
        
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

        # Process documents and create vector store
        chunks = self.processor.smart_chunk(lc_documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3}
        )
        self.processed_files = [getattr(doc, "filename", f"doc_{getattr(doc, 'id', 'unknown')}") for doc in db_documents]
        
        logger.info(f"Successfully loaded {len(lc_documents)} documents with {len(chunks)} chunks")
        return True

    def load_documents_from_content(self, documents: List[Dict[str, Any]]):
        """
        Load documents from content dictionaries
        documents: List of {"content": str, "filename": str, "doc_id": str (optional)}
        """
        logger.info(f"Loading {len(documents)} documents from content...")
        
        lc_documents = []
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            if not content.strip():
                logger.warning(f"Skipping document {i+1}: no content")
                continue
            
            lc_doc = LCDocument(
                page_content=content,
                metadata={
                    "source": f"content_doc_{i+1}",
                    "doc_id": doc_data.get("doc_id", f"doc_{i+1}"),
                    "filename": doc_data.get("filename", f"document_{i+1}.txt"),
                    "type": "text"
                }
            )
            lc_documents.append(lc_doc)

        if not lc_documents:
            raise DocumentProcessingError("No valid documents found to load.")

        # Process documents and create vector store
        chunks = self.processor.smart_chunk(lc_documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3}
        )
        self.processed_files = [doc_data.get("filename", f"document_{i+1}.txt") for i, doc_data in enumerate(documents)]
        
        logger.info(f"Successfully loaded {len(lc_documents)} documents with {len(chunks)} chunks")
        return True

    def process_query(self, query: str) -> ProcessingResult:
        """
        Internal method for processing a single query
        """
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
                document_source=doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                page_number=doc.metadata.get("page"),
                confidence_score=0.8
            ))

        decision = self.decision_engine.make_decision(entities, clause_refs)
        referenced_clauses = clause_refs
        if "referenced_clauses" in decision:
            try:
                referenced_clauses = [
                    clause_refs[int(idx.lstrip("clause_")) - 1]
                    for idx in decision["referenced_clauses"]
                    if idx.lstrip("clause_").isdigit() and int(idx.lstrip("clause_")) <= len(clause_refs)
                ]
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing referenced clauses: {e}")
                referenced_clauses = clause_refs

        return ProcessingResult(
            decision=decision.get("decision", "undecidable"),
            justification=decision.get("justification", "No justification provided"),
            referenced_clauses=referenced_clauses,
            extracted_entities=entities
        )

    def get_query_processor(self) -> QueryProcessor:
        """
        Get the query processor instance for handling queries
        Your second endpoint will use this
        """
        return self.query_processor

    def system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "files": self.processed_files,
            "num_chunks": len(self.vector_store.index_to_docstore_id) if self.vector_store else 0,
            "schema_fields": self.query_parser.schema_fields,
            "documents_loaded": len(self.processed_files) > 0,
            "query_history_count": len(self.query_processor.query_history)
        }

    def is_ready(self) -> bool:
        """Check if system is ready to process queries"""
        return self.retriever is not None
