import os
import re
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio

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
    def __init__(self):
        # Pre-compile regex for better performance
        self.section_re = re.compile(
            r'''(^|\n)(SECTION|ARTICLE|PART|CHAPTER|CLAUSE|EXCLUSION|BENEFIT)[\s\dA-Z\.\-\:]+''', 
            re.IGNORECASE
        )
        # Pre-initialize splitter
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def smart_chunk(self, documents: List[LCDocument], chunk_size=1000, chunk_overlap=200) -> List[LCDocument]:
        # Use pre-initialized splitter for optimal speed
        splitter = self.splitter if chunk_size == 1000 and chunk_overlap == 200 else RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        output_chunks = []
        for doc in documents:
            splits = self.section_re.split(doc.page_content)
            if len(splits) > 1:
                temp_chunks = []
                current = ""
                for part in splits:
                    part = part.strip()
                    if self.section_re.match(part) and current:
                        temp_chunks.append(current.strip())
                        current = part + "\n"
                    else:
                        current += "\n" + part
                if current.strip():
                    temp_chunks.append(current.strip())
                    
                for chunk in temp_chunks:
                    if len(chunk) > chunk_size * 1.3:  # Reduced threshold for faster processing
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
        # Simplified prompt for faster processing
        self.prompt = PromptTemplate.from_template(
            "Extract fields as JSON: {fields}\nQuery: \"{query}\"\nJSON:"
        )
        self.fields_str = fstr
        # Pre-compile JSON regex
        self.json_regex = re.compile(r"{.*}", re.DOTALL)

    def parse_query(self, query: str) -> ExtractedEntity:
        prompt = self.prompt.format(fields=self.fields_str, query=query)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        match = self.json_regex.search(content)
        try:
            data = json.loads(match.group()) if match else {}
        except Exception:
            data = {}
        return ExtractedEntity(data=data, raw_query=query, confidence_score=1.0 if data else 0.3)

class DirectAnswerEngine:
    def __init__(self, llm):
        self.llm = llm
        # Optimized prompt for direct, factual answers
        self.answer_prompt = PromptTemplate.from_template(
            """Answer the question directly using the document clauses.

Question: {query}
Document Content:
{clauses}

Provide a factual, specific answer with numbers, timeframes, and conditions. Be concise but complete.

Answer:"""
        )

    def get_direct_answer(self, query: str, clauses: List[ClauseReference]) -> str:
        # Optimize clause processing for speed and accuracy
        clause_text = "\n\n".join(
            f"{c.clause_text[:600].strip()}" 
            for c in clauses[:4]  # Top 4 most relevant clauses
        )
        
        prompt = self.answer_prompt.format(query=query, clauses=clause_text)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        return content.strip() if content else "Unable to determine from available information."

class QueryProcessor:
    def __init__(self, qa_system: 'GeneralLLMDocumentQASystem'):
        self.qa_system = qa_system
        self.query_history = []
        # Optimized thread pool
        self.executor = ThreadPoolExecutor(max_workers=6)
    
    def process_single_query(self, query: str, query_id: Optional[str] = None) -> ProcessingResult:
        try:
            if not self.qa_system.retriever:
                raise DocumentProcessingError("No documents loaded in the system. Please load documents first.")
            
            # Process the query using the main QA system
            result = self.qa_system.process_query(query)
            
            # Add query ID if provided
            if query_id:
                result.processing_id = f"{query_id}_{result.processing_id}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise DocumentProcessingError(f"Query processing failed: {e}")
    
    def process_multiple_queries(self, queries: List[Dict[str, str]]) -> List[ProcessingResult]:
        try:
            # Parallel processing for maximum speed
            def process_query_wrapper(query_data):
                i, query_item = query_data
                query = query_item.get("query", "")
                query_id = query_item.get("query_id", f"batch_query_{i+1}")
                
                if not query.strip():
                    return None
                
                try:
                    return self.process_single_query(query, query_id)
                except Exception as e:
                    error_result = ProcessingResult(
                        decision="error",
                        justification=f"Processing error: {str(e)}",
                        referenced_clauses=[],
                        extracted_entities=ExtractedEntity(data={}, raw_query=query)
                    )
                    if query_id:
                        error_result.processing_id = f"{query_id}_{error_result.processing_id}"
                    return error_result
            
            query_data = list(enumerate(queries))
            results = list(self.executor.map(process_query_wrapper, query_data))
            results = [r for r in results if r is not None]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch query processing: {e}")
            raise DocumentProcessingError(f"Batch query processing failed: {e}")
    
    def get_query_history(self) -> List[Dict[str, Any]]:
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
        self.query_history = []

class GeneralLLMDocumentQASystem:
    def __init__(
        self,
        google_api_key: str,
        schema_fields: Optional[List[Dict[str, str]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
        
        # Optimized settings for speed and accuracy
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',  # Use experimental version for speed
            temperature=0.0,  # Zero temperature for consistency
            max_tokens=300,  # Reduced for faster response
            top_p=0.9
        )
        
        # Initialize processors
        self.processor = DocumentProcessor()
        self.query_parser = QueryParser(self.llm, schema_fields=schema_fields)
        self.answer_engine = DirectAnswerEngine(self.llm)
        
        # Vector store and retriever
        self.vector_store = None
        self.retriever = None
        self.processed_files = []
        
        # Initialize query processor
        self.query_processor = QueryProcessor(self)
        
        # High-performance cache
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        logger.info("GeneralLLMDocumentQASystem initialized successfully")

    def load_documents_from_db(self, db_documents: List[Any]):
        logger.info(f"Loading {len(db_documents)} documents from database...")
        
        lc_documents = []
        for doc in db_documents:
            if not hasattr(doc, "content") or not doc.content or not doc.content.strip():
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
        
        # Optimized retriever for speed and accuracy
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.4}  # Optimized parameters
        )
        self.processed_files = [getattr(doc, "filename", f"doc_{getattr(doc, 'id', 'unknown')}") for doc in db_documents]
        
        logger.info(f"Successfully loaded {len(lc_documents)} documents with {len(chunks)} chunks")
        return True

    def load_documents_from_content(self, documents: List[Dict[str, Any]]):
        logger.info(f"Loading {len(documents)} documents from content...")
        
        if not documents:
            raise DocumentProcessingError("No documents provided to load.")
        
        lc_documents = []
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            if not content or not content.strip():
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

        try:
            chunks = self.processor.smart_chunk(lc_documents)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            # Optimized retriever settings
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.4}
            )
            self.processed_files = [doc_data.get("filename", f"document_{i+1}.txt") for i, doc_data in enumerate(documents)]
            
            logger.info(f"Successfully loaded {len(lc_documents)} documents with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise DocumentProcessingError(f"Failed to create vector store: {str(e)}")

    def process_query(self, query: str) -> ProcessingResult:
        """
        Ultra-optimized query processing for speed and accuracy
        """
        if not query.strip():
            raise DocumentProcessingError("Query cannot be empty.")
        if not self.retriever:
            raise DocumentProcessingError("No documents loaded. Please load documents first.")

        # Smart caching
        cache_key = f"query_{hash(query.lower().strip())}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Parallel processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Only retrieve documents for direct answer generation
            docs_future = executor.submit(self.retriever.invoke, query)
            retrieved_docs = docs_future.result()

        # Create optimized clause references
        clause_refs = []
        for i, doc in enumerate(retrieved_docs[:4]):  # Top 4 for optimal speed/accuracy balance
            clause_refs.append(ClauseReference(
                clause_id=f"clause_{i+1}",
                clause_text=doc.page_content[:700],  # Optimized length
                document_source=doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                page_number=doc.metadata.get("page"),
                confidence_score=0.85
            ))

        # Get direct answer
        direct_answer = self.answer_engine.get_direct_answer(query, clause_refs)
        
        # Create minimal entity extraction for compatibility
        entities = ExtractedEntity(data={}, raw_query=query, confidence_score=1.0)

        result = ProcessingResult(
            decision="answered",
            justification=direct_answer,
            referenced_clauses=clause_refs[:3],  # Top 3 for response
            extracted_entities=entities
        )

        # Efficient caching
        with self._cache_lock:
            if len(self._cache) > 50:  # Smaller cache for faster access
                oldest_keys = list(self._cache.keys())[:10]
                for key in oldest_keys:
                    del self._cache[key]
            self._cache[cache_key] = result

        return result

    def process_questions_batch(self, questions: List[str]) -> List[str]:
        """
        Ultra-fast batch processing with parallel execution
        """
        if not self.is_ready():
            raise DocumentProcessingError("No documents loaded. Please load documents first.")
        
        def process_single_question(question):
            try:
                result = self.process_query(question)
                return result.justification
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                return "Unable to process this question due to a technical error."
        
        # Parallel processing with optimized worker count
        with ThreadPoolExecutor(max_workers=min(8, len(questions))) as executor:
            answers = list(executor.map(process_single_question, questions))
        
        return answers

    def get_query_processor(self) -> QueryProcessor:
        return self.query_processor

    def system_info(self) -> Dict[str, Any]:
        return {
            "files": self.processed_files,
            "num_chunks": len(self.vector_store.index_to_docstore_id) if self.vector_store else 0,
            "schema_fields": self.query_parser.schema_fields,
            "documents_loaded": len(self.processed_files) > 0,
            "query_history_count": len(self.query_processor.query_history),
            "cache_size": len(self._cache)
        }

    def is_ready(self) -> bool:
        return self.retriever is not None

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
