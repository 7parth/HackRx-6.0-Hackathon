import os
import json
import re
import threading
import uuid
import numpy as np
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LCDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever  
from langchain.retrievers import EnsembleRetriever  
from ..config import settings 
from ..Utils.cache_utils import (
    CACHE_DIR,
)
from ..Utils.parameter_selector import AdaptiveParameterSelector
from App.Utils.cache_utils import get_file_hash_from_url

@dataclass
class ClauseReference:
    clause_id: str
    clause_text: str
    document_source: str
    page_number: Optional[int] = None
    confidence_score: float = 0.0

@dataclass
class ExtractedEntity:
    data: Dict[str, Any]
    raw_query: str = ""
    confidence_score: float = 0.0
  
@dataclass
class ProcessingResult:
    decision: str
    justification: str
    referenced_clauses: List[ClauseReference]
    extracted_entities: ExtractedEntity
    processing_id: str = ""
    timestamp: str = ""
    


logger = logging.getLogger(__name__)
 
class DocumentQualityAssessor:
    
    def __init__(self):
        # Pre-compile regex patterns for better performance
        self.technical_patterns = [
            re.compile(r'\b\d+\.\d+\b'),
            re.compile(r'\b[A-Z]{2,}\b'),
            re.compile(r'\([^)]*\)'),
            re.compile(r'\b(?:Figure|Table|Section|Chapter|Appendix)\s+\d+', re.IGNORECASE),
            re.compile(r'\b(?:Algorithm|Theorem|Lemma|Proof)\b', re.IGNORECASE),
            re.compile(r'\$[^$]+\$'),
        ]
        
        self.table_patterns = [
            re.compile(r'\t'),
            re.compile(r'\|.*\|'),
            re.compile(r'^\s*\w+\s*:\s*\w+', re.MULTILINE),
        ]
    
    def analyze_content(self, text: str) -> Dict:

        if not text or len(text.strip()) < 50:
            return {
                'avg_paragraph_length': 0,
                'has_technical_content': False,
                'table_density': 0,
                'text_quality_score': 0.0
            }
        
        # Use faster splitting and filtering
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        word_count = len(text.split())
        
        if not paragraphs:
            return {
                'avg_paragraph_length': 0,
                'has_technical_content': False,
                'table_density': 0,
                'text_quality_score': 0.0
            }
        
        # Optimized calculations
        avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs)
        
        # Fast sentence splitting (approximate)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Optimized technical content detection
        technical_matches = 0
        for pattern in self.technical_patterns:
            technical_matches += len(pattern.findall(text))
        
        has_technical_content = technical_matches > (word_count * 0.015)
        
        # Optimized table density calculation
        table_indicators = sum(len(pattern.findall(text)) for pattern in self.table_patterns)
        table_density = min(table_indicators / max(len(paragraphs) * 10, 1), 1.0)
        
        # Simplified quality score calculation
        text_quality_score = min(avg_paragraph_length / 300, 1.0) * 0.7 + min(len(paragraphs) / 50, 1.0) * 0.3
        
        return {
            'avg_paragraph_length': avg_paragraph_length,
            'avg_sentence_length': avg_sentence_length,
            'has_technical_content': has_technical_content,
            'table_density': table_density,
            'text_quality_score': text_quality_score,
            'word_count': word_count,
            'paragraph_count': len(paragraphs)
        }
        

            
            
class EnhancedDocumentProcessor:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
        use_section_headers: bool = True,
        section_header_pattern: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_section_headers = use_section_headers
        
        # Pre-compiled regex for performance
        if section_header_pattern is None:
            self.section_pattern = re.compile(
                r'(?m)^(?:\d+\.?\s*|[IVX]+\.?\s*|[A-Z]\.?\s*)?(?:Chapter|Section|Part|Appendix|Introduction|Conclusion|Summary|Abstract|References)\s*(?:\d+)?[:\.]?\s*',
                re.IGNORECASE | re.MULTILINE
            )
        else:
            self.section_pattern = re.compile(section_header_pattern, re.IGNORECASE | re.MULTILINE)
            
        # self.splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size, 
        #     chunk_overlap=chunk_overlap,
        #     separators=["\n\n", "\n", ". ", " ", ""],
        #     length_function=len,
        #     is_separator_regex=False,
        # )
        self.splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(
            model='models/embedding-001', 
            task_type="retrieval_document",
            normalize=True, # type: ignore
            google_api_key=settings.GEMINI_API_KEY
        ))
        # Pre-compiled cleaning patterns
        self.whitespace_pattern = re.compile(r'\s+')
        self.ocr_patterns = [
            re.compile(r'([a-z])([A-Z])'),
            re.compile(r'(\d+)([A-Za-z])'),
        ]
    
    def smart_chunk(self, documents: List[LCDocument]) -> List[LCDocument]:
        output_chunks = []
        
        # Process documents in parallel for large documents
        if len(documents) == 1 and len(documents[0].page_content) > 50000:
            return self._parallel_chunk_large_document(documents[0])
        
        for doc in documents:
            text = self._preprocess_text(doc.page_content)
            
            # Simplified chunking strategy for performance
            if self.use_section_headers and len(text) > self.chunk_size * 3:
                chunks = self._fast_section_chunking(text, doc.metadata)
            else:
                # Use optimized recursive chunking
                chunk_texts = self.splitter.split_text(text)
                chunks = [LCDocument(page_content=chunk, metadata=doc.metadata) 
                         for chunk in chunk_texts if len(chunk.strip()) > 50]
            
            # Fast post-processing
            for chunk in chunks:
                if len(chunk.page_content.strip()) > 50:
                    output_chunks.append(chunk)
        
        logger.info(f"Created {len(output_chunks)} chunks from {len(documents)} documents")
        return output_chunks
    
    def _parallel_chunk_large_document(self, doc: LCDocument) -> List[LCDocument]:
        text = self._preprocess_text(doc.page_content)
        
        # Split into sections for parallel processing
        section_size = len(text) // 4  # 4 parallel sections
        sections = []
        
        for i in range(0, len(text), section_size):
            section = text[i:i + section_size + self.chunk_overlap]
            sections.append(section)
        
        # Process sections in parallel
        all_chunks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.splitter.split_text, section) for section in sections]
            
            for future in as_completed(futures):
                chunk_texts = future.result()
                for chunk_text in chunk_texts:
                    if len(chunk_text.strip()) > 50:
                        all_chunks.append(LCDocument(page_content=chunk_text, metadata=doc.metadata))
        
        return all_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Optimized text preprocessing"""
        # Fast whitespace normalization
        text = self.whitespace_pattern.sub(' ', text)
        
        # Fast OCR error fixing
        for pattern in self.ocr_patterns:
            text = pattern.sub(r'\1 \2', text)
        
        return text.strip()
    
    def _fast_section_chunking(self, text: str, metadata: Dict) -> List[LCDocument]:
        """Fast section-aware chunking"""
        # Quick section detection
        sections = self.section_pattern.split(text)
        
        if len(sections) <= 2:
            # No clear sections, use regular chunking
            chunk_texts = self.splitter.split_text(text)
            return [LCDocument(page_content=chunk, metadata=metadata) 
                   for chunk in chunk_texts if chunk.strip()]
        
        chunks = []
        current_section = ""
        
        for section in sections:
            if not section.strip():
                continue
                
            # Simple section header detection
            if len(section) < 150 and any(header in section.lower() 
                                        for header in ['chapter', 'section', 'part', 'appendix']):
                current_section = section.strip()
                continue
            
            # Process section content
            section_text = f"{current_section}\n\n{section}" if current_section else section
            
            if len(section_text) > self.chunk_size * 2:
                # Split large sections
                sub_chunks = self.splitter.split_text(section_text)
                for sub_chunk in sub_chunks:
                    if sub_chunk.strip():
                        chunks.append(LCDocument(page_content=sub_chunk, metadata=metadata))
            else:
                if section_text.strip():
                    chunks.append(LCDocument(page_content=section_text, metadata=metadata))
        
        return chunks
    
class DirectAnswerEngine:
    """Enhanced answer generation engine with performance optimizations"""
    
    def __init__(self, llm, max_answer_tokens: int = 512):
        self.llm = llm
        self.max_answer_tokens = max_answer_tokens
        
        # Optimized prompt template
        self.answer_prompt = PromptTemplate.from_template(
            """You are a professional assistant helping users answer questions using insurance policy documents.

                INSTRUCTIONS:
                - Use the provided clauses to answer the user question.
                - If critical details are missing in the clauses, say so, but still answer based on what's available.
                - Avoid guessing or assuming beyond the context.
                - If the answer is partially available, mention what is clear and what is not.
                - Clearly cite any exclusions or conditions if stated.
                - When referring to clauses, don't use clause numbers — just explain in natural language.
                - If a specific policy point is not mentioned at all, say: "The policy does not specify details about [topic]."

                QUESTION:
                {question}

                CLAUSES:
                {context}

                ANSWER:"""

        )
    
    def get_direct_answer(self, query: str, clauses: List[ClauseReference]) -> str:
        """Generate direct answer with performance optimizations"""
        if not clauses:
            return "I cannot find relevant information in the provided document to answer this question."
        
        # Optimized context preparation - take top 5 chunks for speed
        context_parts = []
        total_length = 0
        max_context_length = 3000  # Reduced for faster processing
        
        for i, clause in enumerate(clauses[:5]):  # Reduced from 10 to 5
            clause_text = clause.clause_text.strip()
            
            if total_length + len(clause_text) > max_context_length:
                break
                
            context_parts.append(clause_text)
            total_length += len(clause_text)
        
        if not context_parts:
            return "I cannot find relevant information in the provided document to answer this question."
        
        context = "\n\n".join(context_parts)
        
        try:
            # Generate answer with timeout
            prompt = self.answer_prompt.format(query=query, clauses=context)
            response = self.llm.invoke(prompt)
            
            answer_text = getattr(response, "content", str(response)).strip()
            
            # Quick length check
            if len(answer_text) > self.max_answer_tokens * 3:
                answer_text = answer_text[:self.max_answer_tokens * 3] + "..."
            
            return answer_text if answer_text else "I cannot provide a specific answer based on the available information."
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "API Key exhausted"

class AdaptiveGeneralLLMDocumentQASystem:
    """Enhanced RAG system with performance optimizations and hybrid retrieval"""

    def __init__(
            self,
            google_api_key: str,
            llm_model: str = "gemini-2.0-flash",
            llm_temperature: float = 0.1,
            llm_max_tokens: int = 512,
            top_p: float = 0.95,
            **kwargs
        ):
        # Set up API key
        api_key = google_api_key or settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Google API key is required")
        
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize components with performance settings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001', 
            task_type="retrieval_document",
            normalize=True, # type: ignore
            google_api_key=api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            top_p=top_p,
            timeout=100  # Add timeout for faster responses
        )
        
        # Initialize processors with performance focus
        self.adaptive_processor = AdaptiveParameterSelector()
        self.quality_assessor = DocumentQualityAssessor()
        
        # Initialize with default parameters
        self.processor = EnhancedDocumentProcessor()
        self.answer_engine = DirectAnswerEngine(self.llm, max_answer_tokens=llm_max_tokens)
        
        # Storage
        self.vector_store = None
        self.retriever = None
        self.processed_files = []
        
        # Optimized caching
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 50  # Reduced cache size for memory efficiency
        
        # Index persistence
        self._faiss_index_path = "cache"

        # --- FIXED: Initialize sparse retriever properly ---
        self.sparse_retriever = None  # Will be created later with documents
        logger.info("Initialized with fixed hybrid retrieval system")


        logger.info("Initialized AdaptiveGeneralLLMDocumentQASystem with performance optimizations and hybrid retrieval")
        
    def try_load_from_url_cache(self, url, downloaded_path):
        try:
            file_hash = get_file_hash_from_url(url, downloaded_path)
            logger.info(f"[CACHE LOAD] Attempting to load from cache key: {file_hash}")

            cache_path = os.path.join(CACHE_DIR, file_hash)
            index_path = os.path.join(cache_path, "index")
            if not os.path.exists(index_path):
                logger.info("[CACHE MISS] Index path not found")
                return None

            self.vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"[CACHE HIT] Loaded FAISS index from {index_path}")
            
            # Initialize retriever directly
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 8,  # Default value
                    "score_threshold": 0.4  # Default value
                }
            )

            with open(os.path.join(cache_path, "text.txt"), "r", encoding="utf-8") as f:
                cached_text = f.read()

            with open(os.path.join(cache_path, "metadata.json"), "r", encoding="utf-8") as f:
                cached_metadata = json.load(f)

            return {
                "vector_store": self.vector_store,
                "text": cached_text,
                "metadata": cached_metadata
            }

        except Exception as e:
            logger.error(f"[CACHE ERROR] Failed to load cache: {str(e)}")
            return None

    def save_to_url_cache(self, url: str, text: str, metadata: dict):
        try:
            file_hash = get_file_hash_from_url(url, metadata.get("downloaded_path"))
            cache_path = os.path.join(CACHE_DIR, file_hash)
            os.makedirs(cache_path, exist_ok=True)

            # Save FAISS index
            if self.vector_store:
                self.vector_store.save_local(os.path.join(cache_path, "index"))
                logger.info(f"[CACHE SAVE] Saved FAISS index at: {os.path.join(cache_path, 'index')}")

            # Save raw text
            with open(os.path.join(cache_path, "text.txt"), "w", encoding="utf-8") as f:
                f.write(text)
                logger.info("[CACHE SAVE] Saved raw text")

            # Save metadata
            with open(os.path.join(cache_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
                logger.info("[CACHE SAVE] Saved metadata.json")

        except Exception as e:
            logger.error(f"[CACHE ERROR] Failed to save cache: {str(e)}", exc_info=True)

            
    def load_documents_from_content_adaptive(self, documents: List[Dict[str, Any]]) -> Dict:
        if not documents:
            raise ValueError("No documents provided")
        
        start_time = time.time()
        logger.info(f"Loading {len(documents)} documents with performance-optimized processing")

        all_content_analysis = []
        lc_documents = []

        # Process each document
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            if not content.strip():
                continue
                
            # Get or create content analysis
            if 'content_analysis' in doc_data:
                content_analysis = doc_data['content_analysis']
            else:
                content_analysis = self.quality_assessor.analyze_content(content)
                
            content_analysis['page_count'] = doc_data.get('page_count', 10)
            all_content_analysis.append(content_analysis)
            
            # Normalize content
            normalized_content = re.sub(r'\s+', ' ', content).strip()
            lc_doc = LCDocument(
                page_content=normalized_content,
                metadata={
                    "source": f"content_doc_{i + 1}",
                    "doc_id": doc_data.get("doc_id", f"doc_{i + 1}"),
                    "filename": doc_data.get("filename", f"document_{i + 1}.txt"),
                    "type": "text",
                    "page_count": content_analysis.get('page_count', 10),
                    "quality_score": content_analysis.get('text_quality_score', 0.5)
                }
            )
            lc_documents.append(lc_doc)

        if not lc_documents:
            raise ValueError("No valid documents found after processing")

        # Calculate aggregate document statistics
        max_pages = max(analysis.get('page_count', 10) for analysis in all_content_analysis)
        combined_analysis = {
            'avg_paragraph_length': np.mean([a.get('avg_paragraph_length', 500) for a in all_content_analysis]),
            'has_technical_content': any(a.get('has_technical_content', False) for a in all_content_analysis),
            'table_density': np.mean([a.get('table_density', 0) for a in all_content_analysis]),
            'text_quality_score': np.mean([a.get('text_quality_score', 0.5) for a in all_content_analysis]),
        }

        # Determine optimal processing parameters
        optimal_params = self.adaptive_processor.get_optimal_parameters(
            char_count=sum(len(doc.page_content) for doc in lc_documents),
            page_count=max_pages,
            content_analysis=combined_analysis,
            text=" ".join(doc.page_content for doc in lc_documents)
        )


        # Configure document processor with optimal parameters
        self.processor = EnhancedDocumentProcessor(
            chunk_size=optimal_params['chunk_size'],
            chunk_overlap=optimal_params['chunk_overlap'],
            use_section_headers=True
        )

        logger.info("Processing documents with performance-optimized parameters")
        chunks = self.processor.smart_chunk(lc_documents)
        if not chunks:
            raise ValueError("No chunks created from documents")

        logger.info(f"Created {len(chunks)} chunks, building vector store")
        
        # Create vector store with batching for large documents
        batch_size = min(50, len(chunks))
        if len(chunks) > batch_size:
            initial_chunks = chunks[:batch_size]
            self.vector_store = FAISS.from_documents(initial_chunks, self.embeddings)
            for i in range(batch_size, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [doc.page_content for doc in batch]
                batch_metadatas = [doc.metadata for doc in batch]
                self.vector_store.add_texts(batch_texts, metadatas=batch_metadatas)
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Create sparse retriever with actual documents
        logger.info("Building sparse (BM25) retriever")
        self.sparse_retriever = BM25Retriever.from_documents(
            documents=chunks,
            k=min(8, optimal_params["retriever_k"]),
            relevance_score_fn=lambda score: min(score * 0.7, 1.0)
        )

        # Configure base dense retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": optimal_params["retriever_k"],
                "score_threshold": optimal_params["score_threshold"]
            }
        )

        # Configure multi-query retriever for better recall
        query_prompt = PromptTemplate.from_template(
            "Generate 5 different versions of the following question, each phrased uniquely:\n\n{question}"
        )
        
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                max_tokens=512,
                top_p=0.9,
                timeout=60
            ),
            prompt=query_prompt,
            include_original=True
        )

        # Add compression for precision
        compressor = LLMChainExtractor.from_llm(
            llm=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                max_tokens=512,
                top_p=0.9,
                timeout=60
            )
        )
        compressed_multi_query_retriever = ContextualCompressionRetriever(
            base_retriever=multi_query_retriever,
            base_compressor=compressor
        )

        # --- FIXED: Correct ensemble configuration ---
        logger.info("Configuring ensemble retriever with hybrid retrieval")
        try:
            # Create ensemble with Runnable retrievers directly
            ensemble_retriever = EnsembleRetriever(
                retrievers=[compressed_multi_query_retriever, self.sparse_retriever],
                weights=[0.7, 0.3],
                c=60  # Fusion parameter
            )
            self.retriever = ensemble_retriever
        except Exception as e:
            logger.error(f"Error creating ensemble retriever: {str(e)}")
            logger.warning("Falling back to compressed multi-query retriever only")
            self.retriever = compressed_multi_query_retriever

        # Track processed files
        self.processed_files = [
            doc_data.get("filename", f"document_{i + 1}.txt")
            for i, doc_data in enumerate(documents)
        ]

        # Save index in background thread
        threading.Thread(target=self._save_faiss_index, daemon=True).start()
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully loaded documents in {processing_time:.2f} seconds with parameters: {optimal_params}")
        
        # Return parameters used for processing
        return {
            **optimal_params,
            "total_chunks": len(chunks),
            "sparse_retriever_k": min(8, optimal_params["retriever_k"]),
            "ensemble_weights": [0.7, 0.3]
        }

    def process_query(self, query: str) -> ProcessingResult:
        """Process a single query with performance optimizations and safe fallback"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not self.retriever:
            raise ValueError("No documents loaded. Please load documents first.")
        
        # Fast cache check
        cache_key = f"query_{hash(query.lower().strip())}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        try:
            # --- SAFE RETRIEVAL WITH FALLBACK ---
            try:
                retrieved_docs = self.retriever.invoke(query)
            except ValueError as e:
                if "not enough values to unpack" in str(e).lower():
                    logger.warning("Fallback to sparse retriever due to ensemble error")
                    retrieved_docs = self.sparse_retriever.invoke(query) # type: ignore
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected retrieval error: {str(e)}")
                retrieved_docs = self.sparse_retriever.invoke(query) # type: ignore
            
            # Quick conversion to clause references
            clause_refs = []
            for i, doc in enumerate(retrieved_docs[:8]):  # Limit to top 8 for speed
                # Handle different document formats from retrievers
                if isinstance(doc, tuple) and len(doc) == 2:
                    # (Document, score) format
                    doc_obj, score = doc
                    confidence = max(0.1, min(0.99, float(score)))
                else:
                    # Direct Document format
                    doc_obj = doc
                    confidence = max(0.1, 0.9 - (i * 0.1))
                
                clause_refs.append(ClauseReference(
                    clause_id=f"clause_{i + 1}",
                    clause_text=doc_obj.page_content.strip(),
                    document_source=doc_obj.metadata.get("filename", "unknown"),
                    page_number=doc_obj.metadata.get("page"),
                    confidence_score=confidence
                ))
            
            if not clause_refs:
                result = ProcessingResult(
                    decision="no_answer",
                    justification="I cannot find relevant information in the provided document to answer this question.",
                    referenced_clauses=[],
                    extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=0.0),
                    processing_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                # Fast answer generation
                direct_answer = self.answer_engine.get_direct_answer(query, clause_refs)
                
                result = ProcessingResult(
                    decision="answered",
                    justification=direct_answer,
                    referenced_clauses=clause_refs,
                    extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=1.0),
                    processing_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Fast caching
            with self._cache_lock:
                if len(self._cache) >= self._max_cache_size:
                    # Remove oldest entries
                    oldest_keys = list(self._cache.keys())[:5]
                    for key in oldest_keys:
                        del self._cache[key]
                
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return ProcessingResult(
                decision="error",
                justification="API key exhausted",
                referenced_clauses=[],
                extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=0.0),
                processing_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat()
            )

    def process_questions_batch(self, questions: List[str]) -> List[str]:
        """Process multiple questions with optimized concurrent processing"""
        if not self.is_ready():
            raise ValueError("System is not ready. Please load documents first.")
        
        if not questions:
            return []
        
        logger.info(f"Processing batch of {len(questions)} questions with performance optimization")
        
        def process_single_question(question):
            try:
                result = self.process_query(question)
                return result.justification
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                return "I encountered an error while processing this question. Please try rephrasing it."
        
        # Optimized thread pool with more workers for I/O bound tasks
        max_workers = min(8, len(questions))  # Increased from 6 to 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            answers = list(executor.map(process_single_question, questions))
        
        logger.info(f"Completed batch processing of {len(questions)} questions")
        return answers

    def is_ready(self) -> bool:
        """Check if the system is ready to process queries"""
        return self.retriever is not None
    
    def _save_faiss_index(self):
        """Save FAISS index for potential reuse"""
        if self.vector_store:
            try:
                self.vector_store.save_local(self._faiss_index_path)
                logger.debug("FAISS index saved successfully")
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {str(e)}")
    
    def clear_cache(self):
        """Clear the query cache"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cache cleared")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        with self._cache_lock:
            cache_size = len(self._cache)
        
        return {
            "is_ready": self.is_ready(),
            "processed_files": len(self.processed_files),
            "cache_size": cache_size,
            "vector_store_loaded": self.vector_store is not None,
            "retriever_configured": self.retriever is not None
        }
