import os
import re
import threading
import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document as LCDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever

from ..config import settings

logger = logging.getLogger(__name__)

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

class DocumentQualityAssessor:
    """Analyzes document content characteristics for optimal processing"""
    
    def analyze_content(self, text: str) -> Dict:
        """Analyze document content characteristics"""
        if not text or len(text.strip()) < 50:
            return {
                'avg_paragraph_length': 0,
                'has_technical_content': False,
                'table_density': 0,
                'text_quality_score': 0.0
            }
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not paragraphs:
            return {
                'avg_paragraph_length': 0,
                'has_technical_content': False,
                'table_density': 0,
                'text_quality_score': 0.0
            }
        
        # Calculate average paragraph length
        avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs)
        
        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Detect technical content patterns
        technical_patterns = [
            r'\b\d+\.\d+\b',  # Decimal numbers
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\([^)]*\)',      # Parenthetical references
            r'\b(?:Figure|Table|Section|Chapter|Appendix)\s+\d+',  # References
            r'\b(?:Algorithm|Theorem|Lemma|Proof)\b',  # Mathematical terms
            r'\$[^$]+\$',      # LaTeX math expressions
        ]
        
        technical_matches = 0
        word_count = len(text.split())
        
        for pattern in technical_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            technical_matches += matches
        
        has_technical_content = technical_matches > (word_count * 0.015)  # 1.5% threshold
        
        # Estimate table density
        table_indicators = [
            len(re.findall(r'\t', text)),  # Tab characters
            len(re.findall(r'\|.*\|', text)),  # Pipe-separated values
            len(re.findall(r'^\s*\w+\s*:\s*\w+', text, re.MULTILINE)),  # Key-value pairs
        ]
        table_density = sum(table_indicators) / max(len(paragraphs), 1)
        table_density = min(table_density / 10, 1.0)  # Normalize
        
        # Calculate overall text quality score
        quality_factors = [
            min(avg_paragraph_length / 300, 1.0),  # Ideal paragraph length
            min(avg_sentence_length / 20, 1.0),    # Ideal sentence length
            min(len(paragraphs) / 50, 1.0),        # Document completeness
            1.0 - (technical_matches / word_count if word_count > 0 else 0)  # Readability
        ]
        
        text_quality_score = sum(quality_factors) / len(quality_factors)
        
        return {
            'avg_paragraph_length': avg_paragraph_length,
            'avg_sentence_length': avg_sentence_length,
            'has_technical_content': has_technical_content,
            'table_density': min(table_density, 1.0),
            'text_quality_score': text_quality_score,
            'word_count': word_count,
            'paragraph_count': len(paragraphs)
        }

class AdaptiveDocumentProcessor:
    """Determines optimal processing parameters based on document characteristics"""
    
    def get_optimal_parameters(self, page_count: int, content_analysis: Dict) -> Dict:
        """Dynamically determine optimal parameters based on document characteristics"""
        
        # Base parameters for different document sizes
        if page_count <= 10:
            base_chunk_size = 800
            base_overlap = 200
            base_k = 6
            score_threshold = 0.45
        elif page_count <= 35:
            base_chunk_size = 1000
            base_overlap = 300
            base_k = 8
            score_threshold = 0.4
        elif page_count <= 100:
            base_chunk_size = 1500
            base_overlap = 400
            base_k = 12
            score_threshold = 0.35
        else:  # Very large documents
            base_chunk_size = 2000
            base_overlap = 500
            base_k = 16
            score_threshold = 0.3
        
        # Adjust based on content characteristics
        avg_paragraph_length = content_analysis.get('avg_paragraph_length', 500)
        avg_sentence_length = content_analysis.get('avg_sentence_length', 15)
        has_technical_content = content_analysis.get('has_technical_content', False)
        table_density = content_analysis.get('table_density', 0)
        text_quality_score = content_analysis.get('text_quality_score', 0.5)
        
        # Adjust chunk size based on content characteristics
        if avg_paragraph_length > 800:
            base_chunk_size = int(base_chunk_size * 1.3)  # Larger chunks for long paragraphs
        elif avg_paragraph_length < 200:
            base_chunk_size = int(base_chunk_size * 0.8)  # Smaller chunks for short paragraphs
        
        if has_technical_content:
            base_chunk_size = int(base_chunk_size * 1.2)  # Larger chunks for technical content
            base_overlap = int(base_overlap * 1.3)        # More overlap for technical content
        
        if table_density > 0.1:  # High table density
            base_chunk_size = int(base_chunk_size * 1.4)
            base_overlap = int(base_overlap * 1.2)
        
        # Adjust retriever parameters based on document quality
        if text_quality_score < 0.4:  # Poor quality text
            base_k = int(base_k * 1.5)  # Retrieve more documents
            score_threshold *= 0.8      # Lower threshold
        
        # Ensure reasonable bounds
        base_chunk_size = max(400, min(base_chunk_size, 3000))
        base_overlap = max(50, min(base_overlap, base_chunk_size // 3))
        base_k = max(4, min(base_k, 20))
        score_threshold = max(0.1, min(score_threshold, 0.6))
        
        optimal_params = {
            'chunk_size': base_chunk_size,
            'chunk_overlap': base_overlap,
            'retriever_k': base_k,
            'score_threshold': score_threshold,
            'content_characteristics': {
                'avg_paragraph_length': avg_paragraph_length,
                'has_technical_content': has_technical_content,
                'table_density': table_density,
                'text_quality_score': text_quality_score
            }
        }
        
        logger.info(f"Optimal parameters determined: {optimal_params}")
        return optimal_params

class EnhancedDocumentProcessor:
    """Enhanced document processor with adaptive chunking"""
    
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
        
        # Enhanced section header patterns
        if section_header_pattern is None:
            self.section_pattern = r'(?m)^(?:\d+\.?\s*|[IVX]+\.?\s*|[A-Z]\.?\s*)?(?:Chapter|Section|Part|Appendix|Introduction|Conclusion|Summary|Abstract|References)\s*(?:\d+)?[:\.]?\s*'
        else:
            self.section_pattern = section_header_pattern
            
        self.section_re = re.compile(self.section_pattern, re.IGNORECASE | re.MULTILINE) if self.section_pattern else None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def smart_chunk(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Enhanced smart chunking with section awareness"""
        output_chunks = []
        
        for doc in documents:
            text = self._preprocess_text(doc.page_content)
            
            if self.use_section_headers and self.section_re and len(text) > self.chunk_size * 2:
                # Try section-based chunking for longer documents
                chunks = self._section_aware_chunking(text, doc.metadata)
            else:
                # Fall back to recursive chunking
                chunks = self.splitter.split_text(text)
                chunks = [LCDocument(page_content=chunk, metadata=doc.metadata) for chunk in chunks if chunk.strip()]
            
            # Post-process chunks
            for chunk in chunks:
                processed_chunk = self._post_process_chunk(chunk)
                if processed_chunk and len(processed_chunk.page_content.strip()) > 50:  # Minimum chunk size
                    output_chunks.append(processed_chunk)
        
        logger.info(f"Created {len(output_chunks)} chunks from {len(documents)} documents")
        return output_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before chunking"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces between words
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Numbers attached to words
        
        # Mark potential table structures
        text = self._mark_tables(text)
        
        return text.strip()
    
    def _section_aware_chunking(self, text: str, metadata: Dict) -> List[LCDocument]:
        """Perform section-aware chunking"""
        chunks = []
        sections = self.section_re.split(text)
        
        if len(sections) <= 1:
            # No sections found, use regular chunking
            regular_chunks = self.splitter.split_text(text)
            return [LCDocument(page_content=chunk, metadata=metadata) for chunk in regular_chunks if chunk.strip()]
        
        current_section = ""
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Check if this looks like a section header
            if len(section) < 200 and self.section_re.match(section):
                current_section = section
                continue
            
            # Process section content
            section_text = current_section + "\n\n" + section if current_section else section
            
            if len(section_text) > self.chunk_size * 1.5:
                # Section is too long, split it
                sub_chunks = self.splitter.split_text(section_text)
                for sub_chunk in sub_chunks:
                    if sub_chunk.strip():
                        chunk_metadata = metadata.copy()
                        chunk_metadata['section'] = current_section
                        chunks.append(LCDocument(page_content=sub_chunk, metadata=chunk_metadata))
            else:
                # Section fits in one chunk
                if section_text.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata['section'] = current_section
                    chunks.append(LCDocument(page_content=section_text, metadata=chunk_metadata))
            
            current_section = ""
        
        return chunks
    
    def _mark_tables(self, text: str) -> str:
        """Mark table structures for special handling"""
        # Simple table detection - can be enhanced
        lines = text.split('\n')
        table_lines = []
        
        for i, line in enumerate(lines):
            # Detect table-like patterns
            if ('|' in line and line.count('|') > 2) or ('\t' in line and line.count('\t') > 2):
                table_lines.append(i)
        
        # Group consecutive table lines
        if table_lines:
            # Mark table regions - simplified implementation
            pass
        
        return text
    
    def _post_process_chunk(self, chunk: LCDocument) -> Optional[LCDocument]:
        """Post-process individual chunks"""
        content = chunk.page_content.strip()
        
        if len(content) < 50:  # Skip very short chunks
            return None
        
        # Clean up content
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Normalize line breaks
        content = re.sub(r'^\s+|\s+$', '', content)  # Trim whitespace
        
        # Update chunk
        chunk.page_content = content
        return chunk

class DirectAnswerEngine:
    """Enhanced answer generation engine"""
    
    def __init__(self, llm, max_answer_tokens: int = 512):
        self.llm = llm
        self.max_answer_tokens = max_answer_tokens
        
        # Enhanced prompt template
        self.answer_prompt = PromptTemplate.from_template(
            """You are an intelligent assistant that provides accurate answers based solely on the given context.

Instructions:
1. Use ONLY the information provided in the context below
2. If the answer cannot be found in the context, respond with "I cannot find this information in the provided document"
3. Provide complete, clear, and well-structured answers
4. Use bullet points or numbered lists when appropriate
5. Include relevant details and examples from the context
6. Do not make assumptions or add information not present in the context

Context:
{clauses}

Question: {query}

Answer:"""
        )
    
    def get_direct_answer(self, query: str, clauses: List[ClauseReference]) -> str:
        """Generate direct answer with enhanced context handling"""
        if not clauses:
            return "I cannot find relevant information in the provided document to answer this question."
        
        # Prepare context - take top chunks and ensure we don't exceed token limits
        context_parts = []
        total_length = 0
        max_context_length = 4000  # Conservative limit for context
        
        for i, clause in enumerate(clauses[:10]):  # Consider up to 10 most relevant chunks
            clause_text = f"[Source {i+1}]: {clause.clause_text.strip()}"
            
            if total_length + len(clause_text) > max_context_length:
                break
                
            context_parts.append(clause_text)
            total_length += len(clause_text)
        
        if not context_parts:
            return "I cannot find relevant information in the provided document to answer this question."
        
        context = "\n\n".join(context_parts)
        
        try:
            # Generate answer
            prompt = self.answer_prompt.format(query=query, clauses=context)
            response = self.llm.invoke(prompt)
            
            # Extract text from response
            answer_text = getattr(response, "content", str(response))
            
            # Post-process answer
            answer_text = answer_text.strip()
            
            # Ensure answer is not too long
            if len(answer_text) > self.max_answer_tokens * 4:  # Rough character estimate
                answer_text = answer_text[:self.max_answer_tokens * 4] + "..."
            
            return answer_text if answer_text else "I cannot provide a specific answer based on the available information."
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while processing your question. Please try rephrasing it."

class AdaptiveGeneralLLMDocumentQASystem:
    """Enhanced RAG system with adaptive parameter selection"""
    
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
        
        # Initialize components
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001', 
            task_type="retrieval_document",
            normalize=True
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            top_p=top_p
        )
        
        # Initialize processors
        self.adaptive_processor = AdaptiveDocumentProcessor()
        self.quality_assessor = DocumentQualityAssessor()
        
        # Initialize with default parameters (will be updated)
        self.processor = EnhancedDocumentProcessor()
        self.answer_engine = DirectAnswerEngine(self.llm, max_answer_tokens=llm_max_tokens)
        
        # Storage
        self.vector_store = None
        self.retriever = None
        self.processed_files = []
        
        # Caching
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = settings.CACHE_SIZE_LIMIT
        
        # Index persistence
        self._faiss_index_path = "faiss_index"
        
        logger.info("Initialized AdaptiveGeneralLLMDocumentQASystem")
    
    def load_documents_from_content_adaptive(self, documents: List[Dict[str, Any]]) -> Dict:
        """Load documents with adaptive parameter selection"""
        
        if not documents:
            raise ValueError("No documents provided")
        
        logger.info(f"Loading {len(documents)} documents with adaptive processing")
        
        # Analyze all documents to determine optimal parameters
        all_content_analysis = []
        lc_documents = []
        
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            if not content.strip():
                logger.warning(f"Document {i+1} has empty content, skipping")
                continue
            
            # Get or compute content analysis
            if 'content_analysis' in doc_data:
                content_analysis = doc_data['content_analysis']
            else:
                content_analysis = self.quality_assessor.analyze_content(content)
                content_analysis['page_count'] = doc_data.get('page_count', 10)
            
            all_content_analysis.append(content_analysis)
            
            # Create LangChain document
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
        
        # Determine optimal parameters based on the most complex document
        max_pages = max(analysis.get('page_count', 10) for analysis in all_content_analysis)
        
        # Aggregate content characteristics
        combined_analysis = {
            'avg_paragraph_length': np.mean([a.get('avg_paragraph_length', 500) for a in all_content_analysis]),
            'avg_sentence_length': np.mean([a.get('avg_sentence_length', 15) for a in all_content_analysis]),
            'has_technical_content': any(a.get('has_technical_content', False) for a in all_content_analysis),
            'table_density': np.mean([a.get('table_density', 0) for a in all_content_analysis]),
            'text_quality_score': np.mean([a.get('text_quality_score', 0.5) for a in all_content_analysis])
        }
        
        # Get optimal parameters
        optimal_params = self.adaptive_processor.get_optimal_parameters(max_pages, combined_analysis)
        
        # Update processor with optimal parameters
        self.processor = EnhancedDocumentProcessor(
            chunk_size=optimal_params['chunk_size'],
            chunk_overlap=optimal_params['chunk_overlap'],
            use_section_headers=True
        )
        
        # Process documents
        logger.info("Processing documents with optimal parameters")
        chunks = self.processor.smart_chunk(lc_documents)
        
        if not chunks:
            raise ValueError("No chunks created from documents")
        
        logger.info(f"Created {len(chunks)} chunks, building vector store")
        
        # Build vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Create retriever with optimal parameters
        self.retriever = MultiQueryRetriever.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": optimal_params['retriever_k'],
                    "score_threshold": optimal_params['score_threshold']
                }
            )
        )
        
        # Update processed files list
        self.processed_files = [
            doc_data.get("filename", f"document_{i + 1}.txt") 
            for i, doc_data in enumerate(documents)
        ]
        
        # Save index for potential reuse
        self._save_faiss_index()
        
        logger.info(f"Successfully loaded documents with parameters: {optimal_params}")
        return optimal_params
    
    def process_query(self, query: str) -> ProcessingResult:
        """Process a single query with enhanced error handling"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not self.retriever:
            raise ValueError("No documents loaded. Please load documents first.")
        
        # Check cache first
        cache_key = f"query_{hash(query.lower().strip())}"
        with self._cache_lock:
            if cache_key in self._cache:
                logger.debug("Returning cached result")
                return self._cache[cache_key]
        
        try:
            # Retrieve relevant documents
            logger.debug(f"Processing query: {query[:100]}...")
            retrieved_docs = self.retriever.invoke(query)
            
            # Convert to clause references
            clause_refs = []
            for i, doc in enumerate(retrieved_docs):
                confidence = 0.9 - (i * 0.1)  # Decreasing confidence for lower-ranked results
                clause_refs.append(ClauseReference(
                    clause_id=f"clause_{i + 1}",
                    clause_text=doc.page_content.strip(),
                    document_source=doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                    page_number=doc.metadata.get("page"),
                    confidence_score=max(confidence, 0.1)
                ))
            
            if not clause_refs:
                result = ProcessingResult(
                    decision="no_answer",
                    justification="I cannot find relevant information in the provided document to answer this question.",
                    referenced_clauses=[],
                    extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=0.0)
                )
            else:
                # Generate answer
                direct_answer = self.answer_engine.get_direct_answer(query, clause_refs)
                
                result = ProcessingResult(
                    decision="answered",
                    justification=direct_answer,
                    referenced_clauses=clause_refs,
                    extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=1.0)
                )
            
            # Cache the result
            with self._cache_lock:
                if len(self._cache) >= self._max_cache_size:
                    # Remove oldest entries
                    oldest_keys = list(self._cache.keys())[:10]
                    for key in oldest_keys:
                        del self._cache[key]
                
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return ProcessingResult(
                decision="error",
                justification="I encountered an error while processing your question. Please try rephrasing it.",
                referenced_clauses=[],
                extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=0.0)
            )
    
    def process_questions_batch(self, questions: List[str]) -> List[str]:
        """Process multiple questions efficiently"""
        if not self.is_ready():
            raise ValueError("System is not ready. Please load documents first.")
        
        if not questions:
            return []
        
        logger.info(f"Processing batch of {len(questions)} questions")
        
        def process_single_question(question):
            try:
                result = self.process_query(question)
                return result.justification
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                return "I encountered an error while processing this question. Please try rephrasing it."
        
        # Use thread pool for concurrent processing
        max_workers = min(6, len(questions))
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
