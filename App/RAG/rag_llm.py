import os
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
    def __init__(self):
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
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        word_count = len(text.split())
        if not paragraphs:
            return {
                'avg_paragraph_length': 0,
                'has_technical_content': False,
                'table_density': 0,
                'text_quality_score': 0.0
            }
        avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = word_count / max(sentence_count, 1)
        technical_matches = 0
        for pattern in self.technical_patterns:
            technical_matches += len(pattern.findall(text))
        has_technical_content = technical_matches > (word_count * 0.015)
        table_indicators = sum(len(pattern.findall(text)) for pattern in self.table_patterns)
        table_density = min(table_indicators / max(len(paragraphs) * 10, 1), 1.0)
        text_quality_score = min(avg_paragraph_length / 300, 1.0) * 0.7 + min(len(paragraphs) / 50, 1.0) * 0.3
        return {
            'avg_paragraph_length': avg_paragraph_length,
            'avg_sentence_length': avg_sentence_length,
            'has_technical_content': has_technical_content,
            'table_density': table_density,
            'text_quality_score': text_quality_score,
            'word_count': word_count,
            'paragraph_count': len(paragraphs),
        }

class AdaptiveDocumentProcessor:
    def get_optimal_parameters(self, page_count: int, content_analysis: Dict) -> Dict:
        if page_count <= 10:
            base_chunk_size = 1200  # Increased for fewer chunks
            base_overlap = 200
            base_k = 6
            score_threshold = 0.4
        elif page_count <= 35:
            base_chunk_size = 1800
            base_overlap = 300
            base_k = 8
            score_threshold = 0.35
        elif page_count <= 100:
            base_chunk_size = 2500
            base_overlap = 400
            base_k = 10
            score_threshold = 0.3
        else:
            base_chunk_size = 3000
            base_overlap = 500
            base_k = 12
            score_threshold = 0.25

        avg_paragraph_length = content_analysis.get('avg_paragraph_length', 500)
        has_technical_content = content_analysis.get('has_technical_content', False)
        table_density = content_analysis.get('table_density', 0)

        # Removed all upper caps on chunk size and others

        if has_technical_content and page_count > 50:
            base_chunk_size = int(base_chunk_size * 1.1)
        if table_density > 0.1:
            base_chunk_size = int(base_chunk_size * 1.2)

        # Only enforce minimum bounds for sanity
        base_chunk_size = max(800, base_chunk_size)
        base_overlap = max(100, base_overlap)
        base_k = max(6, base_k)
        score_threshold = max(0.2, score_threshold)

        optimal_params = {
            'chunk_size': base_chunk_size,
            'chunk_overlap': base_overlap,
            'retriever_k': base_k,
            'score_threshold': score_threshold,
            'content_characteristics': {
                'avg_paragraph_length': avg_paragraph_length,
                'has_technical_content': has_technical_content,
                'table_density': table_density,
            }
        }
        logger.info(f"Optimal parameters determined: {optimal_params}")
        return optimal_params

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
        if section_header_pattern is None:
            self.section_pattern = re.compile(
                r'(?m)^(?:\d+\.?\s*|[IVX]+\.?\s*|[A-Z]\.?\s*)?(?:Chapter|Section|Part|Appendix|Introduction|Conclusion|Summary|Abstract|References)\s*(?:\d+)?[:\.]?\s*',
                re.IGNORECASE | re.MULTILINE,
            )
        else:
            self.section_pattern = re.compile(section_header_pattern, re.IGNORECASE | re.MULTILINE)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        self.whitespace_pattern = re.compile(r'\s+')
        self.ocr_patterns = [
            re.compile(r'([a-z])([A-Z])'),
            re.compile(r'(\d+)([A-Za-z])'),
        ]

    def smart_chunk(self, documents: List[LCDocument]) -> List[LCDocument]:
        output_chunks = []
        if len(documents) == 1 and len(documents[0].page_content) > 50000:
            return self._parallel_chunk_large_document(documents[0])
        for doc in documents:
            text = self._preprocess_text(doc.page_content)
            if self.use_section_headers and len(text) > self.chunk_size * 3:
                chunks = self._fast_section_chunking(text, doc.metadata)
            else:
                chunk_texts = self.splitter.split_text(text)
                chunks = [LCDocument(page_content=chunk, metadata=doc.metadata) for chunk in chunk_texts if len(chunk.strip()) > 50]
            for chunk in chunks:
                if len(chunk.page_content.strip()) > 50:
                    output_chunks.append(chunk)
        logger.info(f"Created {len(output_chunks)} chunks from {len(documents)} documents")
        return output_chunks

    def _parallel_chunk_large_document(self, doc: LCDocument) -> List[LCDocument]:
        text = self._preprocess_text(doc.page_content)
        section_size = len(text) // 4
        sections = []
        for i in range(0, len(text), section_size):
            section = text[i:i + section_size + self.chunk_overlap]
            sections.append(section)
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
        text = self.whitespace_pattern.sub(' ', text)
        for pattern in self.ocr_patterns:
            text = pattern.sub(r'\1 \2', text)
        return text.strip()

    def _fast_section_chunking(self, text: str, metadata: Dict) -> List[LCDocument]:
        sections = self.section_pattern.split(text)
        if len(sections) <= 2:
            chunk_texts = self.splitter.split_text(text)
            return [LCDocument(page_content=chunk, metadata=metadata) for chunk in chunk_texts if chunk.strip()]
        chunks = []
        current_section = ""
        for section in sections:
            if not section.strip():
                continue
            if len(section) < 150 and any(header in section.lower() for header in ['chapter', 'section', 'part', 'appendix']):
                current_section = section.strip()
                continue
            section_text = f"{current_section}\n\n{section}" if current_section else section
            if len(section_text) > self.chunk_size * 2:
                sub_chunks = self.splitter.split_text(section_text)
                for sub_chunk in sub_chunks:
                    if sub_chunk.strip():
                        chunks.append(LCDocument(page_content=sub_chunk, metadata=metadata))
            else:
                if section_text.strip():
                    chunks.append(LCDocument(page_content=section_text, metadata=metadata))
        return chunks

class DirectAnswerEngine:
    def __init__(self, llm, max_answer_tokens: int = 512):
        self.llm = llm
        self.max_answer_tokens = max_answer_tokens
        self.answer_prompt = PromptTemplate.from_template(
            """Role: Expert insurance analyst. Use only the provided clauses{clauses} and question{query}. 
            If something is missing, say "The policy does not specify [topic]."
            Answer precisely, clearly, and naturally like a human expert. Avoid jargon unless explained.
            Structure answer as:
            1. Direct concise answer leading with key facts
            2. Up to 3 bullet points: Critical limits/exceptions, Requirements, Coverage scope with section references
            3. Verification: List exact section numbers
            Maximum 100 words total.
            """
        )

    def get_direct_answer(self, query: str, clauses: List[ClauseReference]) -> str:
        if not clauses:
            return "I cannot find relevant information in the provided document to answer this question."
        context_parts = []
        total_length = 0
        max_context_length = 3000  # Reduced for faster processing
        for i, clause in enumerate(clauses[:5]):  # Reduced from 10 to 5 for performance
            clause_text = clause.clause_text.strip()
            if total_length + len(clause_text) > max_context_length:
                break
            context_parts.append(clause_text)
            total_length += len(clause_text)
        if not context_parts:
            return "I cannot find relevant information in the provided document to answer this question."
        context = "\n\n".join(context_parts)
        try:
            prompt = self.answer_prompt.format(query=query, clauses=context)
            response = self.llm.invoke(prompt)
            answer_text = getattr(response, "content", str(response)).strip()
            if len(answer_text) > self.max_answer_tokens * 3:
                answer_text = answer_text[:self.max_answer_tokens * 3] + "..."
            return answer_text if answer_text else "I cannot provide a specific answer based on the available information."
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while processing your question. Please try rephrasing it."

class AdaptiveGeneralLLMDocumentQASystem:
    def __init__(
        self,
        google_api_key: str,
        llm_model: str = "gemini-2.0-flash",
        llm_temperature: float = 0.1,
        llm_max_tokens: int = 512,
        top_p: float = 0.95,
        **kwargs
    ):
        api_key = google_api_key or settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("Google API key is required")
        os.environ["GOOGLE_API_KEY"] = api_key

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            task_type="retrieval_document",
            normalize=True
        )

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            top_p=top_p,
            timeout=30  # Timeout to improve performance
        )

        self.adaptive_processor = AdaptiveDocumentProcessor()
        self.quality_assessor = DocumentQualityAssessor()
        self.processor = EnhancedDocumentProcessor()
        self.answer_engine = DirectAnswerEngine(self.llm, max_answer_tokens=llm_max_tokens)
        self.vector_store = None
        self.retriever = None
        self.processed_files = []
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 50
        self._faiss_index_path = "faiss_index"
        logger.info("Initialized AdaptiveGeneralLLMDocumentQASystem with performance optimizations")

    def load_documents_from_content_adaptive(self, documents: List[Dict[str, Any]]) -> Dict:
        if not documents:
            raise ValueError("No documents provided")
        start_time = time.time()
        logger.info(f"Loading {len(documents)} documents with performance-optimized processing")

        all_content_analysis = []
        lc_documents = []

        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            if not content.strip():
                continue
            if 'content_analysis' in doc_data:
                content_analysis = doc_data['content_analysis']
            else:
                content_analysis = self.quality_assessor.analyze_content(content)
            content_analysis['page_count'] = doc_data.get('page_count', 10)
            all_content_analysis.append(content_analysis)
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

        max_pages = max(analysis.get('page_count', 10) for analysis in all_content_analysis)
        combined_analysis = {
            'avg_paragraph_length': np.mean([a.get('avg_paragraph_length', 500) for a in all_content_analysis]),
            'has_technical_content': any(a.get('has_technical_content', False) for a in all_content_analysis),
            'table_density': np.mean([a.get('table_density', 0) for a in all_content_analysis]),
            'text_quality_score': np.mean([a.get('text_quality_score', 0.5) for a in all_content_analysis]),
        }

        optimal_params = self.adaptive_processor.get_optimal_parameters(max_pages, combined_analysis)

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

        self.retriever = MultiQueryRetriever(self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": optimal_params['retriever_k'],
                "score_threshold": optimal_params['score_threshold']
            }
        ))

        self.processed_files = [
            doc_data.get("filename", f"document_{i + 1}.txt")
            for i, doc_data in enumerate(documents)
        ]

        threading.Thread(target=self._save_faiss_index, daemon=True).start()
        processing_time = time.time() - start_time
        logger.info(f"Successfully loaded documents in {processing_time:.2f} seconds with parameters: {optimal_params}")
        return optimal_params

    def process_query(self, query: str) -> ProcessingResult:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not self.retriever:
            raise ValueError("No documents loaded. Please load documents first.")

        cache_key = f"query_{hash(query.lower().strip())}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            retrieved_docs = self.retriever.invoke(query)
            clause_refs = []
            for i, doc in enumerate(retrieved_docs[:8]):
                confidence = 0.9 - (i * 0.1)
                clause_refs.append(ClauseReference(
                    clause_id=f"clause_{i + 1}",
                    clause_text=doc.page_content.strip(),
                    document_source=doc.metadata.get("filename", "unknown"),
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
                direct_answer = self.answer_engine.get_direct_answer(query, clause_refs)
                result = ProcessingResult(
                    decision="answered",
                    justification=direct_answer,
                    referenced_clauses=clause_refs,
                    extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=1.0)
                )
            with self._cache_lock:
                if len(self._cache) >= self._max_cache_size:
                    oldest_keys = list(self._cache.keys())[:5]
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

        max_workers = min(8, len(questions))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            answers = list(executor.map(process_single_question, questions))
        logger.info(f"Completed batch processing of {len(questions)} questions")
        return answers

    def is_ready(self) -> bool:
        return self.retriever is not None

    def _save_faiss_index(self):
        if self.vector_store:
            try:
                self.vector_store.save_local(self._faiss_index_path)
                logger.debug("FAISS index saved successfully")
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {str(e)}")

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cache cleared")

    def get_system_stats(self) -> Dict[str, Any]:
        with self._cache_lock:
            cache_size = len(self._cache)
        return {
            "is_ready": self.is_ready(),
            "processed_files": len(self.processed_files),
            "cache_size": cache_size,
            "vector_store_loaded": self.vector_store is not None,
            "retriever_configured": self.retriever is not None
        }
