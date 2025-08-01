import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document as LCDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from ..config import settings

# --------- Data classes ---------
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

# --------- Document Processing ---------
class DocumentProcessor:
    def __init__(self):
        self.section_re = re.compile(
            r'(^|\n)(SECTION|ARTICLE|PART|CHAPTER|CLAUSE|EXCLUSION|BENEFIT)[\s\dA-Z\.\-\:]+', 
            re.IGNORECASE
        )
        # Reduced chunk size & overlap for speed
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    def smart_chunk(self, documents: List[LCDocument], chunk_size=512, chunk_overlap=64) -> List[LCDocument]:
        splitter = self.splitter if (chunk_size == 512 and chunk_overlap == 64) else RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
                    if len(chunk) > chunk_size * 1.2:
                        sub_chunks = splitter.split_text(chunk)
                        for sub in sub_chunks:
                            output_chunks.append(LCDocument(page_content=sub, metadata=doc.metadata))
                    else:
                        output_chunks.append(LCDocument(page_content=chunk, metadata=doc.metadata))
            else:
                output_chunks.extend(splitter.split_documents([doc]))

        return output_chunks

# --------- Direct Answer Engine ---------
class DirectAnswerEngine:
    def __init__(self, llm):
        self.llm = llm
        self.answer_prompt = PromptTemplate.from_template(
            "Answer the question in 1-2 sentences using ONLY the information below. Be as concise and fact-based as possible.\n"
            "Question: {query}\nDocument:\n{clauses}\nAnswer:"
        )

    def get_direct_answer(self, query: str, clauses: List[ClauseReference]) -> str:
        clause_text = "\n\n".join(c.clause_text[:350].strip() for c in clauses[:3])
        prompt = self.answer_prompt.format(query=query, clauses=clause_text)
        response = self.llm.invoke(prompt)
        text = getattr(response, "content", str(response))
        return text[:300].strip() if len(text) > 300 else text.strip()

# --------- Main RAG System ---------
class GeneralLLMDocumentQASystem:
    def __init__(self, google_api_key: str, chunk_size: int = 512, chunk_overlap: int = 64):
        os.environ["GOOGLE_API_KEY"] = google_api_key or settings.GEMINI_API_KEY
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            temperature=0.0,
            max_tokens=128,
            top_p=0.9
        )

        self.processor = DocumentProcessor()
        self.answer_engine = DirectAnswerEngine(self.llm)

        self.vector_store = None
        self.retriever = None
        self.processed_files = []

        self._cache = {}
        self._cache_lock = threading.Lock()

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # File path to save/load FAISS index for caching embeddings
        self._faiss_index_path = "faiss_index.index"

    # Save FAISS index to disk to avoid repeated embedding in later runs
    def _save_faiss_index(self):
        if self.vector_store:
            self.vector_store.save_local(self._faiss_index_path)

    # --- UPDATED FUNCTION BELOW ---
    # Load FAISS index from disk if exists
    def _load_faiss_index(self) -> bool:
        if os.path.exists(self._faiss_index_path):
            self.vector_store = FAISS.load_local(
                self._faiss_index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # <-- THIS IS THE KEY CHANGE
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 2, "score_threshold": 0.6}
            )
            return True
        return False

    def load_documents_from_content(self, documents: List[Dict[str, Any]]):
        # Try to load cached FAISS index first
        if self._load_faiss_index():
            return

        lc_documents = []
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            if not content.strip():
                continue
            lc_doc = LCDocument(
                page_content=content,
                metadata={
                    "source": f"content_doc_{i + 1}",
                    "doc_id": doc_data.get("doc_id", f"doc_{i + 1}"),
                    "filename": doc_data.get("filename", f"document_{i + 1}.txt"),
                    "type": "text"
                }
            )
            lc_documents.append(lc_doc)

        if not lc_documents:
            raise Exception("No valid documents found to load.")

        # Chunk documents with smaller chunks for speed
        chunks = self.processor.smart_chunk(
            lc_documents,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        # Parallel embedding of chunks to speed up vector store creation
        embeddings_cache = {}

        def embed_chunk(chunk: LCDocument):
            return self.embeddings.embed_query(chunk.page_content)

        with ThreadPoolExecutor(max_workers=8) as executor:
            embedded_vectors = list(executor.map(embed_chunk, chunks))

        # Build FAISS vector store manually from chunks and embedded vectors
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Wrapping retriever with reduced k and increased threshold for speed
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 2, "score_threshold": 0.6}
        )

        self.processed_files = [doc_data.get("filename", f"document_{i+1}.txt") for i, doc_data in enumerate(documents)]

        # Save FAISS index for future use
        self._save_faiss_index()

    def process_query(self, query: str) -> ProcessingResult:
        if not query.strip():
            raise Exception("Query cannot be empty.")
        if not self.retriever:
            raise Exception("No documents loaded. Please load documents first.")

        cache_key = f"query_{hash(query.lower().strip())}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        retrieved_docs = self.retriever.invoke(query)
        clause_refs = []
        for i, doc in enumerate(retrieved_docs[:2]):  # Reduced to top 2 for speed
            clause_refs.append(ClauseReference(
                clause_id=f"clause_{i + 1}",
                clause_text=doc.page_content[:350],
                document_source=doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                page_number=doc.metadata.get("page"),
                confidence_score=0.85
            ))

        direct_answer = self.answer_engine.get_direct_answer(query, clause_refs)
        entities = ExtractedEntity(data={}, raw_query=query, confidence_score=1.0)
        result = ProcessingResult(
            decision="answered",
            justification=direct_answer,
            referenced_clauses=clause_refs,
            extracted_entities=entities
        )

        with self._cache_lock:
            # Keep cache size sane, evict oldest 10 entries if over 25
            if len(self._cache) > 25:
                oldest_keys = list(self._cache.keys())[:10]
                for key in oldest_keys:
                    del self._cache[key]
            self._cache[cache_key] = result

        return result

    def process_questions_batch(self, questions: List[str]) -> List[str]:
        if not self.is_ready():
            raise Exception("No documents loaded. Please load documents first.")

        def process_single_question(q):
            try:
                result = self.process_query(q)
                return result.justification
            except Exception:
                return "Unable to process this question."

        with ThreadPoolExecutor(max_workers=min(6, len(questions))) as executor:
            answers = list(executor.map(process_single_question, questions))

        return answers

    def is_ready(self) -> bool:
        return self.retriever is not None

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()

    def clear_faiss_index_cache(self):
        if os.path.exists(self._faiss_index_path):
            try:
                os.unlink(self._faiss_index_path)
            except Exception:
                pass
        self.vector_store = None
        self.retriever = None
