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
from langchain.retrievers.multi_query import MultiQueryRetriever
from ..config import settings

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

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
        use_section_headers: bool = False,
        section_header_pattern: Optional[str] = None,
    ):
        """
        - chunk_size and chunk_overlap control splitting.
        - use_section_headers: if True, tries to split on generic section headers.
        - section_header_pattern: regex pattern for section headers; can be None for no such splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_section_headers = use_section_headers
        self.section_re = re.compile(section_header_pattern, re.IGNORECASE) if section_header_pattern else None
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def smart_chunk(self, documents: List[LCDocument]) -> List[LCDocument]:
        output_chunks = []

        for doc in documents:
            text = self._mark_tables(doc.page_content)

            if self.use_section_headers and self.section_re:
                splits = self.section_re.split(text)
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
                        if "[TABLE_START]" in chunk and "[TABLE_END]" in chunk:
                            parts = re.split(r"(\[TABLE_START\].*?\[TABLE_END\])", chunk, flags=re.DOTALL)
                            for part in parts:
                                part = part.strip()
                                if not part:
                                    continue
                                if part.startswith("[TABLE_START]"):
                                    clean = part.replace("[TABLE_START]", "").replace("[TABLE_END]", "").strip()
                                    output_chunks.append(LCDocument(page_content=clean, metadata=doc.metadata))
                                else:
                                    for sub in self.splitter.split_text(part):
                                        output_chunks.append(LCDocument(page_content=sub, metadata=doc.metadata))
                        else:
                            if len(chunk) > self.chunk_size * 1.5:
                                sub_chunks = self.splitter.split_text(chunk)
                                for sub in sub_chunks:
                                    output_chunks.append(LCDocument(page_content=sub, metadata=doc.metadata))
                            else:
                                output_chunks.append(LCDocument(page_content=chunk, metadata=doc.metadata))
                else:
                    output_chunks.extend(self.splitter.split_documents([doc]))
            else:
                # No section header splitting (default for generic docs)
                output_chunks.extend(self.splitter.split_documents([doc]))

        return output_chunks

    def _mark_tables(self, text: str) -> str:
        # Placeholder for future table recognition if needed
        return text

class DirectAnswerEngine:
    def __init__(self, llm, max_answer_tokens: int = 384):
        self.llm = llm
        # General-purpose prompt without any domain jargon
        self.answer_prompt = PromptTemplate.from_template(
            "You are an assistant. Use ONLY the below information to answer the question accurately.\n"
            "If the answer is not contained in the provided context, reply exactly 'Not found in document'.\n"
            "Provide complete and clear answers. Use bullet points or enumeration if appropriate.\n"
            "Context:\n{clauses}\n\nQuestion: {query}\nAnswer:"
        )
        self.max_answer_tokens = max_answer_tokens

    def get_direct_answer(self, query: str, clauses: List[ClauseReference]) -> str:
        # Join up to top 7 chunks fully, avoid truncation
        clause_text = "\n\n---\n\n".join(c.clause_text.strip() for c in clauses[:7])
        prompt = self.answer_prompt.format(query=query, clauses=clause_text)
        response = self.llm.invoke(prompt)
        text = getattr(response, "content", str(response))
        return text[:self.max_answer_tokens].strip() if len(text) > self.max_answer_tokens else text.strip()

class GeneralLLMDocumentQASystem:
    def __init__(
        self,
        google_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
        retriever_k: int = 7,
        retriever_score_threshold: float = 0.5,
        section_header_pattern: Optional[str] = None,
        use_section_headers_in_chunking: bool = False,
        llm_model: str = "gemini-2.0-flash",
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 384,
        top_p: float = 0.9,
    ):
        os.environ["GOOGLE_API_KEY"] = google_api_key or settings.GEMINI_API_KEY
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', normalize=True)
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            top_p=top_p
        )

        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_section_headers=use_section_headers_in_chunking,
            section_header_pattern=section_header_pattern,
        )
        self.answer_engine = DirectAnswerEngine(self.llm, max_answer_tokens=llm_max_tokens)

        self.vector_store = None
        self.retriever = None
        self.processed_files = []

        self._cache = {}
        self._cache_lock = threading.Lock()

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._retriever_k = retriever_k
        self._retriever_score_threshold = retriever_score_threshold

        self._faiss_index_path = "faiss_index.index"

    def _save_faiss_index(self):
        if self.vector_store:
            self.vector_store.save_local(self._faiss_index_path)

    def _load_faiss_index(self) -> bool:
        if os.path.exists(self._faiss_index_path):
            self.vector_store = FAISS.load_local(
                self._faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = MultiQueryRetriever.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": self._retriever_k,
                        "score_threshold": self._retriever_score_threshold,
                    }
                )
            )
            return True
        return False

    def load_documents_from_content(self, documents: List[Dict[str, Any]]):
        if self._load_faiss_index():
            return

        lc_documents = []
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            if not content.strip():
                continue
            normalized_content = re.sub(r'\s+', ' ', content).strip()
            lc_doc = LCDocument(
                page_content=normalized_content,
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

        chunks = self.processor.smart_chunk(lc_documents)

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": self._retriever_k, "score_threshold": self._retriever_score_threshold}
        )

        self.processed_files = [doc_data.get("filename", f"document_{i + 1}.txt") for i, doc_data in enumerate(documents)]

        self._save_faiss_index()

    def process_query(self, query: str) -> ProcessingResult:
        if not query.strip():
            raise Exception("Query cannot be empty.")
        if not self.retriever:
            raise Exception("No documents loaded. Please load documents first.")

        # Generalized basic synonym expansion (can be expanded or removed)
        expansions = {
            # Add domain-agnostic or optional expansions here if desired
        }
        expanded_query = query.lower()
        for key, syns in expansions.items():
            if key in expanded_query:
                expansion_text = " OR ".join(syns)
                expanded_query += f" OR {expansion_text}"

        cache_key = f"query_{hash(expanded_query.strip())}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        retrieved_docs = self.retriever.invoke(expanded_query)
        clause_refs = []
        for i, doc in enumerate(retrieved_docs[:self._retriever_k]):
            clause_refs.append(ClauseReference(
                clause_id=f"clause_{i + 1}",
                clause_text=doc.page_content.strip(),
                document_source=doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                page_number=doc.metadata.get("page"),
                confidence_score=0.85
            ))

        if not clause_refs:
            result = ProcessingResult(
                decision="no_answer",
                justification="No relevant documents found for this query.",
                referenced_clauses=[],
                extracted_entities=ExtractedEntity(data={}, raw_query=query, confidence_score=0.0)
            )
            with self._cache_lock:
                self._cache[cache_key] = result
            return result

        direct_answer = self.answer_engine.get_direct_answer(query, clause_refs)
        entities = ExtractedEntity(data={}, raw_query=query, confidence_score=1.0)
        result = ProcessingResult(
            decision="answered",
            justification=direct_answer,
            referenced_clauses=clause_refs,
            extracted_entities=entities
        )

        with self._cache_lock:
            if len(self._cache) > 100:
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
