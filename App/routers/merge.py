from fastapi import APIRouter, HTTPException, Header
from typing import List, Dict, Any
import requests
import tempfile
import os
import logging
import re
from docx import Document as DocxDocument
from pydantic import BaseModel, Field
from ..RAG.rag_llm import AdaptiveGeneralLLMDocumentQASystem
import fitz
import time
from concurrent.futures import ThreadPoolExecutor
from .document_relevance_filter import DocumentRelevanceFilter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["hackrx"], prefix="/api/v1")


class HackRXRequest(BaseModel):
    documents: str = Field(..., description="Document URL or plain text content")
    questions: List[str] = Field(..., min_items=1, max_items=50, description="List of questions (max 50)") # type: ignore


class HackRXResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers corresponding to the questions")


hackrx_rag_system = None


def get_hackrx_rag_system():
    """Optimized singleton pattern for RAG system"""
    global hackrx_rag_system
    if hackrx_rag_system is None:
        hackrx_rag_system = AdaptiveGeneralLLMDocumentQASystem("")
    return hackrx_rag_system


class EnhancedDocumentDownloader:
    def __init__(self):
        self.max_file_size = 1024 * 1024 * 1024  # 1 GB
        self.page_number_pattern = re.compile(r'\n\s*\d+\s*\n')
        self.page_header_pattern = re.compile(r'\n\s*Page\s+\d+.*?\n', re.IGNORECASE)
        self.whitespace_pattern = re.compile(r'\s+')
        self.line_break_pattern = re.compile(r'\n\s*\n\s*\n')
            # Add document type mapping

    DOCUMENT_TYPE_MAP = {
        'motorcycle': 'motorcycle or vehicle documentation',
        'vehicle': 'automotive documentation',
        'engine': 'mechanical engineering documentation',
        'automotive': 'automotive documentation',
        'manual': 'technical manual',
        'user guide': 'user guide',
        'physics': 'scientific material',
        'mathematics': 'mathematical content',
        'scientific': 'scientific material',
        'recipe': 'cookbook or recipe',
        'cooking': 'culinary content',
        'entertainment': 'entertainment content',
        'gaming': 'gaming material',
        'sports': 'sports-related content',
        'fiction': 'fictional literature',
        'novel': 'literary work',
        'biography': 'biographical material',
        'technical manual': 'technical documentation',
        'owner manual': 'product manual',
        'service manual': 'technical service manual',
        'maintenance guide': 'technical maintenance guide',
        'academic research': 'academic research paper',
        'thesis': 'academic thesis',
        'dissertation': 'academic dissertation',
        'journal article': 'academic journal article'
    }

    def detect_document_type(self, indicators: List[str]) -> str:
        """Detect document type based on irrelevant indicators"""
        for indicator in indicators:
            doc_type = self.DOCUMENT_TYPE_MAP.get(indicator)
            if doc_type:
                return doc_type
        # Return generic description if no direct match
        if indicators:
            return f"{indicators[0]} related documentation"
        return "non-relevant documentation"

    def download_from_url(self, url: str) -> tuple[str, str]:
        """Download document from URL with enhanced error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=30, stream=True, headers=headers)
            response.raise_for_status()

            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                raise HTTPException(status_code=413, detail=f"File too large (max {self.max_file_size // (1024 * 1024)}MB)")

            content_type = response.headers.get('content-type', '').lower()
            url_lower = url.lower()
            if url_lower.endswith('.pdf') or 'pdf' in content_type:
                ext, file_type = '.pdf', 'pdf'
            elif url_lower.endswith(('.docx', '.doc')) or 'word' in content_type or 'officedocument' in content_type:
                ext, file_type = '.docx', 'docx'
            else:
                ext, file_type = '.pdf', 'pdf'

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            total_size = 0
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    total_size += len(chunk)
                    if total_size > self.max_file_size:
                        temp_file.close()
                        os.unlink(temp_file.name)
                        raise HTTPException(status_code=413, detail=f"File too large (max {self.max_file_size // (1024 * 1024)}MB)")
                    temp_file.write(chunk)
            temp_file.close()
            return temp_file.name, file_type

        except requests.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    def extract_text_from_pdf_enhanced(self, path: str) -> str:
        """Performance-optimized PDF extraction"""
        try:
            doc = fitz.open(path)
            page_count = len(doc)
            logger.info(f"Processing PDF with {page_count} pages")
            max_pages = page_count

            if page_count > 50:
                text_pages = self._extract_pdf_parallel(doc, max_pages)
            else:
                text_pages = self._extract_pdf_sequential(doc, max_pages)

            doc.close()

            combined_text = "\n\n".join(text_pages)
            logger.info(f"Extracted {len(combined_text)} characters from PDF")
            return combined_text

        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF content: {str(e)}")

    def _extract_pdf_parallel(self, doc, max_pages: int) -> List[str]:
        def extract_page(page_num):
            try:
                page = doc[page_num]
                page_text = self._extract_text_with_layout_awareness(page)
                if page_text.strip():
                    return self._clean_text(page_text)
                return ""
            except Exception as e:
                logger.warning(f"Error extracting page {page_num}: {e}")
                return ""

        with ThreadPoolExecutor(max_workers=4) as executor:
            text_pages = list(executor.map(extract_page, range(max_pages)))

        return [page for page in text_pages if page.strip()]

    def _extract_pdf_sequential(self, doc, max_pages: int) -> List[str]:
        text_pages = []
        for page_num in range(max_pages):
            page = doc[page_num]
            page_text = self._extract_text_with_layout_awareness(page)
            if page_text.strip():
                cleaned_text = self._clean_text(page_text)
                text_pages.append(cleaned_text)
        return text_pages

    def extract_text_from_docx(self, path: str) -> str:
        try:
            doc = DocxDocument(path)
            paragraphs = []

            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    paragraphs.append(text)

            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        paragraphs.append(" | ".join(row_text))

            combined_text = "\n\n".join(paragraphs)
            logger.info(f"Extracted {len(combined_text)} characters from DOCX")
            return combined_text

        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract DOCX content: {str(e)}")

    def _extract_text_with_layout_awareness(self, page) -> str:
        try:
            blocks = page.get_text("dict")
            if not blocks.get("blocks"):
                return page.get_text()
            page_width = page.rect.width
            left_blocks = []
            right_blocks = []
            for block in blocks["blocks"]:
                if "lines" not in block:
                    continue

                block_bbox = block["bbox"]
                block_center_x = (block_bbox[0] + block_bbox[2]) / 2
                block_text = ""

                for line in block["lines"]:
                    line_text = ' '.join([span.get("text", "") for span in line.get("spans", [])])
                    if line_text.strip():
                        block_text += line_text + " "

                if block_text.strip():
                    if block_center_x < page_width * 0.55:
                        left_blocks.append((block_bbox[1], block_text.strip()))
                    else:
                        right_blocks.append((block_bbox[1], block_text.strip()))

            if len(left_blocks) > 2 and len(right_blocks) > 2:
                left_blocks.sort()
                right_blocks.sort()
                combined_text = []
                combined_text.extend([text for _, text in left_blocks])
                combined_text.extend([text for _, text in right_blocks])
                return "\n\n".join(combined_text)
            else:
                return page.get_text()

        except Exception:
            return page.get_text()

    def _clean_text(self, text: str) -> str:
        text = self.line_break_pattern.sub('\n\n', text)
        text = self.whitespace_pattern.sub(' ', text)
        text = self.page_number_pattern.sub('\n', text)
        text = self.page_header_pattern.sub('\n', text)
        return text.strip()


class AdaptiveParameterSelector:
    def __init__(self):
        self.technical_pattern = re.compile(r'\b\d+\.\d+\b|\b[A-Z]{2,}\b|\([^)]*\)|\b(?:Figure|Table|Section)\s+\d+')

    def get_optimal_parameters(self, document_text: str) -> Dict[str, Any]:
        word_count = len(document_text.split())
        page_estimate = max(1, word_count // 250)
        technical_matches = len(self.technical_pattern.findall(document_text))
        has_technical_content = technical_matches > (word_count * 0.015)

        if page_estimate <= 10:
            chunk_size, chunk_overlap, retriever_k = 1200, 200, 6
        elif page_estimate <= 35:
            chunk_size, chunk_overlap, retriever_k = 1800, 300, 8
        elif page_estimate <= 100:
            chunk_size, chunk_overlap, retriever_k = 2500, 400, 10
        else:
            chunk_size, chunk_overlap, retriever_k = 3000, 500, 12

        if has_technical_content:
            chunk_size = int(chunk_size * 1.1)

        return {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'retriever_k': retriever_k,
            'estimated_pages': page_estimate,
            'has_technical_content': has_technical_content
        }


@router.post("/hackrx/run", response_model=HackRXResponse)
def process_document_questions(
    request: HackRXRequest,
    authorization: str = Header(..., description="Bearer token for authentication")
    ):
    start_time = time.time()
    expected_token = "Bearer a087324753b37209904afffffa5ad45b8aac7912c74f61420e7e054237778a95"

    # Authorization check
    if authorization != expected_token:
        logger.warning("401 - Invalid authorization token received.")
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    # Field presence checks
    if request.documents is None and request.questions is None:
        logger.warning("400 - Both 'documents' and 'questions' fields are missing.")
        raise HTTPException(status_code=400, detail="Both documents and questions are required")

    if not request.documents:
        logger.warning("400 - 'documents' field is missing or empty.")
        raise HTTPException(status_code=400, detail="'documents' field is required")

    if not request.questions:
        logger.warning("400 - 'questions' field is missing or empty.")
        raise HTTPException(status_code=400, detail="'questions' field is required")

    if not isinstance(request.questions, list):
        logger.warning(f"400 - 'questions' should be a list, got: {type(request.questions)}")
        raise HTTPException(status_code=400, detail="'questions' should be a list of strings")

    if not all(isinstance(q, str) for q in request.questions):
        logger.warning("400 - One or more entries in 'questions' are not strings.")
        raise HTTPException(status_code=400, detail="All questions must be strings.")

    if len(request.questions) > 50:
        logger.warning(f"400 - Too many questions: {len(request.questions)} (max 50 allowed)")
        raise HTTPException(status_code=400, detail="Maximum 50 questions allowed per request")

    # Enhanced logging of request
    logger.info(f"Received request with document source: {'URL' if request.documents.startswith(('http://', 'https://')) else 'TEXT INPUT'}")
    logger.info(f"Document source: {request.documents[:500] + ('...' if len(request.documents) > 500 else '')}")
    logger.info(f"Questions: {request.questions}")

    temp_path = None
    document_text = ""
    relevance_filter = DocumentRelevanceFilter()
    is_relevant = True
    relevance_reason = "Relevance check bypassed for text input"

    try:
        # Process URL-based documents
        if request.documents.startswith(("http://", "https://")):
            downloader = EnhancedDocumentDownloader()
            temp_path, file_type = downloader.download_from_url(request.documents)
            
            # Check document relevance with detailed logging
            logger.info(f"Checking document relevance for {file_type} file")
            is_relevant, relevance_reason, metadata, irrelevant_indicators = relevance_filter.is_document_relevant(
                temp_path, file_type
            )
            
            # Log metadata for debugging
            logger.info(f"Document metadata: {metadata}")
            logger.info(f"Relevance decision: {is_relevant} - Reason: {relevance_reason}")
            logger.info(f"Irrelevant indicators found: {', '.join(irrelevant_indicators)}")
            
            # Extract document title or use default
            doc_name = metadata.get('title', 'the document') or "unnamed document"
            if not doc_name or doc_name == "unnamed document":
                # Try to get filename from URL
                if '/' in request.documents:
                    doc_name = request.documents.rsplit('/', 1)[-1].split('?')[0]
            
            threshold = relevance_filter.min_relevance_threshold
            
            # PAGE COUNT CHECK
            page_count = metadata.get('page_count', 0)
            if page_count > 450:
                error_msg = (f"The document '{doc_name}' has {page_count} pages which exceeds "
                            "the 450-page limit. Please provide a shorter document.")
                return HackRXResponse(answers=[error_msg for _ in request.questions])
            
            if not is_relevant:
                # Detect document type based on irrelevant indicators
                doc_type = downloader.detect_document_type(irrelevant_indicators)
                logger.warning(f"Irrelevant document detected: {doc_type}")
                
                # Extract relevance score from reason string
                score_match = re.search(r"Relevance score: (\d+\.\d+)", relevance_reason)
                score = float(score_match.group(1)) if score_match else 0.0
                
                # Format professional rejection response
                response_msg = (
                    f"The provided document '{doc_name}' has been evaluated and found to have a "
                    f"relevance score of {score:.2f}, which is below the configured threshold of {threshold} "
                    f"for insurance, legal, HR, and compliance domain queries. This document appears to be "
                    f"{doc_type} and is not suitable for processing in the current professional context."
                )
                return HackRXResponse(answers=[response_msg for _ in request.questions])

            # Extract text if relevant
            logger.info(f"Extracting text from {file_type} document")
            if file_type == 'pdf':
                document_text = downloader.extract_text_from_pdf_enhanced(temp_path)
            elif file_type == 'docx':
                document_text = downloader.extract_text_from_docx(temp_path)
        
        # Process text-based input (skip relevance check)
        else:
            logger.info("Processing text input document")
            document_text = request.documents.strip()
            # Log first 500 characters of text input
            logger.info(f"Text input sample: {document_text[:500]}{'...' if len(document_text) > 500 else ''}")

        # Validate extracted text
        if not document_text or len(document_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Document content too short or empty")
        
        logger.info(f"Document contains {len(document_text)} characters")

        parameter_selector = AdaptiveParameterSelector()
        optimal_params = parameter_selector.get_optimal_parameters(document_text)
        logger.info(f"Using performance-optimized parameters: {optimal_params}")

        # Initialize and configure RAG system
        logger.info("Initializing RAG system")
        llm = AdaptiveGeneralLLMDocumentQASystem(
            google_api_key="",
            llm_model="gemini-2.0-flash",
            llm_temperature=0.1,
            llm_max_tokens=400,
            top_p=0.95
        )

        # Prepare document data
        doc_data = [{
            "content": document_text,
            "filename": "hackrx_document.txt",
            "doc_id": "hackrx_doc_1"
        }]

        # Process documents and questions
        logger.info("Loading documents into RAG system")
        llm.load_documents_from_content_adaptive(doc_data)
        
        logger.info(f"Processing {len(request.questions)} questions")
        answers = llm.process_questions_batch(request.questions)

        # Log processing time
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Successfully processed request in {processing_time} seconds")
        
        # Log first 3 answers as sample
        sample_answers = answers[:3]
        if len(answers) > 3:
            sample_answers.append("...")
        logger.info(f"Sample answers: {sample_answers}")

        return HackRXResponse(answers=answers)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")