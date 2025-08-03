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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["hackrx"], prefix="/api/v1")


class HackRXRequest(BaseModel):
    documents: str = Field(..., description="Document URL or plain text content")
    questions: List[str] = Field(..., min_items=1, max_items=50, description="List of questions (max 50)")


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
    if authorization != expected_token:
        logger.warning("Invalid authorization attempt")
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Both documents and questions are required")

    if len(request.questions) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 questions allowed per request")

    # --- LOG THE DOCUMENT URL or text snippet ---
    if request.documents.startswith("http://") or request.documents.startswith("https://"):
        logger.info(f"Request document URL: {request.documents}")
    else:
        snippet = request.documents[:100].replace('\n', ' ')
        logger.info(f"Request document (text snippet): {snippet}...")

    temp_path = None

    try:
        logger.info(f"Processing request with {len(request.questions)} questions")

        if request.documents.startswith("http://") or request.documents.startswith("https://"):
            downloader = EnhancedDocumentDownloader()
            temp_path, file_type = downloader.download_from_url(request.documents)

            if file_type == 'pdf':
                document_text = downloader.extract_text_from_pdf_enhanced(temp_path)
            elif file_type == 'docx':
                document_text = downloader.extract_text_from_docx(temp_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        else:
            document_text = request.documents.strip()

        if not document_text or len(document_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Document content too short or empty")

        parameter_selector = AdaptiveParameterSelector()
        optimal_params = parameter_selector.get_optimal_parameters(document_text)

        logger.info(f"Using performance-optimized parameters: {optimal_params}")

        llm = AdaptiveGeneralLLMDocumentQASystem(
            google_api_key="",
            llm_model="gemini-2.0-flash",  # Fastest model
            llm_temperature=0.1,
            llm_max_tokens=400,  # Reduced for faster generation
            top_p=0.95
        )

        doc_data = [{
            "content": document_text,
            "filename": "hackrx_document.txt",
            "doc_id": "hackrx_doc_1"
        }]

        llm.load_documents_from_content_adaptive(doc_data)

        logger.info(f"Processing {len(request.questions)} questions with performance optimization")
        answers = llm.process_questions_batch(request.questions)

        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Successfully processed request in {processing_time} seconds")

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
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
