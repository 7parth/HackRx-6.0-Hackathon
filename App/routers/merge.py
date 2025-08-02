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

logger = logging.getLogger(__name__)
router = APIRouter(tags=["hackrx"], prefix="/api/v1")

class HackRXRequest(BaseModel):
    documents: str = Field(..., description="Document URL or plain text content")
    questions: List[str] = Field(..., min_items=1, max_items=20, description="List of questions (max 20)")

class HackRXResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers corresponding to the questions")

# Global RAG system instance
hackrx_rag_system = None

def get_hackrx_rag_system():
    """Singleton pattern for RAG system"""
    global hackrx_rag_system
    if hackrx_rag_system is None:
        hackrx_rag_system = AdaptiveGeneralLLMDocumentQASystem("")
    return hackrx_rag_system

class EnhancedDocumentDownloader:
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
    
    def download_from_url(self, url: str) -> tuple[str, str]:
        """Download document from URL with enhanced error handling"""
        try:
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=45, stream=True, headers=headers)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                raise HTTPException(status_code=413, detail="File too large (max 50MB)")
            
            # Determine file type
            content_type = response.headers.get('content-type', '').lower()
            url_lower = url.lower()
            
            if url_lower.endswith('.pdf') or 'pdf' in content_type:
                ext, file_type = '.pdf', 'pdf'
            elif url_lower.endswith(('.docx', '.doc')) or 'word' in content_type or 'officedocument' in content_type:
                ext, file_type = '.docx', 'docx'
            else:
                # Default to PDF if uncertain
                ext, file_type = '.pdf', 'pdf'
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            
            # Download in chunks to handle large files
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_size += len(chunk)
                    if total_size > self.max_file_size:
                        temp_file.close()
                        os.unlink(temp_file.name)
                        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
                    temp_file.write(chunk)
            
            temp_file.close()
            return temp_file.name, file_type
            
        except requests.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    
    def extract_text_from_pdf_enhanced(self, path: str) -> str:
        """Enhanced PDF extraction with layout awareness"""
        try:
            doc = fitz.open(path)
            text_pages = []
            page_count = len(doc)
            
            logger.info(f"Processing PDF with {page_count} pages")
            
            for page_num in range(min(page_count, 200)):  # Limit to 200 pages for performance
                page = doc[page_num]
                
                # Extract text with layout awareness
                page_text = self._extract_text_with_layout_awareness(page)
                
                if page_text.strip():
                    # Clean and normalize text
                    cleaned_text = self._clean_text(page_text)
                    text_pages.append(cleaned_text)
            
            doc.close()
            
            combined_text = "\n\n".join(text_pages)
            logger.info(f"Extracted {len(combined_text)} characters from PDF")
            return combined_text
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF content: {str(e)}")
    
    def extract_text_from_docx(self, path: str) -> str:
        """Enhanced DOCX extraction"""
        try:
            doc = DocxDocument(path)
            paragraphs = []
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    paragraphs.append(text)
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        paragraphs.append(" | ".join(row_text))
            
            combined_text = "\n\n".join(paragraphs)
            logger.info(f"Extracted {len(combined_text)} characters from DOCX")
            return combined_text
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract DOCX content: {str(e)}")
    
    def _extract_text_with_layout_awareness(self, page) -> str:
        """Extract text respecting layout structure"""
        layout_type = self._detect_column_layout(page)
        
        if layout_type == "two_column":
            return self._extract_two_column_text(page)
        else:
            return page.get_text()
    
    def _detect_column_layout(self, page) -> str:
        """Detect if page has single or multi-column layout"""
        try:
            blocks = page.get_text("dict")
            
            if not blocks.get("blocks"):
                return "single"
            
            text_blocks = []
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        if line.get("spans"):
                            bbox = line["bbox"]
                            text = ' '.join([span.get("text", "") for span in line["spans"]])
                            if text.strip():
                                text_blocks.append({
                                    'x0': bbox[0], 'x1': bbox[2],
                                    'y0': bbox[1], 'y1': bbox[3],
                                    'text': text
                                })
            
            if len(text_blocks) < 10:
                return "single"
            
            # Analyze horizontal distribution
            page_width = page.rect.width
            left_blocks = [b for b in text_blocks if b['x1'] < page_width * 0.55]
            right_blocks = [b for b in text_blocks if b['x0'] > page_width * 0.45]
            
            # If significant content on both sides, likely two-column
            if len(left_blocks) > 3 and len(right_blocks) > 3:
                return "two_column"
            
            return "single"
            
        except Exception:
            return "single"
    
    def _extract_two_column_text(self, page) -> str:
        """Extract text from two-column layout maintaining reading order"""
        try:
            blocks = page.get_text("dict")
            page_width = page.rect.width
            
            left_content = []
            right_content = []
            
            for block in blocks.get("blocks", []):
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
                        left_content.append((block_bbox[1], block_text.strip()))
                    else:
                        right_content.append((block_bbox[1], block_text.strip()))
            
            # Sort by y-coordinate
            left_content.sort(key=lambda x: x[0])
            right_content.sort(key=lambda x: x[0])
            
            # Combine left then right column
            combined_text = []
            combined_text.extend([text for _, text in left_content])
            combined_text.extend([text for _, text in right_content])
            
            return "\n\n".join(combined_text)
            
        except Exception:
            return page.get_text()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Simple page numbers
        text = re.sub(r'\n\s*Page\s+\d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        return text.strip()

class AdaptiveParameterSelector:
    """Selects optimal parameters based on document characteristics"""
    
    def get_optimal_parameters(self, document_text: str) -> Dict[str, Any]:
        """Determine optimal parameters based on document size and content"""
        
        # Estimate document characteristics
        word_count = len(document_text.split())
        page_estimate = max(1, word_count // 250)  # Rough estimate: 250 words per page
        paragraphs = [p.strip() for p in document_text.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        
        # Detect technical content
        technical_patterns = r'\b\d+\.\d+\b|\b[A-Z]{2,}\b|\([^)]*\)|\b(?:Figure|Table|Section)\s+\d+'
        technical_matches = len(re.findall(technical_patterns, document_text))
        has_technical_content = technical_matches > (word_count * 0.015)
        
        # Determine optimal parameters
        if page_estimate <= 10:
            chunk_size = 800
            chunk_overlap = 200
            retriever_k = 6
        elif page_estimate <= 35:
            chunk_size = 1000
            chunk_overlap = 300
            retriever_k = 8
        elif page_estimate <= 100:
            chunk_size = 1500
            chunk_overlap = 400
            retriever_k = 12
        else:  # Very large documents
            chunk_size = 2000
            chunk_overlap = 500
            retriever_k = 16
        
        # Adjust based on content characteristics
        if has_technical_content:
            chunk_size = int(chunk_size * 1.2)
            chunk_overlap = int(chunk_overlap * 1.3)
        
        if avg_paragraph_length > 800:
            chunk_size = int(chunk_size * 1.3)
        elif avg_paragraph_length < 200:
            chunk_size = int(chunk_size * 0.8)
        
        # Ensure reasonable bounds
        chunk_size = max(400, min(chunk_size, 3000))
        chunk_overlap = max(50, min(chunk_overlap, chunk_size // 3))
        retriever_k = max(4, min(retriever_k, 20))
        
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
    """
    Enhanced endpoint for processing documents and answering questions with adaptive parameters
    """
    start_time = time.time()
    
    # Validate authorization
    expected_token = "Bearer a087324753b37209904afffffa5ad45b8aac7912c74f61420e7e054237778a95"
    if authorization != expected_token:
        logger.warning("Invalid authorization attempt")
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    # Validate input
    if not request.documents or not request.questions:
        raise HTTPException(
            status_code=400, 
            detail="Both documents and questions are required"
        )
    
    if len(request.questions) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 questions allowed per request"
        )
    
    temp_path = None
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Process document
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
            # Plain text input
            document_text = request.documents.strip()
        
        if not document_text or len(document_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Document content too short or empty")
        
        # Get adaptive parameters
        parameter_selector = AdaptiveParameterSelector()
        optimal_params = parameter_selector.get_optimal_parameters(document_text)
        
        logger.info(f"Using adaptive parameters: {optimal_params}")
        
        # Initialize RAG system with optimal parameters
        llm = AdaptiveGeneralLLMDocumentQASystem(
            google_api_key="",
            chunk_size=optimal_params['chunk_size'],
            chunk_overlap=optimal_params['chunk_overlap'],
            retriever_k=optimal_params['retriever_k'],
            retriever_score_threshold=0.35 if optimal_params['estimated_pages'] > 50 else 0.4,
            use_section_headers_in_chunking=optimal_params['has_technical_content']
        )
        
        doc_data = [{
            "content": document_text,
            "filename": "hackrx_document.txt",
            "doc_id": "hackrx_doc_1"
        }]
        
        # Load documents
        llm.load_documents_from_content_adaptive(doc_data)
        
        # Process questions
        logger.info(f"Processing {len(request.questions)} questions")
        answers = llm.process_questions_batch(request.questions)
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Successfully processed request in {processing_time} seconds")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
