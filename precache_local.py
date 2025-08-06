import os
import logging
from App.RAG.rag_llm import AdaptiveGeneralLLMDocumentQASystem
from App.routers.merge import EnhancedDocumentDownloader, FileMetadataExtractor
from App.Utils.cache_utils import (
    get_file_hash,
    get_cache_path_for_file,
    is_cached_file,
    save_to_cache
)

# Set up logging
logging.basicConfig(level=logging.INFO)

DOCUMENT_FOLDER = "documents"
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".xlsx"]

llm = AdaptiveGeneralLLMDocumentQASystem(google_api_key="")
downloader = EnhancedDocumentDownloader()
extractor = FileMetadataExtractor()

for filename in os.listdir(DOCUMENT_FOLDER):
    path = os.path.join(DOCUMENT_FOLDER, filename)

    if not os.path.isfile(path) or not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        continue

    if is_cached_file(path):
        logging.info(f"[SKIP] Already cached: {filename}")
        continue

    try:
        logging.info(f"[PROCESSING] {filename}")
        metadata = extractor.extract_metadata(path)
        file_type = metadata.get("file_type")

        # Extract text
        if file_type == 'pdf':
            text = downloader.extract_text_from_pdf_enhanced(path)
        elif file_type in ['docx', 'doc']:
            text = downloader.extract_text_from_docx(path)
        elif file_type in ['xlsx', 'xls']:
            text = downloader.extract_text_from_xlsx(path)
        elif file_type in ['pptx', 'ppt']:
            text = downloader.extract_text_from_pptx(path)
        elif file_type in ['jpg', 'jpeg', 'png', 'gif']:
            text = downloader.extract_text_from_image(path)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

        if not text or len(text.strip()) < 50:
            logging.warning(f"[SKIP] Document too small: {filename}")
            continue

        doc_data = [{
            "content": text,
            "filename": filename,
            "doc_id": get_file_hash(path)
        }]

        llm.load_documents_from_content_adaptive(doc_data)
        save_to_cache(path, text, metadata, llm.vector_store)
        logging.info(f"[SUCCESS] Cached: {filename}")

    except Exception as e:
        logging.error(f"[FAIL] Error processing {filename}: {e}")
