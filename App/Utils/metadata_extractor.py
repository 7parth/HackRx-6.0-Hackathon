import os
import logging
import zipfile
from PIL import Image
from pptx import Presentation

logger = logging.getLogger(__name__)
class FileMetadataExtractor:
    def __init__(self):
        self.signatures = [
            (b'%PDF-', 'pdf'),
            (b'\xFF\xD8\xFF', 'jpg'),
            (b'\x89PNG\r\n\x1a\n', 'png'),
            (b'GIF87a', 'gif'),
            (b'GIF89a', 'gif'),
            (b'\x50\x4B\x03\x04', 'zip'),  # zip or OOXML
        ]

    def get_file_type(self, path: str) -> str:
        try:
            with open(path, 'rb') as f:
                header = f.read(32)
            for sig, ftype in self.signatures:
                if header.startswith(sig):
                    if ftype == 'zip':
                        # Check OOXML types
                        try:
                            with zipfile.ZipFile(path, 'r') as z:
                                namelist = z.namelist()
                                if any(name.startswith('word/') for name in namelist):
                                    return 'docx'
                                if any(name.startswith('xl/') for name in namelist):
                                    return 'xlsx'
                                if any(name.startswith('ppt/') for name in namelist):
                                    return 'pptx'
                            return 'zip'
                        except Exception:
                            return 'zip'
                    return ftype

            ext = os.path.splitext(path)[1].lower()
            if ext:
                return ext[1:]
            return 'unknown'
        except Exception as e:
            logger.error(f"File type detection failed: {str(e)}")
            return 'unknown'