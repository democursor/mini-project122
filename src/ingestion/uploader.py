import uuid
import logging
import tempfile
from pathlib import Path

from .validator import PDFValidator
from .storage import PDFStorage

logger = logging.getLogger(__name__)


def generate_document_id() -> str:
    return f"doc_{uuid.uuid4()}"


class PDFUploader:
    def __init__(self, validator: PDFValidator, storage: PDFStorage):
        self.validator = validator
        self.storage = storage
    
    def upload(self, file, filename: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file.read())
            temp_path = Path(tmp.name)
        
        try:
            result = self.validator.validate(temp_path)
            
            if not result.is_valid:
                raise ValueError(result.errors[0])
            
            document_id = generate_document_id()
            self.storage.store(temp_path, document_id)
            
            logger.info(f"Uploaded: {filename} as {document_id}")
            return document_id
            
        finally:
            temp_path.unlink(missing_ok=True)
