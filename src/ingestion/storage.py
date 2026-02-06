import shutil
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class StorageError(Exception):
    pass


class PDFStorage:
    def __init__(self, base_dir: str = './data/pdfs'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def store(self, source_path: Path, document_id: str) -> Path:
        upload_date = datetime.now()
        year = upload_date.strftime('%Y')
        month = upload_date.strftime('%m')
        
        storage_dir = self.base_dir / year / month
        storage_dir.mkdir(parents=True, exist_ok=True)
        dest_path = storage_dir / f"{document_id}.pdf"
        
        try:
            shutil.copy2(str(source_path), str(dest_path))
            logger.info(f"Stored: {document_id}")
        except Exception as e:
            raise StorageError(f"Failed to store file: {e}")
        
        return dest_path
    
    def retrieve(self, document_id: str) -> Path:
        matches = list(self.base_dir.glob(f'*/*/{document_id}.pdf'))
        if not matches:
            raise FileNotFoundError(f"Document not found: {document_id}")
        return matches[0]
    
    def delete(self, document_id: str) -> bool:
        try:
            file_path = self.retrieve(document_id)
            file_path.unlink()
            logger.info(f"Deleted: {document_id}")
            return True
        except FileNotFoundError:
            logger.warning(f"Document not found for deletion: {document_id}")
            return False
