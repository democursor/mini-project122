"""PDF validation module"""
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of PDF validation"""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ValidationError(Exception):
    """PDF validation error"""
    pass


class PDFValidator:
    """Validates uploaded PDF files"""
    
    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate(self, file_path: Path) -> ValidationResult:
        """Run all validation checks"""
        result = ValidationResult()
        
        try:
            self.validate_format(file_path)
            self.validate_size(file_path)
            self.validate_integrity(file_path)
            result.is_valid = True
            logger.info(f"Validation successful: {file_path.name}")
        except ValidationError as e:
            result.is_valid = False
            result.errors.append(str(e))
            logger.error(f"Validation failed: {file_path.name} - {e}")
        
        return result
    
    def validate_format(self, file_path: Path) -> bool:
        """Validate file format"""
        # Check extension
        if not file_path.suffix.lower() == '.pdf':
            raise ValidationError("File must have .pdf extension")
        
        # Check magic number
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    raise ValidationError("File is not a valid PDF (invalid header)")
        except IOError as e:
            raise ValidationError(f"Cannot read file: {e}")
        
        return True
    
    def validate_size(self, file_path: Path) -> bool:
        """Validate file size"""
        try:
            file_size = file_path.stat().st_size
        except OSError as e:
            raise ValidationError(f"Cannot get file size: {e}")
        
        if file_size == 0:
            raise ValidationError("File is empty")
        
        if file_size > self.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            raise ValidationError(
                f"File too large ({size_mb:.1f}MB). Maximum size is {self.max_file_size_mb}MB"
            )
        
        return True
    
    def validate_integrity(self, file_path: Path) -> bool:
        """Validate PDF integrity"""
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF not installed, skipping integrity check")
            return True
        
        try:
            doc = fitz.open(file_path)
            
            # Check if encrypted
            if doc.is_encrypted:
                doc.close()
                raise ValidationError(
                    "PDF is password-protected. Please upload an unencrypted version."
                )
            
            # Check if has pages
            if doc.page_count == 0:
                doc.close()
                raise ValidationError("PDF has no pages")
            
            # Try to read first page
            first_page = doc[0]
            _ = first_page.get_text()
            
            doc.close()
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"PDF integrity check failed: {str(e)}")
