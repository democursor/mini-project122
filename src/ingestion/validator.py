"""PDF validation module with enhanced error handling"""
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of PDF validation"""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    file_size_mb: Optional[float] = None
    page_count: Optional[int] = None


class ValidationError(Exception):
    """PDF validation error"""
    pass


class PDFValidator:
    """Validates uploaded PDF files with robust error handling"""
    
    # Supported PDF versions
    SUPPORTED_PDF_VERSIONS = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '2.0']
    
    def __init__(self, max_file_size_mb: int = 50, max_pages: int = 1000):
        """
        Initialize validator with limits
        
        Args:
            max_file_size_mb: Maximum file size in MB
            max_pages: Maximum number of pages allowed
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_pages = max_pages
    
    def validate(self, file_path: Path) -> ValidationResult:
        """
        Run all validation checks with comprehensive error handling
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ValidationResult with detailed information
        """
        result = ValidationResult()
        
        # Check if file exists
        if not file_path.exists():
            result.is_valid = False
            result.errors.append(f"File not found: {file_path}")
            return result
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            result.is_valid = False
            result.errors.append(f"Path is not a file: {file_path}")
            return result
        
        try:
            # Run validation checks
            self.validate_format(file_path, result)
            self.validate_size(file_path, result)
            self.validate_integrity(file_path, result)
            
            if not result.errors:
                result.is_valid = True
                logger.info(f"Validation successful: {file_path.name}")
            else:
                result.is_valid = False
                logger.error(f"Validation failed: {file_path.name} - {result.errors}")
                
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unexpected validation error: {str(e)}")
            logger.error(f"Unexpected validation error: {file_path.name} - {e}", exc_info=True)
        
        return result
    
    def validate_format(self, file_path: Path, result: ValidationResult) -> None:
        """
        Validate file format with detailed checks
        
        Args:
            file_path: Path to file
            result: ValidationResult to update
        """
        # Check extension
        if not file_path.suffix.lower() == '.pdf':
            result.errors.append(f"Invalid file extension: {file_path.suffix}. Must be .pdf")
            return
        
        # Check magic number (PDF header)
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                
                if not header.startswith(b'%PDF'):
                    result.errors.append("File is not a valid PDF (invalid header)")
                    return
                
                # Extract PDF version
                try:
                    version_str = header.decode('ascii', errors='ignore')
                    version = version_str[5:8]  # Extract version like "1.4"
                    
                    if version not in self.SUPPORTED_PDF_VERSIONS:
                        result.warnings.append(f"PDF version {version} may not be fully supported")
                        
                except Exception as e:
                    logger.debug(f"Could not extract PDF version: {e}")
                    
        except IOError as e:
            result.errors.append(f"Cannot read file: {str(e)}")
        except Exception as e:
            result.errors.append(f"Format validation error: {str(e)}")
    
    def validate_size(self, file_path: Path, result: ValidationResult) -> None:
        """
        Validate file size with detailed reporting
        
        Args:
            file_path: Path to file
            result: ValidationResult to update
        """
        try:
            file_size = file_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            result.file_size_mb = round(size_mb, 2)
            
        except OSError as e:
            result.errors.append(f"Cannot get file size: {str(e)}")
            return
        
        if file_size == 0:
            result.errors.append("File is empty (0 bytes)")
            return
        
        if file_size > self.max_file_size_bytes:
            result.errors.append(
                f"File too large ({size_mb:.1f}MB). Maximum size is {self.max_file_size_mb}MB"
            )
            return
        
        # Warning for large files
        if size_mb > self.max_file_size_mb * 0.8:
            result.warnings.append(
                f"Large file ({size_mb:.1f}MB). Processing may take longer."
            )
    
    def validate_integrity(self, file_path: Path, result: ValidationResult) -> None:
        """
        Validate PDF integrity with comprehensive checks
        
        Args:
            file_path: Path to file
            result: ValidationResult to update
        """
        try:
            import fitz
        except ImportError:
            result.warnings.append("PyMuPDF not installed, skipping integrity check")
            return
        
        doc = None
        try:
            # Try to open document
            try:
                doc = fitz.open(file_path)
            except Exception as e:
                result.errors.append(f"Cannot open PDF: {str(e)}")
                return
            
            # Check if encrypted
            if doc.is_encrypted:
                result.errors.append("PDF is password-protected. Please upload an unencrypted version.")
                return
            
            # Check page count
            if doc.page_count == 0:
                result.errors.append("PDF has no pages")
                return
            
            result.page_count = doc.page_count
            
            # Warning for very large documents
            if doc.page_count > self.max_pages:
                result.warnings.append(
                    f"Document has {doc.page_count} pages. Only first {self.max_pages} will be processed."
                )
            
            # Try to read first and last page
            try:
                first_page = doc[0]
                text = first_page.get_text()
                
                if not text or len(text.strip()) == 0:
                    result.warnings.append("First page appears to be empty or contains only images")
                    
            except Exception as e:
                result.warnings.append(f"Could not extract text from first page: {str(e)}")
            
            # Check last page for completeness
            try:
                last_page = doc[-1]
                _ = last_page.get_text()
            except Exception as e:
                result.warnings.append(f"Could not read last page: {str(e)}")
            
        except Exception as e:
            result.errors.append(f"PDF integrity check failed: {str(e)}")
        finally:
            if doc:
                doc.close()
