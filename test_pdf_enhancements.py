"""
Test suite for PDF parsing enhancements
Tests optimized parsing for large documents and robust error handling
"""
import logging
import sys
from pathlib import Path
import tempfile
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.parser import PDFParser, ParsingError
from src.ingestion.validator import PDFValidator, ValidationResult


class TestPDFEnhancements:
    """Test class for PDF parsing enhancements"""
    
    def __init__(self):
        self.parser = PDFParser(max_pages=1000, chunk_size=50)
        self.validator = PDFValidator(max_file_size_mb=50, max_pages=1000)
        self.test_results = []
    
    def run_all_tests(self):
        """Run all test cases"""
        logger.info("=" * 80)
        logger.info("Starting PDF Enhancement Tests")
        logger.info("=" * 80)
        
        # Test 1: Validator with valid PDF
        self.test_validator_valid_pdf()
        
        # Test 2: Validator with invalid format
        self.test_validator_invalid_format()
        
        # Test 3: Validator with empty file
        self.test_validator_empty_file()
        
        # Test 4: Validator with non-existent file
        self.test_validator_nonexistent_file()
        
        # Test 5: Parser with valid PDF
        self.test_parser_valid_pdf()
        
        # Test 6: Parser with non-existent file
        self.test_parser_nonexistent_file()
        
        # Test 7: Parser error handling
        self.test_parser_error_handling()
        
        # Print summary
        self.print_summary()
    
    def test_validator_valid_pdf(self):
        """Test validator with a valid PDF"""
        test_name = "Validator - Valid PDF"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Find a test PDF in data/pdfs
            pdf_dir = Path("data/pdfs")
            pdf_files = list(pdf_dir.rglob("*.pdf"))
            
            if not pdf_files:
                logger.warning("No PDF files found in data/pdfs for testing")
                self.test_results.append((test_name, "SKIPPED", "No test PDFs available"))
                return
            
            test_pdf = pdf_files[0]
            logger.info(f"Testing with: {test_pdf}")
            
            result = self.validator.validate(test_pdf)
            
            logger.info(f"Validation Result:")
            logger.info(f"  Valid: {result.is_valid}")
            logger.info(f"  File Size: {result.file_size_mb} MB")
            logger.info(f"  Page Count: {result.page_count}")
            logger.info(f"  Errors: {result.errors}")
            logger.info(f"  Warnings: {result.warnings}")
            
            if result.is_valid:
                self.test_results.append((test_name, "PASSED", "PDF validated successfully"))
            else:
                self.test_results.append((test_name, "FAILED", f"Errors: {result.errors}"))
                
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))

    def test_validator_invalid_format(self):
        """Test validator with invalid file format"""
        test_name = "Validator - Invalid Format"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create a temporary non-PDF file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is not a PDF file")
                temp_file = Path(f.name)
            
            try:
                result = self.validator.validate(temp_file)
                
                logger.info(f"Validation Result:")
                logger.info(f"  Valid: {result.is_valid}")
                logger.info(f"  Errors: {result.errors}")
                
                if not result.is_valid and any('extension' in err.lower() for err in result.errors):
                    self.test_results.append((test_name, "PASSED", "Correctly rejected non-PDF"))
                else:
                    self.test_results.append((test_name, "FAILED", "Should reject non-PDF files"))
            finally:
                temp_file.unlink()
                
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_validator_empty_file(self):
        """Test validator with empty file"""
        test_name = "Validator - Empty File"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create empty PDF file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
                temp_file = Path(f.name)
            
            try:
                result = self.validator.validate(temp_file)
                
                logger.info(f"Validation Result:")
                logger.info(f"  Valid: {result.is_valid}")
                logger.info(f"  Errors: {result.errors}")
                
                if not result.is_valid and any('empty' in err.lower() for err in result.errors):
                    self.test_results.append((test_name, "PASSED", "Correctly rejected empty file"))
                else:
                    self.test_results.append((test_name, "FAILED", "Should reject empty files"))
            finally:
                temp_file.unlink()
                
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_validator_nonexistent_file(self):
        """Test validator with non-existent file"""
        test_name = "Validator - Non-existent File"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            fake_path = Path("nonexistent_file_12345.pdf")
            result = self.validator.validate(fake_path)
            
            logger.info(f"Validation Result:")
            logger.info(f"  Valid: {result.is_valid}")
            logger.info(f"  Errors: {result.errors}")
            
            if not result.is_valid and any('not found' in err.lower() for err in result.errors):
                self.test_results.append((test_name, "PASSED", "Correctly handled missing file"))
            else:
                self.test_results.append((test_name, "FAILED", "Should detect missing files"))
                
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))

    def test_parser_valid_pdf(self):
        """Test parser with a valid PDF"""
        test_name = "Parser - Valid PDF"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Find a test PDF
            pdf_dir = Path("data/pdfs")
            pdf_files = list(pdf_dir.rglob("*.pdf"))
            
            if not pdf_files:
                logger.warning("No PDF files found for parsing test")
                self.test_results.append((test_name, "SKIPPED", "No test PDFs available"))
                return
            
            test_pdf = pdf_files[0]
            logger.info(f"Parsing: {test_pdf}")
            
            parsed_doc = self.parser.parse(test_pdf, "test_doc_001")
            
            logger.info(f"Parsing Result:")
            logger.info(f"  Document ID: {parsed_doc.document_id}")
            logger.info(f"  Title: {parsed_doc.metadata.title}")
            logger.info(f"  Authors: {parsed_doc.metadata.authors}")
            logger.info(f"  Page Count: {parsed_doc.page_count}")
            logger.info(f"  Sections: {len(parsed_doc.sections)}")
            logger.info(f"  Full Text Length: {len(parsed_doc.full_text)} characters")
            
            if parsed_doc.page_count > 0 and len(parsed_doc.full_text) > 0:
                self.test_results.append((test_name, "PASSED", "PDF parsed successfully"))
            else:
                self.test_results.append((test_name, "FAILED", "Parsed document is empty"))
                
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_parser_nonexistent_file(self):
        """Test parser with non-existent file"""
        test_name = "Parser - Non-existent File"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            fake_path = Path("nonexistent_file_12345.pdf")
            
            try:
                parsed_doc = self.parser.parse(fake_path, "test_doc_002")
                self.test_results.append((test_name, "FAILED", "Should raise ParsingError"))
            except ParsingError as e:
                logger.info(f"Correctly raised ParsingError: {e}")
                self.test_results.append((test_name, "PASSED", "Correctly handled missing file"))
                
        except Exception as e:
            logger.error(f"Test failed with unexpected exception: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_parser_error_handling(self):
        """Test parser error handling with corrupted file"""
        test_name = "Parser - Error Handling"
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create a file with PDF extension but invalid content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
                f.write("%PDF-1.4\nThis is corrupted PDF content")
                temp_file = Path(f.name)
            
            try:
                try:
                    parsed_doc = self.parser.parse(temp_file, "test_doc_003")
                    self.test_results.append((test_name, "FAILED", "Should raise ParsingError for corrupted PDF"))
                except ParsingError as e:
                    logger.info(f"Correctly raised ParsingError: {e}")
                    self.test_results.append((test_name, "PASSED", "Correctly handled corrupted PDF"))
            finally:
                temp_file.unlink()
                
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASSED")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAILED")
        skipped = sum(1 for _, status, _ in self.test_results if status == "SKIPPED")
        total = len(self.test_results)
        
        for test_name, status, message in self.test_results:
            status_symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "⊘"
            logger.info(f"{status_symbol} {test_name}: {status} - {message}")
        
        logger.info("\n" + "-" * 80)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"Success Rate: {(passed/total*100) if total > 0 else 0:.1f}%")
        logger.info("=" * 80)


if __name__ == "__main__":
    tester = TestPDFEnhancements()
    tester.run_all_tests()
