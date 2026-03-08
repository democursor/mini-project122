"""Quick test for Phase 1 implementation"""
import sys
from pathlib import Path

print("Testing Phase 1 Implementation...")
print("-" * 50)

# Test imports
try:
    from src.utils.config import Config
    from src.ingestion import PDFValidator, PDFStorage, PDFUploader
    from src.parsing import PDFParser
    from src.orchestration import DocumentProcessor
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test configuration
try:
    config = Config()
    max_size = config.get('storage.max_file_size_mb', 50)
    print(f"✓ Configuration loaded (max file size: {max_size}MB)")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

# Test component initialization
try:
    validator = PDFValidator(max_size)
    storage = PDFStorage('./data/pdfs')
    uploader = PDFUploader(validator, storage)
    parser = PDFParser()
    processor = DocumentProcessor(config)
    print("✓ All components initialized")
except Exception as e:
    print(f"✗ Initialization error: {e}")
    sys.exit(1)

# Check directory structure
data_dir = Path('./data')
if not data_dir.exists():
    data_dir.mkdir()
    print("✓ Created data directory")
else:
    print("✓ Data directory exists")

print("-" * 50)
print("Phase 1 implementation is ready!")
print("\nRun 'python main.py' to upload and process PDFs")
