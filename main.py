"""Phase 1: PDF Ingestion and Parsing - Main Entry Point"""
import logging
from pathlib import Path
from tkinter import Tk, filedialog

from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.ingestion import PDFValidator, PDFStorage, PDFUploader
from src.parsing import PDFParser
from src.orchestration import DocumentProcessor


def select_pdf_file():
    """Open file browser to select PDF"""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


def main():
    # Setup
    config = Config()
    setup_logging(
        log_level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.file')
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Phase 1: PDF Ingestion and Parsing ===")
    
    # Initialize components
    validator = PDFValidator(config.get('storage.max_file_size_mb', 50))
    storage = PDFStorage(config.get('storage.pdf_directory', './data/pdfs'))
    uploader = PDFUploader(validator, storage)
    processor = DocumentProcessor(config)
    
    # Example: Upload and process a PDF
    print("\n--- PDF Upload ---")
    print("Opening file browser...")
    print("(Select a PDF file from the dialog)")
    
    pdf_path = select_pdf_file()
    
    if pdf_path and Path(pdf_path).exists():
        try:
            print(f"\n✓ Selected: {Path(pdf_path).name}")
            
            with open(pdf_path, 'rb') as f:
                document_id = uploader.upload(f, Path(pdf_path).name)
            
            print(f"✓ Upload successful!")
            print(f"  Document ID: {document_id}")
            
            # Process the document
            print("\n--- Processing Document ---")
            result = processor.process_document(document_id)
            
            if result['status'] == 'complete':
                print(f"✓ Processing complete!")
                print(f"\nParsed Data:")
                parsed = result['parsed_data']
                print(f"  Title: {parsed['metadata']['title']}")
                print(f"  Authors: {', '.join(parsed['metadata']['authors'])}")
                print(f"  Pages: {parsed['page_count']}")
                print(f"  Sections: {len(parsed['sections'])}")
                
                # Phase 2 results
                if result.get('chunks'):
                    print(f"\nChunking Results:")
                    print(f"  Total chunks: {len(result['chunks'])}")
                    print(f"  Avg tokens per chunk: {sum(c['token_count'] for c in result['chunks']) // len(result['chunks'])}")
                
                if result.get('concepts'):
                    print(f"\nConcept Extraction Results:")
                    total_entities = sum(len(c['entities']) for c in result['concepts'])
                    total_keyphrases = sum(len(c['keyphrases']) for c in result['concepts'])
                    print(f"  Total entities: {total_entities}")
                    print(f"  Total keyphrases: {total_keyphrases}")
                    
                    if result['concepts'] and result['concepts'][0]['entities']:
                        print(f"\n  Sample entities:")
                        for ent in result['concepts'][0]['entities'][:3]:
                            print(f"    - {ent['text']} ({ent['label']})")
                    
                    if result['concepts'] and result['concepts'][0]['keyphrases']:
                        print(f"\n  Sample keyphrases:")
                        for kp in result['concepts'][0]['keyphrases'][:3]:
                            print(f"    - {kp['phrase']} (score: {kp['score']:.2f})")
                
                # Save parsed output
                output_dir = Path('./data/parsed')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{document_id}.json"
                
                import json
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"\n✓ Saved complete results to: {output_file}")
            else:
                print(f"✗ Processing failed: {result.get('error_message')}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            logger.error(f"Upload/processing error: {e}", exc_info=True)
    else:
        print("No file selected or file not found.")
    
    print("\n=== Phase 1 & 2 Complete ===")
    print("\nComponents implemented:")
    print("  ✓ PDF validation (format, size, integrity)")
    print("  ✓ PDF storage (organized by year/month)")
    print("  ✓ PDF parsing (text, metadata, sections)")
    print("  ✓ Semantic chunking (boundary detection)")
    print("  ✓ Concept extraction (NER + keyphrases)")
    print("  ✓ LangGraph orchestration")
    print("  ✓ File browser for easy PDF selection")
    print("\nNext: Proceed to Phase 3?")


if __name__ == "__main__":
    main()
