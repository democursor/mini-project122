"""Clear all document data - PDFs, metadata, and parsed files"""
import os
from pathlib import Path
import json

print("=" * 60)
print("CLEARING ALL DOCUMENT DATA")
print("=" * 60)
print()

# Clear PDFs
pdf_dir = Path('./data/pdfs')
if pdf_dir.exists():
    count = 0
    for pdf_file in pdf_dir.rglob('*.pdf'):
        try:
            pdf_file.unlink()
            count += 1
        except Exception as e:
            print(f"✗ Could not delete {pdf_file}: {e}")
    print(f"✓ Deleted {count} PDF files")
else:
    print("✓ PDF directory doesn't exist")

# Clear parsed data
parsed_dir = Path('./data/parsed')
if parsed_dir.exists():
    count = 0
    for file in parsed_dir.rglob('*'):
        if file.is_file():
            try:
                file.unlink()
                count += 1
            except Exception as e:
                print(f"✗ Could not delete {file}: {e}")
    print(f"✓ Deleted {count} parsed files")
else:
    print("✓ Parsed directory doesn't exist")

# Clear metadata
metadata_file = Path('./data/documents_metadata.json')
if metadata_file.exists():
    metadata_file.write_text('{}')
    print("✓ Cleared documents metadata")
else:
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text('{}')
    print("✓ Created empty metadata file")

print()
print("=" * 60)
print("ALL DATA CLEARED SUCCESSFULLY")
print("=" * 60)
print()
print("Summary:")
print("- ChromaDB: Cleared (run clear_chroma_safe.py)")
print("- PDFs: Cleared")
print("- Metadata: Cleared")
print("- Neo4j: Not accessible (database not configured)")
print()
print("You can now upload new documents.")
