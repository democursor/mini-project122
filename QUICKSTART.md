# Phase 1 - Quick Start Guide

## Setup (One-time)

```cmd
:: Activate virtual environment
vnv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt
```

## Test Installation

```cmd
python test_phase1.py
```

Expected output:
```
Testing Phase 1 Implementation...
--------------------------------------------------
✓ All imports successful
✓ Configuration loaded (max file size: 50MB)
✓ All components initialized
✓ Data directory exists
--------------------------------------------------
Phase 1 implementation is ready!
```

## Run Phase 1

```cmd
python main.py
```

## What It Does

1. **Validates** PDF files (format, size, integrity)
2. **Stores** PDFs in organized structure (`data/pdfs/YYYY/MM/`)
3. **Parses** PDFs to extract:
   - Text content
   - Metadata (title, authors, abstract)
   - Document structure (sections)
4. **Orchestrates** workflow using LangGraph
5. **Saves** parsed data as JSON (`data/parsed/`)

## Example Workflow

```
Enter PDF path: C:\path\to\research_paper.pdf

✓ Upload successful!
  Document ID: doc_a1b2c3d4-...

--- Processing Document ---
✓ Processing complete!

Parsed Data:
  Title: Attention Is All You Need
  Authors: Vaswani et al.
  Pages: 15
  Sections: 7

✓ Saved parsed data to: data/parsed/doc_a1b2c3d4-....json
```

## Troubleshooting

**Error: "PyMuPDF not installed"**
```cmd
pip install pymupdf
```

**Error: "LangGraph not installed"**
```cmd
pip install langgraph langchain
```

**Error: "File not found"**
- Provide full absolute path to PDF
- Use forward slashes or double backslashes in Windows paths

## Next Steps

After successfully running Phase 1:
1. Check `data/pdfs/` for stored PDFs
2. Check `data/parsed/` for JSON output
3. Review `data/logs/app.log` for processing logs
4. **Proceed to Phase 2** when ready
