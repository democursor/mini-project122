# Phase 1: PDF Ingestion and Parsing

## Implementation Complete

This phase implements:
- PDF validation (format, size, integrity)
- PDF storage (organized by year/month)
- PDF parsing (text extraction, metadata, sections)
- LangGraph workflow orchestration

## Installation

```bash
# Activate virtual environment
vnv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Usage

The program will prompt you to upload a PDF file. Provide the full path to a PDF file on your system.

Example:
```
Enter PDF path: C:\Users\YourName\Documents\paper.pdf
```

## Output

- Uploaded PDFs stored in: `data/pdfs/YYYY/MM/`
- Parsed JSON output in: `data/parsed/`
- Logs in: `data/logs/app.log`

## File Structure

```
src/
├── ingestion/
│   ├── validator.py    # PDF validation
│   ├── storage.py      # File storage
│   └── uploader.py     # Upload handler
├── parsing/
│   ├── models.py       # Data models
│   └── parser.py       # PDF parsing
├── orchestration/
│   └── workflow.py     # LangGraph workflow
└── utils/
    ├── config.py       # Configuration
    └── logging_config.py
```

## Next Steps

After testing Phase 1, proceed to Phase 2 for semantic chunking.
