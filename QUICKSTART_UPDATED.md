# Phase 1 - Quick Start (Updated with File Browser)

## Run Phase 1

```cmd
python main.py
```

## What You'll See

### Step 1: Program Starts
```
--- PDF Upload ---
Opening file browser...
(Select a PDF file from the dialog)
```

### Step 2: File Browser Opens
A window appears where you can:
- Browse your computer
- Select a PDF file
- Click "Open"

### Step 3: Processing Begins
```
✓ Selected: my_research_paper.pdf
✓ Upload successful!
  Document ID: doc_12345678-...

--- Processing Document ---
```

### Step 4: Results Displayed
```
✓ Processing complete!

Parsed Data:
  Title: Your Paper Title
  Authors: Author Names
  Pages: 10
  Sections: 5

✓ Saved parsed data to: data/parsed/doc_12345678-....json
```

## No More Typing Paths!

**Before** (old way):
```
Enter PDF path: C:\Users\YourName\Documents\long\path\to\file.pdf
```

**Now** (new way):
- File browser opens automatically
- Click and select your PDF
- Done!

## Features

✓ **Easy file selection** - Graphical browser
✓ **PDF filtering** - Only shows PDF files
✓ **Validation** - Checks format, size, integrity
✓ **Parsing** - Extracts text, metadata, sections
✓ **JSON output** - Structured data saved
✓ **Logging** - Everything tracked

## Output Locations

- **Stored PDFs**: `data/pdfs/YYYY/MM/doc_<id>.pdf`
- **Parsed JSON**: `data/parsed/doc_<id>.json`
- **Logs**: `data/logs/app.log`

## Try It Now!

```cmd
python main.py
```

Then select any PDF file from your computer!

---

**Phase 1 is ready with easy file browsing!**
