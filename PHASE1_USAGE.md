# Phase 1: Usage Guide with File Browser

## ✓ Updated Feature: Graphical File Selection

Instead of typing file paths, you can now **browse and select PDF files** using a graphical dialog!

## How to Use

### 1. Run the Program

```cmd
python main.py
```

### 2. File Browser Opens Automatically

A file selection dialog will appear:
- **Title**: "Select a PDF file"
- **Filter**: Shows only PDF files by default
- **Browse**: Navigate to your PDF location
- **Select**: Click on the PDF file
- **Open**: Click "Open" button

### 3. Automatic Processing

Once you select a file:
```
✓ Selected: research_paper.pdf
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

## What Happens Behind the Scenes

1. **File Browser** opens (using tkinter)
2. You **select** a PDF file
3. **Validates** the PDF (format, size, integrity)
4. **Stores** it in `data/pdfs/YYYY/MM/`
5. **Parses** text, metadata, and sections
6. **Saves** JSON output to `data/parsed/`
7. **Logs** everything to `data/logs/app.log`

## Features

✓ **Graphical file selection** - No need to type paths
✓ **PDF filter** - Only shows PDF files in browser
✓ **Cross-platform** - Works on Windows, Mac, Linux
✓ **User-friendly** - Familiar file browser interface
✓ **Error handling** - Clear messages if something goes wrong

## Cancel Selection

If you click "Cancel" in the file browser:
```
No file selected or file not found.

=== Phase 1 Complete ===
```

## Technical Details

- Uses **tkinter** (built-in with Python)
- File dialog: `filedialog.askopenfilename()`
- Filters: `*.pdf` files
- Window stays on top for visibility

## Troubleshooting

**Issue**: File browser doesn't appear
- **Solution**: Check if tkinter is installed: `python -m tkinter`

**Issue**: "No file selected" even though I selected one
- **Solution**: Make sure you clicked "Open" not just selected the file

**Issue**: Can't find my PDF file
- **Solution**: Use "All files" filter in the dialog dropdown

## Next Steps

1. Run `python main.py`
2. Select any PDF file from the browser
3. Check output in `data/parsed/` directory
4. Review logs in `data/logs/app.log`

---

**Much easier than typing paths! Ready for Phase 2?**
