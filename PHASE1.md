# Phase 1: PDF Ingestion and Parsing

## Overview

Phase 1 establishes the foundation of the research literature platform by implementing the document ingestion and parsing pipeline. This phase transforms raw PDF files into structured, machine-readable data that downstream components can process.

**Learning Objectives:**
- Understand PDF structure and text extraction challenges
- Learn file validation and security best practices
- Master error handling for file processing
- Design robust storage and metadata management systems

**Key Concepts:**
- PDF document structure (pages, fonts, text positioning)
- File validation (format, size, integrity)
- Metadata extraction (title, authors, dates)
- Structured data representation (JSON schemas)

---

## Table of Contents

1. [PDF Ingestion Module](#pdf-ingestion-module)
2. [PDF Parsing Module](#pdf-parsing-module)
3. [Learning Outcomes](#learning-outcomes)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Success Criteria](#success-criteria)

---

## PDF Ingestion Module

### Purpose

The PDF Ingestion Module is the entry point for all documents entering the system. It handles file upload, validation, storage, and metadata creation.

**Responsibilities:**
- Accept PDF files from users
- Validate file format, size, and integrity
- Store files securely with unique identifiers
- Create initial metadata records
- Trigger downstream processing

### 6.1 File Upload Validation Strategy

#### Validation Rules

**1. File Format Validation**


**Purpose:** Ensure only valid PDF files are accepted

**Implementation Strategy:**
```python
def validate_pdf_format(file_path: Path) -> bool:
    """
    Validate that the file is a genuine PDF.
    
    Checks:
    1. File extension is .pdf
    2. File magic number (header) is %PDF-
    3. File can be opened by PyMuPDF
    """
    # Check extension
    if not file_path.suffix.lower() == '.pdf':
        raise ValidationError("File must have .pdf extension")
    
    # Check magic number (first 4 bytes should be %PDF)
    with open(file_path, 'rb') as f:
        header = f.read(4)
        if header != b'%PDF':
            raise ValidationError("File is not a valid PDF (invalid header)")
    
    # Try opening with PyMuPDF
    try:
        import fitz
        doc = fitz.open(file_path)
        doc.close()
    except Exception as e:
        raise ValidationError(f"Cannot open PDF: {str(e)}")
    
    return True
```

**Why This Matters:**
- Prevents malicious files disguised as PDFs
- Catches corrupted files early
- Protects downstream components from invalid input

**Error Messages:**
- `"File must have .pdf extension"` → User uploaded wrong file type
- `"File is not a valid PDF (invalid header)"` → File renamed but not actually PDF
- `"Cannot open PDF: [details]"` → Corrupted or encrypted PDF

---

**2. File Size Validation**

**Purpose:** Prevent resource exhaustion from extremely large files

**Implementation Strategy:**
```python
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def validate_file_size(file_path: Path) -> bool:
    """
    Validate file size is within acceptable limits.
    
    Limit: 50MB (configurable)
    Rationale: Most research papers are 1-10MB
    """
    file_size = file_path.stat().st_size
    
    if file_size == 0:
        raise ValidationError("File is empty")
    
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ValidationError(
            f"File too large ({size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB"
        )
    
    return True
```

**Why 50MB Limit:**
- Typical research paper: 1-10MB
- Papers with many figures: 10-30MB
- 50MB covers 99% of legitimate papers
- Prevents memory issues during processing

**Error Messages:**
- `"File is empty"` → Upload failed or file corrupted
- `"File too large (75.3MB). Maximum size is 50MB"` → Clear, actionable message

---

**3. File Integrity Validation**

**Purpose:** Detect corrupted or incomplete uploads

**Implementation Strategy:**
```python
def validate_file_integrity(file_path: Path) -> bool:
    """
    Validate PDF structure and integrity.
    
    Checks:
    1. PDF has valid structure (can read pages)
    2. PDF is not encrypted
    3. PDF contains extractable content
    """
    import fitz
    
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
        
        # Try to read first page (catches structural issues)
        first_page = doc[0]
        text = first_page.get_text()
        
        doc.close()
        return True
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"PDF integrity check failed: {str(e)}")
```

**Why This Matters:**
- Encrypted PDFs cannot be processed
- Corrupted PDFs cause downstream failures
- Early detection saves processing time

**Error Messages:**
- `"PDF is password-protected. Please upload an unencrypted version."` → Clear action
- `"PDF has no pages"` → Structural corruption
- `"PDF integrity check failed: [details]"` → Specific technical issue

---

#### Complete Validation Pipeline

```python
class PDFValidator:
    """Validates uploaded PDF files"""
    
    def __init__(self, config: Config):
        self.max_file_size = config.get('storage.max_file_size_mb', 50)
    
    def validate(self, file_path: Path) -> ValidationResult:
        """
        Run all validation checks.
        
        Returns ValidationResult with:
        - is_valid: bool
        - errors: List[str]
        - warnings: List[str]
        """
        result = ValidationResult()
        
        try:
            # 1. Format validation
            self.validate_format(file_path)
            
            # 2. Size validation
            self.validate_size(file_path)
            
            # 3. Integrity validation
            self.validate_integrity(file_path)
            
            result.is_valid = True
            
        except ValidationError as e:
            result.is_valid = False
            result.errors.append(str(e))
        
        return result
```

---

#### Security Considerations

**1. Path Traversal Prevention**
```python
def sanitize_filename(filename: str) -> str:
    """
    Remove dangerous characters from filename.
    
    Prevents: ../../../etc/passwd
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Remove null bytes
    filename = filename.replace('\x00', '')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename
```

**2. Temporary File Handling**
```python
def handle_upload(uploaded_file) -> Path:
    """
    Safely handle uploaded file.
    
    Steps:
    1. Save to temporary location
    2. Validate
    3. Move to permanent storage
    4. Clean up temp file on failure
    """
    import tempfile
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.read())
        temp_path = Path(tmp.name)
    
    try:
        # Validate
        validator = PDFValidator(config)
        result = validator.validate(temp_path)
        
        if not result.is_valid:
            raise ValidationError(result.errors[0])
        
        # Move to permanent storage
        final_path = move_to_storage(temp_path)
        return final_path
        
    except Exception as e:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
        raise
```

---

### 6.2 File Storage Strategy

#### Storage Organization

**Directory Structure:**
```
data/pdfs/
├── 2024/
│   ├── 01/
│   │   ├── doc_abc123.pdf
│   │   └── doc_def456.pdf
│   ├── 02/
│   └── ...
├── 2025/
└── ...
```

**Why This Structure:**
- **Year/Month folders:** Easy to manage and backup
- **Unique IDs:** Prevent filename conflicts
- **Scalability:** Distributes files across directories (filesystem performance)



#### File Naming Convention

```python
import uuid
from datetime import datetime

def generate_document_id() -> str:
    """
    Generate unique document identifier.
    
    Format: doc_<uuid4>
    Example: doc_a3f2b1c4-5d6e-7f8g-9h0i-1j2k3l4m5n6o
    
    Why UUID4:
    - Globally unique (collision probability: ~0)
    - No sequential information (security)
    - URL-safe
    """
    return f"doc_{uuid.uuid4()}"

def get_storage_path(document_id: str, upload_date: datetime) -> Path:
    """
    Determine storage path for document.
    
    Args:
        document_id: Unique document identifier
        upload_date: When document was uploaded
    
    Returns:
        Path: data/pdfs/YYYY/MM/doc_id.pdf
    """
    year = upload_date.strftime('%Y')
    month = upload_date.strftime('%m')
    
    storage_dir = Path('data/pdfs') / year / month
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    return storage_dir / f"{document_id}.pdf"
```

**Benefits:**
- **Unique IDs:** No filename conflicts
- **Chronological organization:** Easy to find recent uploads
- **Scalability:** Thousands of files per directory (not millions)

---

#### Storage Implementation

```python
class PDFStorage:
    """Manages PDF file storage"""
    
    def __init__(self, config: Config):
        self.base_dir = Path(config.get('storage.pdf_directory', './data/pdfs'))
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def store(self, source_path: Path, document_id: str) -> Path:
        """
        Store PDF file in organized structure.
        
        Args:
            source_path: Temporary file location
            document_id: Unique document identifier
        
        Returns:
            Path: Final storage location
        """
        # Determine storage path
        upload_date = datetime.now()
        dest_path = get_storage_path(document_id, upload_date)
        
        # Move file
        shutil.move(str(source_path), str(dest_path))
        
        # Verify file exists
        if not dest_path.exists():
            raise StorageError(f"Failed to store file at {dest_path}")
        
        return dest_path
    
    def retrieve(self, document_id: str) -> Path:
        """
        Retrieve PDF file by document ID.
        
        Searches all year/month directories.
        """
        # Search pattern: data/pdfs/*/*/doc_id.pdf
        pattern = self.base_dir / '*' / '*' / f"{document_id}.pdf"
        matches = list(self.base_dir.glob(f'*/*/{document_id}.pdf'))
        
        if not matches:
            raise FileNotFoundError(f"Document {document_id} not found")
        
        return matches[0]
    
    def delete(self, document_id: str) -> bool:
        """Delete PDF file"""
        try:
            file_path = self.retrieve(document_id)
            file_path.unlink()
            return True
        except FileNotFoundError:
            return False
```

---

#### Backup and Recovery Strategy

**Backup Approach:**
```python
def backup_pdfs(backup_dir: Path):
    """
    Create incremental backup of PDF storage.
    
    Strategy:
    1. Daily incremental backups (only new/modified files)
    2. Weekly full backups
    3. Retain backups for 30 days
    """
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f"pdfs_backup_{timestamp}"
    
    # Copy entire pdfs directory
    shutil.copytree('data/pdfs', backup_path)
    
    print(f"Backup created: {backup_path}")
```

**Recovery Approach:**
```python
def restore_from_backup(backup_path: Path):
    """
    Restore PDFs from backup.
    
    Steps:
    1. Verify backup integrity
    2. Stop processing pipeline
    3. Restore files
    4. Verify restoration
    5. Resume processing
    """
    if not backup_path.exists():
        raise BackupError(f"Backup not found: {backup_path}")
    
    # Create temporary backup of current state
    temp_backup = Path('data/pdfs_temp_backup')
    shutil.copytree('data/pdfs', temp_backup)
    
    try:
        # Remove current files
        shutil.rmtree('data/pdfs')
        
        # Restore from backup
        shutil.copytree(backup_path, 'data/pdfs')
        
        print("Restoration successful")
        
    except Exception as e:
        # Restore from temp backup
        shutil.copytree(temp_backup, 'data/pdfs')
        raise BackupError(f"Restoration failed: {e}")
    
    finally:
        # Clean up temp backup
        shutil.rmtree(temp_backup, ignore_errors=True)
```

---

### 6.3 Concurrent Upload Handling

#### Challenge

Multiple users uploading PDFs simultaneously requires:
- **Isolation:** Each upload processed independently
- **Resource management:** Limit concurrent processing
- **Thread safety:** Prevent race conditions

#### Implementation Strategy

**1. Upload Queue**
```python
import asyncio
from asyncio import Queue, Semaphore

class UploadManager:
    """Manages concurrent PDF uploads"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        self.upload_queue = Queue()
    
    async def upload(self, file_path: Path, user_id: str) -> str:
        """
        Handle PDF upload with concurrency control.
        
        Args:
            file_path: Path to uploaded file
            user_id: User identifier
        
        Returns:
            document_id: Unique document identifier
        """
        async with self.semaphore:
            # Only max_concurrent uploads processed simultaneously
            return await self._process_upload(file_path, user_id)
    
    async def _process_upload(self, file_path: Path, user_id: str) -> str:
        """Process single upload"""
        # Generate unique ID
        document_id = generate_document_id()
        
        # Validate
        validator = PDFValidator(config)
        result = validator.validate(file_path)
        
        if not result.is_valid:
            raise ValidationError(result.errors[0])
        
        # Store
        storage = PDFStorage(config)
        final_path = storage.store(file_path, document_id)
        
        # Create metadata
        await self._create_metadata(document_id, final_path, user_id)
        
        # Trigger processing
        await self._trigger_processing(document_id)
        
        return document_id
```

**2. Isolation Between Uploads**
```python
class UploadContext:
    """Isolated context for each upload"""
    
    def __init__(self, document_id: str):
        self.document_id = document_id
        self.temp_dir = Path(f'temp/{document_id}')
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir, ignore_errors=True)

# Usage
with UploadContext(document_id) as ctx:
    # All operations isolated to this context
    temp_file = ctx.temp_dir / 'processing.pdf'
    # ... process file ...
# Automatic cleanup
```

**3. Resource Management**
```python
class ResourceManager:
    """Manages system resources during uploads"""
    
    def __init__(self):
        self.active_uploads = {}
        self.lock = asyncio.Lock()
    
    async def acquire(self, document_id: str) -> bool:
        """Acquire resources for upload"""
        async with self.lock:
            if document_id in self.active_uploads:
                return False  # Already processing
            
            self.active_uploads[document_id] = {
                'start_time': datetime.now(),
                'status': 'uploading'
            }
            return True
    
    async def release(self, document_id: str):
        """Release resources"""
        async with self.lock:
            if document_id in self.active_uploads:
                del self.active_uploads[document_id]
```

---

## PDF Parsing Module

### Purpose

The PDF Parsing Module extracts text, metadata, and structure from PDF documents, transforming binary PDFs into structured, machine-readable data.

**Responsibilities:**
- Extract all readable text from PDFs
- Identify document structure (sections, paragraphs)
- Extract metadata (title, authors, dates)
- Handle complex layouts (multi-column, tables, images)
- Output structured JSON

### 7.1 Text Extraction Approach

#### PyMuPDF Capabilities

**Why PyMuPDF (fitz)?**
- Fast C-based implementation
- Accurate text extraction with positioning
- Handles complex PDFs (multi-column, rotated text)
- Extracts images and embedded content
- Active development and support



#### Basic Text Extraction

```python
import fitz  # PyMuPDF

class PDFParser:
    """Extracts text and structure from PDF documents"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse PDF and extract all content.
        
        Returns:
            ParsedDocument with text, metadata, and structure
        """
        doc = fitz.open(pdf_path)
        
        try:
            # Extract metadata
            metadata = self._extract_metadata(doc)
            
            # Extract text from all pages
            pages = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_data = self._extract_page(page, page_num)
                pages.append(page_data)
            
            # Identify structure
            structure = self._analyze_structure(pages)
            
            # Combine into document
            parsed_doc = ParsedDocument(
                metadata=metadata,
                pages=pages,
                structure=structure,
                page_count=doc.page_count
            )
            
            return parsed_doc
            
        finally:
            doc.close()
    
    def _extract_page(self, page, page_num: int) -> PageData:
        """Extract text from single page"""
        # Get text with layout preservation
        text = page.get_text("text")
        
        # Get text blocks with positioning
        blocks = page.get_text("blocks")
        
        # Get text with detailed positioning
        words = page.get_text("words")
        
        return PageData(
            page_number=page_num,
            text=text,
            blocks=blocks,
            words=words
        )
```

---

#### Handling Multi-Column Layouts

**Challenge:** Research papers often use 2-column layouts. Simple text extraction reads left-to-right across both columns, mixing content.

**Solution:** Use text positioning to identify columns

```python
def extract_columns(page) -> List[str]:
    """
    Extract text respecting column layout.
    
    Strategy:
    1. Get all text blocks with positions
    2. Identify column boundaries
    3. Sort blocks by column, then by vertical position
    4. Extract text in correct reading order
    """
    blocks = page.get_text("blocks")
    
    if not blocks:
        return []
    
    # Get page dimensions
    page_width = page.rect.width
    
    # Identify columns by x-position clustering
    x_positions = [block[0] for block in blocks]  # x0 coordinate
    
    # Simple heuristic: if blocks cluster around 2 x-positions, it's 2-column
    left_blocks = [b for b in blocks if b[0] < page_width / 2]
    right_blocks = [b for b in blocks if b[0] >= page_width / 2]
    
    # Sort each column by y-position (top to bottom)
    left_blocks.sort(key=lambda b: b[1])  # y0 coordinate
    right_blocks.sort(key=lambda b: b[1])
    
    # Extract text from left column, then right column
    columns = []
    
    if left_blocks:
        left_text = '\n'.join(block[4] for block in left_blocks)
        columns.append(left_text)
    
    if right_blocks:
        right_text = '\n'.join(block[4] for block in right_blocks)
        columns.append(right_text)
    
    return columns
```

**Why This Matters:**
- Preserves reading order
- Prevents mixed content from different columns
- Improves downstream NLP processing

---

#### Structure Preservation

**Goal:** Identify document sections (Abstract, Introduction, Methods, etc.)

```python
def identify_sections(pages: List[PageData]) -> List[Section]:
    """
    Identify document sections using heuristics.
    
    Heuristics:
    1. Section headings are often bold or larger font
    2. Common section names: Abstract, Introduction, Methods, Results, etc.
    3. Headings usually have whitespace before/after
    """
    sections = []
    current_section = None
    
    # Common section headings in research papers
    section_keywords = [
        'abstract', 'introduction', 'background', 'related work',
        'methods', 'methodology', 'approach', 'experiments',
        'results', 'discussion', 'conclusion', 'references'
    ]
    
    for page in pages:
        blocks = page.blocks
        
        for block in blocks:
            text = block[4].strip()
            text_lower = text.lower()
            
            # Check if this looks like a section heading
            is_heading = False
            for keyword in section_keywords:
                if text_lower == keyword or text_lower.startswith(keyword):
                    is_heading = True
                    break
            
            if is_heading:
                # Start new section
                if current_section:
                    sections.append(current_section)
                
                current_section = Section(
                    heading=text,
                    content=[],
                    start_page=page.page_number
                )
            elif current_section:
                # Add to current section
                current_section.content.append(text)
    
    # Add final section
    if current_section:
        sections.append(current_section)
    
    return sections
```

---

### 7.2 Metadata Extraction Strategy

#### Document Metadata

**Goal:** Extract title, authors, publication date, venue, abstract

**Challenge:** Metadata location varies by paper format

**Strategy:** Use multiple approaches and combine results

```python
class MetadataExtractor:
    """Extracts metadata from PDF documents"""
    
    def extract(self, doc: fitz.Document, pages: List[PageData]) -> Metadata:
        """
        Extract metadata using multiple strategies.
        
        Strategies:
        1. PDF metadata fields (if present)
        2. First page text analysis
        3. Pattern matching (regex)
        """
        metadata = Metadata()
        
        # Strategy 1: PDF metadata
        pdf_meta = doc.metadata
        if pdf_meta:
            metadata.title = pdf_meta.get('title', '')
            metadata.author = pdf_meta.get('author', '')
            metadata.creation_date = pdf_meta.get('creationDate', '')
        
        # Strategy 2: First page analysis
        if pages:
            first_page_meta = self._extract_from_first_page(pages[0])
            metadata.merge(first_page_meta)
        
        # Strategy 3: Pattern matching
        pattern_meta = self._extract_with_patterns(pages)
        metadata.merge(pattern_meta)
        
        return metadata
```

---

#### Title Extraction

```python
def extract_title(first_page: PageData) -> str:
    """
    Extract paper title from first page.
    
    Heuristics:
    1. Title is usually the largest text on first page
    2. Title is at the top of the page
    3. Title is often bold or different font
    """
    blocks = first_page.blocks
    
    if not blocks:
        return ""
    
    # Get blocks from top 30% of page
    page_height = max(block[3] for block in blocks)  # y1 coordinate
    top_blocks = [b for b in blocks if b[1] < page_height * 0.3]
    
    if not top_blocks:
        return ""
    
    # Find largest text block (likely title)
    # Font size is not directly available, use block height as proxy
    largest_block = max(top_blocks, key=lambda b: b[3] - b[1])
    
    title = largest_block[4].strip()
    return title
```

---

#### Author Extraction

```python
import re

def extract_authors(first_page: PageData) -> List[str]:
    """
    Extract author names from first page.
    
    Patterns:
    1. Names often appear after title
    2. Multiple authors separated by commas or "and"
    3. May include affiliations (numbers or symbols)
    """
    text = first_page.text
    
    # Pattern: Name1, Name2, and Name3
    # Pattern: Name1 · Name2 · Name3
    # Pattern: Name1, Name2, Name3
    
    # Simple approach: Look for capitalized words after title
    lines = text.split('\n')
    
    authors = []
    in_author_section = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if line looks like author names
        # Heuristic: Capitalized words, possibly with commas
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', line):
            # Extract names
            # Remove affiliations (numbers, symbols)
            clean_line = re.sub(r'[0-9†‡§¶*]+', '', line)
            
            # Split by common separators
            names = re.split(r',\s*|\s+and\s+|·', clean_line)
            authors.extend([n.strip() for n in names if n.strip()])
    
    return authors[:10]  # Limit to reasonable number
```

---

#### Abstract Extraction

```python
def extract_abstract(pages: List[PageData]) -> str:
    """
    Extract abstract from document.
    
    Strategy:
    1. Look for "Abstract" heading
    2. Extract text until next section
    """
    for page in pages[:3]:  # Abstract usually on first 3 pages
        text = page.text
        
        # Find "Abstract" heading
        match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
        
        if match:
            # Extract text after "Abstract"
            start_pos = match.end()
            
            # Find next section (Introduction, etc.)
            next_section = re.search(
                r'\b(Introduction|Background|1\.)\b',
                text[start_pos:],
                re.IGNORECASE
            )
            
            if next_section:
                end_pos = start_pos + next_section.start()
                abstract = text[start_pos:end_pos]
            else:
                # Take next 500 characters
                abstract = text[start_pos:start_pos + 500]
            
            return abstract.strip()
    
    return ""
```

---

### 7.3 Handling Complex PDF Elements

#### Images and Figures

```python
def extract_images(page) -> List[ImageData]:
    """
    Extract images from PDF page.
    
    Returns:
        List of images with position and metadata
    """
    images = []
    image_list = page.get_images()
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        
        # Get image data
        base_image = page.parent.extract_image(xref)
        
        image_data = ImageData(
            xref=xref,
            width=base_image['width'],
            height=base_image['height'],
            format=base_image['ext'],
            size=len(base_image['image'])
        )
        
        images.append(image_data)
    
    return images
```



#### Tables

```python
def detect_tables(page) -> List[TableData]:
    """
    Detect tables in PDF page.
    
    Strategy:
    1. Look for grid-like structures (lines)
    2. Identify cell boundaries
    3. Extract text from cells
    
    Note: Table extraction is complex. For learning purposes,
    we'll use simple detection. Production systems use libraries
    like Camelot or Tabula.
    """
    tables = []
    
    # Get page drawings (lines, rectangles)
    drawings = page.get_drawings()
    
    # Simple heuristic: Many horizontal/vertical lines = table
    horizontal_lines = [d for d in drawings if d['type'] == 'l' and abs(d['y1'] - d['y0']) < 2]
    vertical_lines = [d for d in drawings if d['type'] == 'l' and abs(d['x1'] - d['x0']) < 2]
    
    if len(horizontal_lines) > 3 and len(vertical_lines) > 2:
        # Likely a table
        table_data = TableData(
            page_number=page.number,
            has_table=True,
            row_count=len(horizontal_lines) - 1,
            col_count=len(vertical_lines) - 1
        )
        tables.append(table_data)
    
    return tables
```

**Limitations:**
- Table extraction is complex
- Different table formats (with/without borders)
- Merged cells, nested tables
- For production: Use specialized libraries (Camelot, Tabula, pdfplumber)

---

#### Embedded Text in Images (OCR)

```python
def extract_text_from_images(page) -> str:
    """
    Extract text from images using OCR.
    
    Use case: Scanned PDFs, figures with text
    
    Note: Requires Tesseract OCR installation
    """
    try:
        from PIL import Image
        import pytesseract
        import io
        
        images = page.get_images()
        extracted_text = []
        
        for img in images:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            
            # Convert to PIL Image
            image_bytes = base_image['image']
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run OCR
            text = pytesseract.image_to_string(image)
            
            if text.strip():
                extracted_text.append(text)
        
        return '\n'.join(extracted_text)
        
    except ImportError:
        # Tesseract not installed
        return ""
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""
```

**When to Use OCR:**
- Scanned PDFs (no extractable text)
- Figures with important text
- Screenshots in papers

**Trade-offs:**
- ✅ Extracts text from images
- ❌ Slow (seconds per image)
- ❌ Requires Tesseract installation
- ❌ Accuracy varies with image quality

---

### 7.4 Parser Error Handling

#### Error Categories

**1. Corrupted PDF**
```python
class PDFCorruptedError(Exception):
    """PDF file is corrupted or malformed"""
    pass

def handle_corrupted_pdf(pdf_path: Path, error: Exception):
    """
    Handle corrupted PDF gracefully.
    
    Actions:
    1. Log detailed error
    2. Mark document as failed
    3. Notify user with specific issue
    4. Suggest solutions
    """
    logger.error(f"Corrupted PDF: {pdf_path}", exc_info=True)
    
    # Update database
    db.update_document_status(
        document_id=get_document_id(pdf_path),
        status='failed',
        error_message=f"PDF is corrupted: {str(error)}"
    )
    
    # Notify user
    return {
        'success': False,
        'error': 'PDF file is corrupted or malformed',
        'suggestion': 'Try re-downloading the PDF or use a different version',
        'technical_details': str(error)
    }
```

---

**2. Extraction Failure**
```python
class ExtractionError(Exception):
    """Failed to extract text from PDF"""
    pass

def handle_extraction_failure(pdf_path: Path, page_num: int, error: Exception):
    """
    Handle text extraction failure.
    
    Strategy:
    1. Try alternative extraction method
    2. Skip problematic page if necessary
    3. Continue with remaining pages
    """
    logger.warning(f"Extraction failed for page {page_num}: {error}")
    
    # Try alternative method
    try:
        # Method 1: "text" mode (default)
        # Method 2: "blocks" mode
        # Method 3: "dict" mode (detailed)
        page = fitz.open(pdf_path)[page_num]
        text = page.get_text("dict")
        return text
    except Exception as e:
        logger.error(f"All extraction methods failed for page {page_num}")
        return None  # Skip this page
```

---

**3. Metadata Extraction Failure**
```python
def handle_metadata_failure(pdf_path: Path, error: Exception):
    """
    Handle metadata extraction failure.
    
    Strategy:
    1. Log warning (not critical error)
    2. Continue with partial metadata
    3. Mark fields as "Unknown"
    """
    logger.warning(f"Metadata extraction failed: {error}")
    
    # Return minimal metadata
    return Metadata(
        title="Unknown",
        authors=["Unknown"],
        year=None,
        abstract="",
        extraction_status="partial"
    )
```

---

#### Retry Mechanisms

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class PDFParserWithRetry:
    """PDF Parser with automatic retry logic"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def parse_with_retry(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse PDF with automatic retry on failure.
        
        Retry strategy:
        - Attempt 1: Immediate
        - Attempt 2: Wait 2 seconds
        - Attempt 3: Wait 4 seconds
        - After 3 failures: Give up
        """
        try:
            return self.parse(pdf_path)
        except Exception as e:
            logger.warning(f"Parse attempt failed: {e}")
            raise  # Trigger retry
```

---

#### Fallback Approaches

```python
def parse_with_fallback(pdf_path: Path) -> ParsedDocument:
    """
    Parse PDF with multiple fallback strategies.
    
    Strategy hierarchy:
    1. PyMuPDF (primary)
    2. PDFMiner (fallback 1)
    3. PyPDF2 (fallback 2)
    4. OCR (last resort)
    """
    # Try PyMuPDF
    try:
        return parse_with_pymupdf(pdf_path)
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")
    
    # Try PDFMiner
    try:
        return parse_with_pdfminer(pdf_path)
    except Exception as e:
        logger.warning(f"PDFMiner failed: {e}")
    
    # Try PyPDF2
    try:
        return parse_with_pypdf2(pdf_path)
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")
    
    # Last resort: OCR
    try:
        return parse_with_ocr(pdf_path)
    except Exception as e:
        logger.error(f"All parsing methods failed: {e}")
        raise ParsingError("Unable to parse PDF with any method")
```

---

### 7.5 Structured Output Format

#### JSON Schema

```python
from dataclasses import dataclass, asdict
from typing import List, Optional
import json

@dataclass
class Metadata:
    """Document metadata"""
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    abstract: str
    doi: Optional[str] = None
    keywords: List[str] = None

@dataclass
class Section:
    """Document section"""
    heading: str
    content: str
    start_page: int
    end_page: int

@dataclass
class ParsedDocument:
    """Complete parsed document"""
    document_id: str
    metadata: Metadata
    sections: List[Section]
    full_text: str
    page_count: int
    has_images: bool
    has_tables: bool
    parsing_date: str
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)
    
    def save(self, output_path: Path):
        """Save to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
```

---

#### Example Output

```json
{
  "document_id": "doc_a3f2b1c4-5d6e-7f8g-9h0i-1j2k3l4m5n6o",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": [
      "Ashish Vaswani",
      "Noam Shazeer",
      "Niki Parmar"
    ],
    "year": 2017,
    "venue": "NeurIPS",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
    "doi": "10.48550/arXiv.1706.03762",
    "keywords": ["transformers", "attention", "neural networks"]
  },
  "sections": [
    {
      "heading": "Abstract",
      "content": "The dominant sequence transduction models...",
      "start_page": 0,
      "end_page": 0
    },
    {
      "heading": "Introduction",
      "content": "Recurrent neural networks, long short-term memory...",
      "start_page": 0,
      "end_page": 1
    }
  ],
  "full_text": "Attention Is All You Need\n\nAshish Vaswani...",
  "page_count": 15,
  "has_images": true,
  "has_tables": true,
  "parsing_date": "2024-01-15T10:30:00Z"
}
```

---

## Learning Outcomes

### Skills Learned in Phase 1

**1. File Processing**
- PDF structure and format
- Binary file handling
- File validation techniques
- Secure file storage

**2. Text Extraction**
- PyMuPDF library usage
- Handling complex layouts
- Structure preservation
- Metadata extraction

**3. Error Handling**
- Graceful degradation
- Retry mechanisms
- Fallback strategies
- User-friendly error messages

**4. Data Modeling**
- JSON schema design
- Structured data representation
- Dataclass usage in Python

**5. Asynchronous Programming**
- Concurrent file processing
- Resource management
- Queue-based processing

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: "Cannot open PDF"**
- **Cause:** Corrupted file, encrypted PDF, or unsupported format
- **Solution:** Validate file integrity, check for encryption, try alternative parser

**Issue 2: "No text extracted"**
- **Cause:** Scanned PDF (images only), no extractable text layer
- **Solution:** Use OCR (Tesseract), or inform user that PDF is image-based

**Issue 3: "Garbled text"**
- **Cause:** Encoding issues, custom fonts, or complex formatting
- **Solution:** Try different extraction modes, check font embedding

**Issue 4: "Missing metadata"**
- **Cause:** PDF doesn't contain metadata fields, non-standard format
- **Solution:** Use text-based extraction, pattern matching, or mark as "Unknown"

**Issue 5: "Slow processing"**
- **Cause:** Large PDF, many images, complex layout
- **Solution:** Implement pagination, process pages in parallel, optimize extraction

---

## Success Criteria

Phase 1 is successful when:

✅ **Upload and Validation**
- PDFs can be uploaded successfully
- Invalid files are rejected with clear messages
- Files are stored securely with unique IDs

✅ **Text Extraction**
- Text is extracted from all pages
- Multi-column layouts are handled correctly
- Document structure is preserved

✅ **Metadata Extraction**
- Title, authors, and abstract are extracted (when available)
- Missing metadata is handled gracefully

✅ **Error Handling**
- Corrupted PDFs are detected and reported
- Extraction failures don't crash the system
- Users receive actionable error messages

✅ **Output Quality**
- Structured JSON output is generated
- All required fields are populated
- Output is valid and well-formed

---

## Next Steps

After completing Phase 1, you'll have:
- A working PDF ingestion pipeline
- Structured document data ready for processing
- Understanding of file handling and text extraction

**Phase 2** will build on this foundation by:
- Segmenting documents into semantic chunks
- Extracting concepts and entities using NLP
- Preparing data for knowledge graph construction

---

**Phase 1 demonstrates fundamental skills in file processing, text extraction, and error handling - essential for any document processing system.**

