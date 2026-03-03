# Git Setup Guide for Mini Project

## Prerequisites

1. **GitHub Account**: Create one at https://github.com if you don't have it
2. **Git Installed**: ✅ Already installed (version 2.50.1)
3. **GitHub Authentication**: You'll need either:
   - Personal Access Token (recommended for HTTPS)
   - SSH Key (for SSH connection)

---

## Option 1: Initialize New Repository (Recommended)

### Step 1: Create a New Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `research-literature-platform` (or your choice)
3. Description: "AI-powered research paper processing platform with PDF ingestion, semantic chunking, and concept extraction"
4. Choose: **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 2: Initialize Git in This Folder

Run these commands in your terminal (in the mini project folder):

```powershell
# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Phase 1 & 2 complete - PDF ingestion, parsing, chunking, and extraction"

# Add your GitHub repository as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Option 2: Connect to Existing Repository

If you already have a repository:

```powershell
# Initialize git
git init

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Add and commit files
git add .
git commit -m "Initial commit: Phase 1 & 2 complete"

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## GitHub Authentication Setup

### Method 1: Personal Access Token (HTTPS)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name: "Mini Project Access"
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)

When pushing, use:
- Username: Your GitHub username
- Password: The personal access token (NOT your GitHub password)

### Method 2: SSH Key (More Secure)

```powershell
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
type ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

Then use SSH URL: `git@github.com:YOUR_USERNAME/REPO_NAME.git`

---

## What Files Will Be Pushed?

Based on your `.gitignore`, these will be **excluded**:
- `vnv/` (virtual environment)
- `data/` (PDF files and parsed data)
- `__pycache__/` (Python cache)
- `.pyc` files

These will be **included**:
- All source code (`src/`)
- Configuration files (`config/`)
- Documentation (all `.md` files)
- `requirements.txt`
- `main.py`, test files
- `.gitignore`

---

## Quick Commands Reference

```powershell
# Check status
git status

# Add specific files
git add filename.py

# Add all changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull latest changes
git pull

# View commit history
git log --oneline

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main
```

---

## Recommended First Commit Message

```
Initial commit: Phase 1 & 2 complete

- Phase 1: PDF ingestion and parsing
  - PDF validation (format, size, integrity)
  - Organized storage (year/month structure)
  - Text extraction with metadata
  - File browser for easy upload

- Phase 2: Semantic chunking and concept extraction
  - Semantic boundary detection using sentence-transformers
  - Named entity recognition with SpaCy
  - Keyphrase extraction with KeyBERT
  - LangGraph orchestration pipeline

Tech stack: Python, LangGraph, PyMuPDF, SpaCy, KeyBERT, sentence-transformers
```

---

## Need Help?

After creating your GitHub repository, tell me:
1. Your GitHub username
2. Your repository name

And I can help you run the exact commands!
