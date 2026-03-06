# 🚀 Start Your Research Platform

Complete guide to run both backend and frontend.

## Prerequisites

✅ Python 3.12 with virtual environment activated
✅ Node.js 18+ installed
✅ Neo4j running (optional, for knowledge graph)

---

## Quick Start (2 Steps)

### Step 1: Start Backend API

```bash
# Activate virtual environment
vnv\Scripts\activate

# Start FastAPI server
python run_api.py
```

**Backend will run on:** `http://localhost:8000`
**API Docs:** `http://localhost:8000/docs`

### Step 2: Start Frontend

```bash
# Open new terminal
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**Frontend will run on:** `http://localhost:3000`

---

## Detailed Setup

### Backend Setup

1. **Activate Virtual Environment**
   ```bash
   vnv\Scripts\activate
   ```

2. **Install Dependencies** (if not already installed)
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Neo4j** (optional)
   - Open Neo4j Desktop
   - Start the `miniproject` database
   - Or skip if you don't need knowledge graph features

4. **Start API Server**
   ```bash
   python run_api.py
   ```

   You should see:
   ```
   ============================================================
   Research Literature Processing API
   ============================================================
   
   Starting FastAPI server...
   API Documentation: http://localhost:8000/docs
   Alternative Docs: http://localhost:8000/redoc
   Health Check: http://localhost:8000/health
   
   Press CTRL+C to stop the server
   ============================================================
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

### Frontend Setup

1. **Navigate to Frontend**
   ```bash
   cd frontend
   ```

2. **Install Dependencies** (first time only)
   ```bash
   npm install
   ```

   This will install:
   - React 18
   - Vite
   - Tailwind CSS
   - React Query
   - React Router
   - Axios
   - And more...

3. **Start Development Server**
   ```bash
   npm run dev
   ```

   You should see:
   ```
   VITE v5.0.8  ready in 500 ms
   
   ➜  Local:   http://localhost:3000/
   ➜  Network: use --host to expose
   ➜  press h to show help
   ```

4. **Open Browser**
   
   Navigate to `http://localhost:3000`

---

## Testing the Application

### 1. Check Backend Health

Open `http://localhost:8000/health` in browser

You should see:
```json
{
  "status": "healthy",
  "services": {
    "api": "running",
    "neo4j": "checking...",
    "chromadb": "checking..."
  }
}
```

### 2. Check API Documentation

Open `http://localhost:8000/docs`

You'll see interactive API documentation with all endpoints.

### 3. Test Frontend

Open `http://localhost:3000`

You should see:
- Dashboard with statistics
- Sidebar navigation
- Clean, modern interface

### 4. Upload a Document

1. Click "Upload" in sidebar
2. Drag & drop a PDF or click to browse
3. Click "Upload and Process"
4. Watch the processing happen in background

### 5. Search

1. Click "Search" in sidebar
2. Enter a query like "machine learning"
3. See semantic search results

### 6. Chat with AI

1. Click "AI Assistant" in sidebar
2. Ask a question about your papers
3. Get AI-powered answers with citations

---

## Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem:** Port 8000 already in use
```bash
# Solution: Kill the process or change port in run_api.py
```

**Problem:** Neo4j connection failed
```bash
# Solution: Either start Neo4j or ignore (graph features won't work)
```

### Frontend Issues

**Problem:** `npm install` fails
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Problem:** CORS errors
```bash
# Solution: Make sure backend CORS is configured for localhost:3000
# Already configured in src/api/main.py
```

**Problem:** API connection failed
```bash
# Solution: Ensure backend is running on http://localhost:8000
```

---

## Development Workflow

### Making Changes

**Backend Changes:**
1. Edit Python files in `src/`
2. Server auto-reloads (if using `--reload`)
3. Test at `http://localhost:8000/docs`

**Frontend Changes:**
1. Edit React files in `frontend/src/`
2. Vite hot-reloads automatically
3. See changes instantly in browser

### Stopping Servers

**Backend:**
- Press `CTRL+C` in terminal

**Frontend:**
- Press `CTRL+C` in terminal

---

## Production Build

### Backend

```bash
# Use production ASGI server
pip install gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend

```bash
cd frontend
npm run build
```

Output will be in `frontend/dist/`

Deploy to:
- Vercel
- Netlify
- AWS S3 + CloudFront
- Any static hosting

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Browser                         │
│              http://localhost:3000                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                        │
│              http://localhost:8000                      │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Documents │  │  Search  │  │   Graph  │            │
│  │   API    │  │   API    │  │   API    │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Data Access
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Data Layer                             │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │   PDFs   │  │  Neo4j   │  │ ChromaDB │            │
│  │  Files   │  │  Graph   │  │ Vectors  │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
```

---

## Features Available

✅ **Upload Documents** - Drag & drop PDF upload
✅ **Document Management** - View, list, delete papers
✅ **Semantic Search** - Natural language search
✅ **Knowledge Graph** - Explore paper relationships
✅ **AI Assistant** - Chat with RAG
✅ **Real-time Stats** - Dashboard with metrics
✅ **Background Processing** - Non-blocking uploads
✅ **Auto-generated Docs** - Interactive API docs

---

## Next Steps

1. **Upload some PDFs** to test the system
2. **Try semantic search** to find relevant papers
3. **Explore the knowledge graph** to see relationships
4. **Chat with AI assistant** to ask questions
5. **Check API docs** at `/docs` to understand endpoints

---

## Support

If you encounter issues:

1. Check this guide
2. Review `FASTAPI_BACKEND_SETUP.md`
3. Review `FRONTEND_SETUP.md`
4. Check console logs for errors
5. Verify all prerequisites are met

---

**You're all set! Enjoy your Research Platform! 🎉**
