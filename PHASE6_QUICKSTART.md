# Phase 6 Quick Start Guide

## 🚀 Launch the Web App

```cmd
streamlit run web_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📖 First Time Setup

### 1. Check System Status
- Go to **⚙️ Settings** page
- Verify all statistics show correct counts
- Check configuration is correct

### 2. Upload a Paper (if needed)
- Go to **📤 Upload** page
- Click "Choose a PDF file"
- Select a research paper
- Click "🚀 Process Document"
- Wait for processing (10-15 seconds)

### 3. Try the Chat
- Go to **💬 Chat** page
- Ask a question like:
  - "What is causal inference?"
  - "What methods are discussed?"
  - "Summarize the main findings"
- View the AI response with sources and citations

### 4. Search Documents
- Go to **🔍 Search** page
- Enter a search query
- Adjust number of results
- Click "🔍 Search"
- View results with relevance scores

### 5. Browse Library
- Go to **📚 Library** page
- View all uploaded documents
- See metadata (title, authors, year)

## 💡 Tips

### Chat Tips
- Ask specific questions for better answers
- Use follow-up questions for deeper insights
- Click "🗑️ Clear Chat" to start fresh
- Expand "📚 Sources" to see where answers come from

### Upload Tips
- Only PDF files are supported
- Max file size: 50MB
- Processing takes 10-15 seconds per paper
- You'll see progress through all 4 phases

### Search Tips
- Use natural language queries
- Semantic search finds meaning, not just keywords
- Adjust top-k slider for more/fewer results
- Higher scores = more relevant

## 🎯 Example Workflow

### Research a Topic

1. **Upload papers** on your topic
2. **Search** to find relevant sections
3. **Chat** to ask specific questions
4. **Review sources** to verify information
5. **Continue conversation** for deeper understanding

### Example Session

```
📤 Upload: "causal_inference_paper.pdf"
   ✅ Processed successfully!

🔍 Search: "statistical methods"
   Found 5 results

💬 Chat: "What statistical methods are discussed?"
   🤖 The papers discuss several methods including...
   📚 3 sources cited
   📝 2 citations (100% accuracy)

💬 Follow-up: "Can you explain the first method in detail?"
   🤖 The first method, regression analysis, is...
```

## ⚡ Keyboard Shortcuts

- **Ctrl+K** - Focus search (in browser)
- **Enter** - Send chat message
- **Esc** - Close modals/expanders

## 🔧 Troubleshooting

### App Won't Start
```cmd
# Check if Streamlit is installed
pip install streamlit

# Try running again
streamlit run web_app.py
```

### No Documents Showing
- Upload papers using the Upload page
- Check that `data/parsed/` directory has JSON files
- Verify processing completed successfully

### Chat Not Working
- Check API key is set in `.env` file
- Verify Google Gemini API key is valid
- Check Settings page for configuration

### Search Returns No Results
- Ensure documents are uploaded and processed
- Check that ChromaDB has vectors (Settings page)
- Try different search queries

## 📊 Understanding the Interface

### Chat Page
- **User messages** - Your questions (right side)
- **Assistant messages** - AI responses (left side)
- **Sources** - Expandable section showing source documents
- **Citations** - Expandable section showing citation validation

### Upload Page
- **File uploader** - Drag & drop or click to select
- **Progress bar** - Shows processing progress
- **Status messages** - Success/error notifications

### Library Page
- **Document cards** - Each paper shown as a card
- **Metadata** - Title, authors, year, document ID
- **Total count** - Number of documents at top

### Search Page
- **Search box** - Enter your query
- **Top-K slider** - Number of results (1-20)
- **Result cards** - Each result with score and preview

### Settings Page
- **Statistics** - 4 metric cards at top
- **Configuration** - Current settings display
- **Actions** - Buttons for reload, clear cache, view logs

## 🎨 UI Elements

### Status Messages
- ✅ **Green** - Success
- ❌ **Red** - Error
- ⚠️ **Orange** - Warning
- ℹ️ **Blue** - Information

### Icons
- 💬 Chat
- 📤 Upload
- 📚 Library
- 🔍 Search
- ⚙️ Settings
- 📄 Document
- 🤖 AI Assistant
- 📊 Statistics

## 🔄 Refresh & Reload

### When to Reload Components
- After uploading new documents
- After changing configuration
- If something seems stuck

### How to Reload
1. Go to **⚙️ Settings**
2. Click **🔄 Reload Components**
3. Wait for confirmation
4. Components will reinitialize

### Clear Cache
- Clears all cached data
- Forces fresh load of everything
- Use if experiencing issues

## 📱 Browser Compatibility

**Recommended Browsers:**
- ✅ Chrome/Edge (best performance)
- ✅ Firefox
- ✅ Safari
- ⚠️ Internet Explorer (not supported)

## 🎯 Best Practices

### For Best Results

1. **Upload Quality Papers**
   - Clear, readable PDFs
   - Research papers work best
   - Avoid scanned images (OCR quality varies)

2. **Ask Clear Questions**
   - Be specific
   - Use complete sentences
   - Reference topics in your papers

3. **Review Sources**
   - Always check source citations
   - Verify information accuracy
   - Use multiple sources for important facts

4. **Manage Your Library**
   - Keep papers organized
   - Remove duplicates
   - Upload related papers together

## 🆘 Getting Help

### Check Logs
1. Go to **⚙️ Settings**
2. Click **📊 View Logs**
3. Review recent log entries
4. Look for error messages

### Common Issues

**"No documents found"**
- Upload papers first
- Check Upload page for errors

**"API key not found"**
- Set GOOGLE_API_KEY in `.env` file
- Restart the app

**"Model not found"**
- Check internet connection
- Verify API key is valid
- Try reloading components

## 🎉 You're Ready!

Start exploring your AI Research Assistant:

1. **Upload** your first paper
2. **Chat** to ask questions
3. **Search** to find information
4. **Enjoy** your intelligent research assistant!

---

**Happy Researching! 🤖📚**
