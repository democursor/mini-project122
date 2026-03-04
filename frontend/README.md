# Research Platform Frontend

Modern, fast, and beautiful React frontend for the Research Literature Processing Platform.

## Features

- 📊 **Dashboard** - Overview of your research collection
- 📤 **Upload** - Drag & drop PDF upload with progress tracking
- 📚 **Documents** - Manage your research papers
- 🔍 **Semantic Search** - Find relevant papers using natural language
- 🕸️ **Knowledge Graph** - Explore relationships between papers and concepts
- 💬 **AI Assistant** - Chat with your research collection using RAG

## Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first CSS framework
- **React Query** - Data fetching and caching
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Lucide React** - Beautiful icons
- **React Hot Toast** - Toast notifications

## Setup

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The production build will be in the `dist/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.js          # API client and endpoints
│   ├── components/
│   │   └── Layout.jsx         # Main layout with sidebar
│   ├── pages/
│   │   ├── Dashboard.jsx      # Dashboard page
│   │   ├── Upload.jsx         # Upload page
│   │   ├── Documents.jsx      # Documents list
│   │   ├── Search.jsx         # Semantic search
│   │   ├── KnowledgeGraph.jsx # Knowledge graph
│   │   └── Chat.jsx           # AI assistant
│   ├── App.jsx                # Main app component
│   ├── main.jsx               # Entry point
│   └── index.css              # Global styles
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000/api`.

All API calls are handled through the `src/api/client.js` module:

- `documentsAPI` - Document management
- `searchAPI` - Semantic search
- `graphAPI` - Knowledge graph queries
- `chatAPI` - AI assistant

## Features in Detail

### Dashboard
- Real-time statistics
- Quick actions
- Recent documents

### Upload
- Drag & drop interface
- File validation
- Upload progress
- Processing status

### Documents
- List all documents
- View metadata
- Delete documents
- Status indicators

### Search
- Natural language queries
- Semantic similarity
- Relevance scores
- Highlighted excerpts

### Knowledge Graph
- Graph statistics
- Papers explorer
- Top concepts
- Related papers

### AI Assistant
- Chat interface
- RAG-powered answers
- Source citations
- Conversation history

## Performance

- **Fast Initial Load** - Vite's optimized bundling
- **Code Splitting** - Lazy loading for routes
- **Optimistic Updates** - React Query caching
- **Smooth Animations** - Tailwind transitions
- **Responsive Design** - Mobile-friendly

## Customization

### Colors

Edit `tailwind.config.js` to change the primary color:

```js
theme: {
  extend: {
    colors: {
      primary: {
        // Your color palette
      }
    }
  }
}
```

### API URL

Edit `src/api/client.js` to change the API base URL:

```js
const API_BASE_URL = 'your-api-url'
```

## Troubleshooting

### CORS Issues

If you encounter CORS errors, make sure the backend has CORS enabled for `http://localhost:3000`.

### API Connection

Ensure the backend is running on `http://localhost:8000` before starting the frontend.

### Build Errors

Clear node_modules and reinstall:

```bash
rm -rf node_modules package-lock.json
npm install
```

## License

MIT
