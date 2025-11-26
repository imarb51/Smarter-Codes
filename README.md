# AI Scraper - Semantic Search Application

A full-stack semantic search application that intelligently indexes and searches web content using AI embeddings. Built with React (frontend) and Flask (backend), powered by Google Gemini AI and Pinecone vector database.

![AI Scraper](https://img.shields.io/badge/AI-Semantic%20Search-blue) ![Python](https://img.shields.io/badge/Python-3.9+-green) ![React](https://img.shields.io/badge/React-18+-61DAFB) ![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6)

## ✨ Features

### Smart Caching
- **10-20x faster** for cached URLs
- Automatic detection of already-indexed content
- **~95% reduction** in API costs for repeated searches
- Force re-index option for updated content

### Intelligent Search
- **Semantic understanding** using Google Gemini embeddings (768 dimensions)
- **Chunk-based indexing** with 200-token segments and 50-token overlap
- **Relevance scoring** with configurable threshold (default: 0.3)
- **Top-K results** with customizable result count

### Modern UI
- Clean, responsive React interface
- **Text preview** with expand/collapse (300 char preview)
- **Card-based results** with rank badges and scores
- Real-time search with loading states
- Clickable source URLs

### Production Ready
- Comprehensive error handling
- URL validation and sanitization
- Support for large HTML pages (up to 500k characters)
- Automatic encoding detection
- In-memory fallback for development

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           React Frontend                │
│  (TypeScript + Vite + TailwindCSS)      │
│     localhost:5173                      │
└──────────────┬──────────────────────────┘
               │ REST API
               ▼
┌─────────────────────────────────────────┐
│          Flask Backend                  │
│      (Python + Gemini AI)               │
│     localhost:5000                      │
└──────────────┬──────────────────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌──────────┐      ┌──────────────┐
│  Gemini  │      │   Pinecone   │
│   AI     │      │  Vector DB   │
│ (768dim) │      │              │
└──────────┘      └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **Pinecone API Key** ([Get one free](https://www.pinecone.io/))
- **Google Gemini API Key** ([Get one free](https://makersuite.google.com/app/apikey))
- **HuggingFace Token** (optional, for tokenizer)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp ../.env.example .env
# Edit .env with your API keys

# Run the server
python app.py
```

Backend will start on **http://localhost:5000**

### Frontend Setup

```bash
# Navigate to frontend directory  
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will start on **http://localhost:5173**

---

## 🔧 Configuration

Create a `.env` file in the `backend/` directory (or copy from `.env.example`):

```env
# Required API Keys
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here

# Pinecone Configuration
PINECONE_INDEX_NAME=html-semantic-search
PINECONE_DIMENSION=768

# Search Configuration
CHUNK_SIZE=200              # Tokens per chunk
CHUNK_OVERLAP=50            # Overlap between chunks
MAX_CHUNKS=1000             # Max chunks to prevent OOM
MIN_SCORE_THRESHOLD=0.3     # Minimum similarity score
TOP_K_RESULTS=10            # Number of results to return

# Application Settings
MAX_HTML_SIZE=500000        # Max HTML size in characters
USE_GEMINI=true             # Use Gemini for embeddings
USE_TIKTOKEN=false          # Use tiktoken (faster) vs transformers

# Flask Server
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

---

## 📚 API Endpoints

### `POST /api/index`
Index a URL for semantic search.

**Request:**
```json
{
  "url": "https://example.com",
  "force_reindex": false  // optional, default: false
}
```

**Response:**
```json
{
  "success": true,
  "url": "https://example.com",
  "html_size": 45000,
  "text_size": 12000,
  "chunk_count": 60,
  "indexed_count": 60,
  "cached": false
}
```

---

### `POST /api/search`
Search through already-indexed content.

**Request:**
```json
{
  "query": "machine learning",
  "top_k": 10,                // optional, default: 10
  "score_threshold": 0.3      // optional, default: 0.3
}
```

**Response:**
```json
{
  "success": true,
  "query": "machine learning",
  "results": [
    {
      "rank": 1,
      "score": 0.8542,
      "text": "Machine learning is a subset of...",
      "url": "https://example.com",
      "chunk_id": 5,
      "token_count": 198
    }
  ],
  "total_results": 10
}
```

---

### `POST /api/index-and-search`
Index a URL and immediately search it (most common use case).

**Request:**
```json
{
  "url": "https://example.com",
  "query": "machine learning",
  "top_k": 10,                // optional
  "score_threshold": 0.3,     // optional
  "force_reindex": false      // optional
}
```

**Response:**
```json
{
  "success": true,
  "url": "https://example.com",
  "query": "machine learning",
  "processing": {
    "cached": true,
    "message": "URL already indexed, skipped processing"
  },
  "results": [...],
  "total_results": 10
}
```

---

### `DELETE /api/index/<url>`
Delete all indexed chunks for a URL.

**Example:**
```bash
curl -X DELETE "http://localhost:5000/api/index/https%3A%2F%2Fexample.com"
```

---

### `GET /api/stats`
Get vector database statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_vector_count": 1250,
    "dimension": 768,
    "index_fullness": 0.05
  }
}
```

---

### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "service_ready": true
}
```

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest test_backend.py -v
```

### Manual API Testing

```bash
# Index and search a URL
curl -X POST http://localhost:5000/api/index-and-search \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "query": "machine learning applications"
  }'

# Force re-index
curl -X POST http://localhost:5000/api/index-and-search \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "query": "test",
    "force_reindex": true
  }'
```

---

## 📁 Project Structure

```
AI-Scraper/
├── backend/
│   ├── app.py                    # Flask application
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment variables (not in git)
│   ├── modules/
│   │   ├── fetcher.py           # HTML fetching
│   │   ├── parser.py            # HTML parsing
│   │   ├── tokenizer.py         # Text chunking
│   │   ├── embedder.py          # Gemini embeddings
│   │   ├── vector_db.py         # Pinecone integration
│   │   └── search_service.py    # Search orchestration
│   └── utils/
│       └── logger.py            # Logging setup
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main component
│   │   ├── components/
│   │   │   ├── Header.tsx       # Header component
│   │   │   ├── SearchForm.tsx   # Search form
│   │   │   ├── ResultCard.tsx   # Result card with expand/collapse
│   │   │   └── ResultsList.tsx  # Results list
│   │   ├── services/
│   │   │   └── api.ts           # API client
│   │   └── types/
│   │       └── index.ts         # TypeScript types
│   ├── package.json
│   └── vite.config.ts
│
├── .env.example                  # Environment template
├── .gitignore
└── README.md
```

---

## 🛠️ Technology Stack

### Backend
- **Flask 3.0** - Web framework
- **Google Gemini AI** - Text embeddings (768 dimensions)
- **Pinecone** - Vector database
- **BeautifulSoup4** - HTML parsing
- **Transformers** - Tokenization

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **Lucide React** - Icons

---

## 💡 How It Works

### Indexing Pipeline
1. **Fetch** HTML content from URL
2. **Parse** and clean HTML (remove scripts, styles, duplicates)
3. **Chunk** text into 200-token segments with 50-token overlap
4. **Check Cache** - Skip if URL already indexed
5. **Generate Embeddings** using Gemini AI (768 dimensions)
6. **Store** vectors in Pinecone with metadata

### Search Pipeline
1. **Generate Query Embedding** using Gemini AI
2. **Search Pinecone** for similar vectors (cosine similarity)
3. **Filter** by score threshold (default: 0.3)
4. **Rank** and return top-K results (default: 10)

### Caching Behavior
- **First Request**: Full indexing (~10-20s)
- **Subsequent Requests**: Instant cache hit (~1-2s) ⚡
- **Force Refresh**: Use `force_reindex: true` to update

---

## 🔒 Security Notes

- ✅ `.env` files are in `.gitignore` (never commit API keys!)
- ✅ CORS configured for local development only
- ✅ Input validation on all endpoints
- ✅ URL sanitization and domain validation
- ⚠️ **Before deploying to production:**
  - Change `FLASK_DEBUG=False`
  - Use a production WSGI server (gunicorn/waitress)
  - Set up proper CORS origins
  - Enable HTTPS
  - Rotate API keys if accidentally committed

---

## 📈 Performance

| Metric | First Request | Cached Request |
|--------|--------------|----------------|
| **Time** | 10-20 seconds | 1-2 seconds |
| **Speed** | Baseline | **10-20x faster** ⚡ |
| **API Calls** | 200+ embeddings | 1 query embedding |
| **Cost** | Baseline | **~95% cheaper** 💰 |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- [Google Gemini AI](https://ai.google.dev/) for powerful embeddings
- [Pinecone](https://www.pinecone.io/) for scalable vector database
- [Flask](https://flask.palletsprojects.com/) for the robust backend framework
- [React](https://react.dev/) for the amazing UI library

---

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ using AI-powered semantic search**
