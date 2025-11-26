# API Testing Guide

## Quick Start

1. **Start the Flask Server**
   ```bash
   cd backend
   python app.py
   ```

2. **Test Endpoints with curl**

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Index a URL
```bash
curl -X POST http://localhost:5000/api/index \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### Search
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search query",
    "top_k": 10,
    "score_threshold": 0.3
  }'
```

### Index and Search
```bash
curl -X POST http://localhost:5000/api/index-and-search \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "query": "your search query",
    "top_k": 10
  }'
```

### Get Stats
```bash
curl http://localhost:5000/api/stats
```

### Delete Indexed URL
```bash
curl -X DELETE "http://localhost:5000/api/index/https://example.com"
```

## Frontend Integration (React/Vite)

### Example: Search Component

```javascript
import { useState } from 'react';

function SearchComponent() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  
  const handleSearch = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          top_k: 10,
          score_threshold: 0.3
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setResults(data.results);
      } else {
        console.error('Search failed:', data.error);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter search query..."
      />
      <button onClick={handleSearch} disabled={loading}>
        {loading ? 'Searching...' : 'Search'}
      </button>
      
      <div className="results">
        {results.map((result, index) => (
          <div key={index} className="result-item">
            <h3>#{result.rank} - Score: {result.score}</h3>
            <p>{result.text}</p>
            <small>
              URL: {result.url} | Chunk: {result.chunk_id} | Tokens: {result.token_count}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SearchComponent;
```

### Example: Index Component

```javascript
import { useState } from 'react';

function IndexComponent() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const handleIndex = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/index', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setResult({
          success: true,
          message: `Indexed ${data.chunk_count} chunks from ${data.url}`
        });
      } else {
        setResult({
          success: false,
          message: data.error
        });
      }
    } catch (error) {
      setResult({
        success: false,
        message: error.message
      });
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <input
        type="text"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="Enter URL to index..."
      />
      <button onClick={handleIndex} disabled={loading}>
        {loading ? 'Indexing...' : 'Index URL'}
      </button>
      
      {result && (
        <div className={result.success ? 'success' : 'error'}>
          {result.message}
        </div>
      )}
    </div>
  );
}

export default IndexComponent;
```

## Response Formats

### Success Response
```json
{
  "success": true,
  "query": "search query",
  "results": [...],
  "total_results": 10
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message",
  "details": "Optional detailed error info"
}
```

## Error Codes

- `200` - Success
- `201` - Created (successful indexing)
- `400` - Bad Request (validation error)
- `404` - Not Found
- `405` - Method Not Allowed
- `500` - Internal Server Error
- `503` - Service Unavailable (search service not initialized)
