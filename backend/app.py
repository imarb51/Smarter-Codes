"""
Flask REST API for HTML Semantic Search
Provides endpoints for indexing URLs and performing semantic searches.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from typing import Dict, Any
import traceback

from modules.search_service import SearchService, SearchServiceError

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for frontend (React)
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv('CORS_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize search service (singleton)
try:
    search_service = SearchService(
        use_gemini=os.getenv('USE_GEMINI', 'true').lower() == 'true',
        use_tiktoken=os.getenv('USE_TIKTOKEN', 'false').lower() == 'true'
    )
    print("✅ Search service initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize search service: {e}")
    search_service = None


# Helper function for error responses
def error_response(message: str, status_code: int = 400, details: str = None) -> tuple:
    """
    Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        details: Optional detailed error info
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'success': False,
        'error': message
    }
    if details:
        response['details'] = details
    
    return jsonify(response), status_code


# Helper function for success responses
def success_response(data: Dict[str, Any], status_code: int = 200) -> tuple:
    """
    Create standardized success response
    
    Args:
        data: Response data
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'success': True,
        **data
    }
    return jsonify(response), status_code


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return success_response({
        'status': 'healthy',
        'service_ready': search_service is not None
    })


@app.route('/api/index', methods=['POST'])
def index_url():
    """
    Index a URL for semantic search
    
    Request Body:
    {
        "url": "https://example.com",
        "force_reindex": false  // optional, default: false
    }
    
    Response:
    {
        "success": true,
        "url": "https://example.com",
        "html_size": 45000,
        "text_size": 12000,
        "chunk_count": 24,
        "indexed_count": 24,
        "cached": false
    }
    """
    if not search_service:
        return error_response("Search service not initialized", 503)
    
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return error_response("Request body must be JSON")
        
        url = data.get('url')
        if not url:
            return error_response("Missing required field: 'url'")
        
        if not isinstance(url, str) or len(url.strip()) == 0:
            return error_response("Invalid URL: must be a non-empty string")
        
        # Optional parameter
        force_reindex = data.get('force_reindex', False)
        if not isinstance(force_reindex, bool):
            return error_response("Invalid force_reindex: must be boolean")
        
        # Process and index
        result = search_service.process_and_index(url, force_reindex=force_reindex)
        
        return success_response(result, 201)
        
    except SearchServiceError as e:
        return error_response(str(e), 500)
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/index: {traceback.format_exc()}")
        return error_response("Internal server error", 500, str(e))


@app.route('/api/search', methods=['POST'])
def search():
    """
    Search indexed content
    
    Request Body:
    {
        "query": "search query text",
        "top_k": 10,  // optional, default: 10
        "score_threshold": 0.3  // optional, default: 0.3
    }
    
    Response:
    {
        "success": true,
        "query": "search query text",
        "results": [
            {
                "rank": 1,
                "score": 0.8542,
                "text": "matching text chunk",
                "url": "https://example.com",
                "chunk_id": 5,
                "token_count": 450
            },
            ...
        ],
        "total_results": 10
    }
    """
    if not search_service:
        return error_response("Search service not initialized", 503)
    
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return error_response("Request body must be JSON")
        
        query = data.get('query')
        if not query:
            return error_response("Missing required field: 'query'")
        
        if not isinstance(query, str) or len(query.strip()) == 0:
            return error_response("Invalid query: must be a non-empty string")
        
        # Optional parameters
        top_k = data.get('top_k')
        score_threshold = data.get('score_threshold')
        
        # Validate optional parameters
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
                return error_response("Invalid top_k: must be integer between 1 and 100")
        
        if score_threshold is not None:
            if not isinstance(score_threshold, (int, float)) or score_threshold < 0 or score_threshold > 1:
                return error_response("Invalid score_threshold: must be number between 0 and 1")
        
        # Perform search
        results = search_service.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        return success_response({
            'query': query,
            'results': results,
            'total_results': len(results)
        })
        
    except SearchServiceError as e:
        return error_response(str(e), 500)
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/search: {traceback.format_exc()}")
        return error_response("Internal server error", 500, str(e))


@app.route('/api/index-and-search', methods=['POST'])
def index_and_search():
    """
    Index a URL and immediately search it
    
    Request Body:
    {
        "url": "https://example.com",
        "query": "search query text",
        "top_k": 10,  // optional, default: 10
        "score_threshold": 0.3,  // optional, default: 0.3
        "force_reindex": false  // optional, default: false
    }
    
    Response:
    {
        "success": true,
        "url": "https://example.com",
        "query": "search query text",
        "processing": {
            "html_size": 45000,
            "text_size": 12000,
            "chunk_count": 24,
            "indexed_count": 24,
            "cached": false
        },
        "results": [...],
        "total_results": 10
    }
    """
    if not search_service:
        return error_response("Search service not initialized", 503)
    
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return error_response("Request body must be JSON")
        
        url = data.get('url')
        query = data.get('query')
        
        if not url:
            return error_response("Missing required field: 'url'")
        if not query:
            return error_response("Missing required field: 'query'")
        
        if not isinstance(url, str) or len(url.strip()) == 0:
            return error_response("Invalid URL: must be a non-empty string")
        
        if not isinstance(query, str) or len(query.strip()) == 0:
            return error_response("Invalid query: must be a non-empty string")
        
        # Optional parameters
        top_k = data.get('top_k')
        score_threshold = data.get('score_threshold')
        force_reindex = data.get('force_reindex', False)
        
        # Validate optional parameters
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1 or top_k > 100):
            return error_response("Invalid top_k: must be integer between 1 and 100")
        
        if score_threshold is not None and (not isinstance(score_threshold, (int, float)) or score_threshold < 0 or score_threshold > 1):
            return error_response("Invalid score_threshold: must be number between 0 and 1")
        
        if not isinstance(force_reindex, bool):
            return error_response("Invalid force_reindex: must be boolean")
        
        # Process and search
        result = search_service.process_and_search(
            url=url,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            force_reindex=force_reindex
        )
        
        return success_response(result, 201)
        
    except SearchServiceError as e:
        return error_response(str(e), 500)
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/index-and-search: {traceback.format_exc()}")
        return error_response("Internal server error", 500, str(e))


@app.route('/api/index/<path:url>', methods=['DELETE'])
def delete_index(url: str):
    """
    Delete indexed content for a URL
    
    URL Parameter:
        url: URL to delete (URL-encoded)
    
    Response:
    {
        "success": true,
        "message": "Deleted all chunks for URL",
        "url": "https://example.com"
    }
    """
    if not search_service:
        return error_response("Search service not initialized", 503)
    
    try:
        if not url or len(url.strip()) == 0:
            return error_response("Invalid URL: must be non-empty")
        
        # Delete from index
        search_service.delete_url(url)
        
        return success_response({
            'message': 'Deleted all chunks for URL',
            'url': url
        })
        
    except SearchServiceError as e:
        return error_response(str(e), 500)
    except Exception as e:
        app.logger.error(f"Unexpected error in DELETE /api/index: {traceback.format_exc()}")
        return error_response("Internal server error", 500, str(e))


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get vector database statistics
    
    Response:
    {
        "success": true,
        "stats": {
            "total_vector_count": 1024,
            "dimension": 768,
            "index_fullness": 0.05,
            "namespaces": {}
        }
    }
    """
    if not search_service:
        return error_response("Search service not initialized", 503)
    
    try:
        stats = search_service.get_stats()
        
        return success_response({
            'stats': stats
        })
        
    except SearchServiceError as e:
        return error_response(str(e), 500)
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/stats: {traceback.format_exc()}")
        return error_response("Internal server error", 500, str(e))


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return error_response("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return error_response("Method not allowed", 405)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal server error: {error}")
    return error_response("Internal server error", 500)


# Main entry point
if __name__ == '__main__':
    # Get configuration from environment
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n🚀 Starting Flask API Server")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"🌐 CORS Origins: {os.getenv('CORS_ORIGINS', '*')}")
    print(f"\n📚 API Endpoints:")
    print(f"  GET    /api/health")
    print(f"  POST   /api/index")
    print(f"  POST   /api/search")
    print(f"  POST   /api/index-and-search")
    print(f"  DELETE /api/index/<url>")
    print(f"  GET    /api/stats")
    print(f"\n✨ Ready to serve!\n")
    
    app.run(host=host, port=port, debug=debug)
