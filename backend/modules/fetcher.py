"""
HTML Fetcher Module
Handles fetching and validating HTML content from URLs with comprehensive error handling.
"""

import re
import requests
from urllib.parse import urlparse
from typing import Tuple, Optional
import os


class HTMLFetchError(Exception):
    """Custom exception for HTML fetching errors"""
    pass


class HTMLFetcher:
    """Fetches and validates HTML content from URLs"""
    
    def __init__(self, max_html_size: int = 500000):
        """
        Initialize HTML Fetcher
        
        Args:
            max_html_size: Maximum HTML content size in characters (default: 500k)
        """
        self.max_html_size = max_html_size or int(os.getenv('MAX_HTML_SIZE', 500000))
        self.timeout = 10  # seconds
        self.max_redirects = 10
        
        # User agent to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def validate_url(self, url: str) -> str:
        """
        Validate and sanitize URL
        
        Args:
            url: URL to validate
            
        Returns:
            Sanitized URL
            
        Raises:
            HTMLFetchError: If URL is invalid
        """
        # Strip whitespace
        url = url.strip()
        
        # Add schema if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise HTMLFetchError(f"Invalid URL format: {str(e)}")
        
        # Validate domain
        if not parsed.netloc:
            raise HTMLFetchError("Invalid URL: No domain found")
        
        # Check for valid domain pattern
        domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-_.]+[a-zA-Z0-9]$'
        if not re.match(domain_pattern, parsed.netloc.split(':')[0]):
            raise HTMLFetchError(f"Invalid domain: {parsed.netloc}")
        
        return url
    
    def detect_encoding(self, response: requests.Response) -> str:
        """
        Detect content encoding from response
        
        Args:
            response: HTTP response object
            
        Returns:
            Encoding string (utf-8, latin-1, etc.)
        """
        # Try to get encoding from headers
        if response.encoding:
            return response.encoding
        
        # Try to detect from content
        content_type = response.headers.get('content-type', '').lower()
        
        # Check for charset in content-type header
        if 'charset=' in content_type:
            charset = content_type.split('charset=')[-1].split(';')[0].strip()
            return charset
        
        # Default to utf-8
        return 'utf-8'
    
    def fetch_html(self, url: str) -> Tuple[str, str]:
        """
        Fetch HTML content from URL with comprehensive error handling
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (html_content, final_url)
            
        Raises:
            HTMLFetchError: For various fetch errors
        """
        # Validate URL first
        url = self.validate_url(url)
        
        try:
            # Create a session to handle max_redirects
            with requests.Session() as session:
                session.max_redirects = self.max_redirects
                
                # Make request with timeout and redirect limits
                response = session.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                    stream=True  # Stream to check size before loading
                )
            
            # Check for authentication errors
            if response.status_code == 401:
                raise HTMLFetchError("URL requires authentication (401 Unauthorized)")
            
            if response.status_code == 403:
                raise HTMLFetchError("Access forbidden (403 Forbidden)")
            
            # Check for other HTTP errors
            if response.status_code >= 400:
                raise HTMLFetchError(f"HTTP error {response.status_code}: {response.reason}")
            
            # Validate content type (must be HTML)
            content_type = response.headers.get('content-type', '').lower()
            
            # Check for non-HTML content
            if 'application/pdf' in content_type:
                raise HTMLFetchError("Content is PDF, not HTML")
            elif 'application/json' in content_type:
                raise HTMLFetchError("Content is JSON, not HTML")
            elif 'application/xml' in content_type or 'text/xml' in content_type:
                raise HTMLFetchError("Content is XML, not HTML")
            elif 'text/html' not in content_type and 'application/xhtml' not in content_type:
                # Allow if no content-type specified (some servers don't send it)
                if content_type and content_type.strip():
                    raise HTMLFetchError(f"Content type is not HTML: {content_type}")
            
            # Check content size from headers if available
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_html_size:
                raise HTMLFetchError(f"HTML content too large: {content_length} characters (max: {self.max_html_size})")
            
            # Detect encoding
            encoding = self.detect_encoding(response)
            
            # Read content in chunks to enforce size limit
            html_content = ""
            chunk_size = 8192
            
            try:
                for chunk in response.iter_content(chunk_size=chunk_size, decode_unicode=True):
                    if chunk:
                        html_content += chunk
                        
                        # Check size limit
                        if len(html_content) > self.max_html_size:
                            html_content = html_content[:self.max_html_size]
                            break
            except UnicodeDecodeError:
                # Try with different encoding
                response.encoding = 'latin-1'
                html_content = response.text[:self.max_html_size]
            
            # Validate we got some content
            if not html_content or len(html_content.strip()) == 0:
                raise HTMLFetchError("No content received from URL")
            
            # Return content and final URL (after redirects)
            return html_content, response.url
            
        except requests.exceptions.TooManyRedirects:
            raise HTMLFetchError(f"Too many redirects (max: {self.max_redirects})")
        
        except requests.exceptions.Timeout:
            raise HTMLFetchError(f"Request timeout after {self.timeout} seconds")
        
        except requests.exceptions.ConnectionError as e:
            raise HTMLFetchError(f"Connection error: {str(e)}")
        
        except requests.exceptions.RequestException as e:
            raise HTMLFetchError(f"Request failed: {str(e)}")
        
        except Exception as e:
            raise HTMLFetchError(f"Unexpected error: {str(e)}")


# Convenience function for easy usage
def fetch_html_from_url(url: str, max_size: int = None) -> Tuple[str, str]:
    """
    Fetch HTML content from URL
    
    Args:
        url: URL to fetch
        max_size: Maximum HTML size (optional, defaults to env var or 500k)
        
    Returns:
        Tuple of (html_content, final_url)
    """
    if max_size is None:
        max_size = int(os.getenv('MAX_HTML_SIZE', 500000))
    
    fetcher = HTMLFetcher(max_html_size=max_size)
    return fetcher.fetch_html(url)
