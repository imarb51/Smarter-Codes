"""
HTML Parser Module
Cleans and extracts meaningful text from HTML content.
"""

from bs4 import BeautifulSoup, Comment
import re
from typing import List


class HTMLParseError(Exception):
    """Custom exception for HTML parsing errors"""
    pass


class HTMLParser:
    """Parses and cleans HTML content to extract meaningful text"""
    
    def __init__(self):
        """Initialize HTML Parser"""
        # Tags to extract text from (semantic content)
        self.content_tags = [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'article', 'section', 'main', 'div', 'span',
            'li', 'td', 'th', 'blockquote', 'pre',
            'strong', 'em', 'b', 'i', 'mark',
            'code', 'figcaption', 'caption'
        ]
        
        # Tags to completely remove
        self.remove_tags = [
            'script', 'style', 'noscript', 'iframe',
            'nav', 'footer', 'header', 'aside',
            'form', 'button', 'input', 'select', 'textarea',
            'meta', 'link', 'svg', 'canvas', 'video', 'audio',
            'picture', 'object', 'embed', 'applet', 'param',
            'track', 'source', 'area', 'map'
        ]
    
    def remove_unwanted_elements(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Remove unwanted elements from HTML
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Cleaned BeautifulSoup object
        """
        
        # Remove unwanted tags
        for tag_name in self.remove_tags:
            tags = soup.find_all(tag_name)
            for tag in tags:
                tag.decompose()
        
        # Remove HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        # Remove inline event attributes (onclick, onload, etc.)
        tags_with_attrs = soup.find_all()
        for tag in tags_with_attrs:
            if tag is None:
                continue
            for attr in list(tag.attrs):
                if attr.startswith("on"):
                    del tag.attrs[attr]
        
        # Collect tags to remove for invisible CSS (display:none, visibility:hidden)
        tags_to_remove = []
        style_tags = soup.find_all(style=True)
        for tag in style_tags:
            if tag is None:
                continue
            style_attr = tag.get('style')
            if isinstance(style_attr, str):
                style = style_attr.lower()
                if 'display:none' in style.replace(' ', '') or 'visibility:hidden' in style.replace(' ', ''):
                    tags_to_remove.append(tag)
        
        # Collect tags to remove for hidden class (common pattern)
        class_tags = soup.find_all(class_=True)
        for tag in class_tags:
            if tag is None:
                continue
            classes = tag.get('class')
            if isinstance(classes, list):
                if any(isinstance(cls, str) and cls.lower() in ['hidden', 'invisible', 'd-none'] for cls in classes):
                    tags_to_remove.append(tag)
        
        # Remove all collected tags
        for tag in tags_to_remove:
            tag.decompose()
        
        return soup
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract meaningful text from HTML
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text
        """
        texts = []
        
        # Extract text from content tags
        content_tags_found = soup.find_all(self.content_tags)
        for tag in content_tags_found:
            if tag is None:
                continue
            text = tag.get_text(separator=' ', strip=True)
            if text and len(text.strip()) > 0:
                texts.append(text)
        
        # If no content tags found, get all text
        if not texts:
            text = soup.get_text(separator=' ', strip=True)
            if text:
                texts.append(text)
        
        # Join all texts
        full_text = '\n\n'.join(texts)
        
        return full_text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph separator)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def remove_special_characters(self, text: str) -> str:
        """
        Handle special characters while preserving meaningful punctuation
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Replace non-breaking spaces and other space variants
        text = text.replace('\xa0', ' ')  # Non-breaking space
        text = text.replace('\u2007', ' ')  # Figure space
        text = text.replace('\u202f', ' ')  # Narrow no-break space
        
        # Improved quote normalization - handle all unicode quote variants
        # Single quotes
        text = text.replace(''', "'").replace(''', "'")  # Curly single quotes
        text = text.replace('‚', "'").replace('‛', "'")  # Low-9 and reversed
        text = text.replace('`', "'").replace('´', "'")  # Grave and acute
        
        # Double quotes
        text = text.replace('"', '"').replace('"', '"')  # Curly double quotes
        text = text.replace('„', '"').replace('‟', '"')  # Low-9 and reversed
        text = text.replace('«', '"').replace('»', '"')  # Guillemets
        
        # Replace em/en dashes with regular dash
        text = text.replace('—', '-').replace('–', '-')
        text = text.replace('−', '-')  # Minus sign
        
        # Remove emojis (optional but useful)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        
        # Remove URLs (optional but useful for clean text)
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = url_pattern.sub('', text)
        
        # Remove any remaining control characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Remove long garbage strings (50+ consecutive non-space chars)
        text = re.sub(r'\b\S{50,}\b', '', text)
        
        return text
    
    def remove_duplicates(self, text: str) -> str:
        """
        Remove duplicate consecutive sentences (common in headers/footers)
        
        Args:
            text: Input text
            
        Returns:
            Text with duplicates removed
        """
        # Improved sentence splitting that handles abbreviations
        # Common abbreviations that should not end a sentence
        # We must group them by length because Python's re module requires fixed-width lookbehinds
        abbrevs_list = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'Inc', 'Ltd', 'Co', 'Corp']
        abbrevs_by_len = {}
        for abbr in abbrevs_list:
            length = len(abbr)
            if length not in abbrevs_by_len:
                abbrevs_by_len[length] = []
            abbrevs_by_len[length].append(abbr)
        
        # Construct lookbehinds for each length
        lookbehinds = ''
        for length, abbrs in abbrevs_by_len.items():
            pattern = '|'.join(abbrs)
            lookbehinds += f'(?<!\\b(?:{pattern}))'
            
        # Split on sentence boundaries (. ! ?) but not after abbreviations
        sentence_pattern = lookbehinds + r'([.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Reconstruct sentences (split creates alternating content and punctuation)
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                reconstructed.append(sentences[i] + sentences[i + 1])
            else:
                reconstructed.append(sentences[i])
        
        # If we have leftover, add it
        if len(sentences) % 2 == 1:
            reconstructed.append(sentences[-1])
        
        # Remove consecutive duplicates
        unique_sentences = []
        prev_sentence = None
        
        for sentence in reconstructed:
            # Normalize for comparison
            normalized = sentence.strip().lower()
            
            # Only add if not duplicate of previous and not too short
            if normalized and normalized != prev_sentence and len(normalized) > 3:
                unique_sentences.append(sentence.strip())
                prev_sentence = normalized
        
        # Rejoin sentences with proper spacing
        result = ' '.join(unique_sentences)
        
        # Fix punctuation (add period at end if missing)
        if result and not result[-1] in '.!?':
            result += '.'
        
        return result
    
    def clean_html(self, html_content: str) -> str:
        """
        Main method to clean and extract text from HTML
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned text content
            
        Raises:
            HTMLParseError: If parsing fails
        """
        try:
            # Try lxml parser first (fastest), fallback to html.parser (safer)
            try:
                soup = BeautifulSoup(html_content, 'lxml')
            except Exception as e:
                # Fallback to html.parser if lxml fails
                soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text
            text = self.extract_text(soup)
            
            # Handle special characters
            text = self.remove_special_characters(text)
            
            # Normalize whitespace
            text = self.normalize_whitespace(text)
            
            # Remove duplicates
            text = self.remove_duplicates(text)
            
            # Validate we have content
            if not text or len(text.strip()) < 10:
                return ""  # Fallback to empty string if parsing yields no content
            
            return text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if isinstance(e, HTMLParseError):
                raise
            raise HTMLParseError(f"Failed to parse HTML: {str(e)}")


# Convenience function
def parse_html(html_content: str) -> str:
    """
    Parse and clean HTML content
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Cleaned text
    """
    parser = HTMLParser()
    return parser.clean_html(html_content)
