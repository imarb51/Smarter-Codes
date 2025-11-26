import React, { useState } from 'react';
import type { SearchResult } from '../types';

interface ResultCardProps {
  result: SearchResult;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  const [showFullText, setShowFullText] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  // Truncate text to first 300 characters for preview
  const truncatedText = result.text.slice(0, 300);
  const needsTruncation = result.text.length > 300;

  return (
    <div className="result-card">
      <div className="result-header">
        <span className="result-rank">#{result.rank}</span>
        <span className="result-score">Score: {result.score.toFixed(4)}</span>
      </div>
      
      <p className="result-text">
        {showFullText ? result.text : truncatedText}
        {needsTruncation && !showFullText && '...'}
      </p>
      
      {needsTruncation && (
        <button 
          className="expand-text-btn"
          onClick={() => setShowFullText(!showFullText)}
        >
          {showFullText ? '← Show Less' : 'Read More →'}
        </button>
      )}
      
      <div className="result-footer">
        <button 
          className="toggle-html-btn"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? 'Hide Details' : 'View Details'}
        </button>
        {showDetails && (
          <div className="html-preview">
            <div className="detail-row">
              <strong>Chunk ID:</strong> {result.chunk_id}
            </div>
            <div className="detail-row">
              <strong>Tokens:</strong> {result.token_count}
            </div>
            <div className="detail-row">
              <strong>URL:</strong> <a href={result.url} target="_blank" rel="noopener noreferrer">{result.url}</a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
