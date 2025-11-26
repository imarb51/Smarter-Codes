import React from 'react';
import type { SearchResult } from '../types';
import { ResultCard } from './ResultCard';

interface ResultsListProps {
  results: SearchResult[];
}

export const ResultsList: React.FC<ResultsListProps> = ({ results }) => {
  if (results.length === 0) return null;

  return (
    <div className="results-list">
      {results.map((result, index) => (
        // Using index as fallback key if chunk_id is not unique or present, though it should be.
        // Combining chunk_id and index to be safe.
        <ResultCard key={`${result.chunk_id}-${index}`} result={result} />
      ))}
    </div>
  );
};
