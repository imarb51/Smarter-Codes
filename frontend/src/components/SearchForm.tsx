import React, { useState } from 'react';
import { Search, Globe } from 'lucide-react';

interface SearchFormProps {
  onSearch: (url: string, query: string) => void;
  isLoading: boolean;
}

export const SearchForm: React.FC<SearchFormProps> = ({ onSearch, isLoading }) => {
  const [url, setUrl] = useState('');
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(url, query);
  };

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <div className="input-group">
        <Globe className="input-icon" size={20} />
        <input
          type="url"
          placeholder="https://example.com"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="search-input"
        />
      </div>
      <div className="input-group">
        <Search className="input-icon" size={20} />
        <input
          type="text"
          placeholder="Ask a question about the content..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="search-input"
          required
        />
      </div>
      <button type="submit" className="search-button" disabled={isLoading}>
        {isLoading ? 'Searching...' : 'Search'}
      </button>
    </form>
  );
};
