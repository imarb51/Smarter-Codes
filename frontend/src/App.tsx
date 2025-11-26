import { useState } from 'react';
import { Header } from './components/Header';
import { SearchForm } from './components/SearchForm';
import { ResultsList } from './components/ResultsList';
import { api } from './services/api';
import type { SearchResult } from './types';

function App() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (url: string, query: string) => {
    setIsLoading(true);
    setError(null);
    setResults([]);

    try {
      let response;
      if (url) {
        response = await api.indexAndSearch({ url, query });
      } else {
        response = await api.search({ query });
      }

      if (response.success) {
        setResults(response.results);
      } else {
        setError(response.error || 'An error occurred');
      }
    } catch (err) {
      setError('Failed to connect to the server');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      <main className="main-content">
        <SearchForm onSearch={handleSearch} isLoading={isLoading} />
        
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        <ResultsList results={results} />
      </main>
    </div>
  );
}

export default App;
