export interface SearchResult {
  rank: number;
  score: number;
  text: string;
  url: string;
  chunk_id: number;
  token_count: number;
}

export interface SearchResponse {
  success: boolean;
  query: string;
  results: SearchResult[];
  total_results: number;
  processing?: {
    html_size: number;
    text_size: number;
    chunk_count: number;
    indexed_count: number;
  };
  error?: string;
  details?: string;
}

export interface IndexRequest {
  url: string;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  score_threshold?: number;
}

export interface IndexAndSearchRequest {
  url: string;
  query: string;
  top_k?: number;
  score_threshold?: number;
}
