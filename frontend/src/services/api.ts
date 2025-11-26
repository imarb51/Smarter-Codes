import type { IndexAndSearchRequest, SearchRequest, SearchResponse } from '../types';

// Base URL for the backend API. Can be overridden via REACT_APP_API_BASE_URL environment variable.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000/api';

/**
 * Helper to perform a POST request and parse JSON response.
 * Throws an error if the network request fails or the response is not ok.
 */
async function post<T>(endpoint: string, data: any): Promise<T> {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(`API error ${response.status}: ${errorBody}`);
        }
        return (await response.json()) as T;
    } catch (err) {
        console.error('Network or parsing error:', err);
        throw err;
    }
}

export const api = {
    search: async (data: SearchRequest): Promise<SearchResponse> => {
        return post<SearchResponse>('/search', data);
    },

    indexAndSearch: async (data: IndexAndSearchRequest): Promise<SearchResponse> => {
        return post<SearchResponse>('/index-and-search', data);
    },
};
