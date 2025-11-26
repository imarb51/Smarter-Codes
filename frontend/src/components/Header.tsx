import React from 'react';
import { Globe } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="logo-container">
        <Globe className="logo-icon" size={32} />
        <h1>Semantic Search</h1>
      </div>
      <p className="subtitle">Index and search through web content with AI</p>
    </header>
  );
};
