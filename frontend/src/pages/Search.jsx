import { useState } from 'react'
import { useMutation } from 'react-query'
import { Search as SearchIcon, FileText, MessageSquare } from 'lucide-react'
import { searchAPI } from '../api/client'
import { useNavigate } from 'react-router-dom'

export default function Search() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const navigate = useNavigate()

  const searchMutation = useMutation(searchAPI.search, {
    onSuccess: (data) => {
      setResults(data.data.results || [])
    },
  })

  const handleSearch = (e) => {
    e.preventDefault()
    if (query.trim()) {
      searchMutation.mutate(query)
    }
  }

  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion)
    searchMutation.mutate(suggestion)
  }

  const suggestions = [
    'tumor microenvironment',
    'immunotherapy',
    'precision medicine',
    'AI in oncology'
  ]

  // Highlight matched terms in excerpt
  const highlightText = (text, query) => {
    if (!query) return text
    const parts = text.split(new RegExp(`(${query})`, 'gi'))
    return parts.map((part, i) => 
      part.toLowerCase() === query.toLowerCase() ? (
        <mark 
          key={i}
          style={{
            background: 'rgba(245,158,11,0.15)',
            color: 'var(--accent-amber)',
            borderRadius: '3px',
            padding: '0 2px'
          }}
        >
          {part}
        </mark>
      ) : part
    )
  }

  return (
    <div>
      {/* Page Header */}
      <div className="mb-7">
        <h1 className="page-title mb-1">Semantic Search</h1>
        <p className="metadata">Find relevant research papers using natural language</p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch}>
        <div 
          className="card mb-4 transition-fast"
          style={{
            padding: '0',
            height: '52px',
            display: 'flex',
            alignItems: 'center',
            paddingLeft: '16px',
            paddingRight: '16px',
            gap: '12px',
            border: '1px solid var(--border-default)'
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = 'var(--accent-amber)'
            e.currentTarget.style.boxShadow = '0 0 0 3px rgba(245,158,11,0.08)'
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = 'var(--border-default)'
            e.currentTarget.style.boxShadow = 'none'
          }}
        >
          <SearchIcon size={18} style={{ color: 'var(--text-muted)' }} />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your research library..."
            className="flex-1 bg-transparent border-none outline-none text-base"
            style={{ 
              color: 'var(--text-primary)',
              fontSize: '16px'
            }}
          />
          <button 
            type="submit"
            className="btn-primary"
            style={{ padding: '7px 16px' }}
            disabled={!query.trim() || searchMutation.isLoading}
          >
            Search
          </button>
        </div>
      </form>

      {/* Suggestion Chips */}
      {!query && !searchMutation.isLoading && results.length === 0 && (
        <div className="mb-6">
          <div 
            className="text-xs mb-2"
            style={{ color: 'var(--text-muted)' }}
          >
            Try searching for:
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {suggestions.map((suggestion, i) => (
              <button
                key={i}
                onClick={() => handleSuggestionClick(suggestion)}
                className="text-xs px-3 py-1.5 rounded-full transition-fast"
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--accent-amber-border)'
                  e.currentTarget.style.color = 'var(--accent-amber)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-subtle)'
                  e.currentTarget.style.color = 'var(--text-secondary)'
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Empty State - Before Search */}
      {!query && !searchMutation.isLoading && results.length === 0 && (
        <div className="card text-center" style={{ padding: '60px 20px' }}>
          <div 
            className="relative w-32 h-32 mx-auto mb-6"
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            {/* Concentric circles */}
            <div 
              className="absolute inset-0 rounded-full"
              style={{ background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.08)' }}
            />
            <div 
              className="absolute inset-4 rounded-full"
              style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.12)' }}
            />
            <div 
              className="absolute inset-8 rounded-full"
              style={{ background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.18)' }}
            />
            <SearchIcon size={32} style={{ color: 'var(--accent-amber)', position: 'relative' }} />
          </div>
          <div 
            className="text-base font-medium mb-2"
            style={{ color: 'var(--text-primary)' }}
          >
            Search your research library
          </div>
          <div 
            className="text-sm"
            style={{ color: 'var(--text-muted)' }}
          >
            Ask anything using natural language
          </div>
        </div>
      )}

      {/* Loading State */}
      {searchMutation.isLoading && (
        <div className="space-y-2.5">
          {[1, 2, 3].map(i => (
            <div key={i} className="skeleton" style={{ height: '120px' }} />
          ))}
        </div>
      )}

      {/* Search Results */}
      {!searchMutation.isLoading && results.length > 0 && (
        <div className="space-y-2.5">
          {results.map((result, index) => {
            const relevanceScore = result.score ? Math.round(result.score * 100) : 0
            
            return (
              <div
                key={index}
                className="card transition-fast"
                style={{
                  padding: '16px 20px',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderLeft = '3px solid var(--accent-amber)'
                  e.currentTarget.style.paddingLeft = '17px'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderLeft = '1px solid var(--border-subtle)'
                  e.currentTarget.style.paddingLeft = '20px'
                }}
              >
                {/* Top Row */}
                <div className="flex items-start justify-between mb-2">
                  <div 
                    className="text-[13px] font-medium mono"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {result.document_title || result.filename || `Document ${result.document_id}`}
                  </div>
                  {relevanceScore > 0 && (
                    <div 
                      className="text-[11px] px-2 py-0.5 rounded-full flex-shrink-0 ml-3"
                      style={{
                        background: 'var(--accent-teal-bg)',
                        color: 'var(--accent-teal)'
                      }}
                    >
                      {relevanceScore}% match
                    </div>
                  )}
                </div>

                {/* Excerpt */}
                <div 
                  className="text-[13px] mb-3"
                  style={{ 
                    color: 'var(--text-secondary)',
                    lineHeight: '1.6'
                  }}
                >
                  {highlightText(result.text || result.excerpt || 'No excerpt available', query)}
                </div>

                {/* Bottom Row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {result.keywords?.slice(0, 3).map((keyword, i) => (
                      <span
                        key={i}
                        className="text-[10px] px-2 py-0.5 rounded-full"
                        style={{
                          background: 'var(--bg-elevated)',
                          color: 'var(--text-muted)',
                          border: '1px solid var(--border-subtle)'
                        }}
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                  <button
                    onClick={() => navigate('/chat')}
                    className="text-xs transition-fast flex items-center gap-1"
                    style={{ color: 'var(--accent-amber)' }}
                    onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
                    onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
                  >
                    Ask AI →
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* No Results */}
      {!searchMutation.isLoading && results.length === 0 && query && (
        <div className="card text-center" style={{ padding: '40px 20px' }}>
          <SearchIcon size={48} className="mx-auto mb-4" style={{ color: 'var(--text-faint)' }} />
          <div 
            className="text-sm mb-1"
            style={{ color: 'var(--text-muted)' }}
          >
            No results found
          </div>
          <div 
            className="text-xs"
            style={{ color: 'var(--text-faint)' }}
          >
            Try different keywords or upload more documents
          </div>
        </div>
      )}
    </div>
  )
}
