import { useState } from 'react'
import { useMutation } from 'react-query'
import { Search as SearchIcon, FileText, ExternalLink } from 'lucide-react'
import { searchAPI } from '../api/client'

export default function Search() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])

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

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Semantic Search</h1>
        <p className="text-gray-600 mt-2">
          Find relevant research papers using natural language
        </p>
      </div>

      {/* Search Bar */}
      <div className="card mb-6">
        <form onSubmit={handleSearch} className="flex gap-3">
          <div className="flex-1 relative">
            <SearchIcon 
              className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" 
              size={20} 
            />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for papers, concepts, methods..."
              className="input pl-12"
            />
          </div>
          <button
            type="submit"
            disabled={searchMutation.isLoading || !query.trim()}
            className="btn-primary"
          >
            {searchMutation.isLoading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {/* Results */}
      {searchMutation.isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      )}

      {results.length > 0 && (
        <div>
          <div className="mb-4 text-sm text-gray-600">
            Found {results.length} relevant results
          </div>
          
          <div className="space-y-4">
            {results.map((result, index) => (
              <div key={result.chunk_id} className="card hover:shadow-md transition-shadow">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                      <span className="text-primary-700 font-semibold">
                        {index + 1}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {result.title}
                      </h3>
                      <span className="text-sm text-gray-500 ml-4">
                        Score: {(result.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    {result.authors && result.authors.length > 0 && (
                      <p className="text-sm text-gray-600 mb-3">
                        {result.authors.join(', ')}
                      </p>
                    )}
                    
                    <p className="text-gray-700 leading-relaxed mb-3">
                      {result.excerpt}
                    </p>
                    
                    <div className="flex items-center gap-2 text-sm text-primary-600">
                      <FileText size={14} />
                      <span>Document ID: {result.document_id}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!searchMutation.isLoading && results.length === 0 && query && (
        <div className="card text-center py-12">
          <SearchIcon className="mx-auto text-gray-400 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No results found
          </h3>
          <p className="text-gray-600">
            Try different keywords or upload more documents
          </p>
        </div>
      )}

      {!query && !searchMutation.isLoading && (
        <div className="card text-center py-12">
          <SearchIcon className="mx-auto text-gray-400 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Start searching
          </h3>
          <p className="text-gray-600">
            Enter a query to find relevant research papers
          </p>
        </div>
      )}
    </div>
  )
}
