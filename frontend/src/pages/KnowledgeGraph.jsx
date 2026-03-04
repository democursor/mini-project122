import { useState } from 'react'
import { useQuery } from 'react-query'
import { Network, FileText, Tag, TrendingUp } from 'lucide-react'
import { graphAPI } from '../api/client'

export default function KnowledgeGraph() {
  const [selectedPaper, setSelectedPaper] = useState(null)
  
  const { data: stats } = useQuery('graphStats', graphAPI.stats)
  const { data: papers } = useQuery('papers', () => graphAPI.papers(20))
  const { data: concepts } = useQuery('concepts', () => graphAPI.concepts(30))
  const { data: relatedPapers } = useQuery(
    ['relatedPapers', selectedPaper],
    () => graphAPI.relatedPapers(selectedPaper),
    { enabled: !!selectedPaper }
  )

  const statsData = stats?.data || {}
  const papersList = papers?.data || []
  const conceptsList = concepts?.data || []

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Knowledge Graph</h1>
        <p className="text-gray-600 mt-2">
          Explore relationships between papers and concepts
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Papers</p>
              <p className="text-2xl font-bold mt-1">{statsData.total_papers || 0}</p>
            </div>
            <FileText className="text-blue-500" size={32} />
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Concepts</p>
              <p className="text-2xl font-bold mt-1">{statsData.total_concepts || 0}</p>
            </div>
            <Tag className="text-green-500" size={32} />
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Mentions</p>
              <p className="text-2xl font-bold mt-1">{statsData.total_mentions || 0}</p>
            </div>
            <TrendingUp className="text-purple-500" size={32} />
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Relationships</p>
              <p className="text-2xl font-bold mt-1">{statsData.total_relationships || 0}</p>
            </div>
            <Network className="text-orange-500" size={32} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Papers */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <FileText size={20} />
            Papers in Graph
          </h2>
          
          {papersList.length === 0 ? (
            <p className="text-gray-500 text-center py-8">
              No papers in graph yet
            </p>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {papersList.map((paper) => (
                <div
                  key={paper.id}
                  onClick={() => setSelectedPaper(paper.id)}
                  className={`
                    p-3 rounded-lg cursor-pointer transition-all
                    ${selectedPaper === paper.id 
                      ? 'bg-primary-50 border-2 border-primary-500' 
                      : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'}
                  `}
                >
                  <h3 className="font-medium text-gray-900 mb-1">
                    {paper.title}
                  </h3>
                  {paper.authors && paper.authors.length > 0 && (
                    <p className="text-sm text-gray-600">
                      {paper.authors.join(', ')}
                    </p>
                  )}
                  {paper.year && (
                    <p className="text-xs text-gray-500 mt-1">
                      {paper.year}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Top Concepts */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Tag size={20} />
            Top Concepts
          </h2>
          
          {conceptsList.length === 0 ? (
            <p className="text-gray-500 text-center py-8">
              No concepts extracted yet
            </p>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {conceptsList.map((concept, index) => (
                <div
                  key={concept.name}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-medium text-gray-500">
                      #{index + 1}
                    </span>
                    <span className="font-medium text-gray-900">
                      {concept.name}
                    </span>
                  </div>
                  <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                    {concept.frequency}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Related Papers */}
      {selectedPaper && relatedPapers && (
        <div className="card mt-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Network size={20} />
            Related Papers
          </h2>
          
          {relatedPapers.data?.length === 0 ? (
            <p className="text-gray-500 text-center py-8">
              No related papers found
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {relatedPapers.data?.map((item) => (
                <div key={item.paper.id} className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-900 mb-2">
                    {item.paper.title}
                  </h3>
                  {item.paper.authors && item.paper.authors.length > 0 && (
                    <p className="text-sm text-gray-600 mb-2">
                      {item.paper.authors.join(', ')}
                    </p>
                  )}
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-primary-600">
                      {item.shared_concepts} shared concepts
                    </span>
                    <span className="text-gray-500">
                      {(item.similarity_score * 100).toFixed(0)}% similar
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
