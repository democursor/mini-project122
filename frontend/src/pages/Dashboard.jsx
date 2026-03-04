import { useQuery } from 'react-query'
import { FileText, Network, Search as SearchIcon, MessageSquare } from 'lucide-react'
import { documentsAPI, graphAPI } from '../api/client'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const { data: documents } = useQuery('documents', documentsAPI.list)
  const { data: graphStats } = useQuery('graphStats', graphAPI.stats)

  const stats = [
    {
      name: 'Total Documents',
      value: documents?.data?.total || 0,
      icon: FileText,
      color: 'bg-blue-500',
      link: '/documents'
    },
    {
      name: 'Papers in Graph',
      value: graphStats?.data?.total_papers || 0,
      icon: Network,
      color: 'bg-green-500',
      link: '/graph'
    },
    {
      name: 'Concepts Extracted',
      value: graphStats?.data?.total_concepts || 0,
      icon: SearchIcon,
      color: 'bg-purple-500',
      link: '/graph'
    },
    {
      name: 'Relationships',
      value: graphStats?.data?.total_relationships || 0,
      icon: MessageSquare,
      color: 'bg-orange-500',
      link: '/graph'
    },
  ]

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Welcome to your research literature platform
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <Link
              key={stat.name}
              to={stat.link}
              className="card hover:shadow-md transition-shadow cursor-pointer"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">{stat.name}</p>
                  <p className="text-3xl font-bold mt-2">{stat.value}</p>
                </div>
                <div className={`${stat.color} p-3 rounded-lg`}>
                  <Icon className="text-white" size={24} />
                </div>
              </div>
            </Link>
          )
        })}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            to="/upload"
            className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all text-center"
          >
            <FileText className="mx-auto mb-2 text-primary-600" size={32} />
            <h3 className="font-medium">Upload Document</h3>
            <p className="text-sm text-gray-600 mt-1">
              Add new research papers
            </p>
          </Link>

          <Link
            to="/search"
            className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all text-center"
          >
            <SearchIcon className="mx-auto mb-2 text-primary-600" size={32} />
            <h3 className="font-medium">Semantic Search</h3>
            <p className="text-sm text-gray-600 mt-1">
              Find relevant papers
            </p>
          </Link>

          <Link
            to="/chat"
            className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all text-center"
          >
            <MessageSquare className="mx-auto mb-2 text-primary-600" size={32} />
            <h3 className="font-medium">AI Assistant</h3>
            <p className="text-sm text-gray-600 mt-1">
              Ask questions about papers
            </p>
          </Link>
        </div>
      </div>

      {/* Recent Documents */}
      <div className="card mt-6">
        <h2 className="text-xl font-semibold mb-4">Recent Documents</h2>
        {documents?.data?.documents?.length > 0 ? (
          <div className="space-y-3">
            {documents.data.documents.slice(0, 5).map((doc) => (
              <div
                key={doc.document_id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <FileText className="text-primary-600" size={20} />
                  <div>
                    <p className="font-medium">{doc.title || doc.filename}</p>
                    <p className="text-sm text-gray-600">
                      {doc.authors?.join(', ') || 'Unknown authors'}
                    </p>
                  </div>
                </div>
                <span className={`
                  px-3 py-1 rounded-full text-xs font-medium
                  ${doc.status === 'completed' ? 'bg-green-100 text-green-700' : 
                    doc.status === 'processing' ? 'bg-yellow-100 text-yellow-700' : 
                    'bg-gray-100 text-gray-700'}
                `}>
                  {doc.status}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No documents yet. Upload your first paper to get started!
          </p>
        )}
      </div>
    </div>
  )
}
