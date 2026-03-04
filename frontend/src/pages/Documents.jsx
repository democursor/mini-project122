import { useQuery, useMutation, useQueryClient } from 'react-query'
import { FileText, Trash2, Calendar, User } from 'lucide-react'
import toast from 'react-hot-toast'
import { documentsAPI } from '../api/client'

export default function Documents() {
  const queryClient = useQueryClient()
  const { data, isLoading } = useQuery('documents', documentsAPI.list)

  const deleteMutation = useMutation(documentsAPI.delete, {
    onSuccess: () => {
      toast.success('Document deleted successfully')
      queryClient.invalidateQueries('documents')
    },
    onError: () => {
      toast.error('Failed to delete document')
    },
  })

  const handleDelete = (id, filename) => {
    if (window.confirm(`Delete "${filename}"?`)) {
      deleteMutation.mutate(id)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const documents = data?.data?.documents || []

  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Documents</h1>
          <p className="text-gray-600 mt-2">
            Manage your uploaded research papers
          </p>
        </div>
        <div className="text-sm text-gray-600">
          Total: {documents.length} documents
        </div>
      </div>

      {documents.length === 0 ? (
        <div className="card text-center py-12">
          <FileText className="mx-auto text-gray-400 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No documents yet
          </h3>
          <p className="text-gray-600 mb-4">
            Upload your first research paper to get started
          </p>
          <a href="/upload" className="btn-primary inline-block">
            Upload Document
          </a>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {documents.map((doc) => (
            <div key={doc.document_id} className="card hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4 flex-1">
                  <div className="p-3 bg-primary-50 rounded-lg">
                    <FileText className="text-primary-600" size={24} />
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 mb-1">
                      {doc.title || doc.filename}
                    </h3>
                    
                    {doc.authors && doc.authors.length > 0 && (
                      <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
                        <User size={14} />
                        <span>{doc.authors.join(', ')}</span>
                      </div>
                    )}
                    
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      {doc.year && (
                        <span className="flex items-center gap-1">
                          <Calendar size={14} />
                          {doc.year}
                        </span>
                      )}
                      {doc.pages && (
                        <span>{doc.pages} pages</span>
                      )}
                      {doc.upload_date && (
                        <span>
                          Uploaded: {new Date(doc.upload_date).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                    
                    <div className="mt-3">
                      <span className={`
                        px-3 py-1 rounded-full text-xs font-medium
                        ${doc.status === 'completed' ? 'bg-green-100 text-green-700' : 
                          doc.status === 'processing' ? 'bg-yellow-100 text-yellow-700' : 
                          doc.status === 'failed' ? 'bg-red-100 text-red-700' :
                          'bg-gray-100 text-gray-700'}
                      `}>
                        {doc.status}
                      </span>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={() => handleDelete(doc.document_id, doc.filename)}
                  disabled={deleteMutation.isLoading}
                  className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                  title="Delete document"
                >
                  <Trash2 size={20} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
