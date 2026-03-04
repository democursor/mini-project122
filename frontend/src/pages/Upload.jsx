import { useState } from 'react'
import { useMutation, useQueryClient } from 'react-query'
import { Upload as UploadIcon, FileText, CheckCircle, XCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import { documentsAPI } from '../api/client'

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const queryClient = useQueryClient()

  const uploadMutation = useMutation(documentsAPI.upload, {
    onSuccess: (data) => {
      toast.success('Document uploaded successfully!')
      setSelectedFile(null)
      queryClient.invalidateQueries('documents')
    },
    onError: (error) => {
      toast.error(error.response?.data?.detail || 'Upload failed')
    },
  })

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      if (file.type === 'application/pdf') {
        setSelectedFile(file)
      } else {
        toast.error('Please upload a PDF file')
      }
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      if (file.type === 'application/pdf') {
        setSelectedFile(file)
      } else {
        toast.error('Please upload a PDF file')
      }
    }
  }

  const handleUpload = () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile)
    }
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Upload Document</h1>
        <p className="text-gray-600 mt-2">
          Upload research papers in PDF format for processing
        </p>
      </div>

      <div className="max-w-2xl mx-auto">
        <div className="card">
          {/* Upload Area */}
          <div
            className={`
              border-2 border-dashed rounded-xl p-12 text-center transition-all
              ${dragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300'}
              ${selectedFile ? 'bg-green-50 border-green-500' : ''}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {selectedFile ? (
              <div>
                <CheckCircle className="mx-auto text-green-600 mb-4" size={48} />
                <p className="text-lg font-medium text-gray-900 mb-2">
                  {selectedFile.name}
                </p>
                <p className="text-sm text-gray-600 mb-4">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-red-600 hover:text-red-700 text-sm"
                >
                  Remove file
                </button>
              </div>
            ) : (
              <div>
                <UploadIcon className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-lg font-medium text-gray-900 mb-2">
                  Drop your PDF here, or click to browse
                </p>
                <p className="text-sm text-gray-600 mb-4">
                  Maximum file size: 50 MB
                </p>
                <label className="btn-primary cursor-pointer inline-block">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  Select PDF File
                </label>
              </div>
            )}
          </div>

          {/* Upload Button */}
          {selectedFile && (
            <button
              onClick={handleUpload}
              disabled={uploadMutation.isLoading}
              className="btn-primary w-full mt-6"
            >
              {uploadMutation.isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Uploading...
                </span>
              ) : (
                'Upload and Process'
              )}
            </button>
          )}

          {/* Processing Info */}
          <div className="mt-8 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium text-blue-900 mb-2">
              What happens after upload?
            </h3>
            <ul className="text-sm text-blue-800 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">1.</span>
                <span>PDF is validated and stored securely</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">2.</span>
                <span>Text and metadata are extracted</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">3.</span>
                <span>Document is chunked semantically</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">4.</span>
                <span>Concepts and entities are extracted</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">5.</span>
                <span>Knowledge graph is updated</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-0.5">6.</span>
                <span>Vector embeddings are generated</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
