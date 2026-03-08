import { useState } from 'react'
import { useMutation, useQueryClient } from 'react-query'
import { Upload as UploadIcon, FileText, CheckCircle, Loader2, Sparkles, Brain, Database, Network } from 'lucide-react'
import toast from 'react-hot-toast'
import { documentsAPI } from '../api/client'

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const queryClient = useQueryClient()

  const processingStages = [
    { icon: FileText, label: 'Validating PDF', color: 'text-cyan-400' },
    { icon: FileText, label: 'Extracting Text', color: 'text-blue-400' },
    { icon: Sparkles, label: 'Semantic Chunking', color: 'text-purple-400' },
    { icon: Brain, label: 'Extracting Concepts', color: 'text-pink-400' },
    { icon: Network, label: 'Building Graph', color: 'text-violet-400' },
    { icon: Database, label: 'Generating Embeddings', color: 'text-cyan-400' },
  ]

  const uploadMutation = useMutation(documentsAPI.upload, {
    onSuccess: () => {
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
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold gradient-text neon-text mb-3">Upload Document</h1>
        <p className="text-gray-400 text-lg">
          Upload research papers in PDF format for AI-powered processing
        </p>
      </div>

      <div className="max-w-3xl mx-auto">
        {/* Main Upload Card */}
        <div className="card-gradient mb-6 border-2 border-slate-700/50">
          {/* Upload Area */}
          <div
            className={`
              border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-300 relative overflow-hidden
              ${dragActive ? 'border-cyan-500 bg-cyan-500/10 scale-105 shadow-lg shadow-cyan-500/20' : 'border-slate-600'}
              ${selectedFile ? 'bg-gradient-to-br from-emerald-900/30 to-green-900/20 border-emerald-500/50' : ''}
              ${uploadMutation.isLoading ? 'border-purple-500/50 bg-gradient-to-br from-purple-900/20 to-pink-900/20' : ''}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {uploadMutation.isLoading ? (
              /* Processing State */
              <div className="space-y-6">
                <div className="relative">
                  <Loader2 className="mx-auto text-cyan-400 animate-spin drop-shadow-lg" size={64} />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-20 h-20 rounded-full bg-cyan-500/20 animate-pulse"></div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-2xl font-bold text-gray-100 mb-2">
                    Processing Document
                  </h3>
                  <p className="text-gray-400 mb-6">
                    AI is analyzing your research paper...
                  </p>
                </div>
              </div>
            ) : selectedFile ? (
              /* File Selected State */
              <div className="space-y-4">
                <div className="relative inline-block">
                  <CheckCircle className="mx-auto text-emerald-400 drop-shadow-lg" size={64} />
                  <div className="absolute inset-0 bg-emerald-400 rounded-full blur-2xl opacity-20 animate-pulse"></div>
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-100 mb-2">
                    {selectedFile.name}
                  </p>
                  <p className="text-lg text-gray-400 mb-4">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="text-red-400 hover:text-red-300 font-medium hover:underline transition-all"
                  >
                    Remove file
                  </button>
                </div>
              </div>
            ) : (
              /* Upload State */
              <div className="space-y-6">
                <div className="relative inline-block">
                  <UploadIcon className="mx-auto text-gray-500" size={64} />
                  <div className="absolute inset-0 bg-cyan-400 rounded-full blur-3xl opacity-0 group-hover:opacity-20 transition-opacity"></div>
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-100 mb-2">
                    Drop your PDF here, or click to browse
                  </p>
                  <p className="text-gray-400 mb-6">
                    Maximum file size: 50 MB
                  </p>
                  <label className="btn-primary cursor-pointer inline-flex items-center gap-2 text-lg px-8 py-4">
                    <UploadIcon size={20} />
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    Select PDF File
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Upload Button */}
          {selectedFile && !uploadMutation.isLoading && (
            <button
              onClick={handleUpload}
              className="btn-primary w-full mt-6 text-lg py-4 flex items-center justify-center gap-2 neon-glow"
            >
              <Sparkles size={20} />
              Upload and Process with AI
            </button>
          )}
        </div>

        {/* Processing Info Card */}
        {!uploadMutation.isLoading && (
          <div className="card bg-gradient-to-br from-slate-800/80 to-blue-900/40 border-2 border-cyan-500/20">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg shadow-lg shadow-cyan-500/30">
                <Sparkles className="text-white" size={20} />
              </div>
              <h3 className="text-xl font-bold text-gray-100">
                AI Processing Pipeline
              </h3>
            </div>
            <ul className="space-y-3">
              {processingStages.map((stage, index) => {
                const StageIcon = stage.icon
                return (
                  <li key={index} className="flex items-center gap-3 text-gray-300">
                    <div className="p-2 bg-slate-800/80 rounded-lg shadow-sm border border-slate-700">
                      <StageIcon className={stage.color} size={18} />
                    </div>
                    <span className="font-medium">{stage.label}</span>
                  </li>
                )
              })}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
