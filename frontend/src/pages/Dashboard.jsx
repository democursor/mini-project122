import { useQuery } from 'react-query'
import { useState, useEffect } from 'react'
import { FileText, Network, Tag, GitBranch, Upload, Eye, Search as SearchIcon, Trash2 } from 'lucide-react'
import { documentsAPI, graphAPI } from '../api/client'
import { Link, useNavigate } from 'react-router-dom'

// Animated counter component
function AnimatedCounter({ value, duration = 800 }) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let startTime
    let animationFrame

    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp
      const progress = Math.min((timestamp - startTime) / duration, 1)
      
      // easeOut animation
      const easeOut = 1 - Math.pow(1 - progress, 3)
      setCount(Math.floor(easeOut * value))
      
      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate)
      }
    }

    animationFrame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationFrame)
  }, [value, duration])

  return <span>{count}</span>
}

// Format relative time
function getRelativeTime(dateString) {
  if (!dateString) return 'Unknown'
  const date = new Date(dateString)
  const now = new Date()
  const diffInSeconds = Math.floor((now - date) / 1000)
  
  if (diffInSeconds < 60) return 'Just now'
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`
  return date.toLocaleDateString()
}

export default function Dashboard() {
  const { data: documents } = useQuery('documents', documentsAPI.list)
  const { data: graphStats } = useQuery('graphStats', graphAPI.stats)
  const [hoveredDoc, setHoveredDoc] = useState(null)
  const navigate = useNavigate()

  const stats = [
    {
      name: 'Total Documents',
      value: documents?.data?.total || 0,
      icon: FileText,
      borderColor: '#F59E0B'
    },
    {
      name: 'Papers in Graph',
      value: graphStats?.data?.total_papers || 0,
      icon: Network,
      borderColor: '#14B8A6'
    },
    {
      name: 'Concepts Extracted',
      value: graphStats?.data?.total_concepts || 0,
      icon: Tag,
      borderColor: '#8B5CF6'
    },
    {
      name: 'Relationships',
      value: graphStats?.data?.total_relationships || 0,
      icon: GitBranch,
      borderColor: '#F43F5E'
    },
  ]

  const quickActions = [
    {
      title: 'Upload Document',
      description: 'Add new research papers to your library',
      icon: Upload,
      iconBg: 'rgba(245,158,11,0.12)',
      iconColor: '#F59E0B',
      link: '/upload'
    },
    {
      title: 'Semantic Search',
      description: 'Find relevant papers with AI-powered search',
      icon: SearchIcon,
      iconBg: 'rgba(139,92,246,0.12)',
      iconColor: '#8B5CF6',
      link: '/search'
    },
    {
      title: 'AI Assistant',
      description: 'Ask questions and get insights from your papers',
      icon: FileText,
      iconBg: 'rgba(20,184,166,0.12)',
      iconColor: '#14B8A6',
      link: '/chat'
    },
  ]

  return (
    <div className="space-y-8">
      {/* Page Header - PROBLEM 1 FIXED */}
      <div>
        <h1 style={{ fontSize: '22px', fontWeight: 600, color: '#F5F4F0', letterSpacing: '-0.4px', marginBottom: '4px' }}>
          Dashboard
        </h1>
        <p style={{ fontSize: '13px', color: '#6B6A65' }}>
          Overview of your research library
        </p>
      </div>

      {/* Stats Cards Row - PROBLEM 2 FIXED */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <div
              key={stat.name}
              className="transition-fast"
              style={{
                background: '#141416',
                border: '1px solid rgba(255,255,255,0.06)',
                borderTop: `2px solid ${stat.borderColor}`,
                borderRadius: '12px',
                padding: '18px 20px'
              }}
            >
              <div className="flex items-start justify-between mb-2">
                <div 
                  style={{ 
                    fontSize: '11px',
                    textTransform: 'uppercase',
                    color: '#6B6A65',
                    letterSpacing: '0.6px'
                  }}
                >
                  {stat.name}
                </div>
                <Icon 
                  size={18} 
                  style={{ 
                    color: stat.borderColor,
                    opacity: 0.3
                  }} 
                />
              </div>
              <div 
                style={{ 
                  fontSize: '28px',
                  fontWeight: 600,
                  color: '#F5F4F0',
                  letterSpacing: '-1px',
                  marginTop: '8px'
                }}
              >
                <AnimatedCounter value={stat.value} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Quick Actions - PROBLEM 3 & 5 FIXED */}
      <div>
        <h2 
          style={{
            fontSize: '13px',
            fontWeight: 500,
            color: '#A8A79F',
            marginBottom: '16px'
          }}
        >
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {quickActions.map((action) => {
            const Icon = action.icon
            return (
              <Link
                key={action.title}
                to={action.link}
                className="transition-fast"
                style={{
                  background: '#141416',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '12px',
                  padding: '22px 20px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  display: 'block',
                  textDecoration: 'none'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(245,158,11,0.25)'
                  e.currentTarget.style.background = '#161618'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
                  e.currentTarget.style.background = '#141416'
                }}
              >
                <div 
                  style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '10px',
                    background: action.iconBg,
                    margin: '0 auto 12px auto',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <Icon size={24} style={{ color: action.iconColor }} />
                </div>
                <h3 
                  style={{
                    fontSize: '14px',
                    fontWeight: 500,
                    color: '#F5F4F0',
                    marginBottom: '4px'
                  }}
                >
                  {action.title}
                </h3>
                <p 
                  style={{
                    fontSize: '12px',
                    color: '#6B6A65'
                  }}
                >
                  {action.description}
                </p>
              </Link>
            )
          })}
        </div>
      </div>

      {/* Recent Documents - PROBLEM 4 & 5 FIXED */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 
            style={{
              fontSize: '13px',
              fontWeight: 500,
              color: '#A8A79F'
            }}
          >
            Recent Documents
          </h2>
          <Link 
            to="/documents"
            style={{
              fontSize: '12px',
              color: '#F59E0B',
              textDecoration: 'none'
            }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
          >
            View all →
          </Link>
        </div>

        {documents?.data?.documents?.length > 0 ? (
          <div className="space-y-2">
            {documents.data.documents.slice(0, 5).map((doc) => (
              <div
                key={doc.document_id}
                className="transition-fast"
                style={{
                  background: '#141416',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '10px',
                  padding: '12px 16px',
                  cursor: 'pointer'
                }}
                onMouseEnter={() => setHoveredDoc(doc.document_id)}
                onMouseLeave={() => setHoveredDoc(null)}
              >
                <div className="flex items-center gap-3">
                  {/* File Icon */}
                  <div 
                    style={{
                      width: '34px',
                      height: '34px',
                      borderRadius: '7px',
                      background: 'rgba(245,158,11,0.10)',
                      color: '#F59E0B',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0
                    }}
                  >
                    <FileText size={16} />
                  </div>

                  {/* File Info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div 
                      style={{
                        fontSize: '13px',
                        fontWeight: 500,
                        color: '#F5F4F0',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}
                      title={doc.title || doc.filename}
                    >
                      {doc.title || doc.filename}
                    </div>
                    <div 
                      style={{
                        fontSize: '12px',
                        color: '#6B6A65',
                        marginTop: '2px'
                      }}
                    >
                      Uploaded {getRelativeTime(doc.upload_date || doc.created_at)}
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div 
                    style={{
                      background: 'rgba(20,184,166,0.10)',
                      color: '#14B8A6',
                      border: '1px solid rgba(20,184,166,0.20)',
                      borderRadius: '20px',
                      padding: '3px 10px',
                      fontSize: '11px',
                      fontWeight: 500,
                      flexShrink: 0
                    }}
                  >
                    {doc.status || 'completed'}
                  </div>

                  {/* Action Icons - Show on hover */}
                  {hoveredDoc === doc.document_id && (
                    <div className="flex items-center gap-1.5" style={{ flexShrink: 0 }}>
                      <button 
                        style={{
                          width: '28px',
                          height: '28px',
                          borderRadius: '6px',
                          background: '#1A1A1E',
                          border: '1px solid rgba(255,255,255,0.06)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease'
                        }}
                        onClick={(e) => {
                          e.stopPropagation()
                          navigate(`/documents/${doc.document_id}`)
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(245,158,11,0.10)'
                          e.currentTarget.style.borderColor = 'rgba(245,158,11,0.20)'
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = '#1A1A1E'
                          e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
                        }}
                      >
                        <Eye size={14} style={{ color: '#6B6A65' }} />
                      </button>
                      <button 
                        style={{
                          width: '28px',
                          height: '28px',
                          borderRadius: '6px',
                          background: '#1A1A1E',
                          border: '1px solid rgba(255,255,255,0.06)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease'
                        }}
                        onClick={(e) => {
                          e.stopPropagation()
                          navigate('/search')
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(139,92,246,0.10)'
                          e.currentTarget.style.borderColor = 'rgba(139,92,246,0.20)'
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = '#1A1A1E'
                          e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
                        }}
                      >
                        <SearchIcon size={14} style={{ color: '#6B6A65' }} />
                      </button>
                      <button 
                        style={{
                          width: '28px',
                          height: '28px',
                          borderRadius: '6px',
                          background: '#1A1A1E',
                          border: '1px solid rgba(255,255,255,0.06)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease'
                        }}
                        onClick={(e) => {
                          e.stopPropagation()
                          // Handle delete
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(244,63,94,0.10)'
                          e.currentTarget.style.borderColor = 'rgba(244,63,94,0.20)'
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = '#1A1A1E'
                          e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
                        }}
                      >
                        <Trash2 size={14} style={{ color: '#6B6A65' }} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div 
            style={{
              background: '#141416',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '12px',
              padding: '40px 20px',
              textAlign: 'center'
            }}
          >
            <div 
              style={{
                width: '48px',
                height: '48px',
                margin: '0 auto 16px',
                borderRadius: '8px',
                background: '#1A1A1E',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <FileText size={24} style={{ color: '#3D3D3A' }} />
            </div>
            <div 
              style={{
                fontSize: '14px',
                color: '#6B6A65',
                marginBottom: '4px'
              }}
            >
              No documents yet
            </div>
            <div 
              style={{
                fontSize: '12px',
                color: '#3D3D3A',
                marginBottom: '16px'
              }}
            >
              Upload your first research paper to get started
            </div>
            <Link to="/upload">
              <button 
                className="btn-primary"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <Upload size={16} />
                Upload Paper
              </button>
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
