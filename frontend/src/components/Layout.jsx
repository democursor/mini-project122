import { Link, useLocation } from 'react-router-dom'
import {
  LayoutGrid, Upload, FileText, Search,
  Network, MessageSquare, Sparkles, LogOut
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const navigation = [
  {
    section: 'Workspace',
    items: [
      { name: 'Dashboard',       path: '/',          icon: LayoutGrid },
      { name: 'Upload',          path: '/upload',    icon: Upload },
      { name: 'Documents',       path: '/documents', icon: FileText },
      { name: 'Search',          path: '/search',    icon: Search },
    ]
  },
  {
    section: 'AI Tools',
    items: [
      { name: 'Knowledge Graph', path: '/graph', icon: Network },
      { name: 'AI Assistant',    path: '/chat',  icon: MessageSquare },
    ]
  },
]

export default function Layout({ children }) {
  const location = useLocation()
  const { user, signOut } = useAuth()

  const displayName = user?.user_metadata?.full_name
    || user?.email?.split('@')[0]
    || 'Researcher'

  const initials = displayName
    .split(' ')
    .map(w => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--bg-base)' }}>
      {/* Background */}
      <div className="bg-orbs" />
      <div className="bg-orb-cyan" />

      {/* ── Sidebar ── */}
      <aside
        className="w-[232px] min-w-[232px] fixed h-full flex flex-col z-20"
        style={{
          background: 'rgba(8,8,18,0.92)',
          backdropFilter: 'blur(32px)',
          WebkitBackdropFilter: 'blur(32px)',
          borderRight: '1px solid var(--border-subtle)',
          boxShadow: '4px 0 32px rgba(0,0,0,0.4)',
        }}
      >
        {/* Sidebar top glow */}
        <div
          style={{
            position: 'absolute',
            top: 0, left: 0, right: 0,
            height: '1px',
            background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.5), rgba(245,158,11,0.5), transparent)',
          }}
        />

        {/* Logo */}
        <div
          className="px-4 pt-5 pb-4"
          style={{ borderBottom: '1px solid var(--border-subtle)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 relative"
              style={{
                background: 'linear-gradient(135deg, #F59E0B, #D97706)',
                boxShadow: '0 0 24px rgba(245,158,11,0.4)',
              }}
            >
              <Sparkles size={20} color="#06060E" />
              {/* Pulse ring */}
              <div
                style={{
                  position: 'absolute',
                  inset: '-4px',
                  borderRadius: '15px',
                  border: '1px solid rgba(245,158,11,0.3)',
                  animation: 'glow-pulse 3s ease-in-out infinite',
                }}
              />
            </div>
            <div>
              <div
                style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: '15px',
                  fontWeight: 800,
                  letterSpacing: '-0.3px',
                  background: 'linear-gradient(135deg, #EEEEFF, #A8A8CC)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Research AI
              </div>
              <div
                style={{
                  fontSize: '9px',
                  color: 'var(--text-faint)',
                  letterSpacing: '0.15em',
                  textTransform: 'uppercase',
                  fontFamily: 'var(--font-display)',
                }}
              >
                Literature Intel
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-5 overflow-y-auto">
          {navigation.map((group) => (
            <div key={group.section}>
              <div
                className="section-title px-3 mb-2"
                style={{ paddingLeft: '12px', marginBottom: '6px' }}
              >
                {group.section}
              </div>
              <div className="space-y-0.5">
                {group.items.map((item) => {
                  const Icon = item.icon
                  const isActive = location.pathname === item.path
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className={`nav-item${isActive ? ' active' : ''}`}
                    >
                      <Icon
                        size={15}
                        style={{
                          color: isActive ? 'var(--amber)' : 'currentColor',
                          flexShrink: 0,
                        }}
                      />
                      <span>{item.name}</span>
                    </Link>
                  )
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* User Profile Card */}
        <div
          className="px-3 pb-4"
          style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}
        >
          <div className="user-card">
            <div className="flex items-center gap-3">
              {/* Avatar */}
              <div className="user-avatar">
                {initials}
              </div>

              {/* Name + email */}
              <div className="flex-1 min-w-0">
                <div
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: '12px',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    letterSpacing: '-0.1px',
                  }}
                >
                  {displayName}
                </div>
                <div
                  style={{
                    fontSize: '10px',
                    color: 'var(--text-faint)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  {user?.email}
                </div>
              </div>

              {/* Logout button */}
              <button
                className="logout-btn"
                onClick={signOut}
                title="Sign out"
              >
                <LogOut size={13} style={{ color: 'var(--rose)' }} />
              </button>
            </div>
          </div>

          {/* Powered by */}
          <div className="flex items-center gap-2 mt-3 px-1">
            <div className="dot-pulse" style={{ width: '6px', height: '6px' }} />
            <span
              style={{
                fontSize: '10px',
                color: 'var(--text-faint)',
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.02em',
              }}
            >
              Gemini · ChromaDB · Neo4j
            </span>
          </div>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main
        className="flex-1 ml-[232px] relative z-10 page-enter"
        style={{ padding: '32px 36px', minHeight: '100vh', overflowY: 'auto' }}
      >
        {children}
      </main>
    </div>
  )
}
