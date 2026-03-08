import { Link, useLocation } from 'react-router-dom'
import { 
  Home, Upload, FileText, Search, Network, MessageSquare, Sparkles 
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', path: '/', icon: Home },
  { name: 'Upload', path: '/upload', icon: Upload },
  { name: 'Documents', path: '/documents', icon: FileText },
  { name: 'Search', path: '/search', icon: Search },
  { name: 'Knowledge Graph', path: '/graph', icon: Network },
  { name: 'AI Assistant', path: '/chat', icon: MessageSquare },
]

export default function Layout({ children }) {
  const location = useLocation()

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-950">
      {/* Sidebar */}
      <aside className="w-72 glass fixed h-full shadow-2xl border-r border-slate-700/50">
        <div className="p-6 border-b border-slate-700/50">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-500/30 neon-glow">
              <Sparkles className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-2xl font-bold gradient-text neon-text">
                Research AI
              </h1>
              <p className="text-xs text-cyan-400 font-medium">
                Literature Intelligence Platform
              </p>
            </div>
          </div>
        </div>
        
        <nav className="px-4 py-6 space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group
                  ${isActive 
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/30' 
                    : 'text-gray-300 hover:bg-slate-800/60 hover:shadow-md hover:text-cyan-400'
                  }
                `}
              >
                <Icon size={20} className={isActive ? '' : 'group-hover:scale-110 transition-transform'} />
                <span className="font-medium">{item.name}</span>
              </Link>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-700/50">
          <div className="text-xs text-gray-400 text-center">
            <p className="font-medium text-cyan-400">Powered by AI</p>
            <p className="text-gray-500">Google Gemini • ChromaDB • Neo4j</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="ml-72 flex-1 p-8">
        <div className="max-w-7xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}
