import { Link, useLocation } from 'react-router-dom'
import { 
  Home, Upload, FileText, Search, Network, MessageSquare 
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
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 fixed h-full">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-primary-600">
            Research Platform
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            AI-Powered Literature Analysis
          </p>
        </div>
        
        <nav className="px-3 space-y-1">
          {navigation.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  flex items-center gap-3 px-3 py-2 rounded-lg transition-colors
                  ${isActive 
                    ? 'bg-primary-50 text-primary-700 font-medium' 
                    : 'text-gray-700 hover:bg-gray-50'
                  }
                `}
              >
                <Icon size={20} />
                <span>{item.name}</span>
              </Link>
            )
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="ml-64 flex-1 p-8">
        {children}
      </main>
    </div>
  )
}
