import { useState, useRef, useEffect } from 'react'
import { useMutation } from 'react-query'
import { Send, Bot, User, FileText, Trash2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { chatAPI } from '../api/client'

export default function Chat() {
  // Load messages from localStorage on mount
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('chatMessages')
    return saved ? JSON.parse(saved) : []
  })
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)

  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages))
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const chatMutation = useMutation(chatAPI.ask, {
    onSuccess: (data) => {
      const response = data.data
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.answer,
        citations: response.citations
      }])
    },
    onError: () => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        error: true
      }])
    },
  })

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      setMessages([])
      localStorage.removeItem('chatMessages')
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || chatMutation.isLoading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')

    const conversationHistory = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }))

    chatMutation.mutate(input, conversationHistory)
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">AI Research Assistant</h1>
          <p className="text-gray-600 mt-2">
            Ask questions about your research papers
          </p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={handleClearHistory}
            className="btn-secondary flex items-center gap-2"
          >
            <Trash2 size={18} />
            Clear History
          </button>
        )}
      </div>

      {/* Chat Container */}
      <div className="flex-1 card flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-center">
              <div>
                <Bot className="mx-auto text-gray-400 mb-4" size={48} />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Start a conversation
                </h3>
                <p className="text-gray-600 mb-4">
                  Ask me anything about your research papers
                </p>
                <div className="space-y-2 text-sm text-gray-600">
                  <p>Try asking:</p>
                  <div className="space-y-1">
                    <p className="text-primary-600">"What are the main findings?"</p>
                    <p className="text-primary-600">"Summarize the methodology"</p>
                    <p className="text-primary-600">"What datasets were used?"</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {message.role === 'assistant' && (
                    <div className="flex-shrink-0 w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                      <Bot size={18} className="text-primary-600" />
                    </div>
                  )}
                  
                  <div className={`
                    max-w-3xl rounded-lg p-4
                    ${message.role === 'user' 
                      ? 'bg-primary-600 text-white' 
                      : message.error
                        ? 'bg-red-50 text-red-900'
                        : 'bg-gray-100 text-gray-900'}
                  `}>
                    {message.role === 'assistant' ? (
                      <div>
                        <ReactMarkdown className="prose prose-sm max-w-none">
                          {message.content}
                        </ReactMarkdown>
                        
                        {message.citations && message.citations.length > 0 && (
                          <div className="mt-4 pt-4 border-t border-gray-300">
                            <p className="text-sm font-medium mb-2">Sources:</p>
                            <div className="space-y-2">
                              {message.citations.map((citation, idx) => (
                                <div key={idx} className="text-sm bg-white p-2 rounded">
                                  <div className="flex items-start gap-2">
                                    <FileText size={14} className="mt-0.5 text-primary-600" />
                                    <div>
                                      <p className="font-medium">{citation.title}</p>
                                      <p className="text-gray-600 text-xs mt-1">
                                        {citation.excerpt}
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p>{message.content}</p>
                    )}
                  </div>
                  
                  {message.role === 'user' && (
                    <div className="flex-shrink-0 w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                      <User size={18} className="text-white" />
                    </div>
                  )}
                </div>
              ))}
              
              {chatMutation.isLoading && (
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                    <Bot size={18} className="text-primary-600" />
                  </div>
                  <div className="bg-gray-100 rounded-lg p-4">
                    <div className="flex gap-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="input flex-1"
            disabled={chatMutation.isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || chatMutation.isLoading}
            className="btn-primary"
          >
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  )
}
