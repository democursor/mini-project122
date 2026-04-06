import axios from 'axios'
import { supabase } from '../lib/supabase'

const API_BASE_URL = 'http://localhost:8000/api'

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor to include JWT token
client.interceptors.request.use(
  async (config) => {
    const { data: { session } } = await supabase.auth.getSession()
    if (session?.access_token) {
      config.headers.Authorization = `Bearer ${session.access_token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Add response interceptor to handle 401 errors
client.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Token might be expired, try to refresh
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        // No valid session, redirect to login
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

// Documents API
export const documentsAPI = {
  upload: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return client.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  list: () => client.get('/documents/'),
  get: (id) => client.get(`/documents/${id}`),
  delete: (id) => client.delete(`/documents/${id}`),
}

// Search API
export const searchAPI = {
  search: (query, topK = 10) => 
    client.post('/search/', { query, top_k: topK }),
  findSimilar: (documentId, topK = 10) => 
    client.get(`/search/similar/${documentId}`, { params: { top_k: topK } }),
}

// Graph API
export const graphAPI = {
  stats: () => client.get('/graph/stats'),
  papers: (limit = 50) => client.get('/graph/papers', { params: { limit } }),
  relatedPapers: (paperId, limit = 10) => 
    client.get(`/graph/papers/${paperId}/related`, { params: { limit } }),
  concepts: (limit = 50) => client.get('/graph/concepts', { params: { limit } }),
  searchConcept: (conceptName, limit = 10) => 
    client.post('/graph/concepts/search', { concept_name: conceptName, limit }),
}

// Chat API
export const chatAPI = {
  ask: (question, conversationHistory = []) => 
    client.post('/chat/', { question, conversation_history: conversationHistory }),
}

export default client
