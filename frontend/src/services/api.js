import axios from 'axios'

// Create axios instance with default config
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor for auth
api.interceptors.request.use(
  config => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  response => response.data,
  error => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Monitor API endpoints
export const monitorAPI = {
  getCoverage: () => api.get('/monitor/coverage'),
  getSchemas: () => api.get('/monitor/schemas'),
  getStats: (schema) => api.get(`/monitor/stats/${schema}`),
  getSymbolDetails: (schema) => api.get(`/monitor/symbols/${schema}`),
  getDailyUpdateSummary: () => api.get('/monitor/daily-update-summary'),
  triggerDailyUpdate: () => api.post('/monitor/trigger-daily-update')
}

// Chat API endpoints
export const chatAPI = {
  sendQuery: (query) => api.post('/chat/query', { query }),
  getHistory: (limit = 20) => api.get(`/chat/history?limit=${limit}`),
  saveQuery: (query, sql, results) => api.post('/chat/save', { query, sql, results })
}

// WebSocket connection for real-time updates
export const createWebSocket = (onMessage) => {
  const wsUrl = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/monitor')
  const ws = new WebSocket(wsUrl)
  
  ws.onopen = () => {
    console.log('WebSocket connected')
  }
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    onMessage(data)
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }
  
  ws.onclose = () => {
    console.log('WebSocket disconnected')
    // Reconnect after 5 seconds
    setTimeout(() => createWebSocket(onMessage), 5000)
  }
  
  return ws
}

export default api