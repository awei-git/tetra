<template>
  <div class="h-full bg-gray-950 flex justify-center">
    <div class="chat-interface w-full max-w-4xl h-full flex flex-col">
    <!-- Compact Header -->
    <div class="bg-gray-900 shadow-sm border-b border-gray-800 px-4 py-2">
      <div class="flex items-center justify-between">
        <h1 class="text-lg font-bold text-gray-100">Database Chat</h1>
        <!-- View toggle -->
        <div class="flex items-center space-x-4 text-sm">
          <label class="flex items-center text-gray-300">
            <input type="radio" v-model="defaultViewType" value="table" class="mr-2">
            Table
          </label>
          <label class="flex items-center text-gray-300">
            <input type="radio" v-model="defaultViewType" value="chart" class="mr-2">
            Chart
          </label>
        </div>
      </div>
    </div>
    
    <!-- Main content area -->
    <div class="flex-1 flex flex-col overflow-hidden">
      <!-- Results viewing area -->
      <div class="flex-1 overflow-y-auto p-3 scroll-smooth relative" ref="messagesContainer">
        <div v-if="messages.length === 0" class="text-center text-gray-400 mt-8">
          <p class="text-lg mb-4">Ask me about your data!</p>
          <div class="space-y-2 text-sm">
            <p>Try questions like:</p>
            <p class="italic">"Show me AAPL price data for the last month"</p>
            <p class="italic">"What are the top performing stocks this week?"</p>
            <p class="italic">"Compare GDP growth over the last 10 years"</p>
            <p class="italic">"List all economic indicators we track"</p>
          </div>
        </div>
          
        <!-- Messages -->
        <div v-else class="space-y-3">
          <div v-for="(message, index) in messages" :key="index" :id="`message-${index}`">
            <!-- User message -->
            <div v-if="message.type === 'user'" class="mb-2">
              <div class="flex items-center gap-2 text-gray-400 text-sm mb-2">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                </svg>
                <span>You</span>
              </div>
              <div class="bg-gray-800 text-gray-100 rounded-lg px-4 py-2 inline-block">
                {{ message.content }}
              </div>
            </div>
              
            <!-- Assistant response -->
            <div v-else class="space-y-2">
              <div class="flex items-center gap-2 text-gray-400 text-xs mb-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                </svg>
                <span>Assistant</span>
              </div>
              
              <!-- SQL Query -->
              <div v-if="message.sql" class="bg-gray-800 border border-gray-700 rounded p-2">
                <div class="flex justify-between items-center mb-1">
                  <p class="font-semibold text-xs text-gray-300">SQL:</p>
                  <button @click="copySql(message.sql)" 
                          class="text-xs text-blue-400 hover:text-blue-300">
                    Copy
                  </button>
                </div>
                <pre class="language-sql overflow-x-auto bg-gray-900 p-2 rounded text-xs"><code>{{ message.sql }}</code></pre>
              </div>
                
              <!-- Results -->
              <div v-if="message.results">
                <!-- Table view -->
                <div v-if="message.viewType === 'table'" class="bg-gray-800 border border-gray-700 rounded shadow overflow-hidden">
                  <div class="overflow-x-auto max-h-64">
                    <table class="min-w-full text-xs">
                      <thead class="bg-gray-900 sticky top-0">
                        <tr>
                          <th v-for="col in message.columns" :key="col" 
                              class="px-2 py-1 text-left font-medium text-gray-400 uppercase">
                            {{ col }}
                          </th>
                        </tr>
                      </thead>
                      <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr v-for="(row, idx) in message.results.slice(0, 50)" :key="idx" class="hover:bg-gray-700">
                          <td v-for="col in message.columns" :key="col" 
                              class="px-2 py-1 whitespace-nowrap text-gray-300">
                            {{ formatCellValue(col, row[col]) }}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div v-if="message.results.length > 50" class="px-2 py-1 bg-gray-900 text-xs text-gray-400">
                    Showing first 50 of {{ message.results.length }} rows
                  </div>
                </div>
                  
                <!-- Chart view -->
                <div v-else-if="message.viewType === 'chart'" class="bg-gray-800 border border-gray-700 rounded shadow p-2">
                  <div :id="`chart-${index}`" class="w-full h-48"></div>
                </div>
              </div>
              
              <!-- Analysis/Explanation -->
              <div v-if="message.analysis" class="bg-blue-900 border border-blue-700 rounded p-2">
                <p class="text-xs text-blue-200">{{ message.analysis }}</p>
              </div>
              
              <!-- Error -->
              <div v-if="message.error" class="bg-red-900 border border-red-700 rounded p-2">
                <p class="text-red-300 font-semibold text-xs">Error:</p>
                <p class="text-red-200 text-xs">{{ message.error }}</p>
              </div>
            </div>
          </div>
        </div>
          
        <!-- Loading indicator -->
        <div v-if="loading" class="flex items-center space-x-2 text-gray-400">
          <div class="spinner w-4 h-4"></div>
          <span class="text-sm">Thinking...</span>
        </div>
        
        <!-- Scroll anchor -->
        <div ref="scrollAnchor" style="height: 1px;"></div>
      </div>
      
      <!-- Bottom area with input and history -->
      <div class="bg-gray-900 border-t border-gray-800">
        <!-- Query history (collapsed by default) -->
        <div v-if="showHistory" class="border-b border-gray-800 px-4 py-2 max-h-32 overflow-y-auto">
          <div v-if="queryHistory.length === 0" class="text-gray-500 text-xs">
            No history
          </div>
          <div v-else class="space-y-1">
            <div v-for="(item, index) in queryHistory.slice(0, 5)" 
                 :key="index"
                 @click="loadHistoryItem(item)"
                 class="py-1 text-xs text-gray-400 hover:text-gray-200 cursor-pointer truncate">
              {{ item.query }} <span class="text-gray-600">- {{ formatTime(item.timestamp) }}</span>
            </div>
          </div>
        </div>
        
        <!-- Input area -->
        <div class="p-2">
          <form @submit.prevent="sendQuery" class="flex space-x-1.5">
            <button @click.prevent="showHistory = !showHistory" type="button" class="text-gray-400 hover:text-gray-200 p-1">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
            </button>
            <input
              v-model="currentQuery"
              :disabled="loading"
              type="text"
              placeholder="Ask about your data..."
              class="flex-1 px-2 py-0.5 text-sm bg-gray-800 border border-gray-700 text-gray-100 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-transparent placeholder-gray-500"
            />
            <button
              type="submit"
              :disabled="loading || !currentQuery.trim()"
              class="bg-blue-600 text-white px-3 py-0.5 text-sm rounded hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, watch } from 'vue'
import { chatAPI } from '../services/api'
import Prism from 'prismjs'
import 'prismjs/components/prism-sql'
import Plotly from 'plotly.js-dist'

// State
const messages = ref([])
const currentQuery = ref('')
const loading = ref(false)
const queryHistory = ref([])
const defaultViewType = ref('table')
const messagesContainer = ref(null)
const scrollAnchor = ref(null)
const showHistory = ref(false)

// Send query to backend
const sendQuery = async () => {
  if (!currentQuery.value.trim() || loading.value) return
  
  const query = currentQuery.value.trim()
  currentQuery.value = ''
  
  // Add user message
  messages.value.push({
    type: 'user',
    content: query
  })
  
  // Immediate scroll after user message
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
  
  loading.value = true
  
  try {
    const response = await chatAPI.sendQuery(query)
    
    // Process response
    const assistantMessage = {
      type: 'assistant',
      sql: response.sql,
      results: response.data,
      columns: response.columns,
      analysis: response.analysis,
      viewType: determineViewType(response)
    }
    
    messages.value.push(assistantMessage)
    
    // Add to history
    queryHistory.value.unshift({
      query,
      sql: response.sql,
      timestamp: new Date()
    })
    
    // Save to backend
    await chatAPI.saveQuery(query, response.sql, response.data)
    
    // Render chart if needed
    if (assistantMessage.viewType === 'chart') {
      await nextTick()
      renderChart(messages.value.length - 1, response)
    }
    
    // Highlight SQL
    await nextTick()
    Prism.highlightAll()
    
    // Scroll to bottom to show latest message
    await nextTick()
    scrollToBottom()
    
  } catch (error) {
    messages.value.push({
      type: 'assistant',
      error: error.response?.data?.message || error.message
    })
    await nextTick()
    scrollToBottom()
  } finally {
    loading.value = false
  }
}

// Determine best view type for results
const determineViewType = (response) => {
  if (!response.data || response.data.length === 0) return 'table'
  
  // If user preference is set, use it
  if (defaultViewType.value === 'chart' && canRenderChart(response)) {
    return 'chart'
  }
  
  // Auto-detect based on data
  if (response.columns.includes('date') || response.columns.includes('timestamp')) {
    if (response.columns.some(col => ['price', 'close', 'value', 'amount'].includes(col.toLowerCase()))) {
      return 'chart'
    }
  }
  
  return 'table'
}

// Check if data can be rendered as chart
const canRenderChart = (response) => {
  return response.columns.length >= 2 && response.data.length > 1
}

// Render Plotly chart
const renderChart = (messageIndex, response) => {
  const chartDiv = document.getElementById(`chart-${messageIndex}`)
  if (!chartDiv) return
  
  // Detect x and y columns
  const xCol = response.columns.find(col => ['date', 'timestamp', 'time'].includes(col.toLowerCase())) || response.columns[0]
  const yCol = response.columns.find(col => ['price', 'close', 'value', 'amount'].includes(col.toLowerCase())) || response.columns[1]
  const volumeCol = response.columns.find(col => col.toLowerCase() === 'volume')
  
  const traces = []
  
  // Main price trace
  traces.push({
    x: response.data.map(row => row[xCol]),
    y: response.data.map(row => row[yCol]),
    type: 'scatter',
    mode: 'lines+markers',
    name: yCol,
    line: { color: '#3B82F6' },
    marker: { color: '#3B82F6' },
    yaxis: 'y'
  })
  
  // Add volume bars if volume column exists
  if (volumeCol) {
    traces.push({
      x: response.data.map(row => row[xCol]),
      y: response.data.map(row => row[volumeCol]),
      type: 'bar',
      name: 'Volume',
      yaxis: 'y2',
      marker: { 
        color: '#6B7280',
        opacity: 0.5
      }
    })
  }
  
  const layout = {
    title: {
      text: volumeCol ? `${yCol} and Volume over ${xCol}` : `${yCol} over ${xCol}`,
      font: { color: '#E5E7EB' }
    },
    xaxis: { 
      title: xCol,
      gridcolor: '#374151',
      titlefont: { color: '#9CA3AF' },
      tickfont: { color: '#9CA3AF' },
      type: xCol.toLowerCase().includes('date') || xCol.toLowerCase().includes('time') ? 'date' : 'linear'
    },
    yaxis: { 
      title: yCol,
      gridcolor: '#374151',
      titlefont: { color: '#9CA3AF' },
      tickfont: { color: '#9CA3AF' },
      side: 'left'
    },
    paper_bgcolor: '#1F2937',
    plot_bgcolor: '#111827',
    margin: { t: 40, r: 60, b: 40, l: 60 },
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      font: { color: '#9CA3AF' }
    }
  }
  
  // Add secondary y-axis for volume if present
  if (volumeCol) {
    layout.yaxis2 = {
      title: 'Volume (millions)',
      titlefont: { color: '#9CA3AF' },
      tickfont: { color: '#9CA3AF' },
      overlaying: 'y',
      side: 'right',
      showgrid: false,
      rangemode: 'tozero',
      tickformat: '.2f',  // Show as decimal since already in millions
      ticksuffix: 'M'
    }
    
    // Adjust the volume axis to take up bottom 30% of chart
    layout.yaxis.domain = [0.3, 1]
    layout.yaxis2.domain = [0, 0.25]
  }
  
  Plotly.newPlot(chartDiv, traces, layout, { responsive: true })
}

// Copy SQL to clipboard
const copySql = (sql) => {
  navigator.clipboard.writeText(sql)
  // Could add toast notification here
}

// Load history item
const loadHistoryItem = (item) => {
  currentQuery.value = item.query
}

// Format timestamp
const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date
  
  if (diff < 60000) return 'Just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return date.toLocaleDateString()
}

// Format cell value based on column name
const formatCellValue = (columnName, value) => {
  if (value === null || value === undefined) return '-'
  
  const colLower = columnName.toLowerCase()
  
  // Format volume columns (already in millions from backend)
  if (colLower === 'volume') {
    return `${parseFloat(value).toFixed(2)}M`
  }
  
  // Format price columns
  if (['open', 'high', 'low', 'close', 'price', 'vwap'].includes(colLower)) {
    return `$${parseFloat(value).toFixed(2)}`
  }
  
  // Format timestamps
  if (colLower === 'timestamp' || colLower.includes('date') || colLower.includes('time')) {
    const date = new Date(value)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString()
  }
  
  // Format percentages
  if (colLower.includes('percent') || colLower.includes('performance')) {
    return `${parseFloat(value).toFixed(2)}%`
  }
  
  return value
}

// Scroll to bottom of messages
const scrollToBottom = () => {
  if (messages.value.length > 0) {
    const lastMessageId = `message-${messages.value.length - 1}`
    const lastMessage = document.getElementById(lastMessageId)
    if (lastMessage) {
      lastMessage.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }
}

// Load query history on mount
onMounted(async () => {
  try {
    const history = await chatAPI.getHistory()
    queryHistory.value = history
  } catch (error) {
    console.error('Error loading history:', error)
  }
})

// Watch for SQL code blocks to highlight and scroll to bottom
watch(messages, () => {
  nextTick(() => {
    Prism.highlightAll()
    scrollToBottom()
  })
}, { deep: true })

// Also watch loading state to scroll when content finishes loading
watch(loading, (newVal, oldVal) => {
  if (oldVal === true && newVal === false) {
    // Loading just finished, scroll to bottom
    scrollToBottom()
  }
})
</script>

<style scoped>
/* Chat-specific styles */
.chat-interface {
  font-family: system-ui, -apple-system, sans-serif;
}

/* Ensure messages container is scrollable */
.flex-1.overflow-y-auto {
  min-height: 0;
  max-height: 100%;
  flex: 1 1 auto;
}

/* SQL syntax highlighting */
:deep(pre[class*="language-"]) {
  @apply text-sm;
}
</style>