<template>
  <div class="data-monitor p-6">
    <div class="flex justify-between items-center mb-8">
      <h1 class="text-3xl font-bold text-gray-100">Database Monitor</h1>
      <button 
        @click="triggerDailyUpdate"
        :disabled="triggeringUpdate"
        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center gap-2"
      >
        <svg v-if="triggeringUpdate" class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span v-if="triggeringUpdate">Running...</span>
        <span v-else>Update Data</span>
      </button>
    </div>
    
    <!-- Summary Stats -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <div class="bg-blue-900 border border-blue-700 rounded-lg p-4">
        <p class="text-sm text-blue-300 font-medium">Total Symbols</p>
        <p class="text-2xl font-bold text-blue-100">{{ totalSymbols }}</p>
      </div>
      <div class="bg-green-900 border border-green-700 rounded-lg p-4">
        <p class="text-sm text-green-300 font-medium">Total Records</p>
        <p class="text-2xl font-bold text-green-100">{{ formatNumber(totalRecords) }}</p>
      </div>
      <div class="bg-purple-900 border border-purple-700 rounded-lg p-4">
        <p class="text-sm text-purple-300 font-medium">Data Sources</p>
        <p class="text-2xl font-bold text-purple-100">{{ topics.length }}</p>
      </div>
      <div :class="systemStatus === 'Healthy' ? 'bg-green-900 border-green-700' : 'bg-orange-900 border-orange-700'" class="border rounded-lg p-4">
        <p :class="systemStatus === 'Healthy' ? 'text-green-300' : 'text-orange-300'" class="text-sm font-medium">System Status</p>
        <p :class="systemStatus === 'Healthy' ? 'text-green-100' : 'text-orange-100'" class="text-2xl font-bold">{{ systemStatus }}</p>
      </div>
    </div>
    
    <!-- Loading state -->
    <div v-if="loading" class="flex justify-center items-center h-64">
      <div class="spinner"></div>
    </div>
    
    <!-- Error state -->
    <div v-else-if="error" class="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded">
      <p class="font-bold">Error loading data</p>
      <p>{{ error.message }}</p>
    </div>
    
    <!-- Data coverage grid -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div v-for="topic in topics" :key="topic.name" class="card">
        <div class="flex justify-between items-start mb-4">
          <h3 class="text-xl font-semibold">{{ topic.name }}</h3>
          <span class="text-sm px-2 py-1 rounded" 
                :class="topic.status === 'healthy' ? 'bg-green-900 text-green-200' : 'bg-yellow-900 text-yellow-200'">
            {{ topic.status }}
          </span>
        </div>
        
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Records:</span>
            <span class="font-medium text-gray-200">{{ formatNumber(topic.recordCount) }}</span>
          </div>
          
          <div class="flex justify-between">
            <span class="text-gray-400">Date Range:</span>
            <span class="font-medium text-gray-200">{{ formatDateRange(topic.dateRange) }}</span>
          </div>
          
          <div class="flex justify-between">
            <span class="text-gray-400">Tables:</span>
            <span class="font-medium text-gray-200">{{ topic.tables.length }}</span>
          </div>
          
          <div v-if="topic.symbols" class="flex justify-between">
            <span class="text-gray-400">Symbols:</span>
            <span class="font-medium text-gray-200">{{ topic.symbols }}</span>
          </div>
          
          <div v-if="topic.sources" class="flex justify-between">
            <span class="text-gray-400">Sources:</span>
            <span class="font-medium text-gray-200">{{ topic.sources }}</span>
          </div>
          
          <div v-if="topic.eventTypes" class="flex justify-between">
            <span class="text-gray-400">Event Types:</span>
            <span class="font-medium text-gray-200">{{ topic.eventTypes }}</span>
          </div>
        </div>
        
        <!-- Details Button (for all data sources) -->
        <div class="mt-4">
          <button @click="showSymbolDetails(topic.name)" 
                  class="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium">
            <span v-if="topic.symbols && topic.symbols > 0">
              View {{ topic.symbols }} Symbol{{ topic.symbols > 1 ? 's' : '' }} Details
            </span>
            <span v-else-if="topic.name === 'News'">
              View News Sources
            </span>
            <span v-else-if="topic.name === 'Events'">
              View Event Types
            </span>
            <span v-else>
              View Details
            </span>
          </button>
        </div>
        
        <!-- Expandable details -->
        <div class="mt-4">
          <button @click="toggleDetails(topic.name)" 
                  class="text-blue-400 hover:text-blue-300 text-sm">
            {{ expandedTopics[topic.name] ? 'Hide' : 'Show' }} Details
          </button>
          
          <div v-if="expandedTopics[topic.name]" class="mt-3 pt-3 border-t">
            <div class="text-sm space-y-1">
              <p class="font-medium mb-2">Tables:</p>
              <ul class="list-disc list-inside text-gray-400">
                <li v-for="table in topic.tables" :key="table">{{ table }}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Notification -->
    <div v-if="notification" 
         class="fixed top-4 right-4 max-w-sm z-50 animate-slide-in"
         :class="notification.type === 'success' ? 'bg-green-900 border-green-700' : 'bg-red-900 border-red-700'"
         style="border-width: 1px; border-radius: 0.5rem; padding: 1rem;">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <svg v-if="notification.type === 'success'" class="h-5 w-5 text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
          </svg>
          <svg v-else class="h-5 w-5 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
          <div>
            <p class="font-medium" :class="notification.type === 'success' ? 'text-green-100' : 'text-red-100'">{{ notification.title }}</p>
            <p class="text-sm" :class="notification.type === 'success' ? 'text-green-300' : 'text-red-300'">{{ notification.message }}</p>
          </div>
        </div>
        <button @click="notification = null" class="text-gray-400 hover:text-gray-200">
          <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
    </div>
    
    <!-- Daily Update Summary -->
    <div class="mt-8 bg-gray-800 border border-gray-700 rounded-lg p-6">
      <h3 class="text-lg font-semibold mb-4 text-gray-100">Data Update Summary</h3>
      
      <div v-if="dailyUpdateSummary" class="space-y-3">
        <!-- Show running status prominently -->
        <div v-if="triggeringUpdate" class="bg-blue-900 border border-blue-700 rounded-lg p-4 mb-4">
          <div class="flex items-center gap-3">
            <svg class="animate-spin h-5 w-5 text-blue-300" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <div>
              <p class="text-blue-100 font-medium">Data update in progress...</p>
              <p class="text-sm text-blue-300">This may take a few minutes depending on the amount of data.</p>
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <p class="text-sm text-gray-400">Last Run</p>
            <p class="text-gray-200 font-medium">
              {{ formatLastRun(dailyUpdateSummary.last_run) }}
              <span v-if="dailyUpdateSummary.run_type" class="text-xs ml-2" 
                    :class="dailyUpdateSummary.run_type === 'manual' ? 'text-yellow-400' : 'text-blue-400'">
                ({{ dailyUpdateSummary.run_type }})
              </span>
            </p>
          </div>
          <div>
            <p class="text-sm text-gray-400">Status</p>
            <div class="flex items-center gap-2">
              <p class="font-medium" :class="getStatusClass(dailyUpdateSummary.status)">{{ formatStatus(dailyUpdateSummary.status) }}</p>
              <span v-if="triggeringUpdate && dailyUpdateSummary.status !== 'success' && dailyUpdateSummary.status !== 'failed'" 
                    class="text-xs text-gray-400">(updating...)</span>
            </div>
          </div>
          <div>
            <p class="text-sm text-gray-400">Records Processed</p>
            <p class="text-gray-200 font-medium">{{ formatNumber(dailyUpdateSummary.records_processed) }}</p>
          </div>
          <div>
            <p class="text-sm text-gray-400">Symbols Updated</p>
            <p class="text-gray-200 font-medium">{{ formatNumber(dailyUpdateSummary.symbols_updated) }}</p>
          </div>
        </div>
        
        <div v-if="dailyUpdateSummary.details && Object.keys(dailyUpdateSummary.details).length > 0" class="pt-3 border-t border-gray-700">
          <p class="text-sm font-medium text-gray-300 mb-2">Details:</p>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div v-for="(detail, key) in dailyUpdateSummary.details" :key="key" class="text-sm">
              <span class="text-gray-400">{{ formatDetailKey(key) }}:</span>
              <span class="text-gray-200 ml-2">{{ formatDetailValue(detail) }}</span>
            </div>
          </div>
        </div>
        
        <div v-if="dailyUpdateSummary.errors && dailyUpdateSummary.errors.length > 0" class="pt-3 border-t border-gray-700">
          <p class="text-sm font-medium text-red-400 mb-2">Errors:</p>
          <ul class="list-disc list-inside text-sm text-red-300">
            <li v-for="(error, index) in dailyUpdateSummary.errors" :key="index">{{ error }}</li>
          </ul>
        </div>
      </div>
      
      <div v-else class="text-sm text-gray-400">
        No daily update information available
      </div>
    </div>
    
    <!-- Symbol Details Modal -->
    <div v-if="showModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center" style="z-index: 9999;" @click.self="closeModal">
      <div class="bg-gray-800 rounded-lg shadow-2xl max-w-6xl w-full mx-4 max-h-[90vh] overflow-hidden border border-gray-700">
        <div class="p-6 border-b border-gray-700">
          <div class="flex justify-between items-center">
            <h2 class="text-2xl font-bold text-gray-100">{{ modalTitle }}</h2>
            <button @click="closeModal" class="text-gray-400 hover:text-gray-200">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
        
        <div class="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          <!-- Loading symbols -->
          <div v-if="loadingSymbols" class="flex justify-center items-center h-32">
            <div class="spinner"></div>
          </div>
          
          <!-- Symbol data table -->
          <div v-else-if="symbolData.length > 0" class="overflow-x-auto">
            <!-- Market Data Table -->
            <table v-if="symbolData[0].type === 'market'" class="min-w-full divide-y divide-gray-700">
              <thead class="bg-gray-900">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Records</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date Range</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trading Days</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Missing Days</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality</th>
                </tr>
              </thead>
              <tbody class="bg-gray-800 divide-y divide-gray-700">
                <tr v-for="symbol in symbolData" :key="symbol.symbol" class="hover:bg-gray-700">
                  <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-100">{{ symbol.symbol }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatNumber(symbol.record_count) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatDateRange(symbol.date_range) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatNumber(symbol.trading_days) || '-' }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ symbol.missing_days || '0' }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                          :class="getQualityClass(symbol.coverage_quality)">
                      {{ symbol.coverage_quality }}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
            
            <!-- Economic Data Table -->
            <table v-else-if="symbolData[0].type === 'economic'" class="min-w-full divide-y divide-gray-700">
              <thead class="bg-gray-900">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Indicator</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Data Points</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date Range</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Frequency</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Missing Points</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality</th>
                </tr>
              </thead>
              <tbody class="bg-gray-800 divide-y divide-gray-700">
                <tr v-for="symbol in symbolData" :key="symbol.symbol" class="hover:bg-gray-700">
                  <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-100">{{ symbol.symbol }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatNumber(symbol.data_points) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatDateRange(symbol.date_range) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ symbol.frequency || '-' }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ symbol.missing_points || '0' }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                          :class="getQualityClass(symbol.coverage_quality)">
                      {{ symbol.coverage_quality }}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
            
            <!-- News Sources Table -->
            <table v-else-if="symbolData[0].type === 'news'" class="min-w-full divide-y divide-gray-700">
              <thead class="bg-gray-900">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Articles</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date Range</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Days with Articles</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality</th>
                </tr>
              </thead>
              <tbody class="bg-gray-800 divide-y divide-gray-700">
                <tr v-for="source in symbolData" :key="source.source" class="hover:bg-gray-700">
                  <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-100">{{ source.source }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatNumber(source.article_count) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatDateRange(source.date_range) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ source.days_with_articles }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                          :class="getQualityClass(source.coverage_quality)">
                      {{ source.coverage_quality }}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
            
            <!-- Events Table -->
            <table v-else-if="symbolData[0].type === 'events'" class="min-w-full divide-y divide-gray-700">
              <thead class="bg-gray-900">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event Type</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date Range</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbols</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Countries</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality</th>
                </tr>
              </thead>
              <tbody class="bg-gray-800 divide-y divide-gray-700">
                <tr v-for="event in symbolData" :key="event.event_type" class="hover:bg-gray-700">
                  <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-100">{{ event.event_type }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatNumber(event.event_count) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ formatDateRange(event.date_range) }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ event.unique_symbols || '0' }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{{ event.unique_countries || '-' }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                          :class="getQualityClass(event.coverage_quality)">
                      {{ event.coverage_quality }}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <!-- No data -->
          <div v-else class="text-center text-gray-500 py-8">
            No symbol data available
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { monitorAPI, createWebSocket } from '../services/api'

// State
const topics = ref([])
const loading = ref(true)
const error = ref(null)
const lastUpdated = ref(new Date().toLocaleString())
const expandedTopics = ref({})
const showModal = ref(false)
const modalTitle = ref('')
const symbolData = ref([])
const loadingSymbols = ref(false)
const dailyUpdateSummary = ref(null)
const triggeringUpdate = ref(false)
const notification = ref(null)

let ws = null

// Computed properties
const totalSymbols = computed(() => {
  return topics.value.reduce((sum, topic) => sum + (topic.symbols || 0), 0)
})

const totalRecords = computed(() => {
  return topics.value.reduce((sum, topic) => sum + topic.recordCount, 0)
})

const systemStatus = computed(() => {
  const allHealthy = topics.value.every(topic => topic.status === 'healthy')
  return allHealthy ? 'Healthy' : 'Warning'
})

// Fetch coverage data
const fetchCoverage = async () => {
  try {
    loading.value = true
    error.value = null
    const data = await monitorAPI.getCoverage()
    
    // Transform data into topics
    topics.value = Object.entries(data).map(([name, info]) => ({
      name: formatTopicName(name),
      status: info.status || 'healthy',
      recordCount: info.record_count || 0,
      dateRange: info.date_range || {},
      tables: info.tables || [],
      symbols: info.symbols || null,
      sources: info.sources || null,
      eventTypes: info.event_types || null
    }))
    
    lastUpdated.value = new Date().toLocaleString()
  } catch (err) {
    error.value = err
    console.error('Error fetching coverage:', err)
  } finally {
    loading.value = false
  }
}

// Fetch daily update summary
const fetchDailyUpdateSummary = async () => {
  try {
    const summary = await monitorAPI.getDailyUpdateSummary()
    dailyUpdateSummary.value = summary
  } catch (err) {
    console.error('Error fetching daily update summary:', err)
  }
}

// Format helpers
const formatTopicName = (name) => {
  return name.split('_').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ')
}

const formatNumber = (num) => {
  if (!num) return '0'
  return num.toLocaleString()
}

const formatDateRange = (range) => {
  if (!range || !range.start || !range.end) return 'No data'
  
  // Handle both date formats: "2025-08-01" and "2025-01-01 05:00:00+00:00"
  let startDate, endDate
  
  if (range.start.includes('T') || range.start.includes(' ')) {
    // Already has time component, parse directly
    startDate = new Date(range.start)
    endDate = new Date(range.end)
  } else {
    // Plain date, add time to treat as local date
    startDate = new Date(range.start + 'T00:00:00')
    endDate = new Date(range.end + 'T00:00:00')
  }
  
  return `${startDate.toLocaleDateString()} - ${endDate.toLocaleDateString()}`
}

// Toggle expanded state
const toggleDetails = (topicName) => {
  expandedTopics.value[topicName] = !expandedTopics.value[topicName]
}

// Show symbol details modal
const showSymbolDetails = async (topicName) => {
  const schemaName = topicName.toLowerCase().replace(' ', '_')
  console.log('Fetching symbols for schema:', schemaName)
  console.log('Button clicked! Topic:', topicName)
  modalTitle.value = `${topicName} - Symbol Details`
  showModal.value = true
  loadingSymbols.value = true
  
  try {
    const response = await monitorAPI.getSymbolDetails(schemaName)
    console.log('Symbol details response:', response)
    symbolData.value = response.symbols || []
  } catch (err) {
    console.error('Error fetching symbol details:', err)
    symbolData.value = []
  } finally {
    loadingSymbols.value = false
  }
}

// Close modal
const closeModal = () => {
  showModal.value = false
  symbolData.value = []
}

// Format volume
const formatVolume = (volume) => {
  if (!volume) return '-'
  if (volume >= 1000000) {
    return (volume / 1000000).toFixed(1) + 'M'
  } else if (volume >= 1000) {
    return (volume / 1000).toFixed(1) + 'K'
  }
  return volume.toFixed(0)
}

// Get quality class for styling
const getQualityClass = (quality) => {
  switch (quality) {
    case 'excellent':
      return 'bg-green-900 text-green-200'
    case 'good':
      return 'bg-blue-900 text-blue-200'
    case 'fair':
      return 'bg-yellow-900 text-yellow-200'
    case 'poor':
      return 'bg-red-900 text-red-200'
    default:
      return 'bg-gray-700 text-gray-300'
  }
}

// Handle WebSocket messages
const handleWebSocketMessage = (data) => {
  if (data.type === 'coverage_update') {
    // Update specific topic
    const topicIndex = topics.value.findIndex(t => t.name === data.topic)
    if (topicIndex >= 0) {
      topics.value[topicIndex] = { ...topics.value[topicIndex], ...data.update }
    }
    lastUpdated.value = new Date().toLocaleString()
  }
}

// Format helpers for daily update summary
const formatLastRun = (lastRun) => {
  if (!lastRun || lastRun === 'None' || lastRun === null) return 'Never'
  const date = new Date(lastRun)
  if (isNaN(date.getTime())) return 'Never'
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  
  if (diffMins < 60) {
    return `${diffMins} minutes ago`
  } else if (diffHours < 24) {
    return `${diffHours} hours ago`
  } else if (diffDays < 7) {
    return `${diffDays} days ago`
  } else {
    return date.toLocaleString()
  }
}

const formatStatus = (status) => {
  if (!status) return 'Unknown'
  return status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')
}

const getStatusClass = (status) => {
  switch (status) {
    case 'success':
      return 'text-green-400'
    case 'failed':
      return 'text-red-400'
    case 'no_data':
      return 'text-yellow-400'
    default:
      return 'text-gray-400'
  }
}

const formatDetailKey = (key) => {
  return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
}

const formatDetailValue = (detail) => {
  if (!detail) return '-'
  if (detail.records !== undefined) {
    return `${formatNumber(detail.records)} records`
  }
  if (detail.symbols !== undefined) {
    return `${formatNumber(detail.symbols)} symbols`
  }
  if (detail.indicators !== undefined) {
    return `${formatNumber(detail.indicators)} indicators`
  }
  if (detail.articles !== undefined) {
    return `${formatNumber(detail.articles)} articles`
  }
  return JSON.stringify(detail)
}

// Trigger daily update
const triggerDailyUpdate = async () => {
  if (triggeringUpdate.value) return
  
  triggeringUpdate.value = true
  const initialStatus = dailyUpdateSummary.value?.status
  let checkCount = 0
  
  try {
    const response = await monitorAPI.triggerDailyUpdate()
    
    // Show success message (you could add a toast notification here)
    console.log('Daily update triggered:', response)
    
    // Function to check status
    const checkStatus = async () => {
      checkCount++
      const previousStatus = dailyUpdateSummary.value?.status
      await fetchDailyUpdateSummary()
      
      const currentStatus = dailyUpdateSummary.value?.status
      
      // Check if the run has completed
      if (currentStatus && currentStatus !== initialStatus) {
        if (currentStatus === 'success') {
          // Update completed successfully
          console.log('✅ Data update completed successfully!')
          notification.value = {
            type: 'success',
            title: 'Update Complete',
            message: `Successfully updated ${dailyUpdateSummary.value.symbols_updated || 0} symbols with ${dailyUpdateSummary.value.records_processed || 0} records`
          }
          triggeringUpdate.value = false
          // Refresh coverage data too
          await fetchCoverage()
          // Auto-hide notification after 5 seconds
          setTimeout(() => notification.value = null, 5000)
          return true
        } else if (currentStatus === 'failed') {
          // Update failed
          console.error('❌ Data update failed. Check errors in the summary.')
          notification.value = {
            type: 'error',
            title: 'Update Failed',
            message: dailyUpdateSummary.value.errors?.[0] || 'Check the summary below for error details'
          }
          triggeringUpdate.value = false
          // Auto-hide notification after 8 seconds
          setTimeout(() => notification.value = null, 8000)
          return true
        }
      }
      
      // Still running
      return false
    }
    
    // Check immediately after 3 seconds
    setTimeout(async () => {
      if (await checkStatus()) return
      
      // Then check every 5 seconds for the first minute
      const fastInterval = setInterval(async () => {
        if (await checkStatus() || checkCount > 12) {
          clearInterval(fastInterval)
          
          // If still running after 1 minute, check every 15 seconds
          if (triggeringUpdate.value) {
            const slowInterval = setInterval(async () => {
              if (await checkStatus() || checkCount > 40) {
                clearInterval(slowInterval)
                triggeringUpdate.value = false
              }
            }, 15000)
          }
        }
      }, 5000)
    }, 3000)
    
    // Force stop after 10 minutes
    setTimeout(() => {
      triggeringUpdate.value = false
      console.warn('⏱️ Update check timed out after 10 minutes')
    }, 600000)
    
  } catch (err) {
    console.error('Error triggering daily update:', err)
    triggeringUpdate.value = false
  }
}

// Lifecycle hooks
onMounted(() => {
  fetchCoverage()
  fetchDailyUpdateSummary()
  // Set up WebSocket for real-time updates
  ws = createWebSocket(handleWebSocketMessage)
  
  // Auto-refresh disabled to prevent flashing
  // To enable auto-refresh, uncomment the following:
  // const interval = setInterval(() => {
  //   fetchCoverage()
  //   fetchDailyUpdateSummary()
  // }, 30000)
  onUnmounted(() => {
    // clearInterval(interval)  // Commented out since auto-refresh is disabled
    if (ws) ws.close()
  })
})
</script>

<style scoped>
/* Component-specific styles if needed */
@keyframes slide-in {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.animate-slide-in {
  animation: slide-in 0.3s ease-out;
}
</style>