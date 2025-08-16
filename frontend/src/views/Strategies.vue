<template>
  <div class="min-h-screen bg-gray-900 text-gray-100 p-6">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-100">Strategy Performance</h1>
      <p class="text-gray-400 mt-2">Live backtest results from assessment pipeline</p>
    </div>

    <!-- Filters -->
    <div class="mb-6 flex gap-4">
      <select v-model="selectedCategory" @change="filterStrategies"
              class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-100">
        <option value="">All Categories</option>
        <option v-for="cat in categories" :key="cat.name" :value="cat.name">
          {{ formatCategoryName(cat.name) }} ({{ cat.count }})
        </option>
      </select>
      
      <button @click="refreshData" 
              class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
        </svg>
        Refresh
      </button>
    </div>

    <!-- Strategy List -->
    <div v-if="loading" class="flex justify-center items-center h-64">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="strategies.length === 0" class="text-center py-12">
      <p class="text-gray-400">No strategies found. Run the assessment pipeline to generate strategy data.</p>
    </div>
    
    <div v-else class="space-y-4">
      <div v-for="strategy in strategies" :key="strategy.strategy_name"
           class="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        
        <!-- Strategy Header -->
        <div @click="toggleStrategy(strategy.strategy_name)"
             class="p-4 cursor-pointer hover:bg-gray-750 transition-colors">
          <div class="flex items-center justify-between">
            <div class="flex-1">
              <div class="flex items-center gap-3">
                <span class="text-2xl font-bold text-yellow-400">
                  #{{ strategy.overall_rank || 'N/A' }}
                </span>
                <div class="flex-1">
                  <h3 class="text-lg font-semibold text-gray-100">
                    {{ strategy.strategy_name }}
                  </h3>
                  <div class="text-xs text-gray-400 mt-1">
                    {{ formatCategoryName(strategy.category) }}
                  </div>
                </div>
              </div>
              
              <!-- Key Metrics -->
              <div class="mt-3 grid grid-cols-6 gap-4 text-sm">
                <div>
                  <span class="text-gray-400">Total Return:</span>
                  <span :class="getReturnClass(strategy.total_return)" class="ml-2 font-bold">
                    {{ formatPercent(strategy.total_return) }}
                  </span>
                </div>
                <div>
                  <span class="text-gray-400">Annual:</span>
                  <span :class="getReturnClass(strategy.annualized_return)" class="ml-2 font-bold">
                    {{ formatPercent(strategy.annualized_return) }}
                  </span>
                </div>
                <div>
                  <span class="text-gray-400">Sharpe:</span>
                  <span class="text-gray-100 ml-2 font-bold">
                    {{ formatNumber(strategy.sharpe_ratio, 2) }}
                  </span>
                </div>
                <div>
                  <span class="text-gray-400">Max DD:</span>
                  <span class="text-red-400 ml-2 font-bold">
                    {{ formatPercent(strategy.max_drawdown) }}
                  </span>
                </div>
                <div>
                  <span class="text-gray-400">Win Rate:</span>
                  <span class="text-gray-100 ml-2 font-bold">
                    {{ formatPercent(strategy.win_rate) }}
                  </span>
                </div>
                <div>
                  <span class="text-gray-400">Trades:</span>
                  <span class="text-gray-100 ml-2 font-bold">
                    {{ strategy.total_trades }}
                  </span>
                </div>
              </div>
            </div>
            
            <!-- Expand/Collapse Icon -->
            <svg class="w-5 h-5 text-gray-400 transition-transform"
                 :class="{ 'rotate-180': expandedStrategies.includes(strategy.strategy_name) }"
                 fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
            </svg>
          </div>
        </div>
        
        <!-- Expanded Details -->
        <div v-if="expandedStrategies.includes(strategy.strategy_name)"
             class="border-t border-gray-700 p-4">
          
          <!-- Additional Metrics -->
          <div class="grid grid-cols-4 gap-4 mb-4">
            <div class="bg-gray-850 p-3 rounded-lg">
              <p class="text-xs text-gray-400">Volatility</p>
              <p class="text-lg font-medium text-gray-100">{{ formatPercent(strategy.volatility) }}</p>
            </div>
            <div class="bg-gray-850 p-3 rounded-lg">
              <p class="text-xs text-gray-400">Sortino Ratio</p>
              <p class="text-lg font-medium text-gray-100">{{ formatNumber(strategy.sortino_ratio, 2) || 'N/A' }}</p>
            </div>
            <div class="bg-gray-850 p-3 rounded-lg">
              <p class="text-xs text-gray-400">Calmar Ratio</p>
              <p class="text-lg font-medium text-gray-100">{{ formatNumber(strategy.calmar_ratio, 2) || 'N/A' }}</p>
            </div>
            <div class="bg-gray-850 p-3 rounded-lg">
              <p class="text-xs text-gray-400">Composite Score</p>
              <p class="text-lg font-medium text-gray-100">{{ formatNumber(strategy.composite_score, 1) }}</p>
            </div>
          </div>

          <!-- Metadata -->
          <div v-if="strategy.metadata" class="bg-gray-850 p-3 rounded-lg">
            <p class="text-xs text-gray-400 mb-2">Additional Metrics</p>
            <pre class="text-xs text-gray-300">{{ formatMetadata(strategy.metadata) }}</pre>
          </div>

          <!-- Last Run -->
          <div class="mt-4 text-xs text-gray-400">
            Last updated: {{ formatDate(strategy.last_run) }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { strategiesAPI } from '../services/api'

// State
const strategies = ref([])
const categories = ref([])
const loading = ref(true)
const selectedCategory = ref('')
const expandedStrategies = ref([])

// Fetch initial data
onMounted(async () => {
  await fetchCategories()
  await fetchStrategies()
})

// Fetch strategy categories
const fetchCategories = async () => {
  try {
    const response = await strategiesAPI.getCategories()
    if (response && response.data && response.data.categories) {
      categories.value = response.data.categories
    } else {
      categories.value = []
    }
  } catch (error) {
    console.error('Error fetching categories:', error)
  }
}

// Fetch strategies list
const fetchStrategies = async () => {
  try {
    loading.value = true
    const response = await strategiesAPI.getStrategiesList(selectedCategory.value)
    
    if (response && response.data && response.data.strategies) {
      strategies.value = response.data.strategies
      
      // Sort by overall rank
      strategies.value.sort((a, b) => (a.overall_rank || 999) - (b.overall_rank || 999))
    } else {
      strategies.value = []
    }
  } catch (error) {
    console.error('Error fetching strategies:', error)
  } finally {
    loading.value = false
  }
}

// Toggle strategy expansion
const toggleStrategy = (strategyName) => {
  const index = expandedStrategies.value.indexOf(strategyName)
  if (index > -1) {
    expandedStrategies.value.splice(index, 1)
  } else {
    expandedStrategies.value.push(strategyName)
  }
}

// Filter strategies
const filterStrategies = async () => {
  await fetchStrategies()
}

// Refresh data
const refreshData = async () => {
  loading.value = true
  await fetchCategories()
  await fetchStrategies()
  loading.value = false
}

// Formatting helpers
const formatPercent = (value) => {
  if (value === null || value === undefined) return 'N/A'
  return (value * 100).toFixed(2) + '%'
}

const formatNumber = (value, decimals = 2) => {
  if (value === null || value === undefined) return 'N/A'
  return value.toFixed(decimals)
}

const formatCategoryName = (category) => {
  if (!category) return 'Unknown'
  return category.split('_').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ')
}

const formatDate = (dateStr) => {
  if (!dateStr) return 'N/A'
  return new Date(dateStr).toLocaleString()
}

const formatMetadata = (metadata) => {
  if (!metadata) return ''
  try {
    const parsed = typeof metadata === 'string' ? JSON.parse(metadata) : metadata
    return JSON.stringify(parsed, null, 2)
  } catch {
    return metadata
  }
}

const getReturnClass = (value) => {
  if (value === null || value === undefined) return 'text-gray-400'
  return value >= 0 ? 'text-green-400' : 'text-red-400'
}
</script>

<style scoped>
.spinner {
  border: 3px solid #f3f3f3;
  border-top: 3px solid #3b82f6;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.bg-gray-850 {
  background-color: #1a1d29;
}

.bg-gray-750 {
  background-color: #2d3142;
}
</style>