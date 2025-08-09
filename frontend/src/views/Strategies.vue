<template>
  <div class="min-h-screen bg-gray-900 text-gray-100 p-6">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-100">Strategy Performance</h1>
      <p class="text-gray-400 mt-2">Analyze and compare trading strategy performance across different market conditions</p>
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
      <p class="text-gray-400">No strategies found. Run the benchmark pipeline to generate strategy data.</p>
    </div>
    
    <div v-else class="space-y-4">
      <div v-for="strategy in strategies" :key="strategy.strategy_name"
           class="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        
        <!-- Strategy Header (Collapsible) -->
        <div @click="toggleStrategy(strategy.strategy_name)"
             class="p-4 cursor-pointer hover:bg-gray-750 transition-colors">
          <div class="flex items-center justify-between">
            <div class="flex-1">
              <div class="flex items-center gap-4">
                <h3 class="text-lg font-semibold text-gray-100">{{ strategy.strategy_name }}</h3>
                <span v-if="strategy.overall_rank" 
                      class="px-2 py-1 bg-blue-900 text-blue-300 rounded text-sm">
                  Rank #{{ strategy.overall_rank }}
                </span>
                <span class="px-2 py-1 bg-gray-700 text-gray-300 rounded text-sm">
                  {{ formatCategoryName(strategy.category) }}
                </span>
              </div>
              
              <!-- Quick Stats -->
              <div class="mt-2 grid grid-cols-5 gap-4 text-sm">
                <div>
                  <span class="text-gray-400">Total Return:</span>
                  <span :class="getReturnClass(strategy.total_return)" class="ml-2 font-medium">
                    {{ formatPercent(strategy.total_return) }}
                  </span>
                </div>
                <div>
                  <span class="text-gray-400">Sharpe:</span>
                  <span class="text-gray-100 ml-2 font-medium">{{ formatNumber(strategy.sharpe_ratio, 2) }}</span>
                </div>
                <div>
                  <span class="text-gray-400">Max DD:</span>
                  <span class="text-red-400 ml-2 font-medium">{{ formatPercent(strategy.max_drawdown) }}</span>
                </div>
                <div>
                  <span class="text-gray-400">Win Rate:</span>
                  <span class="text-gray-100 ml-2 font-medium">{{ formatPercent(strategy.win_rate) }}</span>
                </div>
                <div>
                  <span class="text-gray-400">Trades:</span>
                  <span class="text-gray-100 ml-2 font-medium">{{ strategy.total_trades }}</span>
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
             class="border-t border-gray-700">
          
          <!-- Controls -->
          <div class="p-4 bg-gray-850 border-b border-gray-700">
            <div class="flex gap-4 items-center">
              <div>
                <label class="text-sm text-gray-400 mr-2">Symbol:</label>
                <select v-model="strategySettings[strategy.strategy_name].symbol"
                        @change="updateStrategyPerformance(strategy.strategy_name)"
                        class="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm">
                  <option v-for="sym in availableSymbols" :key="sym" :value="sym">{{ sym }}</option>
                </select>
              </div>
              
              <div>
                <label class="text-sm text-gray-400 mr-2">Window Size:</label>
                <select v-model="strategySettings[strategy.strategy_name].windowSize"
                        @change="updateStrategyPerformance(strategy.strategy_name)"
                        class="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm">
                  <option :value="20">20 days</option>
                  <option :value="50">50 days</option>
                  <option :value="100">100 days</option>
                  <option :value="252">252 days (1 year)</option>
                </select>
              </div>
              
              <div>
                <label class="text-sm text-gray-400 mr-2">Period:</label>
                <select v-model="strategySettings[strategy.strategy_name].period"
                        @change="updateStrategyPerformance(strategy.strategy_name)"
                        class="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm">
                  <option value="1Y">1 Year</option>
                  <option value="3Y">3 Years</option>
                  <option value="5Y">5 Years</option>
                  <option value="10Y">10 Years</option>
                  <option value="ALL">All Time</option>
                </select>
              </div>
              
              <button @click="showScenarios(strategy.strategy_name)"
                      class="ml-auto px-4 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700 transition-colors">
                Market Scenarios
              </button>
            </div>
          </div>
          
          <!-- Performance Chart -->
          <div class="p-4">
            <div v-if="loadingPerformance[strategy.strategy_name]" class="flex justify-center py-8">
              <div class="spinner"></div>
            </div>
            
            <div v-else-if="performanceData[strategy.strategy_name]" class="space-y-6">
              <!-- Equity Curve Chart -->
              <div class="bg-gray-850 p-4 rounded-lg">
                <h4 class="text-sm font-medium text-gray-300 mb-4">Equity Curve</h4>
                <canvas :ref="`equityChart_${strategy.strategy_name}`" height="200"></canvas>
              </div>
              
              <!-- Rolling Metrics -->
              <div class="grid grid-cols-3 gap-4">
                <div class="bg-gray-850 p-4 rounded-lg">
                  <h4 class="text-sm font-medium text-gray-300 mb-4">Rolling Return (1Y)</h4>
                  <canvas :ref="`returnChart_${strategy.strategy_name}`" height="150"></canvas>
                </div>
                
                <div class="bg-gray-850 p-4 rounded-lg">
                  <h4 class="text-sm font-medium text-gray-300 mb-4">Rolling Sharpe (1Y)</h4>
                  <canvas :ref="`sharpeChart_${strategy.strategy_name}`" height="150"></canvas>
                </div>
                
                <div class="bg-gray-850 p-4 rounded-lg">
                  <h4 class="text-sm font-medium text-gray-300 mb-4">Rolling Drawdown</h4>
                  <canvas :ref="`drawdownChart_${strategy.strategy_name}`" height="150"></canvas>
                </div>
              </div>
              
              <!-- Additional Metrics -->
              <div class="grid grid-cols-4 gap-4">
                <div class="bg-gray-850 p-3 rounded-lg">
                  <p class="text-xs text-gray-400">Sortino Ratio</p>
                  <p class="text-lg font-medium text-gray-100">{{ formatNumber(strategy.sortino_ratio, 2) || 'N/A' }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded-lg">
                  <p class="text-xs text-gray-400">Calmar Ratio</p>
                  <p class="text-lg font-medium text-gray-100">{{ formatNumber(strategy.calmar_ratio, 2) || 'N/A' }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded-lg">
                  <p class="text-xs text-gray-400">Profit Factor</p>
                  <p class="text-lg font-medium text-gray-100">{{ formatNumber(strategy.profit_factor, 2) || 'N/A' }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded-lg">
                  <p class="text-xs text-gray-400">Expectancy</p>
                  <p class="text-lg font-medium text-gray-100">{{ formatCurrency(strategy.expectancy) || 'N/A' }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Market Scenarios Modal -->
    <div v-if="showScenariosModal" 
         class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
         @click.self="closeScenariosModal">
      <div class="bg-gray-800 rounded-lg shadow-2xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden">
        <div class="p-6 border-b border-gray-700">
          <h2 class="text-2xl font-bold text-gray-100">Market Scenario Analysis</h2>
          <p class="text-gray-400 mt-1">{{ currentScenarioStrategy }} performance in different market conditions</p>
        </div>
        
        <div class="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          <div v-if="loadingScenarios" class="flex justify-center py-12">
            <div class="spinner"></div>
          </div>
          
          <div v-else-if="scenarioData" class="space-y-6">
            <!-- Summary Stats -->
            <div class="grid grid-cols-4 gap-4">
              <div class="bg-gray-850 p-4 rounded-lg">
                <p class="text-sm text-gray-400">Avg Bear Return</p>
                <p class="text-xl font-medium" :class="getReturnClass(scenarioData.summary.avg_bear_return)">
                  {{ formatPercent(scenarioData.summary.avg_bear_return) }}
                </p>
              </div>
              <div class="bg-gray-850 p-4 rounded-lg">
                <p class="text-sm text-gray-400">Avg Bull Return</p>
                <p class="text-xl font-medium" :class="getReturnClass(scenarioData.summary.avg_bull_return)">
                  {{ formatPercent(scenarioData.summary.avg_bull_return) }}
                </p>
              </div>
              <div class="bg-gray-850 p-4 rounded-lg">
                <p class="text-sm text-gray-400">Bear Resilience</p>
                <p class="text-xl font-medium text-gray-100">
                  {{ formatPercent(scenarioData.summary.bear_market_resilience) }}
                </p>
              </div>
              <div class="bg-gray-850 p-4 rounded-lg">
                <p class="text-sm text-gray-400">Consistency</p>
                <p class="text-xl font-medium text-gray-100">
                  {{ formatNumber(scenarioData.summary.consistency_score, 2) }}
                </p>
              </div>
            </div>
            
            <!-- Scenario Details -->
            <div class="space-y-4">
              <div v-for="(scenario, key) in scenarioData.scenarios" :key="key"
                   class="bg-gray-850 rounded-lg p-4">
                <div class="flex items-center justify-between mb-3">
                  <div>
                    <h4 class="font-medium text-gray-100">{{ scenario.name }}</h4>
                    <p class="text-sm text-gray-400">
                      {{ scenario.start_date }} to {{ scenario.end_date }}
                    </p>
                  </div>
                  <span class="px-3 py-1 rounded text-sm"
                        :class="getScenarioTypeClass(scenario.type)">
                    {{ scenario.type }}
                  </span>
                </div>
                
                <div v-if="!scenario.error" class="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <span class="text-gray-400">Return:</span>
                    <span :class="getReturnClass(scenario.total_return)" class="ml-2 font-medium">
                      {{ formatPercent(scenario.total_return) }}
                    </span>
                  </div>
                  <div>
                    <span class="text-gray-400">Sharpe:</span>
                    <span class="text-gray-100 ml-2 font-medium">{{ formatNumber(scenario.sharpe_ratio, 2) }}</span>
                  </div>
                  <div>
                    <span class="text-gray-400">Max DD:</span>
                    <span class="text-red-400 ml-2 font-medium">{{ formatPercent(scenario.max_drawdown) }}</span>
                  </div>
                  <div>
                    <span class="text-gray-400">Win Rate:</span>
                    <span class="text-gray-100 ml-2 font-medium">{{ formatPercent(scenario.win_rate) }}</span>
                  </div>
                </div>
                
                <div v-else class="text-red-400 text-sm">
                  Error: {{ scenario.error }}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="p-6 border-t border-gray-700">
          <button @click="closeScenariosModal"
                  class="px-6 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors">
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, nextTick } from 'vue'
import { strategiesAPI } from '../services/api'
import Chart from 'chart.js/auto'

// State
const strategies = ref([])
const categories = ref([])
const loading = ref(true)
const selectedCategory = ref('')
const expandedStrategies = ref([])
const strategySettings = ref({})
const performanceData = ref({})
const loadingPerformance = ref({})
const chartInstances = ref({})

// Scenarios Modal
const showScenariosModal = ref(false)
const currentScenarioStrategy = ref('')
const scenarioData = ref(null)
const loadingScenarios = ref(false)

// Available symbols for backtesting
const availableSymbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'VXX', 'USO', 'UNG', 'SLV']

// Fetch initial data
onMounted(async () => {
  await fetchCategories()
  await fetchStrategies()
})

// Fetch strategy categories
const fetchCategories = async () => {
  try {
    const response = await strategiesAPI.getCategories()
    // API returns {success, data: {categories: [...]}}
    // Interceptor gives us the outer object
    if (response && response.data && response.data.categories) {
      categories.value = response.data.categories
    } else {
      categories.value = []
      console.error('Unexpected categories response:', response)
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
    console.log('API Response:', response) // Debug log
    console.log('Response type:', typeof response) // Debug log
    console.log('Response keys:', Object.keys(response || {})) // Debug log
    
    // API returns {success, data: {strategies: [...]}}
    // Interceptor gives us the outer object
    if (response && response.data && response.data.strategies) {
      strategies.value = response.data.strategies
    } else {
      strategies.value = []
      console.error('Unexpected response structure:', response)
    }
    
    console.log('Strategies loaded:', strategies.value.length) // Debug log
    
    // Initialize settings for each strategy
    strategies.value.forEach(strategy => {
      if (!strategySettings.value[strategy.strategy_name]) {
        strategySettings.value[strategy.strategy_name] = {
          symbol: 'SPY',
          windowSize: 252,
          period: '3Y'
        }
      }
    })
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
    // Clean up charts
    cleanupCharts(strategyName)
  } else {
    expandedStrategies.value.push(strategyName)
    // Fetch performance data
    updateStrategyPerformance(strategyName)
  }
}

// Update strategy performance data
const updateStrategyPerformance = async (strategyName) => {
  const settings = strategySettings.value[strategyName]
  if (!settings) return
  
  loadingPerformance.value[strategyName] = true
  
  try {
    // Calculate date range based on period
    const endDate = new Date()
    let startDate = new Date()
    
    switch (settings.period) {
      case '1Y':
        startDate.setFullYear(endDate.getFullYear() - 1)
        break
      case '3Y':
        startDate.setFullYear(endDate.getFullYear() - 3)
        break
      case '5Y':
        startDate.setFullYear(endDate.getFullYear() - 5)
        break
      case '10Y':
        startDate.setFullYear(endDate.getFullYear() - 10)
        break
      case 'ALL':
        startDate.setFullYear(2010)
        break
    }
    
    const response = await strategiesAPI.getStrategyPerformance(
      strategyName,
      settings.symbol,
      settings.windowSize,
      startDate.toISOString().split('T')[0],
      endDate.toISOString().split('T')[0]
    )
    
    // Check response structure and extract data
    if (response && response.data) {
      performanceData.value[strategyName] = response.data
    } else {
      performanceData.value[strategyName] = response
    }
    
    // Update charts
    await nextTick()
    updateCharts(strategyName)
    
  } catch (error) {
    console.error('Error fetching performance data:', error)
  } finally {
    loadingPerformance.value[strategyName] = false
  }
}

// Update charts for a strategy
const updateCharts = (strategyName) => {
  const data = performanceData.value[strategyName]
  if (!data || !data.equity_curve) return
  
  // Clean up existing charts
  cleanupCharts(strategyName)
  
  // Chart configuration
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff'
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#9CA3AF' }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#9CA3AF' }
      }
    }
  }
  
  // Equity Curve Chart
  const equityCanvas = document.querySelector(`canvas[ref="equityChart_${strategyName}"]`)
  if (equityCanvas) {
    const equityChart = new Chart(equityCanvas, {
      type: 'line',
      data: {
        labels: data.equity_curve.dates.map(d => new Date(d).toLocaleDateString()),
        datasets: [{
          data: data.equity_curve.values,
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          tension: 0.1,
          pointRadius: 0
        }]
      },
      options: chartOptions
    })
    chartInstances.value[`equity_${strategyName}`] = equityChart
  }
  
  // Rolling metrics charts
  if (data.rolling_metrics) {
    const metricsLabels = data.rolling_metrics.dates.map(d => new Date(d).toLocaleDateString())
    
    // Return Chart
    const returnCanvas = document.querySelector(`canvas[ref="returnChart_${strategyName}"]`)
    if (returnCanvas) {
      const returnChart = new Chart(returnCanvas, {
        type: 'line',
        data: {
          labels: metricsLabels,
          datasets: [{
            data: data.rolling_metrics.returns.map(r => r * 100),
            borderColor: '#10B981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            tension: 0.1,
            pointRadius: 0
          }]
        },
        options: {
          ...chartOptions,
          scales: {
            ...chartOptions.scales,
            y: {
              ...chartOptions.scales.y,
              ticks: {
                ...chartOptions.scales.y.ticks,
                callback: value => value + '%'
              }
            }
          }
        }
      })
      chartInstances.value[`return_${strategyName}`] = returnChart
    }
    
    // Sharpe Chart
    const sharpeCanvas = document.querySelector(`canvas[ref="sharpeChart_${strategyName}"]`)
    if (sharpeCanvas) {
      const sharpeChart = new Chart(sharpeCanvas, {
        type: 'line',
        data: {
          labels: metricsLabels,
          datasets: [{
            data: data.rolling_metrics.sharpe_ratios,
            borderColor: '#F59E0B',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            borderWidth: 2,
            tension: 0.1,
            pointRadius: 0
          }]
        },
        options: chartOptions
      })
      chartInstances.value[`sharpe_${strategyName}`] = sharpeChart
    }
    
    // Drawdown Chart
    const ddCanvas = document.querySelector(`canvas[ref="drawdownChart_${strategyName}"]`)
    if (ddCanvas) {
      const ddChart = new Chart(ddCanvas, {
        type: 'line',
        data: {
          labels: metricsLabels,
          datasets: [{
            data: data.rolling_metrics.max_drawdowns.map(d => d * 100),
            borderColor: '#EF4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderWidth: 2,
            tension: 0.1,
            pointRadius: 0,
            fill: true
          }]
        },
        options: {
          ...chartOptions,
          scales: {
            ...chartOptions.scales,
            y: {
              ...chartOptions.scales.y,
              ticks: {
                ...chartOptions.scales.y.ticks,
                callback: value => value + '%'
              }
            }
          }
        }
      })
      chartInstances.value[`drawdown_${strategyName}`] = ddChart
    }
  }
}

// Clean up charts
const cleanupCharts = (strategyName) => {
  const charts = ['equity', 'return', 'sharpe', 'drawdown']
  charts.forEach(type => {
    const key = `${type}_${strategyName}`
    if (chartInstances.value[key]) {
      chartInstances.value[key].destroy()
      delete chartInstances.value[key]
    }
  })
}

// Show market scenarios
const showScenarios = async (strategyName) => {
  currentScenarioStrategy.value = strategyName
  showScenariosModal.value = true
  loadingScenarios.value = true
  
  try {
    const settings = strategySettings.value[strategyName]
    const response = await strategiesAPI.getStrategyScenarios(
      strategyName,
      settings.symbol
    )
    // Check response structure and extract data
    if (response && response.data) {
      scenarioData.value = response.data
    } else {
      scenarioData.value = response
    }
  } catch (error) {
    console.error('Error fetching scenarios:', error)
  } finally {
    loadingScenarios.value = false
  }
}

// Close scenarios modal
const closeScenariosModal = () => {
  showScenariosModal.value = false
  currentScenarioStrategy.value = ''
  scenarioData.value = null
}

// Filter strategies
const filterStrategies = () => {
  fetchStrategies()
}

// Refresh data
const refreshData = () => {
  fetchStrategies()
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

const formatCurrency = (value) => {
  if (value === null || value === undefined) return 'N/A'
  return '$' + value.toFixed(2)
}

const formatCategoryName = (category) => {
  if (!category) return 'Unknown'
  return category.split('_').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ')
}

const getReturnClass = (value) => {
  if (value === null || value === undefined) return 'text-gray-400'
  return value >= 0 ? 'text-green-400' : 'text-red-400'
}

const getScenarioTypeClass = (type) => {
  const classes = {
    'bear': 'bg-red-900 text-red-300',
    'bull': 'bg-green-900 text-green-300',
    'neutral': 'bg-gray-700 text-gray-300',
    'volatile': 'bg-purple-900 text-purple-300'
  }
  return classes[type] || 'bg-gray-700 text-gray-300'
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