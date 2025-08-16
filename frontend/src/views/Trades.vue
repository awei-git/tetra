<template>
  <div class="min-h-screen bg-gray-900 text-gray-100 p-6">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-100">Strategy Trade Recommendations</h1>
      <p class="text-gray-400 mt-2">Symbol-specific trades with returns, prices, and execution instructions</p>
    </div>

    <!-- Filters -->
    <div class="mb-6 flex gap-4 flex-wrap">
      <!-- Strategy Filter -->
      <select v-model="selectedStrategy" @change="filterTrades"
              class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-100">
        <option value="">All Strategies</option>
        <option v-for="strategy in strategies" :key="strategy" :value="strategy">
          {{ strategy }} ({{ getStrategyCount(strategy) }})
        </option>
      </select>
      
      <!-- Asset Class Filter -->
      <select v-model="selectedAssetClass" @change="filterTrades"
              class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-100">
        <option value="">All Asset Classes</option>
        <option v-for="ac in assetClasses" :key="ac" :value="ac">
          {{ ac }} ({{ getAssetClassCount(ac) }})
        </option>
      </select>
      
      <!-- Signal Filter -->
      <select v-model="selectedSignal" @change="filterTrades"
              class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-100">
        <option value="">All Signals</option>
        <option value="BUY">BUY Only ({{ getSignalCount('BUY') }})</option>
        <option value="SELL">SELL Only ({{ getSignalCount('SELL') }})</option>
        <option value="HOLD">HOLD Only ({{ getSignalCount('HOLD') }})</option>
      </select>
      
      <!-- Refresh Button -->
      <button @click="fetchTrades" 
              class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
        </svg>
        Refresh
      </button>
      
      <!-- Active Filters Display -->
      <div v-if="activeFiltersCount > 0" class="flex items-center gap-2 text-sm">
        <span class="text-gray-400">Active filters: {{ activeFiltersCount }}</span>
        <button @click="clearFilters" class="text-blue-400 hover:text-blue-300">
          Clear all
        </button>
      </div>
    </div>
    
    <!-- Scoring Formula Display -->
    <div class="mb-6 bg-gray-800 border border-gray-700 rounded-lg p-4">
      <h2 class="text-lg font-semibold text-yellow-400 mb-2">📊 Scoring Formula</h2>
      <p class="text-gray-300 font-mono">
        Score = AVG(30 Real Scenarios) × 100
      </p>
      <p class="text-sm text-gray-400 mt-1">
        Where each scenario score = (0.4×Sharpe + 0.3×Return/Vol + 0.2×WinRate + 0.1×(1-MaxDD))
      </p>
      <p class="text-xs text-gray-500 mt-1">
        Averaged across 30 real data scenarios, with 5 additional stress scenarios for analysis
      </p>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center items-center h-64">
      <div class="spinner"></div>
    </div>
    
    <!-- No Data State -->
    <div v-else-if="trades.length === 0" class="text-center py-12">
      <p class="text-gray-400">No trade recommendations available. Run the assessment pipeline to generate trade data.</p>
    </div>
    
    <!-- Trades List -->
    <div v-else class="space-y-4">
      <div v-for="(trade, index) in trades" :key="`${trade.strategy}-${trade.symbol}`"
           class="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        
        <!-- Trade Header -->
        <div class="p-4 border-b border-gray-700">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-4">
              <!-- Rank Badge -->
              <span class="text-2xl font-bold text-yellow-400">
                #{{ trade.rank || index + 1 }}
              </span>
              
              <!-- Strategy & Symbol -->
              <div>
                <h3 class="text-xl font-semibold text-gray-100">
                  {{ trade.strategy }}
                </h3>
                <p class="text-lg text-blue-400 font-medium">
                  {{ trade.symbol }}
                  <span v-if="trade.asset_class" class="text-sm text-gray-400 ml-2">
                    ({{ trade.asset_class }})
                  </span>
                </p>
              </div>
              
              <!-- Trade Signal -->
              <span :class="getSignalClass(trade.trade_type)"
                    class="px-3 py-1 rounded-full text-sm font-bold">
                {{ trade.trade_type }}
              </span>
            </div>
            
            <!-- Composite Score -->
            <div class="text-right">
              <p class="text-sm text-gray-400">Score</p>
              <p class="text-2xl font-bold" :class="getScoreClass(trade.score)">
                {{ trade.score.toFixed(1) }}
              </p>
            </div>
          </div>
        </div>
        
        <!-- Main Trade Info -->
        <div class="p-4 grid grid-cols-2 gap-6">
          <!-- Left Column: Prices & Returns -->
          <div>
            <!-- Price Information -->
            <div class="mb-4">
              <h4 class="text-sm font-semibold text-gray-400 mb-2">PRICE TARGETS</h4>
              <div class="grid grid-cols-2 gap-3">
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Current Price</p>
                  <p class="text-lg font-bold text-gray-100">${{ trade.current_price.toFixed(2) }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Target Price</p>
                  <p class="text-lg font-bold text-green-400">${{ trade.target_price.toFixed(2) }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Exit Price</p>
                  <p class="text-lg font-bold text-blue-400">${{ trade.exit_price.toFixed(2) }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Stop Loss</p>
                  <p class="text-lg font-bold text-red-400">${{ trade.stop_loss.toFixed(2) }}</p>
                </div>
              </div>
            </div>
            
            <!-- Returns -->
            <div>
              <h4 class="text-sm font-semibold text-gray-400 mb-2">EXPECTED RETURNS</h4>
              <div class="grid grid-cols-3 gap-3">
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">2 Week</p>
                  <p class="text-lg font-bold" :class="getReturnClass(trade.returns['2W'])">
                    {{ formatPercent(trade.returns['2W']) }}
                  </p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">1 Month</p>
                  <p class="text-lg font-bold" :class="getReturnClass(trade.returns['1M'])">
                    {{ formatPercent(trade.returns['1M']) }}
                  </p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">3 Month</p>
                  <p class="text-lg font-bold" :class="getReturnClass(trade.returns['3M'])">
                    {{ formatPercent(trade.returns['3M']) }}
                  </p>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Right Column: Execution & Metrics -->
          <div>
            <!-- Execution Instructions -->
            <div class="mb-4">
              <h4 class="text-sm font-semibold text-gray-400 mb-2">EXECUTION INSTRUCTIONS</h4>
              <div class="bg-gray-850 p-3 rounded">
                <p class="text-sm text-gray-200">{{ trade.execution }}</p>
                <div class="mt-2 flex items-center gap-4 text-xs">
                  <span class="text-gray-400">Position Size:</span>
                  <span class="text-gray-100 font-bold">{{ Math.abs(trade.position_size) }} shares</span>
                  <span class="text-gray-400">Signal Strength:</span>
                  <span class="text-gray-100 font-bold">{{ (trade.signal_strength * 100).toFixed(0) }}%</span>
                </div>
              </div>
            </div>
            
            <!-- Risk Metrics -->
            <div>
              <h4 class="text-sm font-semibold text-gray-400 mb-2">RISK METRICS</h4>
              <div class="grid grid-cols-2 gap-3">
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Sharpe Ratio</p>
                  <p class="text-md font-bold text-gray-100">{{ trade.metrics.sharpe_ratio.toFixed(2) }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Volatility</p>
                  <p class="text-md font-bold text-gray-100">{{ formatPercent(trade.metrics.volatility) }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Max Drawdown</p>
                  <p class="text-md font-bold text-red-400">{{ formatPercent(trade.metrics.max_drawdown) }}</p>
                </div>
                <div class="bg-gray-850 p-3 rounded">
                  <p class="text-xs text-gray-400">Win Probability</p>
                  <p class="text-md font-bold text-green-400">{{ formatPercent(trade.metrics.win_probability) }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Scenario Analysis Section -->
        <div class="border-t border-gray-700 p-4">
          <div class="flex items-center justify-between mb-3">
            <h4 class="text-sm font-semibold text-gray-400">SCENARIO ANALYSIS (35 Scenarios)</h4>
            
            <!-- Scenario Dropdown -->
            <select v-model="selectedScenarios[`${trade.strategy}-${trade.symbol}`]"
                    class="bg-gray-850 border border-gray-600 rounded px-3 py-1 text-sm text-gray-100">
              <option value="summary">Summary View</option>
              <option value="real">Real Data Scenarios (30)</option>
              <option value="stress">Stress Scenarios</option>
              <option value="all">All Scenarios</option>
            </select>
          </div>
          
          <!-- Score Components -->
          <div v-if="selectedScenarios[`${trade.strategy}-${trade.symbol}`] === 'summary'" 
               class="bg-gray-850 p-3 rounded">
            <div class="grid grid-cols-4 gap-4 mb-3">
              <div>
                <p class="text-xs text-gray-400">Sharpe Component</p>
                <p class="text-sm font-bold text-gray-100">
                  {{ trade.score_breakdown.sharpe_component?.toFixed(1) || 'N/A' }}
                </p>
              </div>
              <div>
                <p class="text-xs text-gray-400">Return/Vol Component</p>
                <p class="text-sm font-bold text-gray-100">
                  {{ trade.score_breakdown.return_vol_component?.toFixed(1) || 'N/A' }}
                </p>
              </div>
              <div>
                <p class="text-xs text-gray-400">Win Rate Component</p>
                <p class="text-sm font-bold text-gray-100">
                  {{ trade.score_breakdown.win_rate_component?.toFixed(1) || 'N/A' }}
                </p>
              </div>
              <div>
                <p class="text-xs text-gray-400">Drawdown Component</p>
                <p class="text-sm font-bold text-gray-100">
                  {{ trade.score_breakdown.drawdown_component?.toFixed(1) || 'N/A' }}
                </p>
              </div>
            </div>
            
            <!-- Real Scenario Stats -->
            <div v-if="trade.score_breakdown.real_scenario_avg" class="mt-3 pt-3 border-t border-gray-700">
              <p class="text-xs text-gray-400 mb-1">Real Data Scenarios (30)</p>
              <div class="flex items-center gap-4">
                <span class="text-sm">
                  <span class="text-gray-400">Avg Score:</span>
                  <span class="text-gray-100 font-bold ml-1">{{ trade.score_breakdown.real_scenario_avg.toFixed(2) }}</span>
                </span>
                <span class="text-sm">
                  <span class="text-gray-400">Std Dev:</span>
                  <span class="text-gray-100 font-bold ml-1">{{ trade.score_breakdown.real_scenario_std?.toFixed(2) || 'N/A' }}</span>
                </span>
              </div>
            </div>
          </div>
          
          <!-- Scenario Prices Table -->
          <div v-else-if="selectedScenarios[`${trade.strategy}-${trade.symbol}`] === 'real'" 
               class="bg-gray-850 p-3 rounded">
            <div class="grid grid-cols-3 gap-2 max-h-64 overflow-y-auto">
              <div v-for="(price, scenario) in filterScenarios(trade.scenario_prices, 'Real_')" 
                   :key="scenario" class="bg-gray-800 p-2 rounded">
                <p class="text-xs text-gray-400">{{ formatScenarioName(scenario) }}</p>
                <p class="text-sm font-bold text-gray-100">${{ price.toFixed(2) }}</p>
                <p class="text-xs" :class="getReturnClass(trade.scenarios[scenario])">
                  {{ formatPercent(trade.scenarios[scenario]) }}
                </p>
              </div>
            </div>
          </div>
          
          <!-- Stress Scenarios -->
          <div v-else-if="selectedScenarios[`${trade.strategy}-${trade.symbol}`] === 'stress'" 
               class="bg-gray-850 p-3 rounded">
            <div class="grid grid-cols-3 gap-2">
              <div v-for="(price, scenario) in filterStressScenarios(trade.scenario_prices)" 
                   :key="scenario" class="bg-gray-800 p-2 rounded">
                <p class="text-xs text-gray-400">{{ formatScenarioName(scenario) }}</p>
                <p class="text-sm font-bold text-gray-100">${{ price.toFixed(2) }}</p>
                <p class="text-xs" :class="getReturnClass(trade.scenarios[scenario])">
                  {{ formatPercent(trade.scenarios[scenario]) }}
                </p>
              </div>
            </div>
          </div>
          
          <!-- All Scenarios -->
          <div v-else-if="selectedScenarios[`${trade.strategy}-${trade.symbol}`] === 'all'" 
               class="bg-gray-850 p-3 rounded max-h-64 overflow-y-auto">
            <div class="grid grid-cols-4 gap-2">
              <div v-for="(price, scenario) in trade.scenario_prices" 
                   :key="scenario" class="bg-gray-800 p-2 rounded">
                <p class="text-xs text-gray-400 truncate">{{ formatScenarioName(scenario) }}</p>
                <p class="text-sm font-bold text-gray-100">${{ price.toFixed(2) }}</p>
                <p class="text-xs" :class="getReturnClass(trade.scenarios[scenario])">
                  {{ formatPercent(trade.scenarios[scenario]) }}
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Last Signal Date -->
        <div class="border-t border-gray-700 px-4 py-2 bg-gray-850">
          <p class="text-xs text-gray-400">
            Last Signal: {{ formatDate(trade.last_signal) }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, reactive, computed } from 'vue'
import { strategiesAPI } from '../services/api'

// State
const allTrades = ref([])
const trades = ref([])
const loading = ref(true)
const selectedScenarios = reactive({})
const selectedStrategy = ref('')
const selectedAssetClass = ref('')
const selectedSignal = ref('')

// Computed properties
const strategies = computed(() => {
  const strats = new Set()
  allTrades.value.forEach(trade => {
    if (trade.strategy) {
      strats.add(trade.strategy)
    }
  })
  return Array.from(strats).sort()
})

const assetClasses = computed(() => {
  const classes = new Set()
  allTrades.value.forEach(trade => {
    if (trade.asset_class) {
      classes.add(trade.asset_class)
    }
  })
  return Array.from(classes).sort()
})

const activeFiltersCount = computed(() => {
  let count = 0
  if (selectedStrategy.value) count++
  if (selectedAssetClass.value) count++
  if (selectedSignal.value) count++
  return count
})

// Get count functions
const getStrategyCount = (strategy) => {
  return allTrades.value.filter(t => t.strategy === strategy).length
}

const getAssetClassCount = (assetClass) => {
  return allTrades.value.filter(t => t.asset_class === assetClass).length
}

const getSignalCount = (signal) => {
  return allTrades.value.filter(t => t.trade_type === signal).length
}

// Filter trades based on selections
const filterTrades = () => {
  let filtered = [...allTrades.value]
  
  if (selectedStrategy.value) {
    filtered = filtered.filter(t => t.strategy === selectedStrategy.value)
  }
  
  if (selectedAssetClass.value) {
    filtered = filtered.filter(t => t.asset_class === selectedAssetClass.value)
  }
  
  if (selectedSignal.value) {
    filtered = filtered.filter(t => t.trade_type === selectedSignal.value)
  }
  
  trades.value = filtered
}

// Clear all filters
const clearFilters = () => {
  selectedStrategy.value = ''
  selectedAssetClass.value = ''
  selectedSignal.value = ''
  filterTrades()
}

// Fetch trades on mount
onMounted(async () => {
  await fetchTrades()
})

// Fetch trade recommendations
const fetchTrades = async () => {
  try {
    loading.value = true
    const response = await strategiesAPI.getStrategyTrades()
    
    if (response && response.data && response.data.trades) {
      allTrades.value = response.data.trades
      trades.value = response.data.trades
      
      // Initialize scenario dropdowns
      trades.value.forEach(trade => {
        selectedScenarios[`${trade.strategy}-${trade.symbol}`] = 'summary'
      })
    } else {
      allTrades.value = []
      trades.value = []
    }
  } catch (error) {
    console.error('Error fetching trades:', error)
    allTrades.value = []
    trades.value = []
  } finally {
    loading.value = false
  }
}

// Filter scenarios by prefix
const filterScenarios = (scenarios, prefix) => {
  const filtered = {}
  for (const [key, value] of Object.entries(scenarios)) {
    if (key.startsWith(prefix)) {
      filtered[key] = value
    }
  }
  return filtered
}

// Filter stress scenarios
const filterStressScenarios = (scenarios) => {
  const filtered = {}
  const stressNames = ['COVID_Crash', 'Fed_Pivot', 'AI_Boom', 'Rate_Shock', 'Tech_Bubble']
  for (const [key, value] of Object.entries(scenarios)) {
    if (stressNames.some(name => key.includes(name))) {
      filtered[key] = value
    }
  }
  return filtered
}

// Format scenario name
const formatScenarioName = (name) => {
  return name.replace(/_/g, ' ')
}

// Formatting helpers
const formatPercent = (value) => {
  if (value === null || value === undefined) return 'N/A'
  return (value * 100).toFixed(2) + '%'
}

const formatDate = (dateStr) => {
  if (!dateStr) return 'N/A'
  return new Date(dateStr).toLocaleString()
}

// Style helpers
const getSignalClass = (signal) => {
  switch(signal) {
    case 'BUY':
      return 'bg-green-600 text-white'
    case 'SELL':
      return 'bg-red-600 text-white'
    case 'HOLD':
      return 'bg-gray-600 text-white'
    default:
      return 'bg-gray-600 text-white'
  }
}

const getScoreClass = (score) => {
  if (score >= 80) return 'text-green-400'
  if (score >= 60) return 'text-yellow-400'
  if (score >= 40) return 'text-orange-400'
  return 'text-red-400'
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
</style>