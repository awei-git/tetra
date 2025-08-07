<template>
  <div class="strategies-container">
    <div class="header">
      <h1>Strategy Backtests</h1>
      <div class="controls">
        <select v-model="selectedCategory" @change="fetchRankings" class="category-select">
          <option value="">All Categories</option>
          <option v-for="cat in categories" :key="cat.name" :value="cat.name">
            {{ cat.name }} ({{ cat.count }})
          </option>
        </select>
        <button @click="refreshData" class="refresh-btn" :disabled="loading">
          <i class="fas fa-sync-alt" :class="{ 'fa-spin': loading }"></i>
          Refresh
        </button>
      </div>
    </div>

    <!-- Summary Section -->
    <div v-if="summary" class="summary-section">
      <div class="summary-card">
        <h3>Backtest Summary</h3>
        <div class="summary-grid">
          <div class="summary-item">
            <span class="label">Total Strategies:</span>
            <span class="value">{{ summary.total_strategies }}</span>
          </div>
          <div class="summary-item">
            <span class="label">Successful:</span>
            <span class="value">{{ summary.successful_strategies }}</span>
          </div>
          <div class="summary-item">
            <span class="label">Avg Return:</span>
            <span class="value" :class="getReturnClass(summary.avg_return)">
              {{ formatPercent(summary.avg_return) }}
            </span>
          </div>
          <div class="summary-item">
            <span class="label">Best Return:</span>
            <span class="value positive">{{ formatPercent(summary.best_return) }}</span>
          </div>
          <div class="summary-item">
            <span class="label">Avg Sharpe:</span>
            <span class="value">{{ formatNumber(summary.avg_sharpe) }}</span>
          </div>
          <div class="summary-item">
            <span class="label">Best Sharpe:</span>
            <span class="value">{{ formatNumber(summary.best_sharpe) }}</span>
          </div>
        </div>
        <div class="run-info">
          <i class="fas fa-clock"></i>
          Last run: {{ formatDate(runDate) }}
          <span v-if="summary.execution_time">
            ({{ formatNumber(summary.execution_time) }}s)
          </span>
        </div>
      </div>
    </div>

    <!-- Rankings Table -->
    <div class="rankings-section">
      <h2>Strategy Rankings</h2>
      <div v-if="loading" class="loading">
        <i class="fas fa-spinner fa-spin"></i> Loading strategies...
      </div>
      <div v-else-if="rankings.length === 0" class="no-data">
        No strategy data available
      </div>
      <table v-else class="rankings-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Strategy</th>
            <th>Category</th>
            <th>Total Return</th>
            <th>Sharpe Ratio</th>
            <th>Max Drawdown</th>
            <th>Win Rate</th>
            <th>Trades</th>
            <th>Score</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="strategy in rankings" :key="strategy.strategy_name">
            <td class="rank">
              <span class="rank-badge" :class="getRankClass(strategy.overall_rank)">
                #{{ strategy.overall_rank }}
              </span>
            </td>
            <td class="strategy-name">{{ strategy.strategy_name }}</td>
            <td>
              <span class="category-badge" :class="getCategoryClass(strategy.category)">
                {{ strategy.category }}
              </span>
            </td>
            <td class="numeric" :class="getReturnClass(strategy.total_return)">
              {{ formatPercent(strategy.total_return) }}
            </td>
            <td class="numeric">{{ formatNumber(strategy.sharpe_ratio) }}</td>
            <td class="numeric negative">{{ formatPercent(strategy.max_drawdown) }}</td>
            <td class="numeric">{{ formatPercent(strategy.win_rate) }}</td>
            <td class="numeric">{{ strategy.total_trades }}</td>
            <td class="numeric">
              <span class="score">{{ formatNumber(strategy.composite_score) }}</span>
            </td>
            <td class="actions">
              <button 
                @click="viewDetails(strategy)" 
                class="action-btn"
                title="View Details"
              >
                <i class="fas fa-chart-line"></i>
              </button>
              <button 
                @click="compareStrategy(strategy)" 
                class="action-btn"
                title="Add to Comparison"
                :disabled="comparisonStrategies.length >= 5"
              >
                <i class="fas fa-plus"></i>
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Comparison Section -->
    <div v-if="comparisonStrategies.length > 0" class="comparison-section">
      <h2>Strategy Comparison</h2>
      <div class="comparison-controls">
        <div class="selected-strategies">
          <span 
            v-for="strategy in comparisonStrategies" 
            :key="strategy"
            class="strategy-chip"
          >
            {{ strategy }}
            <i class="fas fa-times" @click="removeFromComparison(strategy)"></i>
          </span>
        </div>
        <button @click="clearComparison" class="clear-btn">Clear All</button>
      </div>
      <div class="comparison-chart">
        <canvas ref="comparisonChart"></canvas>
      </div>
    </div>

    <!-- Strategy Details Modal -->
    <div v-if="selectedStrategy" class="modal-overlay" @click="closeDetails">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h2>{{ selectedStrategy.strategy_name }}</h2>
          <button @click="closeDetails" class="close-btn">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="modal-body">
          <div class="details-grid">
            <div class="detail-item">
              <span class="label">Total Return:</span>
              <span class="value" :class="getReturnClass(selectedStrategy.total_return)">
                {{ formatPercent(selectedStrategy.total_return) }}
              </span>
            </div>
            <div class="detail-item">
              <span class="label">Annualized Return:</span>
              <span class="value" :class="getReturnClass(selectedStrategy.annualized_return)">
                {{ formatPercent(selectedStrategy.annualized_return) }}
              </span>
            </div>
            <div class="detail-item">
              <span class="label">Sharpe Ratio:</span>
              <span class="value">{{ formatNumber(selectedStrategy.sharpe_ratio) }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Sortino Ratio:</span>
              <span class="value">{{ formatNumber(selectedStrategy.metadata?.sortino_ratio) }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Max Drawdown:</span>
              <span class="value negative">{{ formatPercent(selectedStrategy.max_drawdown) }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Volatility:</span>
              <span class="value">{{ formatPercent(selectedStrategy.volatility) }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Win Rate:</span>
              <span class="value">{{ formatPercent(selectedStrategy.win_rate) }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Total Trades:</span>
              <span class="value">{{ selectedStrategy.total_trades }}</span>
            </div>
          </div>
          <div class="equity-curve-section">
            <h3>Equity Curve</h3>
            <canvas ref="equityChart"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import Chart from 'chart.js/auto'

export default {
  name: 'Strategies',
  setup() {
    const rankings = ref([])
    const summary = ref(null)
    const categories = ref([])
    const selectedCategory = ref('')
    const loading = ref(false)
    const runDate = ref(null)
    const selectedStrategy = ref(null)
    const comparisonStrategies = ref([])
    const comparisonChart = ref(null)
    const equityChart = ref(null)

    let comparisonChartInstance = null
    let equityChartInstance = null

    const fetchLatestResults = async () => {
      loading.value = true
      try {
        const response = await axios.get('/api/strategies/latest-results')
        if (response.data.success) {
          rankings.value = response.data.data.results
          runDate.value = response.data.data.run_date
        }
      } catch (error) {
        console.error('Error fetching results:', error)
      } finally {
        loading.value = false
      }
    }

    const fetchSummary = async () => {
      try {
        const response = await axios.get('/api/strategies/summary')
        if (response.data.success) {
          summary.value = response.data.data.summary
        }
      } catch (error) {
        console.error('Error fetching summary:', error)
      }
    }

    const fetchCategories = async () => {
      try {
        const response = await axios.get('/api/strategies/categories')
        if (response.data.success) {
          categories.value = response.data.data.categories
        }
      } catch (error) {
        console.error('Error fetching categories:', error)
      }
    }

    const fetchRankings = async () => {
      loading.value = true
      try {
        const params = {}
        if (selectedCategory.value) {
          params.category = selectedCategory.value
        }
        const response = await axios.get('/api/strategies/rankings', { params })
        if (response.data.success) {
          rankings.value = response.data.data.rankings
        }
      } catch (error) {
        console.error('Error fetching rankings:', error)
      } finally {
        loading.value = false
      }
    }

    const viewDetails = async (strategy) => {
      selectedStrategy.value = strategy
      // Fetch equity curve
      try {
        const response = await axios.get(`/api/strategies/equity-curve/${strategy.strategy_name}`)
        if (response.data.success) {
          const { dates, values } = response.data.data
          drawEquityChart(dates, values)
        }
      } catch (error) {
        console.error('Error fetching equity curve:', error)
      }
    }

    const closeDetails = () => {
      selectedStrategy.value = null
      if (equityChartInstance) {
        equityChartInstance.destroy()
        equityChartInstance = null
      }
    }

    const compareStrategy = (strategy) => {
      if (!comparisonStrategies.value.includes(strategy.strategy_name)) {
        comparisonStrategies.value.push(strategy.strategy_name)
        updateComparisonChart()
      }
    }

    const removeFromComparison = (strategyName) => {
      comparisonStrategies.value = comparisonStrategies.value.filter(s => s !== strategyName)
      if (comparisonStrategies.value.length === 0 && comparisonChartInstance) {
        comparisonChartInstance.destroy()
        comparisonChartInstance = null
      } else {
        updateComparisonChart()
      }
    }

    const clearComparison = () => {
      comparisonStrategies.value = []
      if (comparisonChartInstance) {
        comparisonChartInstance.destroy()
        comparisonChartInstance = null
      }
    }

    const updateComparisonChart = async () => {
      if (comparisonStrategies.value.length === 0) return

      try {
        const response = await axios.get('/api/strategies/comparison', {
          params: {
            strategies: comparisonStrategies.value,
            days: 30
          }
        })

        if (response.data.success) {
          drawComparisonChart(response.data.data.comparison)
        }
      } catch (error) {
        console.error('Error fetching comparison data:', error)
      }
    }

    const drawEquityChart = (dates, values) => {
      const ctx = equityChart.value?.getContext('2d')
      if (!ctx) return

      if (equityChartInstance) {
        equityChartInstance.destroy()
      }

      equityChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: dates,
          datasets: [{
            label: 'Portfolio Value',
            data: values,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            x: {
              type: 'time',
              time: {
                parser: 'YYYY-MM-DD',
                displayFormats: {
                  day: 'MMM D'
                }
              }
            },
            y: {
              beginAtZero: false
            }
          }
        }
      })
    }

    const drawComparisonChart = (data) => {
      const ctx = comparisonChart.value?.getContext('2d')
      if (!ctx) return

      if (comparisonChartInstance) {
        comparisonChartInstance.destroy()
      }

      // Prepare datasets
      const datasets = []
      const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
      
      Object.entries(data).forEach(([strategy, history], index) => {
        datasets.push({
          label: strategy,
          data: history.map(h => ({ x: h.date, y: h.return * 100 })),
          borderColor: colors[index % colors.length],
          backgroundColor: 'transparent',
          tension: 0.1
        })
      })

      comparisonChartInstance = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top'
            },
            title: {
              display: true,
              text: 'Strategy Performance Comparison (Returns %)'
            }
          },
          scales: {
            x: {
              type: 'time',
              time: {
                parser: 'YYYY-MM-DD',
                displayFormats: {
                  day: 'MMM D'
                }
              }
            },
            y: {
              title: {
                display: true,
                text: 'Return %'
              }
            }
          }
        }
      })
    }

    const refreshData = () => {
      fetchLatestResults()
      fetchSummary()
    }

    const formatPercent = (value) => {
      if (value === null || value === undefined) return '-'
      return (value * 100).toFixed(2) + '%'
    }

    const formatNumber = (value) => {
      if (value === null || value === undefined) return '-'
      return value.toFixed(2)
    }

    const formatDate = (dateStr) => {
      if (!dateStr) return '-'
      return new Date(dateStr).toLocaleString()
    }

    const getReturnClass = (value) => {
      if (value === null || value === undefined) return ''
      return value >= 0 ? 'positive' : 'negative'
    }

    const getRankClass = (rank) => {
      if (rank <= 3) return 'top-rank'
      if (rank <= 10) return 'high-rank'
      return ''
    }

    const getCategoryClass = (category) => {
      const categoryClasses = {
        'trend_following': 'cat-trend',
        'mean_reversion': 'cat-mr',
        'volatility': 'cat-vol',
        'passive': 'cat-passive',
        'machine_learning': 'cat-ml'
      }
      return categoryClasses[category] || 'cat-default'
    }

    onMounted(() => {
      fetchLatestResults()
      fetchSummary()
      fetchCategories()
    })

    return {
      rankings,
      summary,
      categories,
      selectedCategory,
      loading,
      runDate,
      selectedStrategy,
      comparisonStrategies,
      comparisonChart,
      equityChart,
      fetchRankings,
      viewDetails,
      closeDetails,
      compareStrategy,
      removeFromComparison,
      clearComparison,
      refreshData,
      formatPercent,
      formatNumber,
      formatDate,
      getReturnClass,
      getRankClass,
      getCategoryClass
    }
  }
}
</script>

<style scoped>
.strategies-container {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.header h1 {
  font-size: 24px;
  font-weight: 600;
  color: #333;
}

.controls {
  display: flex;
  gap: 10px;
  align-items: center;
}

.category-select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  cursor: pointer;
}

.refresh-btn {
  padding: 8px 16px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
}

.refresh-btn:hover {
  background: #2563eb;
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Summary Section */
.summary-section {
  margin-bottom: 30px;
}

.summary-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 20px;
}

.summary-card h3 {
  margin-bottom: 15px;
  font-size: 18px;
  color: #333;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 15px;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background: #f9fafb;
  border-radius: 4px;
}

.summary-item .label {
  color: #6b7280;
  font-size: 14px;
}

.summary-item .value {
  font-weight: 600;
  color: #333;
}

.run-info {
  font-size: 14px;
  color: #6b7280;
  display: flex;
  align-items: center;
  gap: 5px;
}

/* Rankings Table */
.rankings-section {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
}

.rankings-section h2 {
  margin-bottom: 15px;
  font-size: 20px;
  color: #333;
}

.rankings-table {
  width: 100%;
  border-collapse: collapse;
}

.rankings-table th {
  text-align: left;
  padding: 12px;
  border-bottom: 2px solid #e5e7eb;
  font-weight: 600;
  color: #4b5563;
  font-size: 14px;
}

.rankings-table td {
  padding: 12px;
  border-bottom: 1px solid #f3f4f6;
}

.rankings-table tr:hover {
  background: #f9fafb;
}

.rank-badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: 600;
  font-size: 12px;
  background: #e5e7eb;
  color: #4b5563;
}

.rank-badge.top-rank {
  background: #fef3c7;
  color: #92400e;
}

.rank-badge.high-rank {
  background: #dbeafe;
  color: #1e40af;
}

.strategy-name {
  font-weight: 500;
  color: #1f2937;
}

.category-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: 500;
}

.cat-trend { background: #dbeafe; color: #1e40af; }
.cat-mr { background: #d1fae5; color: #065f46; }
.cat-vol { background: #fce7f3; color: #9f1239; }
.cat-passive { background: #e5e7eb; color: #4b5563; }
.cat-ml { background: #ede9fe; color: #5b21b6; }
.cat-default { background: #f3f4f6; color: #6b7280; }

.numeric {
  text-align: right;
  font-family: monospace;
}

.positive { color: #10b981; }
.negative { color: #ef4444; }

.score {
  font-weight: 600;
  color: #3b82f6;
}

.actions {
  text-align: center;
}

.action-btn {
  padding: 6px 10px;
  margin: 0 2px;
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 4px;
  cursor: pointer;
  color: #4b5563;
}

.action-btn:hover {
  background: #e5e7eb;
  color: #1f2937;
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Comparison Section */
.comparison-section {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
}

.comparison-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.selected-strategies {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.strategy-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 6px 12px;
  background: #3b82f6;
  color: white;
  border-radius: 20px;
  font-size: 14px;
}

.strategy-chip i {
  cursor: pointer;
  opacity: 0.8;
}

.strategy-chip i:hover {
  opacity: 1;
}

.clear-btn {
  padding: 6px 12px;
  background: #ef4444;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.clear-btn:hover {
  background: #dc2626;
}

.comparison-chart {
  height: 400px;
}

/* Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e5e7eb;
}

.modal-header h2 {
  font-size: 20px;
  font-weight: 600;
  color: #333;
}

.close-btn {
  padding: 8px;
  background: none;
  border: none;
  cursor: pointer;
  color: #6b7280;
  font-size: 18px;
}

.close-btn:hover {
  color: #1f2937;
}

.modal-body {
  padding: 20px;
  overflow-y: auto;
}

.details-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background: #f9fafb;
  border-radius: 4px;
}

.equity-curve-section {
  margin-top: 30px;
}

.equity-curve-section h3 {
  margin-bottom: 15px;
  font-size: 18px;
  color: #333;
}

.equity-curve-section canvas {
  height: 300px;
}

/* Loading and Empty States */
.loading, .no-data {
  text-align: center;
  padding: 40px;
  color: #6b7280;
}

.loading i {
  font-size: 24px;
  margin-right: 10px;
}
</style>