# Benchmark Pipeline

## Overview
The benchmark pipeline is an end-of-day (EOD) process that runs after the data pipeline to evaluate and rank trading strategies using historical simulation.

### Core Concept: Dynamic Strategy Selection
Instead of testing every strategy on every symbol (which would be inefficient), the pipeline:
1. Analyzes each symbol's market characteristics (volatility, trend, momentum, etc.)
2. Dynamically assigns the most suitable strategy to each symbol
3. Runs backtests on these optimized strategy-symbol pairs
4. Ranks the results to find the best performing combinations
5. Updates the WebGUI daily with the top performers

This approach ensures that:
- Trend-following strategies are tested on trending stocks
- Mean reversion strategies are tested on range-bound stocks
- High volatility strategies are tested on volatile instruments
- Each symbol gets a strategy that matches its behavior

## Key Principles

### 1. Data Source
- **ALL DATA MUST COME FROM THE SIMULATOR** - Never directly query the database
- The simulator provides point-in-time accurate data with proper handling of:
  - Look-ahead bias prevention
  - Corporate actions (splits, dividends)
  - Delisted securities
  - Weekend/holiday handling

### 2. Pipeline Steps

#### Step 1: Strategy Collection
- Gather all benchmark strategies from `src/strats/benchmark.py`
- Filter based on configuration (all, core, by style, etc.)
- Initialize strategy instances with proper parameters

#### Step 2: Simulation Setup
- Initialize the historical simulator for the evaluation period
- Configure simulation parameters:
  - Date range (e.g., last 3 months for daily, 1-5 years for full backtest)
  - Universe of symbols
  - Transaction costs and slippage
  - Initial capital

#### Step 3: Dynamic Strategy-Symbol Assignment
- **For each symbol in the universe, analyze its characteristics:**
  - Volatility (annualized standard deviation of returns)
  - Trend strength (price vs moving averages)
  - Mean reversion indicators (RSI)
  - Volume trends
  - Market cap category
  - Symbol type (ETF, stock, crypto)
  
- **Assign the most suitable strategy based on characteristics:**
  - Index ETFs (SPY, QQQ, IWM) → Buy and Hold
  - High volatility (>30%) → Turtle Trading or Mean Reversion
  - Strong trends (>5% from 50-day MA) → Momentum strategies
  - Oversold (RSI < 30) → Mean Reversion
  - Crypto symbols → Crypto-specific strategies
  - Sector rotation candidates → Sector Rotation strategy
  
- **Create strategy-symbol pairs for testing**
  - Each symbol gets its optimal strategy
  - Some strategies may be assigned to multiple symbols
  - Default to Buy and Hold if no clear match

#### Step 4: Strategy Backtest Execution
- Run each strategy-symbol pair through the simulator
- Execute in parallel batches for efficiency
- Each backtest includes:
  - Portfolio initialization with starting capital
  - Day-by-day simulation through historical data
  - Trade execution with costs and slippage
  - Position tracking and P&L calculation
- Collect all trades and position history
- Track portfolio value over time

#### Step 5: Metrics Calculation
- Calculate performance metrics for each strategy:
  - Total return
  - Annualized return
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
  - Average win/loss
  - Number of trades
- Risk metrics:
  - Volatility
  - Beta to market
  - Value at Risk (VaR)
  - Conditional VaR

#### Step 6: Ranking and Scoring
- Rank strategy-symbol pairs by multiple criteria:
  - Risk-adjusted returns (Sharpe ratio)
  - Absolute returns
  - Consistency (win rate, profit factor)
  - Drawdown management
  - Number of profitable trades
- Create composite scores combining all metrics
- Identify:
  - Top 10 overall performers
  - Best strategy for each category (momentum, mean reversion, etc.)
  - Most consistent performers
  - Best risk-adjusted returns

#### Step 7: Result Storage and Display
- Store backtest results in database tables:
  - `strategies.backtest_results` - Individual backtest metrics
  - `strategies.strategy_rankings` - Daily rankings
  - `strategies.equity_curves` - Portfolio value over time
- Update WebGUI strategies tab with:
  - Top 10 strategy-symbol pairs
  - Performance charts
  - Risk metrics
  - Trade statistics
- Generate daily summary report

## Database Schema

### strategies.backtest_results
- id
- strategy_name
- symbol  -- The specific symbol this strategy was tested on
- run_date
- backtest_start_date
- backtest_end_date
- universe
- initial_capital
- final_value
- total_return
- annualized_return
- sharpe_ratio
- max_drawdown
- volatility
- win_rate
- total_trades
- avg_win
- avg_loss
- profit_factor
- best_trade
- worst_trade
- metadata (JSONB) -- Contains strategy parameters, symbol characteristics
- created_at

### strategies.strategy_rankings
- id
- run_date
- strategy_name
- rank_by_sharpe
- rank_by_return
- rank_by_consistency
- composite_score
- category (trend_following, mean_reversion, etc.)

## Schedule
- Runs daily after market close and data pipeline completion
- Full backtest on weekends (longer historical period)
- Can be triggered manually for specific strategies or periods

## Integration Points
1. **Data Pipeline**: Must complete successfully before benchmark pipeline
2. **Simulator**: All market data access through simulator only
3. **WebGUI**: Results displayed in strategies tab with rankings and charts
4. **Monitoring**: Track pipeline execution and alert on failures

## Important Notes
- NEVER query market_data tables directly - always use simulator
- Ensure strategies can't peek into the future
- Handle missing data gracefully
- Account for survivorship bias in universe selection

## Example Daily Workflow

### 7:00 PM - Pipeline Starts
1. Data pipeline completes, all market data is up to date
2. Benchmark pipeline triggered automatically

### 7:05 PM - Analysis Phase
For universe of 20 symbols:
- AAPL: High volatility (28%), slight downtrend → Assigned "Mean Reversion"
- SPY: Index ETF, stable → Assigned "Buy and Hold"
- NVDA: Strong uptrend (>10%), high volume → Assigned "Momentum Factor"
- BTC-USD: Crypto, volatile → Assigned "Turtle Trading"
- ... (16 more symbols analyzed)

### 7:10 PM - Backtest Phase
Running 20 parallel backtests (last 90 days):
- Mean Reversion on AAPL
- Buy and Hold on SPY
- Momentum Factor on NVDA
- Turtle Trading on BTC-USD
- ... (16 more strategy-symbol pairs)

### 7:20 PM - Results Phase
Top 10 performers identified:
1. Momentum Factor + NVDA: +18.5% return, 1.85 Sharpe
2. Turtle Trading + BTC-USD: +22.3% return, 1.42 Sharpe
3. Mean Reversion + AAPL: +12.1% return, 2.10 Sharpe
... 

### 7:25 PM - WebGUI Update
Strategies tab shows:
- Leaderboard of top 10 strategy-symbol pairs
- Performance charts for each
- Risk metrics and trade statistics
- "Strategy of the Day" highlight

## Implementation Status

### Completed ✅
- Pipeline infrastructure and steps
- Strategy collection from benchmark.py
- Dynamic symbol analysis and strategy assignment
- Database tables created
- Basic backtest execution framework

### In Progress 🚧
- Strategy-simulator interface adapter
- Signal computation integration
- Proper data format conversion

### TODO 📋
- WebGUI integration for results display
- Email notifications for top performers
- Weekend full historical backtests
- Performance tracking over time