# Assessment Pipeline Documentation

## CRITICAL: Assessment Pipeline Requirements

### Core Principle
**The Assessment Pipeline MUST evaluate EVERY combination of:**
- **EVERY strategy** 
- **EVERY scenario**
- **EVERY symbol**

This means if we have:
- 11 strategies
- 32 scenarios  
- 153 symbols

We should run **11 × 32 × 153 = 53,856 backtests**

### Current Implementation

The assessment pipeline runs backtests for all combinations and produces:
- Individual backtest results for each strategy-scenario-symbol combination
- Aggregated metrics per strategy across all scenarios and symbols
- Rankings based on overall performance

### Important Notes

1. **Strategy Count**: We currently have 11 strategies defined in `src/definitions/strategies.py`:
   - Buy and Hold
   - Dollar Cost Averaging
   - Golden Cross
   - Trend Following
   - Mean Reversion
   - RSI Strategy
   - Momentum
   - Dual Momentum
   - Volatility Targeting
   - ML Ensemble
   - Balanced Portfolio

2. **Backtest Results**: Each backtest result includes:
   - Strategy name
   - Symbol tested
   - Scenario name
   - Performance metrics (return, Sharpe, etc.)

3. **Aggregation**: The final ranking aggregates performance across:
   - All symbols for each strategy
   - All scenarios for each strategy
   - This gives an overall score for each strategy

### File Locations
- Strategy definitions: `src/definitions/strategies.py`
- Assessment pipeline: `src/pipelines/assessment_pipeline/`
- Results: `data/assessment/`

### Adding More Strategies

To add more strategies:
1. Define them in `src/definitions/strategies.py` under `DEFAULT_STRATEGIES`
2. Implement the strategy logic in `src/strats/`
3. Re-run the assessment pipeline

### Understanding Results

The assessment results show:
- **Overall Rankings**: Best strategies across all scenarios and symbols
- **Category Rankings**: Best strategy in each category (momentum, mean reversion, etc.)
- **Regime Rankings**: Best strategies for different market conditions
- **Individual Results**: Detailed performance for each combination

Each strategy's final score is an aggregate of its performance across ALL symbols and ALL scenarios, not just one symbol.