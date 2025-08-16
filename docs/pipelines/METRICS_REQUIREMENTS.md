# Comprehensive Metrics Requirements for All Trading Strategies

## Overview
This document details ALL metrics required by every strategy type in the Tetra platform. The Metrics Pipeline (Stage 3) must calculate these metrics for each of the 131 scenarios to support comprehensive strategy backtesting.

## Strategy Categories and Their Requirements

### 1. Signal-Based Strategies

#### Technical Indicator Requirements
```yaml
Moving Averages:
  - sma_5, sma_10, sma_20, sma_50, sma_100, sma_200
  - ema_12, ema_26, ema_50, ema_200
  - wma_10, wma_20, wma_50
  - dema_20, dema_50  # Double EMA
  - tema_20, tema_50  # Triple EMA
  - kama_10, kama_30  # Kaufman Adaptive MA
  - volume_sma_20, volume_sma_50

Momentum Indicators:
  - rsi_14, rsi_21
  - stochastic_k_14, stochastic_d_3
  - macd, macd_signal, macd_histogram
  - cci_20  # Commodity Channel Index
  - williams_r_14
  - roc_10, roc_20  # Rate of Change
  - tsi_25_13  # True Strength Index
  - ultimate_7_14_28  # Ultimate Oscillator
  - mfi_14  # Money Flow Index

Volatility Indicators:
  - bb_upper, bb_middle, bb_lower  # Bollinger Bands (20,2)
  - atr_14  # Average True Range
  - natr_14  # Normalized ATR
  - chandelier_exit_22_3
  - historical_volatility_20, historical_volatility_252
  - garman_klass_20
  - parkinson_20
  - keltner_upper_20, keltner_lower_20

Volume Indicators:
  - obv  # On-Balance Volume
  - cmf_20  # Chaikin Money Flow
  - volume_ratio  # Current volume / avg volume
  - dollar_volume, dollar_volume_20d
  - accumulation_distribution
  - ease_of_movement_14
  - vwap  # Volume-Weighted Average Price

Trend Indicators:
  - adx_14  # Average Directional Index
  - aroon_up_25, aroon_down_25
  - psar  # Parabolic SAR
  - supertrend_10_3
  - donchian_high_20, donchian_low_10, donchian_low_20
  - highest_20, highest_55, lowest_10, lowest_20

Statistical Metrics:
  - returns_1, returns_5, returns_20, returns_60, returns_252
  - log_returns_1, log_returns_5, log_returns_20
  - excess_returns_1, excess_returns_5, excess_returns_20
  - volatility_20, volatility_60, volatility_252
  - correlation_spy_20, correlation_spy_60, correlation_spy_252
  - beta_60, beta_252
  - sharpe_ratio_252
  - sortino_ratio_252
```

#### Derived Signals
```yaml
Crossover Signals:
  - golden_cross  # SMA50 > SMA200
  - death_cross  # SMA50 < SMA200
  - macd_bullish  # MACD crosses above signal
  - macd_bearish  # MACD crosses below signal
  - rsi_oversold  # RSI < 30
  - rsi_overbought  # RSI > 70
  - breakout_20d_high
  - breakout_20d_low
```

### 2. ML-Based Strategies

#### Feature Engineering Requirements
```yaml
Price Features:
  - price_position_20, price_position_50, price_position_200
  - distance_from_sma_20, distance_from_sma_50, distance_from_sma_200
  - price_to_sma_ratio_20, price_to_sma_ratio_50
  - support_level_20, support_level_50
  - resistance_level_20, resistance_level_50
  - fibonacci_0.236, fibonacci_0.382, fibonacci_0.5, fibonacci_0.618, fibonacci_0.786

Pattern Features:
  - candlestick_doji
  - candlestick_hammer
  - candlestick_engulfing
  - candlestick_shooting_star
  - candlestick_morning_star
  - candlestick_evening_star
  - chart_triangle_detected
  - chart_flag_detected
  - chart_head_shoulders_detected
  - breakout_strength_20, breakout_strength_50

Momentum Features:
  - momentum_score  # Composite of multiple momentum indicators
  - trend_strength  # ADX-based trend strength
  - reversal_probability  # ML-computed reversal likelihood
  - momentum_divergence  # Price vs momentum divergence
  - relative_strength_rank  # Cross-sectional momentum rank

Market Regime Features:
  - regime_bull_probability
  - regime_bear_probability  
  - regime_neutral_probability
  - volatility_regime_low
  - volatility_regime_normal
  - volatility_regime_high
  - correlation_regime_high
  - correlation_regime_breaking

Microstructure Features:
  - bid_ask_spread
  - bid_ask_imbalance
  - order_flow_imbalance
  - trade_intensity
  - price_impact
  - kyle_lambda  # Price impact coefficient
  - amihud_illiquidity
  - roll_spread

Sentiment Features:
  - news_sentiment_positive
  - news_sentiment_negative
  - news_sentiment_neutral
  - social_sentiment_score
  - options_put_call_ratio
  - options_implied_volatility
  - options_skew
  - vix_level
  - term_structure_slope

Lag Features (for time series models):
  - returns_lag_1, returns_lag_2, returns_lag_3, returns_lag_5, returns_lag_10
  - volume_lag_1, volume_lag_2, volume_lag_3
  - volatility_lag_1, volatility_lag_2, volatility_lag_5
  - rsi_lag_1, rsi_lag_2, rsi_lag_3

Rolling Statistics:
  - returns_mean_5, returns_mean_20, returns_mean_60
  - returns_std_5, returns_std_20, returns_std_60
  - returns_skew_20, returns_skew_60
  - returns_kurtosis_20, returns_kurtosis_60
  - volume_mean_5, volume_mean_20, volume_mean_60
  - high_low_spread_mean_20
  - close_to_close_volatility_20

Cross-Asset Features:
  - correlation_with_spy
  - correlation_with_qqq
  - correlation_with_iwm
  - correlation_with_vix
  - beta_to_market
  - sector_relative_strength
  - industry_momentum
```

### 3. Time-Based Strategies

#### Time-Specific Metrics
```yaml
Intraday Metrics:
  - gap  # Opening gap from previous close
  - gap_percentage
  - premarket_volume
  - premarket_high
  - premarket_low
  - overnight_return
  - first_30min_return
  - first_hour_return
  - last_hour_return
  - lunch_hour_volume
  - power_hour_momentum  # 3-4 PM momentum
  - close_auction_imbalance

Session Metrics:
  - asian_session_return
  - european_session_return
  - us_session_return
  - session_overlap_volatility
  - session_volume_ratio

Time-of-Day Patterns:
  - hour_of_day_return_avg
  - minute_of_day_volatility
  - time_weighted_vwap
  - opening_range_high
  - opening_range_low
  - opening_range_breakout

Day-of-Week Effects:
  - monday_effect
  - friday_effect
  - weekend_effect
  - turn_of_month_effect
  - option_expiry_effect
  - triple_witching_effect

Calendar Effects:
  - month_end_rebalancing
  - quarter_end_window_dressing
  - january_effect
  - sell_in_may_effect
  - december_tax_loss_effect
  - holiday_effect
```

### 4. Event-Based Strategies

#### Event-Specific Metrics
```yaml
Earnings Metrics:
  - days_to_earnings
  - earnings_surprise_history
  - earnings_revision_trend
  - analyst_consensus_spread
  - earnings_volatility_premium
  - post_earnings_drift_factor
  - earnings_quality_score
  - earnings_momentum

Economic Event Metrics:
  - days_to_fomc
  - days_to_cpi_release
  - days_to_jobs_report
  - fed_funds_probability
  - economic_surprise_index
  - gdp_nowcast
  - inflation_expectations
  - yield_curve_slope

Corporate Action Metrics:
  - days_to_dividend
  - dividend_yield
  - days_to_split
  - split_ratio
  - buyback_intensity
  - insider_trading_score
  - short_interest_ratio
  - days_to_cover

News Event Metrics:
  - news_volume_24h
  - news_sentiment_change
  - news_novelty_score
  - news_reach_score
  - headline_negativity
  - news_momentum
  - social_media_mentions
  - google_trends_score
```

### 5. Composite Strategies

#### Aggregation Metrics
```yaml
Strategy Scoring:
  - strategy_1_signal
  - strategy_2_signal
  - strategy_3_signal
  - weighted_composite_signal
  - voting_composite_signal
  - ml_ensemble_prediction

Consensus Indicators:
  - technical_consensus_score  # Aggregate of all technical indicators
  - fundamental_consensus_score
  - sentiment_consensus_score
  - analyst_consensus_rating
  - quant_factor_score

Multi-Timeframe:
  - signal_1min
  - signal_5min
  - signal_15min
  - signal_1hour
  - signal_daily
  - signal_weekly
  - timeframe_alignment_score
```

## Risk Management Metrics

```yaml
Position Risk:
  - position_var_95  # Value at Risk
  - position_cvar_95  # Conditional VaR
  - position_expected_shortfall
  - position_max_drawdown
  - position_recovery_time
  - position_downside_deviation
  - position_ulcer_index

Portfolio Risk:
  - portfolio_var_95
  - portfolio_cvar_95
  - portfolio_correlation_matrix
  - portfolio_concentration_risk
  - portfolio_sector_exposure
  - portfolio_factor_exposure
  - portfolio_tail_risk

Market Risk:
  - systematic_risk_beta
  - idiosyncratic_risk
  - tracking_error
  - information_ratio
  - treynor_ratio
  - calmar_ratio
  - omega_ratio
  - kappa_three_ratio
```

## Performance Metrics

```yaml
Trade Metrics:
  - win_rate
  - profit_factor
  - avg_win_size
  - avg_loss_size
  - win_loss_ratio
  - consecutive_wins
  - consecutive_losses
  - largest_win
  - largest_loss
  - avg_trade_duration
  - trade_efficiency

Return Metrics:
  - total_return
  - annualized_return
  - risk_adjusted_return
  - excess_return
  - active_return
  - cumulative_return
  - compound_return
  - time_weighted_return
  - money_weighted_return
```

## Market Microstructure Metrics

```yaml
Liquidity Metrics:
  - bid_size
  - ask_size
  - bid_ask_spread
  - effective_spread
  - realized_spread
  - quoted_spread
  - time_weighted_spread
  - volume_weighted_spread

Market Depth:
  - order_book_imbalance
  - market_depth_10bps
  - market_depth_50bps
  - market_depth_100bps
  - level_2_pressure
  - hidden_liquidity_estimate

Price Discovery:
  - information_share
  - common_factor_weight
  - price_discovery_measure
  - hasbrouck_information_share
  - gonzalo_granger_measure

Trade Flow:
  - trade_sign
  - lee_ready_indicator
  - tick_rule
  - bulk_volume_classification
  - order_flow_toxicity
  - pin_probability  # Probability of informed trading
```

## Implementation Priority

### Phase 1 - Core Metrics (Required for Basic Strategies)
- All moving averages (SMA, EMA)
- Basic momentum (RSI, MACD, Stochastic)
- Basic volatility (Bollinger Bands, ATR)
- Basic volume (OBV, Volume Ratio)
- Basic returns and volatility statistics

### Phase 2 - Advanced Technical (Required for Complex Strategies)
- Advanced momentum indicators
- Advanced volatility measures
- Market microstructure basics
- Pattern recognition metrics
- Multi-timeframe indicators

### Phase 3 - ML Features (Required for ML Strategies)
- All feature engineering metrics
- Lag features
- Rolling statistics
- Cross-asset correlations
- Sentiment indicators

### Phase 4 - Event & Time Metrics (Required for Event/Time Strategies)
- Event-specific calculations
- Time-of-day patterns
- Calendar effects
- Session-specific metrics

### Phase 5 - Risk & Performance (Required for Portfolio Management)
- All risk metrics
- Performance attribution
- Portfolio-level calculations

## Storage Estimates

```yaml
Per Scenario Storage:
  technical_indicators: 200 columns × 252 days × 500 symbols = ~100MB
  ml_features: 150 columns × 252 days × 500 symbols = ~75MB
  event_metrics: 50 columns × 252 days × 500 symbols = ~25MB
  risk_metrics: 30 columns × 252 days × 500 symbols = ~15MB
  Total per scenario: ~215MB

Total for 131 Scenarios:
  Uncompressed: 131 × 215MB = ~28GB
  Compressed (Parquet): ~7GB
  With daily overwrite: Constant 7GB
```

## Calculation Performance

```yaml
Estimated Calculation Time:
  Per Symbol:
    technical_indicators: 0.5 seconds
    ml_features: 1.0 seconds
    event_metrics: 0.3 seconds
    risk_metrics: 0.2 seconds
    Total: ~2 seconds per symbol

  Per Scenario (500 symbols):
    Sequential: 500 × 2 = 1000 seconds (~17 minutes)
    Parallel (8 cores): ~2.5 minutes

  All 131 Scenarios:
    Sequential: 131 × 17 = 2227 minutes (~37 hours)
    Parallel (8 cores): 131 × 2.5 = 328 minutes (~5.5 hours)
    Distributed (32 cores): ~1.5 hours
```

## Data Dependencies

```yaml
Required Input Data:
  - OHLCV prices (daily, intraday if available)
  - Volume data
  - Market index data (SPY, QQQ, IWM, VIX)
  - Economic indicators (interest rates, GDP, CPI)
  - Event calendar (earnings, economic releases, corporate actions)
  - News sentiment scores
  - Options data (if available)

Update Frequency:
  - Daily: After market close
  - Intraday: Every 5 minutes for real-time strategies
  - Event-driven: As events occur
```

## Quality Assurance

```yaml
Validation Rules:
  - RSI: Must be between 0 and 100
  - Correlation: Must be between -1 and 1
  - Probabilities: Must be between 0 and 1
  - Returns: Reasonable bounds check (-50% to +50% daily)
  - Volume: Must be non-negative
  - Prices: High >= Low, High >= Close, Low <= Close

Missing Data Handling:
  - Forward fill for prices (max 5 days)
  - Zero fill for volume
  - NaN for indicators until enough data
  - Interpolation for interest rates
  - Last known value for events

Calculation Verification:
  - Unit tests for each indicator
  - Comparison with reference implementations
  - Historical consistency checks
  - Cross-validation with multiple libraries
```