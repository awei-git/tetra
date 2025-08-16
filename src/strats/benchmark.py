"""Benchmark and universe-specific trading strategies for performance comparison."""

from datetime import time
from typing import List, Dict, Optional

from src.strats.signal_based import SignalBasedStrategy, SignalRule, SignalCondition, ConditionOperator, PositionSide
from src.strats.event_based import EventBasedStrategy, EventTrigger, EventType, EventImpact
from src.strats.time_based import TimeBasedStrategy, TradingWindow, TradingSchedule, SessionType
from src.strats.composite import CompositeStrategy, StrategyWeight, CombinationMode
from src.definitions.market_universe import MarketUniverse
from src.strats.ml_based import (
    create_ml_strategies,
    create_ml_benchmark_strategies,
    check_ml_models_available
)


# ============================================================================
# CLASSIC BENCHMARK STRATEGIES
# ============================================================================

def buy_and_hold_strategy():
    """Simple buy and hold benchmark - the baseline to beat."""
    rules = [
        SignalRule(
            name="buy_and_hold",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, 0)  # Always true - buy immediately
            ],
            exit_conditions=[
                # Never exit - hold forever
            ],
            position_side=PositionSide.LONG,
            position_size_factor=1.0  # Use normal position sizing
        )
    ]
    
    return SignalBasedStrategy(
        name="Buy and Hold Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=1.0,  # 100% allocation
        max_positions=1,
        commission=0.0  # Minimal trading costs
    )


def golden_cross_strategy():
    """Classic 50/200 SMA golden cross strategy."""
    rules = [
        SignalRule(
            name="golden_cross",
            entry_conditions=[
                SignalCondition("sma_50", ConditionOperator.CROSSES_ABOVE, "sma_200"),
                SignalCondition("volume", ConditionOperator.GREATER_THAN, "volume_sma_20")
            ],
            exit_conditions=[
                SignalCondition("sma_50", ConditionOperator.CROSSES_BELOW, "sma_200")  # Death cross
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.10  # 10% stop loss
        )
    ]
    
    return SignalBasedStrategy(
        name="Golden Cross Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.95,  # Nearly full allocation when in position
        max_positions=1
    )


def turtle_trading_strategy():
    """Classic Turtle Trading system - 20/55 day breakout."""
    rules = [
        SignalRule(
            name="turtle_entry",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, "highest_20"),
                SignalCondition("atr_14", ConditionOperator.GREATER_THAN, 0)  # Volatility present
            ],
            exit_conditions=[
                SignalCondition("close", ConditionOperator.LESS_THAN, "lowest_10")
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.02,  # 2 ATR stop (simplified to 2%)
            position_size_factor=0.5  # Turtle used volatility-based sizing
        ),
        SignalRule(
            name="turtle_strong",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, "highest_55")
            ],
            exit_conditions=[
                SignalCondition("close", ConditionOperator.LESS_THAN, "lowest_20")
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.02,
            position_size_factor=1.0  # Full size for stronger signal
        )
    ]
    
    return SignalBasedStrategy(
        name="Turtle Trading Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.1,  # 10% base size
        max_positions=4  # Multiple positions allowed
    )


def rsi_mean_reversion_strategy():
    """Classic RSI oversold/overbought mean reversion."""
    rules = [
        SignalRule(
            name="rsi_oversold",
            entry_conditions=[
                SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 30),
                SignalCondition("close", ConditionOperator.GREATER_THAN, "sma_200")  # Long-term uptrend
            ],
            exit_conditions=[
                SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 50)  # Return to neutral
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.05,
            take_profit=0.10
        )
    ]
    
    return SignalBasedStrategy(
        name="RSI Mean Reversion Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.1,
        max_positions=5
    )


def bollinger_band_strategy():
    """Classic Bollinger Band mean reversion strategy."""
    rules = [
        SignalRule(
            name="bb_oversold",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.LESS_THAN, "bb_lower"),
                SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 40)
            ],
            exit_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, "bb_middle")
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.03
        ),
        SignalRule(
            name="bb_overbought",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, "bb_upper"),
                SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 60)
            ],
            exit_conditions=[
                SignalCondition("close", ConditionOperator.LESS_THAN, "bb_middle")
            ],
            position_side=PositionSide.SHORT,
            stop_loss=0.03
        )
    ]
    
    return SignalBasedStrategy(
        name="Bollinger Bands Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.1,
        max_positions=5,
        # allow_shorts=True
    )


def momentum_factor_strategy():
    """Academic momentum factor - buy top performers."""
    rules = [
        SignalRule(
            name="momentum_factor",
            entry_conditions=[
                SignalCondition("returns_252", ConditionOperator.GREATER_THAN, 0.20),  # 20%+ annual return
                SignalCondition("returns_20", ConditionOperator.GREATER_THAN, 0),  # Recent momentum positive
                SignalCondition("dollar_volume_20d", ConditionOperator.GREATER_THAN, 1_000_000)  # Liquid
            ],
            exit_conditions=[
                SignalCondition("returns_20", ConditionOperator.LESS_THAN, -0.05)  # Exit if -5% monthly
            ],
            position_side=PositionSide.LONG
        )
    ]
    
    return SignalBasedStrategy(
        name="Momentum Factor Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.1,  # Equal weight
        max_positions=10
    )


def macd_crossover_strategy():
    """Classic MACD crossover strategy."""
    rules = [
        SignalRule(
            name="macd_bullish",
            entry_conditions=[
                SignalCondition("macd", ConditionOperator.CROSSES_ABOVE, "macd_signal"),
                SignalCondition("macd_histogram", ConditionOperator.GREATER_THAN, 0)
            ],
            exit_conditions=[
                SignalCondition("macd", ConditionOperator.CROSSES_BELOW, "macd_signal")
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.05
        )
    ]
    
    return SignalBasedStrategy(
        name="MACD Crossover Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.2,
        max_positions=3
    )


def donchian_breakout_strategy():
    """Classic Donchian channel breakout."""
    rules = [
        SignalRule(
            name="donchian_breakout",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, "donchian_high_20"),
                SignalCondition("adx_14", ConditionOperator.GREATER_THAN, 20)  # Trending market
            ],
            exit_conditions=[
                SignalCondition("close", ConditionOperator.LESS_THAN, "donchian_low_10")
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.04
        )
    ]
    
    return SignalBasedStrategy(
        name="Donchian Breakout Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.1,
        max_positions=5
    )


def volume_weighted_momentum_strategy():
    """Momentum with volume confirmation."""
    rules = [
        SignalRule(
            name="volume_momentum",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, "vwap"),
                SignalCondition("volume_ratio", ConditionOperator.GREATER_THAN, 1.5),  # 150% of average volume
                SignalCondition("rsi_14", ConditionOperator.BETWEEN, [50, 70])
            ],
            exit_conditions=[
                SignalCondition("close", ConditionOperator.CROSSES_BELOW, "vwap"),
                SignalCondition("volume_ratio", ConditionOperator.LESS_THAN, 0.5)  # 50% of average volume
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.03,
            time_limit=10
        )
    ]
    
    return SignalBasedStrategy(
        name="Volume Momentum Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.15,
        max_positions=4
    )


def dual_momentum_strategy():
    """Gary Antonacci's Dual Momentum - absolute + relative."""
    rules = [
        SignalRule(
            name="dual_momentum",
            entry_conditions=[
                SignalCondition("returns_252", ConditionOperator.GREATER_THAN, 0),  # Absolute momentum
                SignalCondition("relative_strength_vs_bonds", ConditionOperator.GREATER_THAN, 1.0),  # Relative momentum
                SignalCondition("trend_filter_200d", ConditionOperator.EQUAL, "up")
            ],
            exit_conditions=[
                SignalCondition("returns_252", ConditionOperator.LESS_THAN, 0),
                SignalCondition("relative_strength_vs_bonds", ConditionOperator.LESS_THAN, 1.0)
            ],
            position_side=PositionSide.LONG,
            # rebalance_frequency=21
        )
    ]
    
    return SignalBasedStrategy(
        name="Dual Momentum Benchmark",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.95,  # Nearly fully invested
        max_positions=1  # Concentrated
    )


# ============================================================================
# UNIVERSE-SPECIFIC STRATEGIES
# ============================================================================

def create_mega_cap_momentum():
    """Momentum strategy for large cap stocks (AAPL, MSFT, etc.)"""
    rules = [
        SignalRule(
            name="mega_cap_trend",
            entry_conditions=[
                SignalCondition("sma_20", ConditionOperator.GREATER_THAN, "sma_50"),
                SignalCondition("sma_50", ConditionOperator.GREATER_THAN, "sma_200"),
                SignalCondition("rsi_14", ConditionOperator.BETWEEN, [45, 65]),
                SignalCondition("volume", ConditionOperator.GREATER_THAN, "volume_sma_20"),
                SignalCondition("market_cap", ConditionOperator.GREATER_THAN, 500_000_000_000)  # 500B+
            ],
            exit_conditions=[
                SignalCondition("sma_20", ConditionOperator.CROSSES_BELOW, "sma_50"),
                SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 35)
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.05,      # 5% stop - wider for large caps
            take_profit=0.15,    # 15% target
            time_limit=30        # 30 day max hold
        )
    ]
    
    return SignalBasedStrategy(
        name="Mega Cap Momentum",
        signal_rules=rules,
        confirmation_required=2,
        initial_capital=100000,
        position_size=0.15,      # 15% per position - larger for stable stocks
        max_positions=5,
        # symbols_filter=MarketUniverse.LARGE_CAP_STOCKS[:10]  # Top 10 only
    )


def create_ai_growth_strategy():
    """High growth strategy for AI infrastructure stocks."""
    rules = [
        SignalRule(
            name="ai_breakout",
            entry_conditions=[
                SignalCondition("price", ConditionOperator.GREATER_THAN, "high_52w_0.9"),  # Near 52w high
                SignalCondition("rsi_14", ConditionOperator.BETWEEN, [50, 70]),
                SignalCondition("volume_ratio", ConditionOperator.GREATER_THAN, 2.0),  # 200% of average volume
                SignalCondition("revenue_growth", ConditionOperator.GREATER_THAN, 0.20)  # 20% growth
            ],
            exit_conditions=[
                SignalCondition("price", ConditionOperator.LESS_THAN, "sma_50"),
                SignalCondition("rsi_14", ConditionOperator.CROSSES_BELOW, 40)
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.08,      # 8% stop - more volatile
            take_profit=0.25,    # 25% target - higher for growth
            position_size_factor=0.7  # Smaller size due to volatility
        ),
        SignalRule(
            name="ai_pullback_buy",
            entry_conditions=[
                SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 35),
                SignalCondition("price", ConditionOperator.GREATER_THAN, "sma_200"),
                SignalCondition("macd_histogram", ConditionOperator.CROSSES_ABOVE, 0)
            ],
            exit_conditions=[
                SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 65)
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.06
        )
    ]
    
    return SignalBasedStrategy(
        name="AI Infrastructure Growth",
        signal_rules=rules,
        # symbols_filter=MarketUniverse.AI_INFRASTRUCTURE_STOCKS,
        position_size=0.08,  # 8% positions - smaller for volatile stocks
        max_positions=6
    )


def create_sector_rotation_etf():
    """Sector rotation strategy using sector ETFs."""
    rules = [
        SignalRule(
            name="sector_momentum",
            entry_conditions=[
                SignalCondition("relative_strength_rank", ConditionOperator.LESS_THAN, 4),  # Top 3 sectors
                SignalCondition("sma_20", ConditionOperator.GREATER_THAN, "sma_50"),
                SignalCondition("sector_breadth", ConditionOperator.GREATER_THAN, 0.6)  # 60% advancing
            ],
            exit_conditions=[
                SignalCondition("relative_strength_rank", ConditionOperator.GREATER_THAN, 7),  # Falls to bottom half
                SignalCondition("sma_20", ConditionOperator.CROSSES_BELOW, "sma_50")
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.04,      # 4% stop for ETFs
            # rebalance_frequency=20  # Review every 20 days
        )
    ]
    
    return SignalBasedStrategy(
        name="Sector Rotation ETF",
        signal_rules=rules,
        # symbols_filter=MarketUniverse.SECTOR_ETFS,
        position_size=0.20,      # 20% per sector - concentrated
        max_positions=3,         # Top 3 sectors only
        # rebalance_mode='monthly'
    )


def create_volatility_harvesting():
    """Strategy for trading volatility ETFs."""
    rules = [
        SignalRule(
            name="vol_spike_short",
            entry_conditions=[
                SignalCondition("vix", ConditionOperator.GREATER_THAN, 25),
                SignalCondition("vix_term_structure", ConditionOperator.EQUAL, "backwardation"),
                SignalCondition("spy_rsi", ConditionOperator.LESS_THAN, 30)
            ],
            exit_conditions=[
                SignalCondition("vix", ConditionOperator.LESS_THAN, 18),
                SignalCondition("position_days", ConditionOperator.GREATER_THAN, 5)
            ],
            position_side=PositionSide.SHORT,  # Short volatility
            # symbols_filter=['SVXY', 'ZIV'],   # Inverse VIX ETFs
            stop_loss=0.10,      # 10% stop - vol can spike
            position_size_factor=0.5  # Half size for safety
        ),
        SignalRule(
            name="vol_contango_harvest",
            entry_conditions=[
                SignalCondition("vix", ConditionOperator.LESS_THAN, 15),
                SignalCondition("vix_term_structure", ConditionOperator.EQUAL, "contango"),
                SignalCondition("contango_degree", ConditionOperator.GREATER_THAN, 0.05)  # 5% contango
            ],
            exit_conditions=[
                SignalCondition("vix", ConditionOperator.GREATER_THAN, 20),
                SignalCondition("spy_drawdown", ConditionOperator.GREATER_THAN, 0.05)
            ],
            position_side=PositionSide.SHORT,
            # symbols_filter=['VXX', 'VIXY'],
            time_limit=30
        )
    ]
    
    return SignalBasedStrategy(
        name="Volatility Harvesting",
        signal_rules=rules,
        # symbols_filter=MarketUniverse.VOLATILITY_ETFS,
        position_size=0.05,      # 5% max - risky
        max_positions=2
        # require_hedge=True       # Always hedge with SPY puts
    )


def create_crypto_momentum():
    """Cryptocurrency momentum strategy."""
    rules = [
        SignalRule(
            name="crypto_trend",
            entry_conditions=[
                SignalCondition("close", ConditionOperator.GREATER_THAN, "sma_50"),
                SignalCondition("rsi_14", ConditionOperator.BETWEEN, [40, 70]),
                SignalCondition("dollar_volume", ConditionOperator.GREATER_THAN, 10_000_000)  # $10M daily volume
            ],
            exit_conditions=[
                SignalCondition("close", ConditionOperator.LESS_THAN, "sma_20"),  # Below SMA20
                SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 30)
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.10,      # 10% stop - crypto is volatile
            take_profit=0.30,    # 30% target
            time_limit=14        # 2 week max
        )
    ]
    
    return SignalBasedStrategy(
        name="Crypto Momentum",
        signal_rules=rules,
        # symbols_filter=MarketUniverse.CRYPTO_SYMBOLS,
        position_size=0.05,      # 5% positions
        max_positions=4
        # trading_hours='24/7'     # Crypto trades 24/7
    )


def create_dividend_aristocrat():
    """Dividend capture strategy for high-yield ETFs."""
    triggers = [
        EventTrigger(
            event_type=EventType.DIVIDEND,
            impact=EventImpact.MEDIUM,
            pre_event_days=5,    # Enter 5 days before ex-div
            post_event_days=2,   # Hold 2 days after
            entry_conditions={
                'dividend_yield': 0.03,      # Min 3% yield
                'payout_consistency': 0.90,  # 90% consistent
                'price_above_support': True
            },
            # symbols_filter=MarketUniverse.INDEX_ETFS + ['HDV', 'VIG', 'DVY', 'SDY']
        )
    ]
    
    return EventBasedStrategy(
        name="Dividend Aristocrat Capture",
        event_triggers=triggers,
        position_size=0.20  # Large positions for stable dividend stocks
        # hedge_market_risk=True
    )


def create_earnings_surprise():
    """Trade earnings events for large cap tech."""
    triggers = [
        EventTrigger(
            event_type=EventType.EARNINGS,
            impact=EventImpact.HIGH,
            pre_event_days=3,
            post_event_days=1,
            entry_conditions={
                'analyst_revision_trend': 'positive',
                'implied_volatility_percentile': 0.7,  # High IV
                'technical_setup': 'bullish',
                'earnings_streak': 3  # Beat last 3 quarters
            },
            exit_conditions={
                'earnings_reaction': 'complete',  # After initial move
                'iv_crush': True
            },
            # symbols_filter=MarketUniverse.LARGE_CAP_STOCKS[:20]  # Top 20 only
        )
    ]
    
    return EventBasedStrategy(
        name="Big Tech Earnings Play",
        event_triggers=triggers,
        # use_options=True,  # Trade options for leverage
        position_size=0.05
        # max_event_exposure=0.20  # Max 20% in earnings plays
    )


def create_morning_breakout():
    """Intraday breakout strategy for liquid stocks."""
    windows = [
        TradingWindow(
            start_time=time(9, 30),
            end_time=time(10, 30),
            session_type=SessionType.REGULAR
            # entry_rules={
            #     'opening_range_breakout': True,
            #     'min_range': 0.005,      # 0.5% minimum range
            #     'volume_confirmation': 1.5,
            #     'gap_fade': False        # Don't fade gaps
            # },
            # symbols_filter=MarketUniverse.LARGE_CAP_STOCKS
        ),
        TradingWindow(
            start_time=time(10, 30),
            end_time=time(11, 30),
            session_type=SessionType.REGULAR
            # entry_rules={
            #     'continuation_only': True,
            #     'require_trend': True
            # }
        )
    ]
    
    schedule = TradingSchedule(
        windows=windows,
        max_trades_per_day=5,
        force_close_end_of_day=True
        # no_overnight_positions=True
    )
    
    return TimeBasedStrategy(
        name="Morning Momentum Breakout",
        trading_schedule=schedule,
        position_size=0.10
        # time_stops={'morning': 90, 'midday': 120}  # Minutes
    )


def create_global_macro_composite():
    """Composite strategy combining multiple asset classes."""
    # Create component strategies
    equity_momentum = create_sector_rotation_etf()
    bond_macro = EventBasedStrategy(
        name="Bond Macro Events",
        event_triggers=[
            EventTrigger(
                event_type=EventType.FOMC,
                impact=EventImpact.HIGH,
                pre_event_days=2,
                post_event_days=5,
                # symbols_filter=MarketUniverse.BOND_ETFS
            )
        ]
        # trade_direction='both',
        # use_futures=True
    )
    commodity_trend = SignalBasedStrategy(
        name="Commodity Trend",
        signal_rules=[
            SignalRule(
                name="commodity_momentum",
                entry_conditions=[
                    SignalCondition("sma_50", ConditionOperator.GREATER_THAN, "sma_200"),
                    SignalCondition("adx_14", ConditionOperator.GREATER_THAN, 25)
                ],
                exit_conditions=[
                    SignalCondition("sma_50", ConditionOperator.CROSSES_BELOW, "sma_200")
                ],
                position_side=PositionSide.LONG
            )
        ],
        # symbols_filter=MarketUniverse.COMMODITY_ETFS
    )
    
    # Combine with risk parity weights
    return CompositeStrategy(
        name="Global Macro Risk Parity",
        strategies=[
            StrategyWeight(equity_momentum, weight=1.0, min_confidence=0.6),
            StrategyWeight(bond_macro, weight=1.2, min_confidence=0.5),
            StrategyWeight(commodity_trend, weight=0.8, min_confidence=0.7)
        ],
        combination_mode=CombinationMode.WEIGHTED
        # target_volatility=0.10,  # 10% target vol
        # rebalance_frequency='weekly',
        # correlation_limit=0.6,
        # max_drawdown_limit=0.15  # 15% max drawdown
    )


def create_all_weather_defensive():
    """Defensive all-weather strategy."""
    # Quality dividend stocks
    dividend_strategy = SignalBasedStrategy(
        name="Quality Dividend",
        signal_rules=[
            SignalRule(
                name="dividend_quality",
                entry_conditions=[
                    SignalCondition("dividend_yield", ConditionOperator.BETWEEN, [0.02, 0.06]),
                    SignalCondition("payout_ratio", ConditionOperator.LESS_THAN, 0.7),
                    SignalCondition("debt_to_equity", ConditionOperator.LESS_THAN, 1.0),
                    SignalCondition("price", ConditionOperator.GREATER_THAN, "sma_200")
                ],
                exit_conditions=[
                    SignalCondition("dividend_cut_risk", ConditionOperator.GREATER_THAN, 0.3)
                ],
                position_side=PositionSide.LONG,
                stop_loss=0.08
            )
        ],
        # symbols_filter=[s for s in MarketUniverse.LARGE_CAP_STOCKS if s in ['JNJ', 'PG', 'KO', 'PEP', 'WMT']]
    )
    
    # Defensive sectors
    defensive_sectors = SignalBasedStrategy(
        name="Defensive Sectors",
        signal_rules=[
            SignalRule(
                name="defensive_rotation",
                entry_conditions=[
                    SignalCondition("market_volatility", ConditionOperator.GREATER_THAN, 20),
                    SignalCondition("sector_relative_strength", ConditionOperator.GREATER_THAN, 1.0)
                ],
                exit_conditions=[
                    SignalCondition("market_volatility", ConditionOperator.LESS_THAN, 15)
                ],
                position_side=PositionSide.LONG
            )
        ],
        # symbols_filter=['XLP', 'XLU', 'XLV', 'XLRE']  # Staples, Utilities, Healthcare, REITs
    )
    
    # Treasury allocation
    treasury_allocation = SignalBasedStrategy(
        name="Treasury Hedge",
        signal_rules=[
            SignalRule(
                name="flight_to_quality",
                entry_conditions=[
                    SignalCondition("spy_drawdown", ConditionOperator.GREATER_THAN, 0.05),
                    SignalCondition("credit_spreads", ConditionOperator.GREATER_THAN, 200)
                ],
                exit_conditions=[
                    SignalCondition("spy_recovery", ConditionOperator.GREATER_THAN, 0.5)
                ],
                position_side=PositionSide.LONG,
                position_size_factor=2.0  # Double size during stress
            )
        ],
        # symbols_filter=['TLT', 'IEF', 'SHY']
    )
    
    return CompositeStrategy(
        name="All Weather Defensive",
        strategies=[
            StrategyWeight(dividend_strategy, weight=0.4),
            StrategyWeight(defensive_sectors, weight=0.3),
            StrategyWeight(treasury_allocation, weight=0.3)
        ],
        combination_mode=CombinationMode.MAJORITY
        # min_agreement=0.5,
        # max_volatility=0.08,  # 8% max volatility
        # preserve_capital=True
    )


# ============================================================================
# STRATEGY REGISTRY AND UTILITIES
# ============================================================================

BENCHMARK_STRATEGIES = {
    # Classic benchmarks
    'buy_and_hold': buy_and_hold_strategy,
    'golden_cross': golden_cross_strategy,
    'turtle_trading': turtle_trading_strategy,
    'rsi_reversion': rsi_mean_reversion_strategy,
    'bollinger_bands': bollinger_band_strategy,
    'momentum_factor': momentum_factor_strategy,
    'macd_crossover': macd_crossover_strategy,
    'donchian_breakout': donchian_breakout_strategy,
    'volume_momentum': volume_weighted_momentum_strategy,
    'dual_momentum': dual_momentum_strategy,
    
    # Universe-specific strategies
    'mega_cap_momentum': create_mega_cap_momentum,
    'ai_growth': create_ai_growth_strategy,
    'sector_rotation': create_sector_rotation_etf,
    'volatility': create_volatility_harvesting,
    'crypto': create_crypto_momentum,
    'dividend': create_dividend_aristocrat,
    'earnings': create_earnings_surprise,
    'morning_breakout': create_morning_breakout,
    'global_macro': create_global_macro_composite,
    'all_weather': create_all_weather_defensive
}

# Add old signal-based ML strategies
for ml_strategy in create_ml_strategies():
    # Convert strategy name to lowercase with underscores
    strategy_key = 'signal_' + ml_strategy.name.lower().replace(' ', '_')
    BENCHMARK_STRATEGIES[strategy_key] = lambda s=ml_strategy: s

# Add new ML pipeline-based strategies if models are available
if check_ml_models_available():
    ml_benchmark_strategies = create_ml_benchmark_strategies()
    for ml_strategy in ml_benchmark_strategies:
        strategy_key = ml_strategy.name.lower().replace(' ', '_')
        BENCHMARK_STRATEGIES[strategy_key] = lambda s=ml_strategy: s


def get_benchmark_strategy(name: str):
    """Get a benchmark strategy by name."""
    if name in BENCHMARK_STRATEGIES:
        return BENCHMARK_STRATEGIES[name]()
    else:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_STRATEGIES.keys())}")


def get_all_benchmarks():
    """Get all benchmark strategies."""
    return {name: creator() for name, creator in BENCHMARK_STRATEGIES.items()}


def get_core_benchmarks():
    """Get the most important benchmark strategies for comparison."""
    core_names = ['buy_and_hold', 'golden_cross', 'turtle_trading', 'momentum_factor', 'sector_rotation']
    
    # Add ML strategies if available
    if check_ml_models_available():
        core_names.extend(['ml_ensemble', 'ml_xgboost_1-day'])
        
    return {name: get_benchmark_strategy(name) for name in core_names if name in BENCHMARK_STRATEGIES}


def get_benchmarks_by_style():
    """Get benchmarks categorized by trading style."""
    return {
        'passive': ['buy_and_hold'],
        'trend_following': ['golden_cross', 'turtle_trading', 'donchian_breakout', 'dual_momentum'],
        'mean_reversion': ['rsi_reversion', 'bollinger_bands'],
        'momentum': ['momentum_factor', 'macd_crossover', 'volume_momentum', 'mega_cap_momentum', 'ai_growth'],
        'rotation': ['sector_rotation'],
        'volatility': ['volatility'],
        'crypto': ['crypto'],
        'event_driven': ['dividend', 'earnings'],
        'intraday': ['morning_breakout'],
        'multi_asset': ['global_macro', 'all_weather'],
        'machine_learning': [
            # Signal-based ML
            'signal_ml_basic', 'signal_ml_anomaly', 'signal_ml_high_confidence', 'signal_ml_technical_combo',
            # Pipeline-based ML (if available)
            'ml_ensemble', 'ml_xgboost_1-day', 'ml_lightgbm_1-day', 'ml_multi-horizon', 'ml_high_confidence'
        ]
    }


def get_strategies_by_risk_level():
    """Categorize strategies by risk level."""
    return {
        'conservative': ['buy_and_hold', 'all_weather', 'dividend', 'sector_rotation'],
        'moderate': ['golden_cross', 'mega_cap_momentum', 'global_macro', 'ml_technical_combo'],
        'aggressive': ['ai_growth', 'crypto', 'morning_breakout', 'turtle_trading', 'ml_basic', 'ml_anomaly'],
        'very_aggressive': ['volatility', 'earnings', 'ml_high_confidence']
    }


def get_strategies_by_universe():
    """Map strategies to their primary universe."""
    return {
        'large_cap_stocks': ['mega_cap_momentum', 'earnings', 'golden_cross'],
        'ai_stocks': ['ai_growth'],
        'sector_etfs': ['sector_rotation'],
        'volatility_etfs': ['volatility'],
        'crypto': ['crypto'],
        'bonds': ['all_weather'],
        'multi_asset': ['global_macro', 'all_weather'],
        'intraday': ['morning_breakout'],
        'any': ['buy_and_hold', 'turtle_trading', 'rsi_reversion', 'bollinger_bands', 
                'momentum_factor', 'macd_crossover', 'donchian_breakout', 
                'volume_momentum', 'dual_momentum']
    }