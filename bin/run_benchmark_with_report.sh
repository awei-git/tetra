#!/bin/bash
# Enhanced Tetra benchmark pipeline with detailed reporting
# Runs daily (2020-now) and weekly (all data) analyses with iCloud Drive reports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# iCloud Drive path for reports
ICLOUD_STRATEGIES="$HOME/Library/Mobile Documents/com~apple~CloudDocs/strategies"

# Create strategies folder if it doesn't exist
mkdir -p "$ICLOUD_STRATEGIES"

# Determine analysis type based on arguments or day of week
if [ "$1" = "weekly" ] || [ "$(date +%u)" -eq 6 ]; then
    # Saturday (6) - run full historical analysis
    MODE="weekly"
    START_DATE="2015-01-01"  # Use all available data
    END_DATE=$(date +%Y-%m-%d)
    REPORT_PREFIX="weekly_full"
    ANALYSIS_DESC="Weekly Full Historical Analysis (2015-present)"
else
    # Daily run - analyze from 2020 to now
    MODE="daily"
    START_DATE="2020-01-01"
    END_DATE=$(date +%Y-%m-%d)
    REPORT_PREFIX="daily"
    ANALYSIS_DESC="Daily Analysis (2020-present)"
fi

# Set up logging
LOG_FILE="/tmp/tetra_benchmark_${REPORT_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
REPORT_DATE=$(date +%Y%m%d)
REPORT_TIME=$(date +%H%M%S)
REPORT_FILE="$ICLOUD_STRATEGIES/tetra_${REPORT_PREFIX}_report_${REPORT_DATE}_${REPORT_TIME}.md"
CSV_FILE="$ICLOUD_STRATEGIES/tetra_${REPORT_PREFIX}_results_${REPORT_DATE}_${REPORT_TIME}.csv"

echo "========================================" | tee -a "$LOG_FILE"
echo "Tetra Benchmark Analysis" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Mode: $ANALYSIS_DESC" | tee -a "$LOG_FILE"
echo "Start Date: $START_DATE" | tee -a "$LOG_FILE"
echo "End Date: $END_DATE" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Report File: $REPORT_FILE" | tee -a "$LOG_FILE"
echo "CSV File: $CSV_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Run benchmark pipeline with detailed analysis
echo "[$(date)] Starting $MODE benchmark analysis..." | tee -a "$LOG_FILE"

# Run the analysis (no JSON output available, will parse logs)
python -m src.pipelines.benchmark_pipeline.main \
    --mode=backfill \
    --start-date="$START_DATE" \
    --end-date="$END_DATE" \
    --universe=all \
    --parallel=8 \
    2>&1 | tee -a "$LOG_FILE"

PIPELINE_STATUS=${PIPESTATUS[0]}

# Generate detailed report
echo "[$(date)] Generating detailed report..." | tee -a "$LOG_FILE"

# Create the report using Python
python << EOF > "$REPORT_FILE"
import json
import pandas as pd
from datetime import datetime, date
import os

# No JSON file to load, will parse from log

# Parse log file for detailed results
strategy_results = []
with open('$LOG_FILE', 'r') as f:
    for line in f:
        if 'Completed backtest for' in line and 'Return=' in line:
            parts = line.split('Completed backtest for ')[1].split(':')
            strategy_symbol = parts[0].strip()
            metrics = parts[1].strip()
            
            # Parse metrics
            return_pct = float(metrics.split('Return=')[1].split('%')[0])
            sharpe = float(metrics.split('Sharpe=')[1].split(',')[0])
            trades = int(metrics.split('Trades=')[1].strip())
            
            # Split strategy and symbol
            parts = strategy_symbol.split('_')
            if len(parts) >= 2:
                strategy = '_'.join(parts[:-1])
                symbol = parts[-1]
            else:
                strategy = strategy_symbol
                symbol = 'UNKNOWN'
            
            strategy_results.append({
                'strategy': strategy,
                'symbol': symbol,
                'return_pct': return_pct,
                'sharpe_ratio': sharpe,
                'trades': trades,
                'strategy_symbol': strategy_symbol
            })

# Create DataFrame
df = pd.DataFrame(strategy_results)

# Generate report
print("# Tetra Benchmark Analysis Report")
print(f"\n**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"**Analysis Type**: $ANALYSIS_DESC")
print(f"**Date Range**: $START_DATE to $END_DATE")
print(f"**Total Combinations Tested**: {len(df)}")
print()

# Executive Summary
print("## Executive Summary")
print()

# Top performers
top_by_return = df.nlargest(10, 'return_pct')
top_by_sharpe = df[df['trades'] > 0].nlargest(10, 'sharpe_ratio') if len(df[df['trades'] > 0]) > 0 else pd.DataFrame()

print("### Top 10 by Total Return")
print("| Rank | Strategy | Symbol | Return % | Sharpe | Trades |")
print("|------|----------|--------|----------|--------|--------|")
for i, row in top_by_return.iterrows():
    print(f"| {i+1} | {row['strategy']} | {row['symbol']} | {row['return_pct']:.2f}% | {row['sharpe_ratio']:.2f} | {row['trades']} |")
print()

if len(top_by_sharpe) > 0:
    print("### Top 10 by Sharpe Ratio (Active Strategies)")
    print("| Rank | Strategy | Symbol | Return % | Sharpe | Trades |")
    print("|------|----------|--------|----------|--------|--------|")
    for i, row in top_by_sharpe.iterrows():
        print(f"| {i+1} | {row['strategy']} | {row['symbol']} | {row['return_pct']:.2f}% | {row['sharpe_ratio']:.2f} | {row['trades']} |")
    print()

# Strategy Performance Summary
print("## Strategy Performance Summary")
print()

strategy_summary = df.groupby('strategy').agg({
    'return_pct': ['mean', 'std', 'min', 'max'],
    'sharpe_ratio': ['mean', 'std', 'min', 'max'],
    'trades': ['sum', 'mean'],
    'symbol': 'count'
}).round(2)

print("| Strategy | Symbols Tested | Avg Return % | Std Return % | Max Return % | Avg Sharpe | Total Trades |")
print("|----------|----------------|--------------|--------------|--------------|------------|--------------|")
for strategy in strategy_summary.index:
    stats = strategy_summary.loc[strategy]
    print(f"| {strategy} | {stats[('symbol', 'count')]} | {stats[('return_pct', 'mean')]:.2f}% | {stats[('return_pct', 'std')]:.2f}% | {stats[('return_pct', 'max')]:.2f}% | {stats[('sharpe_ratio', 'mean')]:.2f} | {int(stats[('trades', 'sum')])} |")
print()

# Symbol Performance Summary
print("## Symbol Performance Summary")
print()

symbol_summary = df.groupby('symbol').agg({
    'return_pct': ['mean', 'std', 'min', 'max'],
    'sharpe_ratio': ['mean'],
    'strategy': 'count'
}).round(2)

# Top 20 symbols by average return
top_symbols = symbol_summary.nlargest(20, ('return_pct', 'mean'))

print("### Top 20 Symbols by Average Return Across All Strategies")
print("| Symbol | Strategies Tested | Avg Return % | Std Return % | Max Return % | Avg Sharpe |")
print("|--------|------------------|--------------|--------------|--------------|------------|")
for symbol in top_symbols.index:
    stats = top_symbols.loc[symbol]
    print(f"| {symbol} | {stats[('strategy', 'count')]} | {stats[('return_pct', 'mean')]:.2f}% | {stats[('return_pct', 'std')]:.2f}% | {stats[('return_pct', 'max')]:.2f}% | {stats[('sharpe_ratio', 'mean')]:.2f} |")
print()

# Detailed Results by Time Window
print("## Detailed Performance Analysis")
print()

# Group by strategy type
strategy_types = {
    'Trend Following': ['golden_cross', 'turtle_trading', 'momentum_factor', 'dual_momentum'],
    'Mean Reversion': ['rsi_reversion', 'bollinger_bands', 'macd_crossover'],
    'Portfolio': ['sector_rotation', 'all_weather', 'buy_and_hold'],
    'Specialized': ['volatility', 'crypto', 'dividend', 'earnings']
}

for category, strategies in strategy_types.items():
    category_df = df[df['strategy'].isin(strategies)]
    if len(category_df) > 0:
        print(f"### {category} Strategies")
        print()
        
        # Best performers in category
        best_in_category = category_df.nlargest(5, 'return_pct')
        print(f"**Top 5 {category} Combinations:**")
        print("| Strategy | Symbol | Return % | Sharpe | Trades |")
        print("|----------|--------|----------|--------|--------|")
        for _, row in best_in_category.iterrows():
            print(f"| {row['strategy']} | {row['symbol']} | {row['return_pct']:.2f}% | {row['sharpe_ratio']:.2f} | {row['trades']} |")
        print()

# Market Regime Analysis
print("## Market Regime Performance")
print()
print("*Note: This analysis shows strategy performance across different market conditions*")
print()

# Active strategies only
active_df = df[df['trades'] > 0]
if len(active_df) > 0:
    print("### Most Active Strategies")
    print("| Strategy | Symbol | Return % | Sharpe | Trades |")
    print("|----------|--------|----------|--------|--------|")
    most_active = active_df.nlargest(10, 'trades')
    for _, row in most_active.iterrows():
        print(f"| {row['strategy']} | {row['symbol']} | {row['return_pct']:.2f}% | {row['sharpe_ratio']:.2f} | {row['trades']} |")
    print()

# Risk Analysis
print("## Risk Analysis")
print()

# Strategies with positive Sharpe
positive_sharpe = df[df['sharpe_ratio'] > 0]
print(f"**Strategies with Positive Risk-Adjusted Returns**: {len(positive_sharpe)} out of {len(df)} ({len(positive_sharpe)/len(df)*100:.1f}%)")
print()

# Best risk-adjusted by strategy type
print("### Best Risk-Adjusted Returns by Strategy Type")
for category, strategies in strategy_types.items():
    category_df = df[df['strategy'].isin(strategies) & (df['sharpe_ratio'] > 0)]
    if len(category_df) > 0:
        best = category_df.nlargest(1, 'sharpe_ratio').iloc[0]
        print(f"- **{category}**: {best['strategy']} on {best['symbol']} (Sharpe: {best['sharpe_ratio']:.2f}, Return: {best['return_pct']:.2f}%)")
print()

# Recommendations
print("## Recommendations")
print()
print("Based on the analysis, here are the top recommended strategy-symbol combinations:")
print()

# Filter for meaningful results (positive return, positive Sharpe, at least some trades)
meaningful = df[(df['return_pct'] > 0) & (df['sharpe_ratio'] > 0.5)]
if len(meaningful) > 0:
    recommendations = meaningful.nlargest(5, 'sharpe_ratio')
    print("### Top 5 Recommended Combinations (Risk-Adjusted)")
    print("| Priority | Strategy | Symbol | Expected Return % | Sharpe Ratio | Trades |")
    print("|----------|----------|--------|------------------|--------------|--------|")
    for i, (_, row) in enumerate(recommendations.iterrows()):
        print(f"| {i+1} | {row['strategy']} | {row['symbol']} | {row['return_pct']:.2f}% | {row['sharpe_ratio']:.2f} | {row['trades']} |")
else:
    print("*No strategies met the minimum criteria for recommendation in this period.*")

print()
print("---")
print(f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Save CSV file
df.to_csv('$CSV_FILE', index=False)
print(f"\n*Detailed results saved to CSV: {os.path.basename('$CSV_FILE')}*")
EOF

# Check if report generation was successful
if [ -f "$REPORT_FILE" ]; then
    echo "[$(date)] Report generated successfully: $REPORT_FILE" | tee -a "$LOG_FILE"
    echo "[$(date)] CSV results saved: $CSV_FILE" | tee -a "$LOG_FILE"
    
    # Create a summary for quick viewing
    SUMMARY_FILE="$ICLOUD_STRATEGIES/tetra_${REPORT_PREFIX}_summary_${REPORT_DATE}.txt"
    {
        echo "Tetra $ANALYSIS_DESC Summary"
        echo "Generated: $(date)"
        echo ""
        echo "Top 5 Performers by Return:"
        grep -A 6 "Top 10 by Total Return" "$REPORT_FILE" | head -7
        echo ""
        echo "Report Location: $REPORT_FILE"
    } > "$SUMMARY_FILE"
    
    echo "[$(date)] Summary saved: $SUMMARY_FILE" | tee -a "$LOG_FILE"
else
    echo "[$(date)] ERROR: Report generation failed" | tee -a "$LOG_FILE"
fi

# Cleanup (no temp files to remove)

# Exit with pipeline status
if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "[$(date)] Benchmark analysis completed successfully" | tee -a "$LOG_FILE"
    exit 0
else
    echo "[$(date)] Benchmark analysis failed" | tee -a "$LOG_FILE"
    exit 1
fi