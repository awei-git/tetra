"""Router for trading recommendations endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import json
from pathlib import Path

router = APIRouter()

def load_recommendations():
    """Load the latest trading recommendations."""
    try:
        # Path to recommendations file - adjusted for backend location
        rec_path = Path("../data/assessment/TRADING_RECOMMENDATIONS.md")
        
        if not rec_path.exists():
            # Try alternative path
            rec_path = Path("data/assessment/TRADING_RECOMMENDATIONS.md")
            if not rec_path.exists():
                return None
            
        # Read the markdown file
        with open(rec_path, 'r') as f:
            content = f.read()
        
        # Parse the markdown into structured data
        recommendations = parse_recommendations_md(content)
        
        # Add additional data from assessment results
        assessment_path = rec_path.parent / "assessment_pipeline_summary.json"
        if assessment_path.exists():
            with open(assessment_path, 'r') as f:
                assessment = json.load(f)
                recommendations['assessment_summary'] = assessment
        
        # Load detailed backtest results
        results_path = rec_path.parent / "backtesting_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            # Convert to dict, handling NaN values
            recommendations['detailed_results'] = df.fillna(0).to_dict('records')
        
        return recommendations
        
    except Exception as e:
        print(f"Error loading recommendations: {e}")
        return None


def parse_recommendations_md(content: str) -> Dict[str, Any]:
    """Parse markdown recommendations into structured data."""
    
    lines = content.split('\n')
    recommendations = {
        'generated_at': None,
        'symbol_actions': [],
        'portfolio_allocation': [],
        'risk_warnings': [],
        'implementation_notes': [],
        'execution_instructions': []
    }
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Parse generation timestamp
        if line.startswith('Generated:'):
            recommendations['generated_at'] = line.split('Generated:')[1].strip()
        
        # Identify sections
        elif '## 📊 SYMBOL-SPECIFIC ACTIONS' in line:
            current_section = 'symbols'
        elif '## 💰 PORTFOLIO ALLOCATION' in line:
            current_section = 'allocation'
        elif '## ⚠️ RISK WARNINGS' in line:
            current_section = 'risks'
        elif '## 📝 IMPLEMENTATION NOTES' in line:
            current_section = 'implementation'
        
        # Parse symbol recommendations
        elif current_section == 'symbols' and line.startswith('### '):
            symbol_line = line.replace('### ', '').replace('✅ ', '')
            if ':' in symbol_line:
                symbol, strategy = symbol_line.split(':', 1)
                symbol_rec = {
                    'symbol': symbol.strip(),
                    'strategy': strategy.strip().replace('**', ''),
                    'metrics': {}
                }
                recommendations['symbol_actions'].append(symbol_rec)
        
        # Parse metrics for current symbol
        elif current_section == 'symbols' and recommendations['symbol_actions'] and line.startswith('- '):
            metric_line = line[2:]  # Remove '- '
            if ':' in metric_line:
                key, value = metric_line.split(':', 1)
                current_symbol = recommendations['symbol_actions'][-1]
                current_symbol['metrics'][key.strip()] = value.strip()
        
        # Parse allocation table
        elif current_section == 'allocation' and '|' in line and not line.startswith('|---'):
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4 and parts[0] not in ['Asset', '']:
                recommendations['portfolio_allocation'].append({
                    'asset': parts[0],
                    'strategy': parts[1],
                    'allocation': parts[2],
                    'amount': parts[3]
                })
        
        # Parse risk warnings
        elif current_section == 'risks' and line.startswith('- '):
            recommendations['risk_warnings'].append(line[2:])
        
        # Parse implementation notes
        elif current_section == 'implementation' and line and line[0].isdigit():
            recommendations['implementation_notes'].append(line)
    
    # Add execution instructions for each strategy
    recommendations['execution_instructions'] = generate_execution_instructions(recommendations)
    
    return recommendations


def generate_execution_instructions(recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate detailed execution instructions for each recommended strategy."""
    
    instructions = []
    
    for symbol_action in recommendations['symbol_actions']:
        symbol = symbol_action['symbol']
        strategy = symbol_action['strategy']
        metrics = symbol_action.get('metrics', {})
        
        instruction = {
            'symbol': symbol,
            'strategy': strategy,
            'action': 'BUY',
            'timing': 'Market Open',
            'steps': []
        }
        
        # Strategy-specific instructions
        if 'Buy and Hold' in strategy:
            instruction['steps'] = [
                f"Place market order for {symbol} at market open",
                "Use limit order if you want better entry: Set limit at previous close -0.5%",
                "Position size: Follow allocation table (typically 10-20% of portfolio)",
                "No stop loss for buy-and-hold strategy",
                "Hold for minimum 6-12 months",
                "Rebalance quarterly based on new assessments"
            ]
            instruction['risk_management'] = "Monitor daily for 20%+ drawdowns"
            
        elif 'ML' in strategy or 'Machine Learning' in strategy:
            instruction['steps'] = [
                f"Check ML prediction signal for {symbol} (should be positive)",
                "Enter with 50% of allocated position at market open",
                "Add remaining 50% if price moves up 1% from entry",
                "Set stop loss at -5% from entry price",
                "Take partial profits (50%) at +10%",
                "Let remaining position run with trailing stop at -8%"
            ]
            instruction['risk_management'] = "Exit immediately if ML confidence drops below 60%"
            
        elif 'Momentum' in strategy:
            instruction['steps'] = [
                f"Verify {symbol} is above 50-day SMA",
                "Check RSI is between 50-70 (not overbought)",
                "Enter position when price breaks above previous day high",
                "Set initial stop at -3% from entry",
                "Trail stop to breakeven after +5% gain",
                "Exit when price closes below 20-day SMA"
            ]
            instruction['risk_management'] = "Maximum position size: 5% of portfolio"
            
        elif 'Event' in strategy or 'Earnings' in strategy:
            instruction['steps'] = [
                f"Check earnings calendar for {symbol}",
                "Enter position 3-5 days before earnings",
                "Position size: 50% of normal allocation",
                "Exit day before earnings announcement",
                "Re-enter after earnings if surprise > 5%",
                "Hold post-earnings position for 2-3 days maximum"
            ]
            instruction['risk_management'] = "Never hold through earnings with full position"
            
        elif 'Golden Cross' in strategy:
            instruction['steps'] = [
                f"Verify 50-day SMA crossed above 200-day SMA for {symbol}",
                "Wait for pullback to 50-day SMA for entry",
                "Enter with 70% of allocation on pullback",
                "Add remaining 30% on breakout above recent high",
                "Set stop loss at 200-day SMA",
                "Exit when 50-day SMA crosses below 200-day SMA"
            ]
            instruction['risk_management'] = "Reduce position by 50% if drawdown exceeds 10%"
        
        else:
            # Generic instructions
            instruction['steps'] = [
                f"Review current price and volume for {symbol}",
                "Enter position at market open with limit order",
                "Position size as per allocation table",
                "Set stop loss at -5% from entry",
                "Take profits at +15% or based on strategy signals",
                "Monitor daily and adjust based on market conditions"
            ]
            instruction['risk_management'] = "Standard 2% portfolio risk per trade"
        
        # Add entry criteria
        instruction['entry_criteria'] = {
            'price_action': 'Bullish' if 'Expected Return' in metrics and float(metrics.get('Expected Return', '0%').replace('%', '')) > 0 else 'Bearish',
            'volume': 'Above average volume preferred',
            'market_conditions': 'Avoid entry during high VIX (>25)'
        }
        
        # Add position sizing
        allocation = next((a for a in recommendations['portfolio_allocation'] 
                          if a['asset'] == symbol), None)
        if allocation:
            instruction['position_sizing'] = {
                'recommended_allocation': allocation['allocation'],
                'dollar_amount': allocation['amount'],
                'share_calculation': f"Shares = {allocation['amount']} / Current Price"
            }
        
        instructions.append(instruction)
    
    return instructions


@router.get("/latest")
async def get_latest_recommendations() -> Dict[str, Any]:
    """Get the latest trading recommendations."""
    
    recommendations = load_recommendations()
    
    if not recommendations:
        return {
            "success": False,
            "message": "No recommendations available. Run assessment pipeline first."
        }
    
    # Check if recommendations are stale (older than 24 hours)
    if recommendations.get('generated_at'):
        try:
            gen_time = datetime.fromisoformat(recommendations['generated_at'])
            if datetime.now() - gen_time > timedelta(hours=24):
                recommendations['stale_warning'] = "Recommendations are older than 24 hours. Consider running assessment pipeline."
        except:
            pass
    
    return {
        "success": True,
        "data": recommendations
    }


@router.get("/summary")
async def get_recommendations_summary() -> Dict[str, Any]:
    """Get a summary of recommendations."""
    
    recommendations = load_recommendations()
    
    if not recommendations:
        return {
            "success": True,
            "data": {
                "has_recommendations": False,
                "message": "No recommendations available"
            }
        }
    
    # Create summary
    summary = {
        "has_recommendations": True,
        "generated_at": recommendations.get('generated_at'),
        "total_positions": len(recommendations.get('symbol_actions', [])),
        "top_pick": None,
        "portfolio_stats": {
            "total_allocated": 0,
            "cash_reserve": 0,
            "expected_return": 0
        }
    }
    
    # Get top pick (highest expected return)
    if recommendations.get('symbol_actions'):
        for action in recommendations['symbol_actions']:
            metrics = action.get('metrics', {})
            expected_return = metrics.get('Expected Return', '0%')
            try:
                return_val = float(expected_return.replace('%', ''))
                if not summary['top_pick'] or return_val > float(summary['top_pick']['metrics'].get('Expected Return', '0%').replace('%', '')):
                    summary['top_pick'] = action
            except:
                pass
    
    # Calculate portfolio stats
    total_allocated = 0
    for allocation in recommendations.get('portfolio_allocation', []):
        if 'Cash' not in allocation['asset']:
            amount_str = allocation['amount'].replace('$', '').replace(',', '')
            try:
                total_allocated += float(amount_str)
            except:
                pass
        else:
            amount_str = allocation['amount'].replace('$', '').replace(',', '')
            try:
                summary['portfolio_stats']['cash_reserve'] = float(amount_str)
            except:
                pass
    
    summary['portfolio_stats']['total_allocated'] = total_allocated
    
    return {
        "success": True,
        "data": summary
    }