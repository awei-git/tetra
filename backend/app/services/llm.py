from typing import List, Dict, Any, Optional
import logging
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import json
import pandas as pd
import numpy as np
from sqlalchemy import text

from ..config import settings
from .schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM integration (SQL generation and analysis)"""
    
    def __init__(self):
        if settings.llm_provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API key not found")
            self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        elif settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not found")
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            raise NotImplementedError(f"LLM provider {settings.llm_provider} not implemented")
        
        self.provider = settings.llm_provider
        self.schema_registry = SchemaRegistry()
    
    async def understand_query(self, natural_language_query: str) -> Dict[str, Any]:
        """Understand the intent and determine the appropriate action"""
        
        # Get schema context
        schema_context = self.schema_registry.get_schema_context()
        
        system_prompt = f"""You are a financial data analyst with deep knowledge of market data and trading strategies.
        
You have access to a database with the following schema:
{schema_context}

Your job is to understand what the user is asking and determine the best approach to answer their question.

For each query, you should return a JSON response with:
1. "intent": The type of query (data_retrieval, strategy_analysis, comparison, education, calculation)
2. "approach": How to handle it (sql_query, multi_step_analysis, explanation, calculation)
3. "details": Specific details needed for the approach

Examples:
- "Show me AAPL price" -> {{"intent": "data_retrieval", "approach": "sql_query", "details": {{"query": "SELECT * FROM market_data.ohlcv WHERE symbol = 'AAPL' ORDER BY timestamp DESC LIMIT 100"}}}}
- "How do I trade SPY/TLT pairs?" -> {{"intent": "strategy_analysis", "approach": "multi_step_analysis", "details": {{"strategy": "pair_trading", "symbols": ["SPY", "TLT"], "steps": ["fetch_correlation", "calculate_ratio", "analyze_zscore", "generate_signals"]}}}}
- "What signals for momentum trading?" -> {{"intent": "education", "approach": "explanation", "details": {{"topic": "momentum_indicators", "examples": ["RSI", "MACD", "Moving Averages"]}}}}

Always return valid JSON."""
        
        user_prompt = f"Analyze this query and determine the best approach: {natural_language_query}"
        
        try:
            if self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=settings.llm_model,
                    max_tokens=1000,
                    temperature=0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                result = response.content[0].text.strip()
            else:  # OpenAI
                response = await self.client.chat.completions.create(
                    model=settings.llm_model,
                    max_tokens=1000,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # First clean up the response if it has markdown code blocks
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            
            parsed = json.loads(result.strip())
            logger.info(f"Parsed understanding: {parsed}")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}, Raw result: {result}")
            # Fallback to SQL generation if JSON parsing fails
            return {
                "intent": "data_retrieval",
                "approach": "sql_query",
                "details": {"needs_sql_generation": True}
            }
        except Exception as e:
            logger.error(f"Error understanding query: {str(e)}")
            raise
    
    async def generate_sql(self, natural_language_query: str) -> str:
        """Generate SQL from natural language query"""
        
        # Get schema context
        schema_context = self.schema_registry.get_schema_context()
        
        system_prompt = """You are a PostgreSQL expert helping users query their financial database.
        
Database Schema:
{schema}

IMPORTANT: You can generate sophisticated queries for complex financial analysis:

For pair trading questions (e.g., "how to trade SPY/TLT pair"):
Generate a query that calculates:
- Price ratio between the two symbols
- Z-score of the ratio
- Trading signals based on z-score
Example:
WITH pair_data AS (
    SELECT DATE(timestamp) as date, symbol, close
    FROM market_data.ohlcv
    WHERE symbol IN ('SPY', 'TLT')
    AND timestamp >= CURRENT_DATE - INTERVAL '90 days'
    AND timeframe = '1d'
),
pivoted AS (
    SELECT date,
           MAX(CASE WHEN symbol = 'SPY' THEN close END) as spy_close,
           MAX(CASE WHEN symbol = 'TLT' THEN close END) as tlt_close
    FROM pair_data
    GROUP BY date
    HAVING COUNT(DISTINCT symbol) = 2
),
ratio_calc AS (
    SELECT *,
           spy_close / tlt_close as ratio
    FROM pivoted
),
stats AS (
    SELECT AVG(ratio) as mean_ratio,
           STDDEV(ratio) as std_ratio
    FROM ratio_calc
)
SELECT r.date,
       r.spy_close,
       r.tlt_close,
       r.ratio,
       s.mean_ratio,
       s.std_ratio,
       (r.ratio - s.mean_ratio) / s.std_ratio as z_score,
       CASE 
           WHEN (r.ratio - s.mean_ratio) / s.std_ratio > 2 THEN 'SELL SPY/BUY TLT'
           WHEN (r.ratio - s.mean_ratio) / s.std_ratio < -2 THEN 'BUY SPY/SELL TLT'
           ELSE 'NEUTRAL'
       END as signal
FROM ratio_calc r
CROSS JOIN stats s
ORDER BY date DESC
LIMIT 30;

Important Rules:
1. Generate ONLY valid PostgreSQL SELECT queries
2. Use proper schema prefixes (e.g., market_data.ohlcv)
3. Always use proper date filtering for time-series data
4. Handle market holidays and weekends appropriately
5. Use efficient queries with proper indexes
6. Return only the SQL query, no explanations
7. ALWAYS include a complete query with FROM clause
8. Output MUST be a complete, executable SQL statement
9. When using GROUP BY, ensure ALL non-aggregate columns in SELECT are included in GROUP BY
10. For performance calculations, use proper window functions or aggregate functions
11. DO NOT filter out future dates unless explicitly requested - the database may contain forecast or simulation data
12. When querying "latest" data, use ORDER BY timestamp DESC LIMIT 1 instead of date filters
13. For queries about stock price and volume, always include the volume column when available
14. Use PostgreSQL INTERVAL syntax correctly: '3 months', '90 days', '1 year', NOT '1 quarter'
15. For time periods like "past quarter", use: timestamp >= CURRENT_DATE - INTERVAL '3 months'
16. IMPORTANT: When comparing different stocks over DIFFERENT time periods, calculate each separately with their own date filters
17. For return calculations, use (last_close - first_open) / first_open * 100

Common patterns:
- Latest price: SELECT * FROM market_data.ohlcv WHERE symbol = 'AAPL' ORDER BY timestamp DESC LIMIT 1
- Date range: SELECT * FROM market_data.ohlcv WHERE timestamp BETWEEN '2024-01-01' AND '2025-12-31'
- Past quarter: SELECT * FROM market_data.ohlcv WHERE symbol = 'PLTR' AND timestamp >= CURRENT_DATE - INTERVAL '3 months' ORDER BY timestamp
- Past N days: SELECT * FROM market_data.ohlcv WHERE symbol = 'AAPL' AND timestamp >= CURRENT_DATE - INTERVAL '30 days'
- All data for a symbol: SELECT * FROM market_data.ohlcv WHERE symbol = 'AAPL' ORDER BY timestamp
- Aggregations: SELECT DATE_TRUNC('day', timestamp) as date, AVG(close), SUM(volume) as volume FROM market_data.ohlcv GROUP BY date
- GDP data: SELECT date, value FROM economic_data.economic_data WHERE symbol = 'GDPC1' ORDER BY date DESC LIMIT 10
- GDP growth: SELECT date, value FROM economic_data.economic_data WHERE symbol = 'A191RL1Q225SBEA' ORDER BY date DESC LIMIT 10
- Economic indicators: SELECT DISTINCT symbol FROM economic_data.economic_data ORDER BY symbol
- Stock performance: 
  WITH first_last AS (
    SELECT symbol,
           FIRST_VALUE(open) OVER (PARTITION BY symbol ORDER BY timestamp) as first_open,
           LAST_VALUE(close) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_close
    FROM market_data.ohlcv
    WHERE timestamp >= '2024-04-01'
  )
  SELECT DISTINCT symbol, 
         ((last_close - first_open) / first_open * 100) as performance
  FROM first_last
  ORDER BY performance DESC
  LIMIT 5
- Compare different time periods:
  WITH pltr_1y AS (
    SELECT MIN(open) as first_open, MAX(close) as last_close
    FROM market_data.ohlcv
    WHERE symbol = 'PLTR' AND timestamp >= CURRENT_DATE - INTERVAL '1 year'
  ),
  meta_5y AS (
    SELECT MIN(open) as first_open, MAX(close) as last_close
    FROM market_data.ohlcv
    WHERE symbol = 'META' AND timestamp >= CURRENT_DATE - INTERVAL '5 years'
  )
  SELECT 'PLTR (1 year)' as stock_period, ((last_close - first_open) / first_open * 100) as return_pct FROM pltr_1y
  UNION ALL
  SELECT 'META (5 years)', ((last_close - first_open) / first_open * 100) FROM meta_5y
""".format(schema=schema_context)
        
        user_prompt = f"Generate a complete SQL query for: {natural_language_query}\n\nRemember to include the full query with SELECT, FROM, and any necessary clauses."
        
        try:
            if self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=settings.llm_model,
                    max_tokens=1000,
                    temperature=0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                sql = response.content[0].text.strip()
            else:  # OpenAI
                response = await self.client.chat.completions.create(
                    model=settings.llm_model,
                    max_tokens=1000,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                sql = response.choices[0].message.content.strip()
            
            # Clean up SQL if wrapped in markdown
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            
            sql = sql.strip()
            
            # Fix common SQL generation issues
            sql = self._fix_common_sql_issues(sql)
            
            # Log the raw response for debugging
            logger.info(f"Raw LLM response: {repr(sql)}")
            
            # Validate that we got a complete query
            sql_normalized = sql.replace('\n', ' ').replace('\r', ' ').strip()
            if len(sql_normalized) < 20 or ' FROM ' not in sql_normalized.upper():
                logger.error(f"Incomplete SQL generated: {sql}")
                # Try to generate a more explicit query
                if 'symbol' in natural_language_query.lower():
                    return "SELECT DISTINCT symbol FROM market_data.ohlcv ORDER BY symbol"
                else:
                    return "SELECT * FROM market_data.ohlcv LIMIT 10"
            
            return sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise
    
    async def analyze_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        columns: List[str]
    ) -> str:
        """Analyze query results and provide insights"""
        
        # Format results for LLM
        results_summary = {
            "columns": columns,
            "row_count": len(results),
            "sample_data": results[:5]  # First 5 rows
        }
        
        system_prompt = """You are a financial data analyst providing insights on query results.

Provide a brief, insightful analysis focusing on:
1. Key patterns or trends in the data
2. Notable values or anomalies
3. Practical implications for trading/investing
4. Any data quality observations

Keep the analysis concise (2-3 sentences) and actionable."""
        
        user_prompt = f"""
Original query: {query}

Results summary:
{json.dumps(results_summary, indent=2, default=str)}

Provide analysis of these results.
"""
        
        try:
            if self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=settings.llm_model,
                    max_tokens=500,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text.strip()
            else:  # OpenAI
                response = await self.client.chat.completions.create(
                    model=settings.llm_model,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            return "Unable to analyze results at this time."
    
    def _fix_common_sql_issues(self, sql: str) -> str:
        """Fix common SQL generation issues"""
        import re
        
        # Fix interval syntax - PostgreSQL doesn't support 'quarter'
        sql = re.sub(r"INTERVAL\s+'(\d+)\s+quarter'", r"INTERVAL '\1 months' * 3", sql, flags=re.IGNORECASE)
        sql = re.sub(r"INTERVAL\s+'1\s+quarter'", r"INTERVAL '3 months'", sql, flags=re.IGNORECASE)
        sql = re.sub(r"INTERVAL\s+'(\d+)\s+quarters'", lambda m: f"INTERVAL '{int(m.group(1)) * 3} months'", sql, flags=re.IGNORECASE)
        
        # Fix common timeframe issues
        sql = re.sub(r"timeframe\s*=\s*'daily'", r"timeframe = '1d'", sql, flags=re.IGNORECASE)
        sql = re.sub(r"timeframe\s*=\s*'hourly'", r"timeframe = '1h'", sql, flags=re.IGNORECASE)
        sql = re.sub(r"timeframe\s*=\s*'weekly'", r"timeframe = '1w'", sql, flags=re.IGNORECASE)
        
        return sql
    
    async def execute_strategy_analysis(
        self, 
        strategy: str, 
        details: Dict[str, Any],
        db_session: Any
    ) -> Dict[str, Any]:
        """Execute complex strategy analysis based on LLM understanding"""
        
        if strategy == "pair_trading":
            return await self._analyze_pair_trading(details, db_session)
        elif strategy == "momentum":
            return await self._analyze_momentum(details, db_session)
        elif strategy == "correlation":
            return await self._analyze_correlation(details, db_session)
        else:
            return {"error": f"Unknown strategy: {strategy}"}
    
    async def _analyze_pair_trading(self, details: Dict[str, Any], db_session: Any) -> Dict[str, Any]:
        """Perform pair trading analysis"""
        symbols = details.get("symbols", [])
        if len(symbols) != 2:
            return {"error": "Pair trading requires exactly 2 symbols"}
        
        # Fetch correlation data using parameterized query
        correlation_query = """
        WITH price_data AS (
            SELECT DATE(timestamp) as date, symbol, close
            FROM market_data.ohlcv
            WHERE symbol IN (:symbol1, :symbol2)
            AND timestamp >= CURRENT_DATE - INTERVAL '90 days'
            AND timeframe = '1d'
        ),
        pivoted AS (
            SELECT date,
                   MAX(CASE WHEN symbol = :symbol1 THEN close END) as symbol1_close,
                   MAX(CASE WHEN symbol = :symbol2 THEN close END) as symbol2_close
            FROM price_data
            GROUP BY date
            HAVING COUNT(DISTINCT symbol) = 2
        )
        SELECT * FROM pivoted ORDER BY date
        """
        
        # Execute query with parameters
        result = await db_session.execute(
            text(correlation_query),
            {"symbol1": symbols[0], "symbol2": symbols[1]}
        )
        rows = result.fetchall()
        
        if not rows:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate statistics
        # Convert rows to list of dicts
        data = []
        for row in rows:
            symbol1_close = float(row[1])
            symbol2_close = float(row[2])
            data.append({
                'date': str(row[0]),
                f'{symbols[0]}_close': symbol1_close,
                f'{symbols[1]}_close': symbol2_close,
                'ratio': symbol1_close / symbol2_close if symbol2_close != 0 else 0
            })
        
        df = pd.DataFrame(data)
        # Use the actual column names from the pivoted query
        df['symbol1_close'] = df[f'{symbols[0]}_close']
        df['symbol2_close'] = df[f'{symbols[1]}_close']
        df['ratio'] = df['symbol1_close'] / df['symbol2_close']
        
        correlation = df[f'{symbols[0]}_close'].corr(df[f'{symbols[1]}_close'])
        mean_ratio = df['ratio'].mean()
        std_ratio = df['ratio'].std()
        current_ratio = df['ratio'].iloc[-1]
        z_score = (current_ratio - mean_ratio) / std_ratio
        
        # Generate insights
        insights = await self._generate_pair_trading_insights(
            symbols, correlation, z_score, mean_ratio, current_ratio
        )
        
        return {
            "analysis_type": "pair_trading",
            "symbols": symbols,
            "statistics": {
                "correlation": round(correlation, 3),
                "current_ratio": round(current_ratio, 4),
                "mean_ratio": round(mean_ratio, 4),
                "z_score": round(z_score, 2)
            },
            "insights": insights,
            "data": data[-20:],  # Last 20 data points as dicts
            "columns": ["date", f"{symbols[0]}_close", f"{symbols[1]}_close", "ratio"]
        }
    
    async def _generate_pair_trading_insights(self, symbols, correlation, z_score, mean_ratio, current_ratio):
        """Generate natural language insights for pair trading"""
        
        prompt = f"""
        Generate actionable pair trading insights for {symbols[0]}/{symbols[1]}:
        - Correlation: {correlation:.3f}
        - Current ratio: {current_ratio:.4f}
        - Mean ratio: {mean_ratio:.4f}
        - Z-score: {z_score:.2f}
        
        Provide:
        1. Assessment of pair suitability
        2. Current signal (if any)
        3. Specific entry/exit recommendations
        4. Risk management suggestions
        
        Be specific and actionable.
        """
        
        if self.provider == "anthropic":
            response = await self.client.messages.create(
                model=settings.llm_model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        else:  # OpenAI
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
    
    async def generate_educational_content(self, topic: str, examples: List[str]) -> str:
        """Generate educational content about trading strategies"""
        
        prompt = f"""
        Explain {topic} in the context of trading, focusing on:
        - What it is and why it matters
        - Key indicators to watch: {', '.join(examples)}
        - How to use it in practice
        - Common pitfalls to avoid
        
        Keep it concise but actionable.
        """
        
        if self.provider == "anthropic":
            response = await self.client.messages.create(
                model=settings.llm_model,
                max_tokens=800,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        else:  # OpenAI
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=800,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()