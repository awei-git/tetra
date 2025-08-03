from typing import Dict, List


class SchemaRegistry:
    """Registry of database schemas for LLM context"""
    
    def __init__(self):
        self.schemas = {
            "market_data": {
                "ohlcv": {
                    "description": "OHLCV (Open, High, Low, Close, Volume) price data",
                    "columns": {
                        "symbol": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                        "timestamp": "Timestamp with timezone",
                        "open": "Opening price",
                        "high": "Highest price",
                        "low": "Lowest price",
                        "close": "Closing price",
                        "volume": "Trading volume",
                        "vwap": "Volume weighted average price",
                        "timeframe": "Time interval (e.g., '1d', '1h')"
                    },
                    "indexes": ["symbol", "timestamp"],
                    "example_queries": [
                        "Latest price for AAPL",
                        "AAPL prices for last 30 days",
                        "Top 10 stocks by volume today"
                    ]
                }
            },
            "economic_data": {
                "economic_data": {
                    "description": "Economic indicators and macro data",
                    "columns": {
                        "symbol": "Indicator symbol (e.g., 'GDPC1', 'CPIAUCSL', 'UNRATE')",
                        "date": "Date of the data point",
                        "value": "Indicator value",
                        "source": "Data source (e.g., 'FRED')",
                        "is_preliminary": "Whether the data is preliminary"
                    },
                    "indexes": ["symbol", "date"],
                    "example_queries": [
                        "Latest GDP data",
                        "US unemployment rate history",
                        "CPI inflation data"
                    ]
                }
            },
            "news": {
                "news_articles": {
                    "description": "Financial news articles",
                    "columns": {
                        "title": "Article title",
                        "description": "Article summary",
                        "source": "News source",
                        "published_at": "Publication timestamp",
                        "url": "Article URL"
                    },
                    "indexes": ["published_at", "source"]
                },
                "news_sentiments": {
                    "description": "Sentiment analysis of news articles",
                    "columns": {
                        "article_id": "Reference to news_articles",
                        "symbols": "Related stock symbols",
                        "polarity": "Sentiment score (-1 to 1)",
                        "relevance_score": "Relevance to symbols (0 to 1)"
                    }
                }
            },
            "events": {
                "event_data": {
                    "description": "Market events (earnings, dividends, etc.)",
                    "columns": {
                        "event_type": "Type of event",
                        "event_datetime": "When the event occurs",
                        "affected_symbols": "Symbols affected by event",
                        "description": "Event description"
                    },
                    "indexes": ["event_type", "event_datetime"]
                }
            }
        }
    
    def get_schema_context(self) -> str:
        """Format schema information for LLM context"""
        context_lines = []
        
        for schema_name, tables in self.schemas.items():
            context_lines.append(f"\nSchema: {schema_name}")
            context_lines.append("=" * 40)
            
            for table_name, table_info in tables.items():
                context_lines.append(f"\nTable: {schema_name}.{table_name}")
                context_lines.append(f"Description: {table_info['description']}")
                context_lines.append("\nColumns:")
                
                for col_name, col_desc in table_info["columns"].items():
                    context_lines.append(f"  - {col_name}: {col_desc}")
                
                if "indexes" in table_info:
                    context_lines.append(f"\nIndexed on: {', '.join(table_info['indexes'])}")
                
                if "example_queries" in table_info:
                    context_lines.append("\nExample queries:")
                    for example in table_info["example_queries"]:
                        context_lines.append(f"  - {example}")
        
        return "\n".join(context_lines)
    
    def get_table_list(self) -> List[str]:
        """Get list of all tables with schema prefix"""
        tables = []
        for schema_name, schema_tables in self.schemas.items():
            for table_name in schema_tables:
                tables.append(f"{schema_name}.{table_name}")
        return tables