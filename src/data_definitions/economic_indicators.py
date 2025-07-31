"""Centralized economic indicators definition for all economic data we track"""

from typing import List, Dict, Set, Tuple
from enum import Enum


class UpdateFrequency(Enum):
    """Update frequency for economic indicators"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class EconomicIndicators:
    """Central definition of economic indicators"""
    
    # Interest Rates
    INTEREST_RATES = [
        ("DFF", "Federal Funds Rate", UpdateFrequency.DAILY),
        ("DFEDTARU", "Fed Funds Target Rate - Upper", UpdateFrequency.DAILY),
        ("DFEDTARL", "Fed Funds Target Rate - Lower", UpdateFrequency.DAILY),
        ("TB3MS", "3-Month Treasury Bill", UpdateFrequency.MONTHLY),
        ("DGS2", "2-Year Treasury", UpdateFrequency.DAILY),
        ("DGS5", "5-Year Treasury", UpdateFrequency.DAILY),
        ("DGS10", "10-Year Treasury", UpdateFrequency.DAILY),
        ("DGS30", "30-Year Treasury", UpdateFrequency.DAILY),
        ("TEDRATE", "TED Spread", UpdateFrequency.DAILY),
        ("SOFR", "Secured Overnight Financing Rate", UpdateFrequency.DAILY),
    ]
    
    # Inflation Indicators
    INFLATION_INDICATORS = [
        ("CPIAUCSL", "CPI Urban", UpdateFrequency.MONTHLY),
        ("CPILFESL", "Core CPI", UpdateFrequency.MONTHLY),
        ("PCEPI", "PCE Inflation", UpdateFrequency.MONTHLY),
        ("PCEPILFE", "Core PCE", UpdateFrequency.MONTHLY),
        ("T5YIE", "5-Year Inflation Expectations", UpdateFrequency.DAILY),
        ("T10YIE", "10-Year Inflation Expectations", UpdateFrequency.DAILY),
        ("PPIFIS", "PPI Final Demand", UpdateFrequency.MONTHLY),
    ]
    
    # Economic Growth
    ECONOMIC_GROWTH = [
        ("GDPC1", "Real GDP", UpdateFrequency.QUARTERLY),
        ("A191RL1Q225SBEA", "GDP Growth Rate", UpdateFrequency.QUARTERLY),
        ("INDPRO", "Industrial Production", UpdateFrequency.MONTHLY),
        ("TCU", "Capacity Utilization", UpdateFrequency.MONTHLY),
        ("RSXFS", "Retail Sales", UpdateFrequency.MONTHLY),
        ("PCE", "Personal Consumption Expenditures", UpdateFrequency.MONTHLY),
    ]
    
    # Labor Market
    LABOR_MARKET = [
        ("UNRATE", "Unemployment Rate", UpdateFrequency.MONTHLY),
        ("PAYEMS", "Non-Farm Payrolls", UpdateFrequency.MONTHLY),
        ("ICSA", "Initial Jobless Claims", UpdateFrequency.WEEKLY),
        ("CCSA", "Continuing Claims", UpdateFrequency.WEEKLY),
        ("CES0500000003", "Average Hourly Earnings", UpdateFrequency.MONTHLY),
        ("CIVPART", "Labor Force Participation Rate", UpdateFrequency.MONTHLY),
        ("JTSJOL", "Job Openings", UpdateFrequency.MONTHLY),
    ]
    
    # Housing Market
    HOUSING_MARKET = [
        ("HOUST", "Housing Starts", UpdateFrequency.MONTHLY),
        ("PERMIT", "Building Permits", UpdateFrequency.MONTHLY),
        ("HSN1F", "New Home Sales", UpdateFrequency.MONTHLY),
        ("EXHOSLUSM495S", "Existing Home Sales", UpdateFrequency.MONTHLY),
        ("CSUSHPISA", "Case-Shiller Home Price Index", UpdateFrequency.MONTHLY),
        ("MORTGAGE30US", "30-Year Mortgage Rate", UpdateFrequency.WEEKLY),
    ]
    
    # Consumer Sentiment
    CONSUMER_SENTIMENT = [
        ("UMCSENT", "U Michigan Consumer Sentiment", UpdateFrequency.MONTHLY),
        ("PSAVERT", "Personal Savings Rate", UpdateFrequency.MONTHLY),
    ]
    
    # Manufacturing & Business
    MANUFACTURING_BUSINESS = [
        ("MANEMP", "ISM Manufacturing Employment", UpdateFrequency.MONTHLY),
        ("NMFBAI", "ISM Non-Manufacturing Index", UpdateFrequency.MONTHLY),
        ("DGORDER", "Durable Goods Orders", UpdateFrequency.MONTHLY),
        ("BUSINV", "Business Inventories", UpdateFrequency.MONTHLY),
    ]
    
    # Monetary Policy
    MONETARY_POLICY = [
        ("M2SL", "M2 Money Supply", UpdateFrequency.MONTHLY),
        ("WALCL", "Fed Balance Sheet", UpdateFrequency.WEEKLY),
        ("TOTRESNS", "Bank Reserves", UpdateFrequency.MONTHLY),
        ("M2V", "Velocity of M2", UpdateFrequency.QUARTERLY),
    ]
    
    # Fiscal Policy
    FISCAL_POLICY = [
        ("GFDEBTN", "Federal Debt", UpdateFrequency.QUARTERLY),
        ("GFDEGDQ188S", "Federal Debt to GDP", UpdateFrequency.QUARTERLY),
        ("MTSDS133FMS", "Federal Surplus/Deficit", UpdateFrequency.MONTHLY),
        ("W006RC1Q027SBEA", "Federal Tax Receipts", UpdateFrequency.QUARTERLY),
        ("W018RC1Q027SBEA", "Government Spending", UpdateFrequency.QUARTERLY),
    ]
    
    # Market Indicators
    MARKET_INDICATORS = [
        ("VIXCLS", "VIX", UpdateFrequency.DAILY),
        ("DTWEXBGS", "Trade Weighted Dollar Index", UpdateFrequency.DAILY),
        ("BAMLC0A0CM", "Investment Grade Credit Spread", UpdateFrequency.DAILY),
        ("T10Y2Y", "10Y-2Y Treasury Spread", UpdateFrequency.DAILY),
        ("BAMLH0A0HYM2", "High Yield Spread", UpdateFrequency.DAILY),
    ]
    
    # Commodities
    COMMODITIES = [
        ("DCOILWTICO", "WTI Crude Oil", UpdateFrequency.DAILY),
        ("GOLDAMGBD228NLBM", "Gold Price London Fix", UpdateFrequency.DAILY),
    ]
    
    # Global Indicators
    GLOBAL_INDICATORS = [
        ("GEPUCURRENT", "Global Economic Policy Uncertainty", UpdateFrequency.MONTHLY),
        ("LRHUTTTTEUM156S", "EU Unemployment Rate", UpdateFrequency.MONTHLY),
    ]
    
    @classmethod
    def get_all_indicators(cls) -> List[Tuple[str, str, UpdateFrequency]]:
        """Get all economic indicators"""
        return (
            cls.INTEREST_RATES + 
            cls.INFLATION_INDICATORS + 
            cls.ECONOMIC_GROWTH + 
            cls.LABOR_MARKET + 
            cls.HOUSING_MARKET + 
            cls.CONSUMER_SENTIMENT + 
            cls.MANUFACTURING_BUSINESS + 
            cls.MONETARY_POLICY + 
            cls.FISCAL_POLICY + 
            cls.MARKET_INDICATORS + 
            cls.COMMODITIES + 
            cls.GLOBAL_INDICATORS
        )
    
    @classmethod
    def get_daily_indicators(cls) -> List[Tuple[str, str, UpdateFrequency]]:
        """Get all daily updated indicators"""
        return [ind for ind in cls.get_all_indicators() if ind[2] == UpdateFrequency.DAILY]
    
    @classmethod
    def get_weekly_indicators(cls) -> List[Tuple[str, str, UpdateFrequency]]:
        """Get all weekly updated indicators"""
        return [ind for ind in cls.get_all_indicators() if ind[2] == UpdateFrequency.WEEKLY]
    
    @classmethod
    def get_monthly_indicators(cls) -> List[Tuple[str, str, UpdateFrequency]]:
        """Get all monthly updated indicators"""
        return [ind for ind in cls.get_all_indicators() if ind[2] == UpdateFrequency.MONTHLY]
    
    @classmethod
    def get_quarterly_indicators(cls) -> List[Tuple[str, str, UpdateFrequency]]:
        """Get all quarterly updated indicators"""
        return [ind for ind in cls.get_all_indicators() if ind[2] == UpdateFrequency.QUARTERLY]
    
    @classmethod
    def get_indicators_by_category(cls) -> Dict[str, List[Tuple[str, str, UpdateFrequency]]]:
        """Get indicators organized by category"""
        return {
            "interest_rates": cls.INTEREST_RATES,
            "inflation": cls.INFLATION_INDICATORS,
            "economic_growth": cls.ECONOMIC_GROWTH,
            "labor_market": cls.LABOR_MARKET,
            "housing": cls.HOUSING_MARKET,
            "consumer_sentiment": cls.CONSUMER_SENTIMENT,
            "manufacturing_business": cls.MANUFACTURING_BUSINESS,
            "monetary_policy": cls.MONETARY_POLICY,
            "fiscal_policy": cls.FISCAL_POLICY,
            "market_indicators": cls.MARKET_INDICATORS,
            "commodities": cls.COMMODITIES,
            "global": cls.GLOBAL_INDICATORS,
        }
    
    @classmethod
    def get_indicator_info(cls, symbol: str) -> Dict[str, any]:
        """Get information about a specific indicator"""
        for indicator in cls.get_all_indicators():
            if indicator[0] == symbol:
                return {
                    "symbol": indicator[0],
                    "name": indicator[1],
                    "frequency": indicator[2],
                    "category": cls._get_category_for_indicator(symbol)
                }
        return {
            "symbol": symbol,
            "name": "Unknown",
            "frequency": None,
            "category": "unknown"
        }
    
    @classmethod
    def _get_category_for_indicator(cls, symbol: str) -> str:
        """Get category for a specific indicator"""
        categories = cls.get_indicators_by_category()
        for category, indicators in categories.items():
            if any(ind[0] == symbol for ind in indicators):
                return category
        return "unknown"
    
    @classmethod
    def get_high_priority_indicators(cls) -> List[Tuple[str, str, UpdateFrequency]]:
        """Get high priority indicators for more frequent monitoring"""
        return [
            # Key rates
            ("DFF", "Federal Funds Rate", UpdateFrequency.DAILY),
            ("DGS10", "10-Year Treasury", UpdateFrequency.DAILY),
            ("DGS2", "2-Year Treasury", UpdateFrequency.DAILY),
            # Inflation
            ("CPIAUCSL", "CPI Urban", UpdateFrequency.MONTHLY),
            ("PCEPILFE", "Core PCE", UpdateFrequency.MONTHLY),
            # Growth
            ("GDPC1", "Real GDP", UpdateFrequency.QUARTERLY),
            # Employment
            ("UNRATE", "Unemployment Rate", UpdateFrequency.MONTHLY),
            ("PAYEMS", "Non-Farm Payrolls", UpdateFrequency.MONTHLY),
            ("ICSA", "Initial Jobless Claims", UpdateFrequency.WEEKLY),
            # Market stress
            ("VIXCLS", "VIX", UpdateFrequency.DAILY),
            ("T10Y2Y", "10Y-2Y Treasury Spread", UpdateFrequency.DAILY),
        ]