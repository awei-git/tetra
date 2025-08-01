"""Centralized universe definition for all symbols we track"""

from typing import List, Dict, Set
from datetime import datetime, date


class MarketUniverse:
    """Central definition of trading universe"""
    
    # Major Index ETFs
    INDEX_ETFS = [
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
        "IWM",   # Russell 2000
        "DIA",   # Dow Jones
        "VOO",   # Vanguard S&P 500
        "VTI",   # Total Market
        "HDV",   # iShares Core High Dividend
    ]
    
    # Sector ETFs
    SECTOR_ETFS = [
        "XLF",   # Financials
        "XLE",   # Energy
        "XLK",   # Technology
        "XLV",   # Healthcare
        "XLI",   # Industrials
        "XLP",   # Consumer Staples
        "XLY",   # Consumer Discretionary
        "XLRE",  # Real Estate
        "XLB",   # Materials
        "XLU",   # Utilities
        "XLC",   # Communication Services
    ]
    
    # International ETFs
    INTERNATIONAL_ETFS = [
        "EEM",   # Emerging Markets
        "EFA",   # Developed Markets
        "VEA",   # Developed Markets (Vanguard)
        "IEMG",  # Core Emerging Markets
        "VWO",   # Emerging Markets (Vanguard)
        "FXI",   # China Large-Cap
        "EWJ",   # Japan
        "EWZ",   # Brazil
        "INDA",  # India
        "EWY",   # South Korea
    ]
    
    # Fixed Income ETFs
    BOND_ETFS = [
        "TLT",   # 20+ Year Treasury
        "IEF",   # 7-10 Year Treasury
        "SHY",   # 1-3 Year Treasury
        "AGG",   # Aggregate Bond
        "BND",   # Total Bond Market
        "HYG",   # High Yield Corporate
        "LQD",   # Investment Grade Corporate
        "EMB",   # Emerging Markets Bond
        "TIP",   # TIPS
    ]
    
    # Commodity ETFs
    COMMODITY_ETFS = [
        "GLD",   # Gold
        "SLV",   # Silver
        "USO",   # Oil
        "UNG",   # Natural Gas
        "DBA",   # Agriculture
        "GDX",   # Gold Miners
        "GDXJ",  # Junior Gold Miners
        "SLX",   # Steel
        "COPX",  # Copper Miners
    ]
    
    # Thematic/Growth ETFs
    THEMATIC_ETFS = [
        "ARKK",  # ARK Innovation
        "ARKQ",  # ARK Autonomous Technology
        "ARKW",  # ARK Next Generation Internet
        "ARKG",  # ARK Genomic Revolution
        "ARKF",  # ARK Fintech Innovation
        "ICLN",  # Clean Energy
        "TAN",   # Solar
        "LIT",   # Lithium & Battery Tech
        "QCLN",  # Clean Energy (First Trust)
        "PBW",   # Clean Energy (Invesco)
        "HACK",  # Cybersecurity
        "ROBO",  # Robotics & AI
        "BOTZ",  # Robotics & AI (Global X)
    ]
    
    # REIT ETFs (Data Center & Warehouse focus)
    REIT_ETFS = [
        "VNQ",   # Vanguard Real Estate
        "XLRE",  # Real Estate Select Sector
        "DLR",   # Digital Realty (Data Centers)
        "EQIX",  # Equinix (Data Centers)
        "AMT",   # American Tower
        "CCI",   # Crown Castle
        "PLD",   # Prologis (Warehouses)
    ]
    
    # Volatility Index ETFs/ETNs
    VOLATILITY_ETFS = [
        "VXX",   # iPath S&P 500 VIX Short-Term Futures ETN
        "VIXY",  # ProShares VIX Short-Term Futures ETF
        "UVXY",  # ProShares Ultra VIX Short-Term Futures ETF (1.5x)
        "SVXY",  # ProShares Short VIX Short-Term Futures ETF (-0.5x)
        "VXZ",   # iPath S&P 500 VIX Mid-Term Futures ETN
        "VIXM",  # ProShares VIX Mid-Term Futures ETF
        "VIIX",  # VelocityShares VIX Short-Term ETN
    ]
    
    # Large Cap Stocks (Top 30)
    LARGE_CAP_STOCKS = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet Class A
        "AMZN",  # Amazon
        "NVDA",  # NVIDIA
        "META",  # Meta Platforms
        "TSLA",  # Tesla
        "BRK-B", # Berkshire Hathaway
        "V",     # Visa
        "JNJ",   # Johnson & Johnson
        "WMT",   # Walmart
        "JPM",   # JPMorgan Chase
        "PG",    # Procter & Gamble
        "UNH",   # UnitedHealth
        "HD",    # Home Depot
        "MA",    # Mastercard
        "DIS",   # Disney
        "BAC",   # Bank of America
        "CVX",   # Chevron
        "ABBV",  # AbbVie
        "PFE",   # Pfizer
        "KO",    # Coca-Cola
        "PEP",   # PepsiCo
        "CSCO",  # Cisco
        "TMO",   # Thermo Fisher
        "ABT",   # Abbott
        "NKE",   # Nike
        "MRK",   # Merck
        "VZ",    # Verizon
        "ADBE",  # Adobe
    ]
    
    # Growth Stocks by Sector
    QUANTUM_COMPUTING_STOCKS = [
        "IBM",   # IBM Quantum
        "GOOGL", # Google Quantum AI
        "MSFT",  # Microsoft Azure Quantum
        "IONQ",  # IonQ
        "RGTI",  # Rigetti Computing
        "QBTS",  # D-Wave Quantum
    ]
    
    MILITARY_DEFENSE_STOCKS = [
        "LMT",   # Lockheed Martin
        "RTX",   # Raytheon
        "BA",    # Boeing
        "NOC",   # Northrop Grumman
        "GD",    # General Dynamics
        "LHX",   # L3Harris
        "HII",   # Huntington Ingalls
        "TXT",   # Textron
    ]
    
    AI_INFRASTRUCTURE_STOCKS = [
        "NVDA",  # NVIDIA
        "AMD",   # AMD
        "INTC",  # Intel
        "AVGO",  # Broadcom
        "MRVL",  # Marvell
        "SMCI",  # Super Micro Computer
        "DELL",  # Dell Technologies
        "HPE",   # Hewlett Packard Enterprise
        "ANET",  # Arista Networks
        "CRWD",  # CrowdStrike
        "PLTR",  # Palantir
        "SNOW",  # Snowflake
        "NET",   # Cloudflare
    ]
    
    CONSUMER_PRODUCT_STOCKS = [
        "AAPL",  # Apple
        "TSLA",  # Tesla
        "NKE",   # Nike
        "SBUX",  # Starbucks
        "MCD",   # McDonald's
        "LULU",  # Lululemon
        "CMG",   # Chipotle
        "ABNB",  # Airbnb
        "BKNG",  # Booking Holdings
        "MAR",   # Marriott
    ]
    
    # Crypto-related Stocks
    CRYPTO_STOCKS = [
        "COIN",  # Coinbase
        "MSTR",  # MicroStrategy
        "MARA",  # Marathon Digital
        "RIOT",  # Riot Platforms
        "HUT",   # Hut 8 Mining
        "BITF",  # Bitfarms
        "BTBT",  # Bit Digital
        "GBTC",  # Grayscale Bitcoin Trust
        "IBIT",  # iShares Bitcoin Trust
        "PYPL",  # PayPal
    ]
    
    # Crypto symbols (for providers that support them)
    CRYPTO_SYMBOLS = [
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
        "BNB-USD",   # Binance Coin
        "XRP-USD",   # Ripple
        "ADA-USD",   # Cardano
        "SOL-USD",   # Solana
        "DOGE-USD",  # Dogecoin
        "DOT-USD",   # Polkadot
        "MATIC-USD", # Polygon
        "AVAX-USD",  # Avalanche
    ]
    
    @classmethod
    def get_all_etfs(cls) -> List[str]:
        """Get all ETF symbols"""
        return list(set(
            cls.INDEX_ETFS + 
            cls.SECTOR_ETFS + 
            cls.INTERNATIONAL_ETFS + 
            cls.BOND_ETFS + 
            cls.COMMODITY_ETFS + 
            cls.THEMATIC_ETFS + 
            cls.REIT_ETFS +
            cls.VOLATILITY_ETFS
        ))
    
    @classmethod
    def get_all_stocks(cls) -> List[str]:
        """Get all stock symbols"""
        return list(set(
            cls.LARGE_CAP_STOCKS + 
            cls.QUANTUM_COMPUTING_STOCKS + 
            cls.MILITARY_DEFENSE_STOCKS + 
            cls.AI_INFRASTRUCTURE_STOCKS + 
            cls.CONSUMER_PRODUCT_STOCKS + 
            cls.CRYPTO_STOCKS
        ))
    
    @classmethod
    def get_all_symbols(cls) -> List[str]:
        """Get all symbols including crypto"""
        return cls.get_all_etfs() + cls.get_all_stocks() + cls.CRYPTO_SYMBOLS
    
    @classmethod
    def get_high_priority_symbols(cls) -> List[str]:
        """Get high priority symbols for more frequent updates"""
        return [
            # Major indices
            "SPY", "QQQ", "IWM", "DIA",
            # Top tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            # Key sectors
            "XLF", "XLE", "XLK",
            # Major crypto
            "BTC-USD", "ETH-USD",
        ]
    
    @classmethod
    def get_universe_by_category(cls) -> Dict[str, List[str]]:
        """Get universe organized by category"""
        return {
            "index_etfs": cls.INDEX_ETFS,
            "sector_etfs": cls.SECTOR_ETFS,
            "international_etfs": cls.INTERNATIONAL_ETFS,
            "bond_etfs": cls.BOND_ETFS,
            "commodity_etfs": cls.COMMODITY_ETFS,
            "thematic_etfs": cls.THEMATIC_ETFS,
            "reit_etfs": cls.REIT_ETFS,
            "volatility_etfs": cls.VOLATILITY_ETFS,
            "large_cap_stocks": cls.LARGE_CAP_STOCKS,
            "quantum_computing": cls.QUANTUM_COMPUTING_STOCKS,
            "defense": cls.MILITARY_DEFENSE_STOCKS,
            "ai_infrastructure": cls.AI_INFRASTRUCTURE_STOCKS,
            "consumer": cls.CONSUMER_PRODUCT_STOCKS,
            "crypto_stocks": cls.CRYPTO_STOCKS,
            "crypto": cls.CRYPTO_SYMBOLS,
        }
    
    @classmethod
    def is_crypto(cls, symbol: str) -> bool:
        """Check if symbol is a crypto"""
        return symbol in cls.CRYPTO_SYMBOLS
    
    @classmethod
    def is_etf(cls, symbol: str) -> bool:
        """Check if symbol is an ETF"""
        return symbol in cls.get_all_etfs()
    
    @classmethod
    def get_symbol_info(cls, symbol: str) -> Dict[str, str]:
        """Get information about a symbol"""
        categories = cls.get_universe_by_category()
        
        for category, symbols in categories.items():
            if symbol in symbols:
                return {
                    "symbol": symbol,
                    "category": category,
                    "is_etf": cls.is_etf(symbol),
                    "is_crypto": cls.is_crypto(symbol),
                }
        
        return {
            "symbol": symbol,
            "category": "unknown",
            "is_etf": False,
            "is_crypto": False,
        }