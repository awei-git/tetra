"""Centralized universe definition for all symbols we track"""

from typing import List, Dict, Set, Optional
from datetime import datetime, date


class MarketUniverse:
    """Central definition of trading universe"""
    
    # Symbol start dates (IPO dates for stocks that went public after our data begins)
    # For symbols not in this dict, we assume they existed before 2015-08-03
    SYMBOL_START_DATES: Dict[str, date] = {
        # Recent IPOs (2019-2024)
        "DDOG": date(2019, 9, 19),    # Datadog IPO
        "COIN": date(2021, 4, 14),    # Coinbase IPO
        "RIVN": date(2021, 11, 10),   # Rivian IPO
        "PLTR": date(2020, 9, 30),    # Palantir IPO
        "SNOW": date(2020, 9, 16),    # Snowflake IPO
        "ABNB": date(2020, 12, 10),   # Airbnb IPO
        "DASH": date(2020, 12, 9),    # DoorDash IPO
        "RBLX": date(2021, 3, 10),    # Roblox IPO
        "HOOD": date(2021, 7, 29),    # Robinhood IPO
        "SOFI": date(2021, 6, 1),     # SoFi (via SPAC)
        "LCID": date(2021, 7, 26),    # Lucid Motors (via SPAC)
        "ARM": date(2023, 9, 14),     # ARM Holdings IPO
        "IONQ": date(2021, 10, 1),    # IonQ (via SPAC)
        "PATH": date(2021, 5, 19),    # UiPath IPO
        "CPNG": date(2021, 3, 11),    # Coupang IPO
        "NU": date(2021, 12, 9),      # Nu Holdings IPO
        "GTLB": date(2021, 9, 15),    # GitLab IPO
        "S": date(2021, 6, 23),       # SentinelOne IPO
        "BROS": date(2021, 9, 29),    # Dutch Bros IPO
        "UPST": date(2020, 12, 16),   # Upstart IPO
        "AFRM": date(2021, 1, 13),    # Affirm IPO
        "COMP": date(2020, 12, 10),   # Compass IPO (real estate)
        "BMBL": date(2021, 2, 11),    # Bumble IPO
        "WISH": date(2020, 12, 16),   # ContextLogic (Wish) IPO
        "AI": date(2021, 4, 28),      # C3.ai IPO (actually Dec 2020)
        "SOUN": date(2021, 4, 28),    # SoundHound AI (via SPAC)
        "U": date(2020, 9, 18),       # Unity Software IPO
        "ZI": date(2020, 6, 4),       # ZoomInfo IPO
        "BILL": date(2019, 12, 12),   # Bill.com IPO
        "DOCN": date(2021, 3, 24),    # DigitalOcean IPO
        "MQ": date(2021, 6, 17),      # Marqeta IPO
        "DUOL": date(2021, 7, 28),    # Duolingo IPO
        
        # Chinese EV companies
        "NIO": date(2018, 9, 12),     # NIO IPO
        "LI": date(2020, 7, 30),      # Li Auto IPO  
        "XPEV": date(2020, 8, 27),    # XPeng IPO
        
        # SPACs that merged 2020-2022
        "PSFE": date(2021, 3, 31),    # Paysafe (SPAC)
        "CLOV": date(2021, 1, 8),     # Clover Health (SPAC)
        "OPEN": date(2020, 12, 21),   # Opendoor (SPAC)
        
        # ETFs launched after 2015
        "IBIT": date(2024, 1, 11),    # iShares Bitcoin Trust ETF
        
        # Older companies with relevant pivot dates
        "SQ": date(2015, 11, 19),     # Block/Square IPO
    }
    
    # Default start date for symbols not in the list (when our data begins)
    DEFAULT_START_DATE = date(2015, 8, 3)
    
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
    
    # Mega Cap Stocks (Top 50 by market cap - September 2025 estimates)
    MEGA_CAP_STOCKS = [
        # $4T+ club (AI revolution fully priced)
        "NVDA",  # NVIDIA ~$5T (Blackwell dominance, AI everywhere)
        "MSFT",  # Microsoft ~$4.2T (Copilot monetization)
        "AAPL",  # Apple ~$4T (AI iPhone cycle)
        
        # $2T-$4T club
        "GOOGL", # Alphabet Class A ~$3T (Gemini success)
        "AMZN",  # Amazon ~$2.8T (AWS AI services boom)
        "META",  # Meta Platforms ~$2T (Llama open-source winning)
        
        # $1T-$2T club
        "TSLA",  # Tesla ~$1.8T (FSD + Optimus robots)
        "AVGO",  # Broadcom ~$1.5T (custom AI chips for hyperscalers)
        "BRK-B", # Berkshire Hathaway ~$1.2T
        "LLY",   # Eli Lilly ~$1.1T (GLP-1 blockbuster)
        "ORCL",  # Oracle ~$1T (database AI boom)
        
        # $500B-$1T club
        "AMD",   # AMD ~$800B (MI300 taking GPU share)
        "JPM",   # JPMorgan Chase ~$750B
        "V",     # Visa ~$650B
        "WMT",   # Walmart ~$600B
        "XOM",   # Exxon Mobil ~$520B
        
        # $200B-$500B club  
        "UNH",   # UnitedHealth ~$500B
        "MA",    # Mastercard ~$490B
        "COST",  # Costco ~$480B
        "HD",    # Home Depot ~$450B
        "CRM",   # Salesforce ~$420B (AI agents taking off)
        "PG",    # Procter & Gamble ~$410B
        "NFLX",  # Netflix ~$380B (password sharing success)
        "JNJ",   # Johnson & Johnson ~$370B
        "ABBV",  # AbbVie ~$350B
        "BAC",   # Bank of America ~$340B
        "NOW",   # ServiceNow ~$320B (AI workflows)
        "PLTR",  # Palantir ~$300B (AI platform for enterprises)
        "KO",    # Coca-Cola ~$280B
        "CVX",   # Chevron ~$270B
        "PEP",   # PepsiCo ~$260B
        "MRK",   # Merck ~$250B
        "ADBE",  # Adobe ~$240B (Firefly AI)
        "TMO",   # Thermo Fisher ~$230B
        
        # $100B-$200B club
        "DIS",   # Disney ~$195B
        "CSCO",  # Cisco ~$190B
        "ABT",   # Abbott ~$190B
        "INTU",  # Intuit ~$185B (AI tax/accounting)
        "TXN",   # Texas Instruments ~$180B
        "VZ",    # Verizon ~$175B
        "QCOM",  # Qualcomm ~$175B (AI on edge)
        "IBM",   # IBM ~$170B (watsonx traction)
        "PFE",   # Pfizer ~$160B
        "UBER",  # Uber ~$160B (autonomous ready)
        "GS",    # Goldman Sachs ~$155B
        "SPGI",  # S&P Global ~$150B
        "INTC",  # Intel ~$145B (still struggling)
        "NKE",   # Nike ~$145B
        "AMGN",  # Amgen ~$140B
        "ANET",  # Arista Networks ~$140B (AI networking)
        "CAT",   # Caterpillar ~$135B
        "MS",    # Morgan Stanley ~$135B
        "PANW",  # Palo Alto Networks ~$130B (AI security)
        "GE",    # General Electric ~$130B
        "MU",    # Micron ~$125B (HBM shortage)
        "CRWD",  # CrowdStrike ~$120B (post-outage recovery)
        "SNOW",  # Snowflake ~$115B (AI data platform)
        "MRVL",  # Marvell ~$110B (custom AI chips)
        "ARM",   # Arm Holdings ~$105B (AI edge)
    ]
    
    # High Growth Stocks (regardless of market cap - for growth vs value comparison)
    HIGH_GROWTH_STOCKS = [
        # Software/SaaS growth
        "CRM",   # Salesforce
        "NOW",   # ServiceNow  
        "SNOW",  # Snowflake
        "DDOG",  # Datadog
        "MDB",   # MongoDB
        "NET",   # Cloudflare
        "CRWD",  # CrowdStrike
        "ZS",    # Zscaler
        "OKTA",  # Okta
        "TWLO",  # Twilio
        "DOCN",  # DigitalOcean
        "GTLB",  # GitLab
        
        # Consumer growth
        "ABNB",  # Airbnb
        "DASH",  # DoorDash
        "UBER",  # Uber
        "SHOP",  # Shopify
        "MELI",  # MercadoLibre
        "SE",    # Sea Limited
        "CPNG",  # Coupang
        
        # Fintech growth
        "SQ",    # Block (Square)
        "PYPL",  # PayPal
        "SOFI",  # SoFi
        "AFRM",  # Affirm
        "UPST",  # Upstart
        "HOOD",  # Robinhood
        
        # EV/Clean energy growth
        "TSLA",  # Tesla
        "RIVN",  # Rivian
        "LCID",  # Lucid
        "NIO",   # Nio
        "XPEV",  # XPeng
        "LI",    # Li Auto
        "ENPH",  # Enphase
        "SEDG",  # SolarEdge
        
        # Biotech growth
        "MRNA",  # Moderna
        "BNTX",  # BioNTech
        "REGN",  # Regeneron
        "VRTX",  # Vertex
        "BIIB",  # Biogen
        
        # Gaming/Entertainment growth
        "RBLX",  # Roblox
        "U",     # Unity
        "TTWO",  # Take-Two
        "EA",    # Electronic Arts
        "NFLX",  # Netflix
        "ROKU",  # Roku
        "SPOT",  # Spotify
    ]
    
    # Growth Stocks by Specific Sector
    QUANTUM_COMPUTING_STOCKS = [
        "IONQ",  # IonQ
        "RGTI",  # Rigetti Computing
        "QBTS",  # D-Wave Quantum
        "ARQQ",  # Arqit Quantum
        "IBM",   # IBM Quantum
        "GOOGL", # Google Quantum AI
        "MSFT",  # Microsoft Azure Quantum
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
        # All AI infrastructure plays (including mega-caps for sector analysis)
        "NVDA",  # NVIDIA - AI chips leader
        "AMD",   # AMD - AI chips challenger
        "INTC",  # Intel - AI chips
        "AVGO",  # Broadcom - AI networking
        "MRVL",  # Marvell ~$70B
        "SMCI",  # Super Micro Computer ~$15B
        "DELL",  # Dell Technologies ~$70B
        "HPE",   # Hewlett Packard Enterprise ~$20B
        "ANET",  # Arista Networks ~$100B
        "CRWD",  # CrowdStrike ~$70B
        "PLTR",  # Palantir ~$80B
        "SNOW",  # Snowflake ~$50B
        "NET",   # Cloudflare ~$30B
        "DDOG",  # Datadog ~$40B
        "MDB",   # MongoDB ~$20B
        "PATH",  # UiPath ~$7B
        "AI",    # C3.ai ~$3B
        "SOUN",  # SoundHound AI ~$1B
    ]
    
    CONSUMER_PRODUCT_STOCKS = [
        # Consumer brands (including mega-caps for sector comparison)
        "AAPL",  # Apple
        "TSLA",  # Tesla
        "NKE",   # Nike
        "SBUX",  # Starbucks ~$100B
        "MCD",   # McDonald's ~$200B
        "LULU",  # Lululemon ~$40B
        "CMG",   # Chipotle ~$70B
        "ABNB",  # Airbnb ~$80B
        "BKNG",  # Booking Holdings ~$150B
        "MAR",   # Marriott ~$60B
        "YUM",   # Yum Brands ~$35B
        "DPZ",   # Domino's ~$15B
        "DASH",  # DoorDash ~$50B
        "UBER",  # Uber ~$130B
        "LYFT",  # Lyft ~$5B
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
        """Get all stock symbols (unique, no duplicates)"""
        return list(set(
            cls.MEGA_CAP_STOCKS + 
            cls.HIGH_GROWTH_STOCKS +
            cls.QUANTUM_COMPUTING_STOCKS + 
            cls.MILITARY_DEFENSE_STOCKS + 
            cls.AI_INFRASTRUCTURE_STOCKS + 
            cls.CONSUMER_PRODUCT_STOCKS + 
            cls.CRYPTO_STOCKS
        ))
    
    @classmethod
    def get_all_symbols(cls) -> List[str]:
        """Get all symbols (excluding crypto which needs special handling)"""
        return cls.get_all_etfs() + cls.get_all_stocks()
    
    @classmethod
    def get_all_symbols_with_crypto(cls) -> List[str]:
        """Get all symbols including crypto (use only with crypto-capable providers)"""
        return cls.get_all_etfs() + cls.get_all_stocks() + cls.CRYPTO_SYMBOLS
    
    @classmethod
    def get_high_priority_symbols(cls) -> List[str]:
        """Get high priority symbols for more frequent updates"""
        return [
            # Major indices
            "SPY", "QQQ", "IWM", "DIA",
            # Magnificent 7 (Sept 2025 order by market cap)
            "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA",
            # AI infrastructure leaders
            "AVGO", "AMD", "ORCL", "PLTR",
            # Other mega caps
            "LLY", "BRK-B",
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
            "mega_cap_stocks": cls.MEGA_CAP_STOCKS,
            "high_growth_stocks": cls.HIGH_GROWTH_STOCKS,
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
                    "start_date": cls.get_symbol_start_date(symbol),
                }
        
        return {
            "symbol": symbol,
            "category": "unknown",
            "is_etf": False,
            "is_crypto": False,
            "start_date": cls.get_symbol_start_date(symbol),
        }
    
    @classmethod
    def get_symbol_start_date(cls, symbol: str) -> date:
        """Get the earliest date we should consider for a symbol"""
        return cls.SYMBOL_START_DATES.get(symbol, cls.DEFAULT_START_DATE)
    
    @classmethod
    def is_symbol_active_on_date(cls, symbol: str, check_date: date) -> bool:
        """Check if a symbol was active (post-IPO) on a given date"""
        start_date = cls.get_symbol_start_date(symbol)
        return check_date >= start_date
    
    @classmethod
    def filter_symbols_for_date_range(cls, symbols: List[str], start_date: date, end_date: date) -> List[str]:
        """Filter symbols that were active during a date range"""
        active_symbols = []
        for symbol in symbols:
            symbol_start = cls.get_symbol_start_date(symbol)
            # Symbol is relevant if it started before the scenario ends
            if symbol_start <= end_date:
                active_symbols.append(symbol)
        return active_symbols