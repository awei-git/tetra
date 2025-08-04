#!/usr/bin/env python3
"""Fix import statements after model reorganization"""
import os
import re

# Files and their specific imports to fix
fixes = [
    ("src/api/routers/event_data.py", "from src.models.event_data import EventType, EventImpact, EventStatus", "from src.models import EventType, EventImpact, EventStatus"),
    ("src/api/routers/market_data.py", "from src.models.market_data import OHLCVData", "from src.models import OHLCVData"),
    ("src/api/routers/events.py", "from src.models.event_data import EventType, EventImpact, EventStatus", "from src.models import EventType, EventImpact, EventStatus"),
    ("src/clients/news_sentiment_client.py", "from src.models.news_sentiment import", "from src.models import"),
    ("src/clients/event_data_client.py", "from src.models.event_data import", "from src.models import"),
    ("tests/conftest.py", "from src.models.market_data import OHLCVData", "from src.models import OHLCVData"),
    ("tests/test_economic_data.py", "from src.models.economic_data import EconomicData, EconomicRelease, EconomicForecast", "from src.models import EconomicData, EconomicRelease, EconomicForecast"),
    ("tests/test_event_data.py", "from src.models.event_data import", "from src.models import"),
    ("tests/test_news_sentiment.py", "from src.models.news_sentiment import", "from src.models import"),
    ("tests/test_data_ingester.py", "from src.models.market_data import OHLCVData", "from src.models import OHLCVData"),
    ("tests/test_market_data_client.py", "from src.models.market_data import OHLCVData", "from src.models import OHLCVData"),
]

# Change to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

for file_path, old_import, new_import in fixes:
    if os.path.exists(file_path):
        print(f"Fixing {file_path}")
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace the import
        updated_content = content.replace(old_import, new_import)
        
        if content != updated_content:
            with open(file_path, 'w') as f:
                f.write(updated_content)
            print(f"  ✓ Fixed import")
        else:
            print(f"  - No changes needed")
    else:
        print(f"  ✗ File not found: {file_path}")

print("\nDone!")