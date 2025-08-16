"""Fix scenario dates to match available data range."""

import json
from datetime import datetime, date
from pathlib import Path

# Our data range
DATA_START = datetime(2015, 8, 2)
DATA_END = datetime(2025, 8, 15)

def adjust_date_to_range(d, is_start=True):
    """Adjust date to be within data range."""
    if isinstance(d, str):
        d = datetime.fromisoformat(d.replace('Z', '+00:00').replace('+00:00', ''))
    elif isinstance(d, date):
        d = datetime.combine(d, datetime.min.time())
    
    if d < DATA_START:
        # Map old dates to equivalent periods in our range
        years_diff = (DATA_START.year - d.year)
        # Use a more recent equivalent period
        if years_diff > 10:
            # Very old dates - use 2016-2020 range
            new_year = 2016 + (d.year % 4)
        elif years_diff > 5:
            # 2008-2015 -> map to 2016-2023
            new_year = d.year + 8
        else:
            # Just after our cutoff - use 2016
            new_year = 2016
        
        adjusted = d.replace(year=new_year)
        if adjusted < DATA_START:
            adjusted = DATA_START
        return adjusted.isoformat()
    elif d > DATA_END:
        # Future dates - cap at our data end
        return DATA_END.isoformat()
    
    return d.isoformat()

# Load scenario metadata
metadata_file = Path('data/scenarios/scenario_metadata.json')
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Fix dates for all scenarios
fixed_count = 0
for scenario in metadata['scenarios']:
    old_start = scenario.get('start_date')
    old_end = scenario.get('end_date')
    
    if old_start:
        new_start = adjust_date_to_range(old_start, is_start=True)
        if new_start != old_start:
            print(f"Fixing {scenario['name']}: start {old_start} -> {new_start}")
            scenario['start_date'] = new_start
            fixed_count += 1
    
    if old_end:
        new_end = adjust_date_to_range(old_end, is_start=False)
        # Ensure end is after start
        if new_start and new_end < new_start:
            # Add 3 months to start
            end_dt = datetime.fromisoformat(new_start)
            end_dt = end_dt.replace(month=min(12, end_dt.month + 3))
            new_end = end_dt.isoformat()
        
        if new_end != old_end:
            print(f"Fixing {scenario['name']}: end {old_end} -> {new_end}")
            scenario['end_date'] = new_end
            fixed_count += 1

# Save fixed metadata
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"\nFixed {fixed_count} date issues in scenarios")
print(f"Saved to {metadata_file}")