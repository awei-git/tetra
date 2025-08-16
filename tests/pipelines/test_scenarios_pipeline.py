"""
Comprehensive test suite for Scenarios Pipeline.
Tests all scenario types, validations, and edge cases.
"""

import pytest
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.scenarios_pipeline import (
    ScenariosPipeline,
    EventScenario,
    ALL_SCENARIOS,
    BULL_MARKET_SCENARIOS,
    CRISIS_SCENARIOS,
    FULL_CYCLE_SCENARIOS,
    STRESS_TEST_SCENARIOS,
    SCENARIO_CATEGORIES
)
from src.pipelines.scenarios_pipeline.event_definitions import (
    get_scenarios_by_type,
    get_scenarios_by_year,
    get_scenarios_by_volatility,
    get_tariff_scenarios,
    get_future_scenarios
)


class TestEventDefinitions:
    """Test the event scenario definitions."""
    
    def test_all_scenarios_loaded(self):
        """Test that all scenario dictionaries are properly loaded."""
        assert len(ALL_SCENARIOS) > 0, "No scenarios loaded"
        assert len(BULL_MARKET_SCENARIOS) >= 6, "Missing bull market scenarios"
        assert len(CRISIS_SCENARIOS) >= 3, "Missing crisis scenarios"
        assert len(FULL_CYCLE_SCENARIOS) >= 6, "Missing full cycle scenarios"
        assert len(STRESS_TEST_SCENARIOS) >= 10, "Missing stress test scenarios"
        
        # Verify total count
        total = (len(BULL_MARKET_SCENARIOS) + len(CRISIS_SCENARIOS) + 
                len(FULL_CYCLE_SCENARIOS) + len(STRESS_TEST_SCENARIOS))
        assert len(ALL_SCENARIOS) == total, "ALL_SCENARIOS count mismatch"
    
    def test_scenario_structure(self):
        """Test that all scenarios have required fields."""
        required_fields = ['name', 'scenario_type', 'start_date', 'end_date', 
                          'description', 'volatility_multiplier']
        
        for scenario_name, scenario in ALL_SCENARIOS.items():
            # Check required fields
            for field in required_fields:
                assert hasattr(scenario, field), f"{scenario_name} missing {field}"
            
            # Validate dates
            assert isinstance(scenario.start_date, date), f"{scenario_name} invalid start_date"
            assert isinstance(scenario.end_date, date), f"{scenario_name} invalid end_date"
            assert scenario.start_date <= scenario.end_date, f"{scenario_name} dates inverted"
            
            # Validate volatility multiplier
            assert scenario.volatility_multiplier > 0, f"{scenario_name} invalid volatility"
            assert scenario.volatility_multiplier <= 10, f"{scenario_name} unrealistic volatility"
            
            # Validate expected return if present
            if scenario.expected_return is not None:
                assert -1.0 <= scenario.expected_return <= 2.0, f"{scenario_name} unrealistic return"
    
    def test_bull_market_scenarios(self):
        """Test bull market scenario characteristics."""
        for name, scenario in BULL_MARKET_SCENARIOS.items():
            assert scenario.scenario_type == "bull", f"{name} not marked as bull"
            
            # Most bull markets should have positive expected returns
            if scenario.expected_return is not None:
                assert scenario.expected_return > 0, f"{name} bull market with negative return"
            
            # Bull markets typically have lower volatility
            assert scenario.volatility_multiplier <= 3.0, f"{name} bull market too volatile"
    
    def test_crisis_scenarios(self):
        """Test crisis scenario characteristics."""
        for name, scenario in CRISIS_SCENARIOS.items():
            assert scenario.scenario_type == "crisis", f"{name} not marked as crisis"
            
            # Crises should have negative expected returns
            if scenario.expected_return is not None:
                assert scenario.expected_return < 0, f"{name} crisis with positive return"
            
            # Crises should have elevated volatility
            assert scenario.volatility_multiplier >= 1.5, f"{name} crisis volatility too low"
    
    def test_full_cycle_scenarios(self):
        """Test full cycle scenario characteristics."""
        for name, scenario in FULL_CYCLE_SCENARIOS.items():
            assert scenario.scenario_type == "full_cycle", f"{name} not marked as full_cycle"
            
            # Full cycles should have longer duration (>180 days)
            duration = (scenario.end_date - scenario.start_date).days
            assert duration >= 180, f"{name} full cycle too short ({duration} days)"
            
            # Should have phases metadata
            if scenario.metadata:
                assert 'phases' in scenario.metadata, f"{name} missing phases metadata"
    
    def test_stress_test_scenarios(self):
        """Test stress test scenario characteristics."""
        for name, scenario in STRESS_TEST_SCENARIOS.items():
            assert scenario.scenario_type == "stress", f"{name} not marked as stress"
            
            # Stress tests should have negative expected returns
            if scenario.expected_return is not None:
                assert scenario.expected_return <= 0, f"{name} stress test with positive return"
            
            # Should have scenario_type in metadata
            if scenario.metadata:
                assert 'scenario_type' in scenario.metadata, f"{name} missing scenario_type metadata"
    
    def test_tariff_scenarios(self):
        """Test tariff-specific scenarios."""
        tariff_scenarios = get_tariff_scenarios()
        assert len(tariff_scenarios) >= 4, "Missing tariff scenarios"
        
        expected_tariff_scenarios = [
            'tariff_impact_2025',
            'tariff_escalation_2025',
            'manufacturing_renaissance_2025',
            'tariff_cycle_2025_2026'
        ]
        
        for expected in expected_tariff_scenarios:
            assert expected in tariff_scenarios, f"Missing {expected} scenario"
        
        # Test tariff scenario characteristics
        for name, scenario in tariff_scenarios.items():
            if '2025' in name or '2026' in name:
                assert scenario.start_date.year >= 2025, f"{name} wrong year"
            
            # Check for trade-related metadata
            if scenario.metadata:
                trade_indicators = ['tariff', 'trade', 'manufacturing', 'reshoring']
                has_trade_metadata = any(
                    indicator in str(scenario.metadata).lower() 
                    for indicator in trade_indicators
                )
                assert has_trade_metadata, f"{name} missing trade-related metadata"
    
    def test_scenario_categories(self):
        """Test scenario categorization."""
        assert 'bull_markets' in SCENARIO_CATEGORIES
        assert 'crisis_events' in SCENARIO_CATEGORIES
        assert 'full_cycles' in SCENARIO_CATEGORIES
        assert 'stress_tests' in SCENARIO_CATEGORIES
        assert 'high_volatility' in SCENARIO_CATEGORIES
        
        # Verify category counts
        assert len(SCENARIO_CATEGORIES['bull_markets']) == len(BULL_MARKET_SCENARIOS)
        assert len(SCENARIO_CATEGORIES['crisis_events']) == len(CRISIS_SCENARIOS)
        assert len(SCENARIO_CATEGORIES['full_cycles']) == len(FULL_CYCLE_SCENARIOS)
        assert len(SCENARIO_CATEGORIES['stress_tests']) == len(STRESS_TEST_SCENARIOS)
    
    def test_key_dates(self):
        """Test that scenarios with key dates have valid date entries."""
        for name, scenario in ALL_SCENARIOS.items():
            if scenario.key_dates:
                for date_key, description in scenario.key_dates.items():
                    assert isinstance(date_key, date), f"{name} invalid key date type"
                    assert isinstance(description, str), f"{name} invalid key date description"
                    assert len(description) > 0, f"{name} empty key date description"
                    
                    # Key dates should be within scenario period
                    assert scenario.start_date <= date_key <= scenario.end_date, \
                        f"{name} key date {date_key} outside scenario period"


class TestScenarioFilters:
    """Test scenario filtering functions."""
    
    def test_get_scenarios_by_type(self):
        """Test filtering scenarios by type."""
        bull_scenarios = get_scenarios_by_type("bull")
        assert len(bull_scenarios) == len(BULL_MARKET_SCENARIOS)
        
        crisis_scenarios = get_scenarios_by_type("crisis")
        assert len(crisis_scenarios) == len(CRISIS_SCENARIOS)
        
        stress_scenarios = get_scenarios_by_type("stress")
        assert len(stress_scenarios) == len(STRESS_TEST_SCENARIOS)
    
    def test_get_scenarios_by_year(self):
        """Test filtering scenarios by year."""
        # Test 2020 scenarios (should include COVID)
        scenarios_2020 = get_scenarios_by_year(2020)
        assert any('covid' in name.lower() for name in scenarios_2020.keys())
        
        # Test 2008 scenarios (should include GFC)
        scenarios_2008 = get_scenarios_by_year(2008)
        assert any('gfc' in name.lower() or '2008' in name for name in scenarios_2008.keys())
        
        # Test 2025 scenarios (should include tariffs)
        scenarios_2025 = get_scenarios_by_year(2025)
        assert any('tariff' in name.lower() or '2025' in name for name in scenarios_2025.keys())
    
    def test_get_scenarios_by_volatility(self):
        """Test filtering scenarios by volatility threshold."""
        # High volatility scenarios (2.5x+)
        high_vol = get_scenarios_by_volatility(2.5)
        for name, scenario in high_vol.items():
            assert scenario.volatility_multiplier >= 2.5, f"{name} volatility too low"
        
        # Extreme volatility scenarios (4x+)
        extreme_vol = get_scenarios_by_volatility(4.0)
        assert len(extreme_vol) > 0, "No extreme volatility scenarios found"
        for name, scenario in extreme_vol.items():
            assert scenario.volatility_multiplier >= 4.0, f"{name} not extreme volatility"
    
    def test_get_future_scenarios(self):
        """Test filtering future scenarios."""
        future_scenarios = get_future_scenarios()
        
        for name, scenario in future_scenarios.items():
            assert scenario.start_date >= date(2025, 1, 1), f"{name} not a future scenario"
        
        # Should include tariff scenarios
        assert any('tariff' in name.lower() for name in future_scenarios.keys())


class TestScenarioPipeline:
    """Test the Scenarios Pipeline implementation."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        config = {
            'scenario_types': ['historical'],
            'storage': {
                'save_to_database': False,  # Don't save during tests
                'save_timeseries': False,
                'save_metadata': True
            }
        }
        return ScenariosPipeline(config)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert pipeline.name == "Scenarios Pipeline"
        assert 'historical' in pipeline.config['scenario_types']
        assert pipeline.config['storage']['save_to_database'] == False
    
    def test_pipeline_configuration(self):
        """Test different pipeline configurations."""
        # Test with all scenario types
        config = {'scenario_types': ['historical', 'regime', 'stochastic', 'stress']}
        pipeline = ScenariosPipeline(config)
        assert len(pipeline.config['scenario_types']) == 4
        
        # Test with single scenario type
        config = {'scenario_types': ['stress']}
        pipeline = ScenariosPipeline(config)
        assert pipeline.config['scenario_types'] == ['stress']
    
    @pytest.mark.asyncio
    async def test_pipeline_setup(self, pipeline):
        """Test pipeline setup method."""
        context = await pipeline.setup()
        
        assert context is not None
        assert 'config' in context.data
        assert 'scenarios' in context.data
        assert 'scenario_definitions' in context.data
        assert 'start_date' in context.data
        assert 'end_date' in context.data
        
        # Check scenario definitions loaded
        definitions = context.data['scenario_definitions']
        assert 'bull_markets' in definitions
        assert 'crises' in definitions
        assert 'full_cycles' in definitions
        assert 'stress_tests' in definitions


class TestScenarioValidation:
    """Test scenario validation logic."""
    
    def test_date_consistency(self):
        """Test that all dates are consistent."""
        for name, scenario in ALL_SCENARIOS.items():
            # Duration should be positive
            duration = (scenario.end_date - scenario.start_date).days
            assert duration >= 0, f"{name} has negative duration"
            
            # Historical scenarios should be in the past (except future projections)
            if scenario.scenario_type in ['crisis', 'bull'] and '2025' not in name:
                assert scenario.start_date < date.today(), f"{name} historical scenario in future"
    
    def test_metadata_consistency(self):
        """Test metadata field consistency."""
        for name, scenario in ALL_SCENARIOS.items():
            if scenario.metadata:
                # Check for common metadata fields
                if 'phases' in scenario.metadata:
                    phases = scenario.metadata['phases']
                    assert isinstance(phases, list), f"{name} phases not a list"
                    assert len(phases) > 0, f"{name} empty phases list"
                
                if 'affected_sectors' in scenario.metadata:
                    sectors = scenario.metadata['affected_sectors']
                    assert isinstance(sectors, list), f"{name} sectors not a list"
                
                if 'catalyst' in scenario.metadata:
                    catalyst = scenario.metadata['catalyst']
                    assert isinstance(catalyst, str), f"{name} catalyst not a string"
    
    def test_return_volatility_relationship(self):
        """Test that returns and volatility have logical relationships."""
        for name, scenario in ALL_SCENARIOS.items():
            if scenario.expected_return is not None:
                # High volatility scenarios should have larger absolute returns
                if scenario.volatility_multiplier >= 3.0:
                    assert abs(scenario.expected_return) >= 0.10, \
                        f"{name} high volatility but low return impact"
                
                # Crisis scenarios should have negative returns with high volatility
                if scenario.scenario_type == 'crisis':
                    assert scenario.expected_return < 0, f"{name} crisis with positive return"
                    assert scenario.volatility_multiplier >= 1.5, f"{name} crisis with low volatility"


class TestScenarioStatistics:
    """Test statistical properties of scenarios."""
    
    def test_scenario_coverage(self):
        """Test that scenarios cover different market conditions."""
        # Check we have both positive and negative scenarios
        positive_returns = [s for s in ALL_SCENARIOS.values() 
                           if s.expected_return and s.expected_return > 0]
        negative_returns = [s for s in ALL_SCENARIOS.values() 
                           if s.expected_return and s.expected_return < 0]
        
        assert len(positive_returns) > 5, "Not enough positive scenarios"
        assert len(negative_returns) > 5, "Not enough negative scenarios"
        
        # Check volatility range
        volatilities = [s.volatility_multiplier for s in ALL_SCENARIOS.values()]
        assert min(volatilities) < 1.0, "No low volatility scenarios"
        assert max(volatilities) > 3.0, "No high volatility scenarios"
        
        # Check duration range
        durations = [(s.end_date - s.start_date).days for s in ALL_SCENARIOS.values()]
        assert min(durations) <= 30, "No short-term scenarios"
        assert max(durations) >= 365, "No long-term scenarios"
    
    def test_scenario_distribution(self):
        """Test distribution of scenario characteristics."""
        # Collect statistics
        returns = [s.expected_return for s in ALL_SCENARIOS.values() 
                  if s.expected_return is not None]
        volatilities = [s.volatility_multiplier for s in ALL_SCENARIOS.values()]
        
        # Basic statistics
        assert len(returns) > 20, "Not enough scenarios with returns"
        
        avg_return = sum(returns) / len(returns)
        assert -0.20 <= avg_return <= 0.20, "Average return seems unrealistic"
        
        avg_volatility = sum(volatilities) / len(volatilities)
        assert 1.0 <= avg_volatility <= 3.0, "Average volatility seems unrealistic"
    
    def test_historical_coverage(self):
        """Test that major historical events are covered."""
        # Major events that should be included
        required_events = [
            ('covid', 2020),
            ('gfc', 2008),
            ('dot_com', 2000),
            ('trump', 2016),
            ('svb', 2023)
        ]
        
        for event_keyword, year in required_events:
            found = False
            for name, scenario in ALL_SCENARIOS.items():
                if event_keyword in name.lower():
                    # Check year matches
                    if scenario.start_date.year <= year <= scenario.end_date.year:
                        found = True
                        break
            assert found, f"Missing historical event: {event_keyword} ({year})"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])