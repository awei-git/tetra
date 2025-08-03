"""Signal configuration management system."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from ..base import SignalConfig
from ..base.base_signal import BaseSignal

logger = logging.getLogger(__name__)


class SignalConfigManager:
    """Manages signal configurations - save, load, and organize."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize config manager.
        
        Args:
            config_dir: Directory to store configurations
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "presets"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, 
                   name: str,
                   config: SignalConfig,
                   description: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   format: str = "json") -> Path:
        """Save a signal configuration.
        
        Args:
            name: Name for the configuration
            config: SignalConfig object
            description: Optional description
            tags: Optional tags for categorization
            format: Save format ('json' or 'yaml')
            
        Returns:
            Path to saved configuration
        """
        # Create config data
        config_data = {
            "name": name,
            "description": description or f"Signal configuration: {name}",
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
            "config": config.to_dict()
        }
        
        # Determine file path
        file_extension = "json" if format == "json" else "yml"
        file_path = self.config_dir / f"{name}.{file_extension}"
        
        # Save configuration
        with open(file_path, 'w') as f:
            if format == "json":
                json.dump(config_data, f, indent=2)
            else:
                yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Saved configuration '{name}' to {file_path}")
        return file_path
    
    def load_config(self, name: str) -> SignalConfig:
        """Load a signal configuration by name.
        
        Args:
            name: Name of the configuration
            
        Returns:
            SignalConfig object
        """
        # Try different file formats
        for ext in ['.json', '.yml', '.yaml']:
            file_path = self.config_dir / f"{name}{ext}"
            if file_path.exists():
                return self._load_from_file(file_path)
        
        raise FileNotFoundError(f"Configuration '{name}' not found")
    
    def _load_from_file(self, file_path: Path) -> SignalConfig:
        """Load configuration from file."""
        with open(file_path, 'r') as f:
            if file_path.suffix == '.json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        # Extract config data
        config_dict = data.get('config', data)
        
        # Create SignalConfig
        return SignalConfig.from_dict(config_dict)
    
    def list_configs(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List available configurations.
        
        Args:
            tags: Filter by tags (if provided)
            
        Returns:
            List of configuration metadata
        """
        configs = []
        
        for file_path in self.config_dir.glob("*.[jy]*"):
            try:
                with open(file_path, 'r') as f:
                    if file_path.suffix == '.json':
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
                
                # Filter by tags if provided
                if tags:
                    config_tags = data.get('tags', [])
                    if not any(tag in config_tags for tag in tags):
                        continue
                
                configs.append({
                    'name': data.get('name', file_path.stem),
                    'description': data.get('description', ''),
                    'created_at': data.get('created_at', ''),
                    'tags': data.get('tags', []),
                    'file': str(file_path)
                })
                
            except Exception as e:
                logger.warning(f"Error reading config {file_path}: {e}")
        
        return sorted(configs, key=lambda x: x['name'])
    
    def delete_config(self, name: str) -> bool:
        """Delete a configuration.
        
        Args:
            name: Name of configuration to delete
            
        Returns:
            True if deleted, False if not found
        """
        for ext in ['.json', '.yml', '.yaml']:
            file_path = self.config_dir / f"{name}{ext}"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted configuration '{name}'")
                return True
        
        return False
    
    def export_configs(self, output_path: Union[str, Path], names: Optional[List[str]] = None):
        """Export configurations to a single file.
        
        Args:
            output_path: Path for exported file
            names: Specific configs to export (None = all)
        """
        output_path = Path(output_path)
        
        # Get configurations to export
        if names:
            configs = []
            for name in names:
                try:
                    config = self.load_config(name)
                    configs.append({
                        'name': name,
                        'config': config.to_dict()
                    })
                except FileNotFoundError:
                    logger.warning(f"Configuration '{name}' not found")
        else:
            configs = []
            for config_info in self.list_configs():
                name = config_info['name']
                config = self.load_config(name)
                configs.append({
                    'name': name,
                    'config': config.to_dict()
                })
        
        # Export
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'configurations': configs
        }
        
        with open(output_path, 'w') as f:
            if output_path.suffix == '.json':
                json.dump(export_data, f, indent=2)
            else:
                yaml.dump(export_data, f, default_flow_style=False)
        
        logger.info(f"Exported {len(configs)} configurations to {output_path}")
    
    def import_configs(self, import_path: Union[str, Path], overwrite: bool = False):
        """Import configurations from file.
        
        Args:
            import_path: Path to import file
            overwrite: Whether to overwrite existing configs
        """
        import_path = Path(import_path)
        
        with open(import_path, 'r') as f:
            if import_path.suffix == '.json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        configurations = data.get('configurations', [])
        imported = 0
        skipped = 0
        
        for config_data in configurations:
            name = config_data['name']
            
            # Check if exists
            exists = any((self.config_dir / f"{name}{ext}").exists() 
                        for ext in ['.json', '.yml', '.yaml'])
            
            if exists and not overwrite:
                logger.info(f"Skipping existing configuration '{name}'")
                skipped += 1
                continue
            
            # Import configuration
            config = SignalConfig.from_dict(config_data['config'])
            self.save_config(name, config)
            imported += 1
        
        logger.info(f"Imported {imported} configurations, skipped {skipped}")


class SignalConfigBuilder:
    """Builder for creating signal configurations programmatically."""
    
    def __init__(self):
        self._config_dict = {}
    
    def with_technical_indicators(self,
                                 rsi_period: Optional[int] = None,
                                 macd_params: Optional[Dict[str, int]] = None,
                                 bb_params: Optional[Dict[str, Any]] = None) -> 'SignalConfigBuilder':
        """Configure technical indicators."""
        if rsi_period:
            self._config_dict['rsi_period'] = rsi_period
        
        if macd_params:
            self._config_dict.update({
                'macd_fast': macd_params.get('fast', 12),
                'macd_slow': macd_params.get('slow', 26),
                'macd_signal': macd_params.get('signal', 9)
            })
        
        if bb_params:
            self._config_dict.update({
                'bb_period': bb_params.get('period', 20),
                'bb_std': bb_params.get('std', 2.0)
            })
        
        return self
    
    def with_statistical_params(self,
                              volatility_window: Optional[int] = None,
                              correlation_window: Optional[int] = None,
                              returns_periods: Optional[List[int]] = None) -> 'SignalConfigBuilder':
        """Configure statistical parameters."""
        if volatility_window:
            self._config_dict['volatility_window'] = volatility_window
        
        if correlation_window:
            self._config_dict['correlation_window'] = correlation_window
        
        if returns_periods:
            self._config_dict['returns_periods'] = returns_periods
        
        return self
    
    def with_ml_params(self,
                      feature_window: Optional[int] = None,
                      prediction_horizon: Optional[int] = None,
                      confidence_threshold: Optional[float] = None) -> 'SignalConfigBuilder':
        """Configure ML parameters."""
        if feature_window:
            self._config_dict['ml_feature_window'] = feature_window
        
        if prediction_horizon:
            self._config_dict['ml_prediction_horizon'] = prediction_horizon
        
        if confidence_threshold:
            self._config_dict['ml_confidence_threshold'] = confidence_threshold
        
        return self
    
    def with_performance_settings(self,
                                 parallel: bool = True,
                                 cache: bool = True,
                                 batch_size: Optional[int] = None) -> 'SignalConfigBuilder':
        """Configure performance settings."""
        self._config_dict['parallel_compute'] = parallel
        self._config_dict['cache_results'] = cache
        
        if batch_size:
            self._config_dict['batch_size'] = batch_size
        
        return self
    
    def with_signal_selection(self,
                            technical: bool = True,
                            statistical: bool = True,
                            ml: bool = True,
                            specific_signals: Optional[Dict[str, List[str]]] = None) -> 'SignalConfigBuilder':
        """Configure which signals to compute."""
        self._config_dict['compute_technical'] = technical
        self._config_dict['compute_statistical'] = statistical
        self._config_dict['compute_ml'] = ml
        
        if specific_signals:
            if 'technical' in specific_signals:
                self._config_dict['technical_signals'] = set(specific_signals['technical'])
            if 'statistical' in specific_signals:
                self._config_dict['statistical_signals'] = set(specific_signals['statistical'])
            if 'ml' in specific_signals:
                self._config_dict['ml_signals'] = set(specific_signals['ml'])
        
        return self
    
    def with_custom_params(self, **kwargs) -> 'SignalConfigBuilder':
        """Add custom parameters."""
        if 'custom_params' not in self._config_dict:
            self._config_dict['custom_params'] = {}
        
        self._config_dict['custom_params'].update(kwargs)
        return self
    
    def build(self) -> SignalConfig:
        """Build the SignalConfig object."""
        return SignalConfig(**self._config_dict)