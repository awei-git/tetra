#!/usr/bin/env python3
"""CLI tool for managing signal configurations."""

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.signals.configs.config_manager import SignalConfigManager, SignalConfigBuilder
from src.signals.base import SignalConfig


def list_configs(args):
    """List available configurations."""
    manager = SignalConfigManager()
    configs = manager.list_configs(tags=args.tags)
    
    if not configs:
        print("No configurations found.")
        return
    
    # Prepare table data
    table_data = []
    for config in configs:
        tags_str = ', '.join(config['tags']) if config['tags'] else '-'
        table_data.append([
            config['name'],
            config['description'][:50] + '...' if len(config['description']) > 50 else config['description'],
            tags_str,
            config['created_at'][:10] if config['created_at'] else '-'
        ])
    
    print("\nAvailable Signal Configurations:")
    print(tabulate(table_data, headers=['Name', 'Description', 'Tags', 'Created'], tablefmt='grid'))


def show_config(args):
    """Show details of a configuration."""
    manager = SignalConfigManager()
    
    try:
        config = manager.load_config(args.name)
        config_dict = config.to_dict()
        
        print(f"\nConfiguration: {args.name}")
        print("=" * 60)
        
        # Group parameters by category
        categories = {
            'Technical': ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal', 
                         'bb_period', 'bb_std', 'ema_periods', 'sma_periods'],
            'Statistical': ['returns_periods', 'volatility_window', 'correlation_window',
                           'sharpe_window', 'var_confidence'],
            'ML': ['ml_feature_window', 'ml_prediction_horizon', 'ml_min_train_samples',
                   'ml_confidence_threshold'],
            'Performance': ['parallel_compute', 'cache_results', 'batch_size'],
            'Signals': ['compute_technical', 'compute_statistical', 'compute_ml',
                       'technical_signals', 'statistical_signals', 'ml_signals']
        }
        
        for category, params in categories.items():
            print(f"\n{category} Parameters:")
            for param in params:
                if param in config_dict:
                    value = config_dict[param]
                    if isinstance(value, list):
                        value = ', '.join(map(str, value))
                    elif isinstance(value, bool):
                        value = '✓' if value else '✗'
                    print(f"  {param}: {value}")
        
        # Custom parameters
        if config_dict.get('custom_params'):
            print("\nCustom Parameters:")
            for key, value in config_dict['custom_params'].items():
                print(f"  {key}: {value}")
                
    except FileNotFoundError:
        print(f"Configuration '{args.name}' not found.")
        sys.exit(1)


def create_config(args):
    """Create a new configuration."""
    builder = SignalConfigBuilder()
    
    # Build configuration based on template
    if args.template == 'scalping':
        builder.with_technical_indicators(
            rsi_period=9,
            macd_params={'fast': 8, 'slow': 17, 'signal': 9},
            bb_params={'period': 10, 'std': 1.5}
        ).with_statistical_params(
            volatility_window=10,
            returns_periods=[1, 5, 10]
        ).with_performance_settings(
            parallel=True,
            cache=True
        )
    
    elif args.template == 'swing':
        builder.with_technical_indicators(
            rsi_period=14,
            macd_params={'fast': 12, 'slow': 26, 'signal': 9},
            bb_params={'period': 20, 'std': 2.0}
        ).with_statistical_params(
            volatility_window=20,
            correlation_window=60,
            returns_periods=[1, 5, 20]
        ).with_ml_params(
            feature_window=20,
            prediction_horizon=5
        )
    
    elif args.template == 'longterm':
        builder.with_technical_indicators(
            rsi_period=21,
            macd_params={'fast': 19, 'slow': 39, 'signal': 9},
            bb_params={'period': 50, 'std': 2.5}
        ).with_statistical_params(
            volatility_window=60,
            correlation_window=252,
            returns_periods=[5, 20, 60, 252]
        ).with_ml_params(
            feature_window=60,
            prediction_horizon=20
        )
    
    else:  # default
        builder.with_technical_indicators().with_statistical_params()
    
    # Apply custom parameters
    if args.params:
        custom_params = {}
        for param in args.params:
            key, value = param.split('=')
            # Try to parse value
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            custom_params[key] = value
        builder.with_custom_params(**custom_params)
    
    # Create configuration
    config = builder.build()
    
    # Save configuration
    manager = SignalConfigManager()
    path = manager.save_config(
        name=args.name,
        config=config,
        description=args.description,
        tags=args.tags,
        format=args.format
    )
    
    print(f"Created configuration '{args.name}' at {path}")


def delete_config(args):
    """Delete a configuration."""
    manager = SignalConfigManager()
    
    if manager.delete_config(args.name):
        print(f"Deleted configuration '{args.name}'")
    else:
        print(f"Configuration '{args.name}' not found")
        sys.exit(1)


def export_configs(args):
    """Export configurations."""
    manager = SignalConfigManager()
    
    try:
        manager.export_configs(args.output, names=args.names)
        print(f"Exported configurations to {args.output}")
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


def import_configs(args):
    """Import configurations."""
    manager = SignalConfigManager()
    
    try:
        manager.import_configs(args.input, overwrite=args.overwrite)
        print(f"Imported configurations from {args.input}")
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Signal Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List configurations')
    list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    list_parser.set_defaults(func=list_configs)
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration details')
    show_parser.add_argument('name', help='Configuration name')
    show_parser.set_defaults(func=show_config)
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new configuration')
    create_parser.add_argument('name', help='Configuration name')
    create_parser.add_argument('-d', '--description', help='Description')
    create_parser.add_argument('-t', '--tags', nargs='+', help='Tags')
    create_parser.add_argument('--template', choices=['scalping', 'swing', 'longterm', 'default'],
                              default='default', help='Template to use')
    create_parser.add_argument('-p', '--params', nargs='+', 
                              help='Custom parameters (key=value)')
    create_parser.add_argument('-f', '--format', choices=['json', 'yaml'],
                              default='json', help='Save format')
    create_parser.set_defaults(func=create_config)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete configuration')
    delete_parser.add_argument('name', help='Configuration name')
    delete_parser.set_defaults(func=delete_config)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configurations')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('--names', nargs='+', help='Specific configs to export')
    export_parser.set_defaults(func=export_configs)
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configurations')
    import_parser.add_argument('input', help='Input file path')
    import_parser.add_argument('--overwrite', action='store_true',
                              help='Overwrite existing configurations')
    import_parser.set_defaults(func=import_configs)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()