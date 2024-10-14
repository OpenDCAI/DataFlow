from typing import Dict, Tuple, Optional
from argparse import ArgumentError
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import PositiveInt


def new_init_config(args=None):
    """Initialize new configuration with updated settings."""
    parser = ArgumentParser(default_env=True, default_config_files=None)
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a base config file', required=True)
    parser.add_argument('--data', type=Dict, help='Data configurations')
    parser.add_argument('--scorers', type=Dict, required=True, help='Scorer configurations')
    # parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation')
    parser.add_argument('--model_cache_path', type=str, default='~/.cache', help='Path to model cache directory')
    parser.add_argument('--num_workers', type=PositiveInt, default=1, help='Number of worker threads')

    try:
        cfg = parser.parse_args(args=args)
        return cfg
    except ArgumentError:
        print('Configuration initialization failed')
