from typing import Dict, Tuple, Optional
from argparse import ArgumentError
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import PositiveInt

def init_config(args=None):
    """Initialize configuration using jsonargparse."""
    parser = ArgumentParser(default_env=True, default_config_files=None)
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a base config file', required=True)
    parser.add_argument('--data', type=Dict, help='Data configurations')
    parser.add_argument('--meta_data_path', type=str, default='', help='Path to dataset metadata file')
    parser.add_argument('--data_path', type=str, default='', help='Path to dataset')
    parser.add_argument('--formatter', type=str, default='', required=True)
    parser.add_argument('--model_cache_path', type=str, default='~/.cache', help='Path to model cache directory')
    parser.add_argument('--num_workers', type=PositiveInt, default=0, help='Number of worker threads')
    parser.add_argument('--image_key', type=str, default='image', help='Key for image file name in dataset')
    parser.add_argument('--image_caption_key', type=str, default='caption', help='Key for image caption in dataset')
    parser.add_argument('--id_key', type=str, default='id', help='Key for ID in dataset')
    parser.add_argument('--pure_image_meta_data_path', type=str, help='Path to pure image metadata file')
    parser.add_argument('--image_caption_meta_data_path', type=str, help='Path to image-caption metadata file')
    parser.add_argument('--ref_pure_image_meta_data_path', type=str, default=None, help='Path to reference pure image metadata file')
    parser.add_argument('--image_folder_path', type=str, help='Path to image folder')
    parser.add_argument('--ref_image_folder_path', type=str, default=None, help='Path to reference image folder')
    parser.add_argument('--text', type=Dict, help='Text scorer configurations')
    parser.add_argument('--image', type=Dict, help='Image scorer configurations')
    parser.add_argument('--video', type=Dict, help='Video scorer configurations')

    try:
        cfg = parser.parse_args(args=args)
        return cfg
    except ArgumentError:
        print('Configuration initialization failed')


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
