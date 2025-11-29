"""
Lighting preset functions.
"""

import os
import numpy as np
from src.dpr import load_dpr_sh_coeffs


def create_lighting_preset(preset_name, return_dpr_format=True):
    """
    Create lighting presets using DPR-format SH coefficients.
    
    Args:
        preset_name: Name of the preset (e.g., 'front', 'red', 'purple')
        return_dpr_format: If True, return DPR-format SH coefficients (9,)
    
    Returns:
        sh_coeffs_dpr: numpy array of shape (9,) with DPR-format SH coefficients
    """
    preset_name = preset_name.lower()
    
    # Get path relative to this file
    src_dir = os.path.dirname(os.path.abspath(__file__))
    dpr_light_dir = os.path.join(src_dir, 'data', 'dpr', 'example_light')
    dpr_colored_dir = os.path.join(src_dir, 'data', 'dpr', 'colored_light')
    
    # Map preset names to DPR file names
    dpr_file_map = {
        'red': 'red_light_00.txt',
        'red-dramatic': 'red_light_00.txt',
        'red-intense': 'red_light_01.txt',
        'purple': 'purple_light_00.txt',
        'purple-dramatic': 'purple_light_00.txt',
        'purple-intense': 'purple_light_01.txt',
        'blue': 'blue_light_00.txt',
        'green': 'green_light_00.txt',
        'orange': 'orange_light_00.txt',
        'yellow': 'yellow_light_00.txt',
        'crimson': 'red_light_00.txt',
        'violet': 'purple_light_00.txt',
        'magenta': 'purple_light_01.txt',
    }
    
    # Check colored light directory first
    if preset_name in dpr_file_map:
        colored_file = os.path.join(dpr_colored_dir, dpr_file_map[preset_name])
        if os.path.exists(colored_file):
            sh_coeffs_dpr = load_dpr_sh_coeffs(colored_file)
            return sh_coeffs_dpr
    
    # Try example_light directory for base lighting
    example_files = ['rotate_light_00.txt', 'rotate_light_01.txt', 'rotate_light_02.txt']
    if preset_name in ['front', 'side-left', 'side-right', 'top']:
        example_file = os.path.join(dpr_light_dir, example_files[0])
        if os.path.exists(example_file):
            sh_coeffs_dpr = load_dpr_sh_coeffs(example_file)
            return sh_coeffs_dpr
    
    # Default: use first example file
    example_file = os.path.join(dpr_light_dir, example_files[0])
    if os.path.exists(example_file):
        sh_coeffs_dpr = load_dpr_sh_coeffs(example_file)
        return sh_coeffs_dpr
    
    # Fallback: return default lighting
    sh_coeffs_dpr = np.zeros(9, dtype=np.float32)
    sh_coeffs_dpr[0] = 0.5  # Ambient term
    return sh_coeffs_dpr

