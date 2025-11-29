"""
DPR (Deep Portrait Relighting) module.
"""

import os
import sys
import numpy as np
import cv2
import torch

# Import DPR model from local src/models
from .models.dpr.defineHourglass_512_gray_skip import HourglassNet


def load_dpr_model(model_path, use_gpu=True):
    """Load and initialize DPR model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'DPR model not found at {model_path}')
    
    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create and load model
    model = HourglassNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f'DPR model loaded successfully on {device}')
    return model, device


def load_dpr_sh_coeffs(file_path):
    """Load DPR-format SH coefficients from a text file."""
    try:
        sh_coeffs = np.loadtxt(file_path)
        if len(sh_coeffs) >= 9:
            return sh_coeffs[:9].astype(np.float32)
        else:
            raise ValueError(f'SH file {file_path} has fewer than 9 coefficients')
    except Exception as e:
        raise ValueError(f'Error loading DPR SH coefficients from {file_path}: {e}')


def convert_sh_coeffs_to_dpr_format(sh_coeffs_rgb):
    """Convert RGB SH coefficients (3, 9) to DPR format (single 9-element vector)."""
    # Convert RGB SH to grayscale using standard weights: 0.299*R + 0.587*G + 0.114*B
    weights = np.array([0.299, 0.587, 0.114])
    sh_coeffs_dpr = np.dot(weights, sh_coeffs_rgb)
    return sh_coeffs_dpr.astype(np.float32)


def relight_subject_neural(image_rgb, dpr_model, device, dpr_sh_coeffs, mask=None, target_size=512):
    """
    Apply neural relighting using DPR model.
    
    Args:
        image_rgb: Input RGB image (H, W, 3) in [0, 255] range
        dpr_model: Loaded DPR model
        device: torch device
        dpr_sh_coeffs: DPR-format SH coefficients (9,)
        mask: Optional alpha matte (H, W) for masking
        target_size: Target size for DPR processing (512 or 1024)
    
    Returns:
        relit_image: Relit RGB image (H, W, 3) in [0, 255] range
    """
    # Store original size
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Normalize image to [0, 1] if needed
    if image_rgb.max() > 1.0:
        image = image_rgb.astype(np.float32) / 255.0
    else:
        image = image_rgb.astype(np.float32)
    
    # Resize to target size (DPR works on 512x512 or 1024x1024)
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert RGB to LAB color space
    image_bgr = cv2.cvtColor((image_resized * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab = lab.astype(np.float32) / 255.0
    
    # Extract L channel (luminance) - DPR expects values in [0, 1]
    L = lab[:, :, 0]  # Shape: (H, W)
    
    # Prepare input for DPR
    inputL = L.transpose((0, 1))  # Match DPR code exactly
    inputL = inputL[None, None, ...]  # Add batch and channel dimensions: (1, 1, H, W)
    inputL_tensor = torch.from_numpy(inputL).to(device)
    
    # Prepare SH coefficients for DPR
    sh_dpr = np.reshape(dpr_sh_coeffs, (1, 9, 1, 1)).astype(np.float32)
    sh_tensor = torch.from_numpy(sh_dpr).to(device)
    
    # Run DPR model
    with torch.no_grad():
        outputL, outputSH = dpr_model(inputL_tensor, sh_tensor, 0)
        outputL = outputL[0].cpu().data.numpy()  # Shape: (1, H, W) from model output
        
        # Process output exactly as DPR test code does
        outputL = outputL.transpose((1, 2, 0))  # From (1, H, W) to (H, W, 1)
        outputL = np.squeeze(outputL)  # Remove singleton dims: (H, W)
        
        # Ensure output is in [0, 1] range
        outputL = np.clip(outputL, 0.0, 1.0)
        
        # Verify dimensions match original L channel
        if outputL.shape != L.shape:
            outputL = cv2.resize(outputL, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Combine new L channel with original A and B channels
    lab[:, :, 0] = outputL
    lab_uint8 = (lab * 255.0).astype(np.uint8)
    
    # Convert back to BGR then RGB
    result_bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize back to original size
    result_rgb = cv2.resize(result_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Apply mask if provided
    if mask is not None:
        if mask.max() > 1.0:
            mask_norm = mask.astype(np.float32) / 255.0
        else:
            mask_norm = mask.astype(np.float32)
        
        if mask_norm.shape[:2] != result_rgb.shape[:2]:
            h, w = result_rgb.shape[:2]
            mask_norm = cv2.resize(mask_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        
        mask_3ch = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
        result_rgb = result_rgb * mask_3ch
    
    # Convert to uint8
    result_rgb = np.clip(result_rgb, 0.0, 255.0).astype(np.uint8)
    
    return result_rgb

