"""
MODNet module for portrait matting.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import MODNet from local src/models
from .models.modnet.modnet import MODNet


def load_modnet(model_path, use_gpu=True):
    """Load and initialize MODNet model."""
    print(f'Loading MODNet from: {model_path}')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'MODNet checkpoint not found at {model_path}')
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    if use_gpu and torch.cuda.is_available():
        print('Using GPU...')
        modnet = modnet.cuda()
        weights = torch.load(model_path)
    else:
        print('Using CPU...')
        use_gpu = False
        weights = torch.load(model_path, map_location=torch.device('cpu'))
    
    modnet.load_state_dict(weights)
    modnet.eval()
    
    return modnet, use_gpu


def segment_image(image, matte):
    """Segment image using alpha matte (extract foreground)."""
    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image_norm = image.astype(np.float32) / 255.0
    else:
        image_norm = image.astype(np.float32)
    
    # Ensure matte is in [0, 1] range
    if matte.max() > 1.0:
        matte = matte.astype(np.float32) / 255.0
    else:
        matte = matte.astype(np.float32)
    
    # Expand matte to 3 channels
    if len(matte.shape) == 2:
        matte_3ch = np.stack([matte, matte, matte], axis=2)
    else:
        matte_3ch = matte
    
    # Ensure image and matte have same shape
    if image_norm.shape[:2] != matte_3ch.shape[:2]:
        h, w = image_norm.shape[:2]
        matte_3ch = cv2.resize(matte_3ch, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Segment foreground (apply matte to image)
    segmented = image_norm * matte_3ch
    
    # Convert back to uint8
    segmented = (np.clip(segmented, 0.0, 1.0) * 255).astype(np.uint8)
    
    return segmented


def process_frame_modnet(frame_rgb, modnet, use_gpu=True, ref_size=512):
    """
    Process a single frame with MODNet to generate alpha matte.
    
    Args:
        frame_rgb: RGB frame (H, W, 3) in [0, 255]
        modnet: Loaded MODNet model
        use_gpu: Whether to use GPU
        ref_size: Reference size for processing
    
    Returns:
        matte: Alpha matte (H, W) in [0, 1]
    """
    # Define transforms
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Convert to PIL Image
    im = Image.fromarray(frame_rgb)
    im = im_transform(im)
    im = im[None, :, :, :]
    
    # Resize for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
    
    # Inference MODNet
    with torch.no_grad():
        _, _, matte = modnet(im.cuda() if use_gpu else im, True)
    
    # Resize matte to original size
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    
    return matte

