"""
DSINE module for surface normal estimation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_dsine_model(model_path, use_gpu=True):
    """Load and initialize DSINE model for surface normal estimation."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'DSINE model not found at {model_path}')
    
    print(f'Loading DSINE model from: {model_path}')
    
    try:
        device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        print(f'Using device: {device}')
        
        # Import DSINE components from local src
        from src.models.dsine.v02 import DSINE_v02 as DSINE
        from src.utils.dsine import utils
        from src.utils.dsine.projection import intrins_from_fov
        
        # Create config object for DSINE
        class DSINEConfig:
            def __init__(self):
                self.NNET_architecture = 'v02'
                self.NNET_output_dim = 3
                self.NNET_output_type = 'R'
                self.NNET_feature_dim = 64
                self.NNET_hidden_dim = 64
                self.NNET_encoder_B = 5
                self.NNET_decoder_NF = 2048
                self.NNET_decoder_BN = False
                self.NNET_decoder_down = 8
                self.NNET_learned_upsampling = False
                self.NRN_prop_ps = 5
                self.NRN_num_iter_train = 5
                self.NRN_num_iter_test = 5
                self.NRN_ray_relu = False
        
        args = DSINEConfig()
        
        # Create model
        model = DSINE(args).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        
        model.eval()
        
        # Create predictor wrapper
        class DSINEPredictor:
            def __init__(self, model, device):
                self.model = model
                self.device = device
                self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            def infer_pil(self, img, intrins=None):
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                _, _, orig_H, orig_W = img.shape
                
                # Pad input
                l, r, t, b = utils.get_padding(orig_H, orig_W)
                img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
                img = self.transform(img)
                
                if intrins is None:
                    intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
                
                intrins[:, 0, 2] += l
                intrins[:, 1, 2] += t
                
                with torch.no_grad():
                    pred_norm = self.model(img, intrins=intrins)[-1]
                    pred_norm = pred_norm[:, :, t:t+orig_H, l:l+orig_W]
                
                return pred_norm
        
        predictor = DSINEPredictor(model, device)
        
        print(f'DSINE model loaded successfully on {device}')
        return predictor, device
    except Exception as e:
        print(f'Error loading DSINE model: {e}')
        import traceback
        traceback.print_exc()
        raise


def generate_surface_normals(image_rgb, dsine_predictor):
    """
    Generate surface normals from RGB image using DSINE.
    
    Args:
        image_rgb: RGB image (H, W, 3) in [0, 255] or PIL Image
        dsine_predictor: Loaded DSINE predictor
    
    Returns:
        normals: Surface normals (H, W, 3) in [0, 255] for visualization
    """
    # Convert to PIL Image if needed
    if isinstance(image_rgb, np.ndarray):
        image_pil = Image.fromarray(image_rgb)
    else:
        image_pil = image_rgb
    
    # Generate normals
    pred_norm = dsine_predictor.infer_pil(image_pil)  # Returns tensor (1, 3, H, W) in [-1, 1]
    pred_np = pred_norm.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
    
    # Convert to [0, 255] range for visualization
    pred_img = ((pred_np + 1) * 127.5).astype(np.uint8)
    pred_img = np.clip(pred_img, 0, 255)
    
    return pred_img

