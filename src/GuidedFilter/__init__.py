"""
Guided Filter implementation based on the C++ GuidedFilter code.
"""

import numpy as np
import cv2


def guided_filter_gray(I, p, r, eps):
    """
    Guided filter for grayscale guidance image.
    
    Args:
        I: Guidance image (H, W) in [0, 1] range, float32
        p: Input image (H, W) in [0, 1] range, float32
        r: Radius of local window
        eps: Regularization parameter
    
    Returns:
        q: Filtered output (H, W) in [0, 1] range, float32
    """
    ksize = (2 * r + 1, 2 * r + 1)
    
    # Compute means
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize, normalize=True)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize, normalize=True)
    
    # Covariance of (I, p)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize, normalize=True)
    cov_Ip = mean_Ip - mean_I * mean_p
    
    # Variance of I
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize, normalize=True)
    var_I = mean_II - mean_I * mean_I
    
    # Compute a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Mean of a and b
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True)
    
    # Output
    q = mean_a * I + mean_b
    
    return q


def guided_filter_color(I, p, r, eps):
    """
    Guided filter for color (BGR) guidance image.
    
    Args:
        I: Guidance image (H, W, 3) in [0, 1] range, float32, BGR order
        p: Input image (H, W) in [0, 1] range, float32
        r: Radius of local window
        eps: Regularization parameter
    
    Returns:
        q: Filtered output (H, W) in [0, 1] range, float32
    """
    ksize = (2 * r + 1, 2 * r + 1)
    H, W = p.shape
    
    # Split BGR channels
    b, g, r_ch = cv2.split(I)
    
    # Compute means
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True)
    mean_g = cv2.boxFilter(g, cv2.CV_32F, ksize, normalize=True)
    mean_r = cv2.boxFilter(r_ch, cv2.CV_32F, ksize, normalize=True)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize, normalize=True)
    
    # Covariance of (I, p) for each channel
    mean_bp = cv2.boxFilter(b * p, cv2.CV_32F, ksize, normalize=True)
    mean_gp = cv2.boxFilter(g * p, cv2.CV_32F, ksize, normalize=True)
    mean_rp = cv2.boxFilter(r_ch * p, cv2.CV_32F, ksize, normalize=True)
    
    cov_bp = mean_bp - mean_b * mean_p
    cov_gp = mean_gp - mean_g * mean_p
    cov_rp = mean_rp - mean_r * mean_p
    
    # Variance of I (3x3 covariance matrix for each pixel)
    mean_bb = cv2.boxFilter(b * b, cv2.CV_32F, ksize, normalize=True)
    mean_bg = cv2.boxFilter(b * g, cv2.CV_32F, ksize, normalize=True)
    mean_br = cv2.boxFilter(b * r_ch, cv2.CV_32F, ksize, normalize=True)
    mean_gg = cv2.boxFilter(g * g, cv2.CV_32F, ksize, normalize=True)
    mean_gr = cv2.boxFilter(g * r_ch, cv2.CV_32F, ksize, normalize=True)
    mean_rr = cv2.boxFilter(r_ch * r_ch, cv2.CV_32F, ksize, normalize=True)
    
    var_bb = mean_bb - mean_b * mean_b
    var_bg = mean_bg - mean_b * mean_g
    var_br = mean_br - mean_b * mean_r
    var_gg = mean_gg - mean_g * mean_g
    var_gr = mean_gr - mean_g * mean_r
    var_rr = mean_rr - mean_r * mean_r
    
    # Solve 3x3 linear system for each pixel: sigma * a = cov
    A_b = np.zeros((H, W), dtype=np.float32)
    A_g = np.zeros((H, W), dtype=np.float32)
    A_r = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            # Build 3x3 covariance matrix
            sigma = np.array([
                [var_bb[row, col], var_bg[row, col], var_br[row, col]],
                [var_bg[row, col], var_gg[row, col], var_gr[row, col]],
                [var_br[row, col], var_gr[row, col], var_rr[row, col]]
            ], dtype=np.float32)
            
            # Add regularization
            sigma += eps * np.eye(3, dtype=np.float32)
            
            # Right-hand side
            cov = np.array([
                [cov_bp[row, col]],
                [cov_gp[row, col]],
                [cov_rp[row, col]]
            ], dtype=np.float32)
            
            # Solve: sigma * a = cov
            a = np.linalg.solve(sigma, cov)
            
            A_b[row, col] = a[0, 0]
            A_g[row, col] = a[1, 0]
            A_r[row, col] = a[2, 0]
            
            B[row, col] = mean_p[row, col] - a[0, 0] * mean_b[row, col] - a[1, 0] * mean_g[row, col] - a[2, 0] * mean_r[row, col]
    
    # Mean of a and b
    mean_A_b = cv2.boxFilter(A_b, cv2.CV_32F, ksize, normalize=True)
    mean_A_g = cv2.boxFilter(A_g, cv2.CV_32F, ksize, normalize=True)
    mean_A_r = cv2.boxFilter(A_r, cv2.CV_32F, ksize, normalize=True)
    mean_B = cv2.boxFilter(B, cv2.CV_32F, ksize, normalize=True)
    
    # Output
    q = mean_A_b * b + mean_A_g * g + mean_A_r * r_ch + mean_B
    
    return q


def guided_filter(I, p, r, eps):
    """
    Guided filter - automatically chooses grayscale or color version.
    
    Args:
        I: Guidance image (H, W) or (H, W, 3) in [0, 1] range, float32, BGR order
        p: Input image (H, W) in [0, 1] range, float32
        r: Radius of local window
        eps: Regularization parameter
    
    Returns:
        q: Filtered output (H, W) in [0, 1] range, float32
    """
    if len(I.shape) == 3 and I.shape[2] == 3:
        return guided_filter_color(I, p, r, eps)
    elif len(I.shape) == 2:
        return guided_filter_gray(I, p, r, eps)
    else:
        raise ValueError(f'Unsupported guidance image shape: {I.shape}')


def apply_guided_filter(segmented_frame, relit_frame, radius=8, eps=0.01):
    """
    Apply guided filter to transfer relit lighting style to segmented frame.
    Wrapper function matching the old interface.
    
    Args:
        segmented_frame: Original segmented frame (H, W, 3) in [0, 255] range, BGR
        relit_frame: Relit frame with lighting effects (H, W, 3) in [0, 255] range, BGR
        radius: Guided filter radius (default: 8)
        eps: Guided filter regularization (default: 0.01)
    
    Returns:
        filtered_frame: Frame with relit style applied (H, W, 3) in [0, 255] range, BGR
    """
    # Ensure same dimensions
    if segmented_frame.shape[:2] != relit_frame.shape[:2]:
        h, w = segmented_frame.shape[:2]
        relit_frame = cv2.resize(relit_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to float32 and normalize to [0, 1]
    if segmented_frame.dtype != np.float32:
        segmented_norm = segmented_frame.astype(np.float32) / 255.0
    else:
        segmented_norm = segmented_frame.copy()
        if segmented_norm.max() > 1.0:
            segmented_norm = segmented_norm / 255.0
    
    if relit_frame.dtype != np.float32:
        relit_norm = relit_frame.astype(np.float32) / 255.0
    else:
        relit_norm = relit_frame.copy()
        if relit_norm.max() > 1.0:
            relit_norm = relit_norm / 255.0
    
    # Convert segmented to grayscale for filtering input
    if len(segmented_norm.shape) == 3:
        p = cv2.cvtColor(segmented_norm, cv2.COLOR_BGR2GRAY)
    else:
        p = segmented_norm
    
    # Apply guided filter: relit is guide (I), segmented grayscale is input (p)
    q = guided_filter(relit_norm, p, radius, eps)
    
    # Convert back to BGR and scale to [0, 255]
    if len(q.shape) == 2:
        # Grayscale output, convert to BGR
        q_bgr = cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    else:
        q_bgr = q
    
    # Clamp and convert to uint8
    q_bgr = np.clip(q_bgr, 0.0, 1.0)
    q_uint8 = (q_bgr * 255.0).astype(np.uint8)
    
    return q_uint8

