"""
Albedo estimation module.
"""

import numpy as np
import cv2


def compute_spherical_harmonics_basis(normals):
    """Compute spherical harmonics basis functions for given normals."""
    H, W = normals.shape[:2]
    nx, ny, nz = normals[:, :, 0], normals[:, :, 1], normals[:, :, 2]
    
    # Using the standard SH basis functions
    sh_basis = np.zeros((H, W, 9), dtype=np.float32)
    
    # Order 0 (l=0, m=0)
    sh_basis[:, :, 0] = 0.282095  # Y_0^0 = 1/(2*sqrt(pi))
    
    # Order 1 (l=1)
    sh_basis[:, :, 1] = 0.488603 * ny  # Y_1^{-1} = sqrt(3/(4*pi)) * y
    sh_basis[:, :, 2] = 0.488603 * nz  # Y_1^0 = sqrt(3/(4*pi)) * z
    sh_basis[:, :, 3] = 0.488603 * nx  # Y_1^1 = sqrt(3/(4*pi)) * x
    
    # Order 2 (l=2)
    sh_basis[:, :, 4] = 1.092548 * nx * ny  # Y_2^{-2} = sqrt(15/(4*pi)) * x*y
    sh_basis[:, :, 5] = 1.092548 * ny * nz  # Y_2^{-1} = sqrt(15/(4*pi)) * y*z
    sh_basis[:, :, 6] = 0.315392 * (3 * nz * nz - 1)  # Y_2^0 = sqrt(5/(16*pi)) * (3*z^2 - 1)
    sh_basis[:, :, 7] = 1.092548 * nx * nz  # Y_2^1 = sqrt(15/(4*pi)) * x*z
    sh_basis[:, :, 8] = 0.546274 * (nx * nx - ny * ny)  # Y_2^2 = sqrt(15/(16*pi)) * (x^2 - y^2)
    
    return sh_basis


def estimate_lighting(image, normals, mask):
    """Estimate spherical harmonics lighting coefficients from image and surface normals."""
    # Compute SH basis
    sh_basis = compute_spherical_harmonics_basis(normals)  # (H, W, 9)
    
    # Flatten and apply mask
    valid_mask = mask > 0.1  # Threshold to avoid near-zero values
    sh_flat = sh_basis[valid_mask]  # (N, 9)
    img_flat = image[valid_mask]  # (N, 3)
    
    if sh_flat.shape[0] < 9:
        # Not enough valid pixels, return default lighting (uniform)
        sh_coeffs = np.zeros((3, 9), dtype=np.float32)
        sh_coeffs[:, 0] = 0.5  # Ambient term
        return sh_coeffs
    
    # Solve for SH coefficients using least squares: image = sh_basis * coeffs
    sh_coeffs = np.zeros((3, 9), dtype=np.float32)
    
    for c in range(3):
        # Solve: img_c = sh_basis @ coeffs_c
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(sh_flat, img_flat[:, c], rcond=None)
            sh_coeffs[c, :] = coeffs[:9]
        except np.linalg.LinAlgError:
            # Fallback to default if solving fails
            sh_coeffs[c, 0] = 0.5
    
    return sh_coeffs


def compute_shading(normals, sh_coeffs):
    """Compute shading from surface normals and spherical harmonics coefficients."""
    sh_basis = compute_spherical_harmonics_basis(normals)  # (H, W, 9)
    H, W = normals.shape[:2]
    
    # Compute shading for each color channel
    shading = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        shading[:, :, c] = np.dot(sh_basis, sh_coeffs[c, :])
    
    # Ensure non-negative
    shading = np.maximum(shading, 0.0)
    
    return shading


def estimate_albedo(segmented_image, surface_normals, alpha_matte):
    """
    Estimate albedo map from segmented image, surface normals, and alpha matte.
    
    Args:
        segmented_image: numpy array of shape (H, W, 3) with RGB segmented image in [0, 255] range
        surface_normals: numpy array of shape (H, W, 3) with surface normals in [0, 255] range (RGB visualization)
        alpha_matte: numpy array of shape (H, W) with alpha matte in [0, 255] or [0, 1] range
    
    Returns:
        albedo: numpy array of shape (H, W, 3) with albedo map in [0, 255] range
        normals: numpy array of shape (H, W, 3) with normalized normals in [-1, 1] range
    """
    # Normalize inputs
    if segmented_image.max() > 1.0:
        image = segmented_image.astype(np.float32) / 255.0
    else:
        image = segmented_image.astype(np.float32)
    
    if alpha_matte.max() > 1.0:
        mask = alpha_matte.astype(np.float32) / 255.0
    else:
        mask = alpha_matte.astype(np.float32)
    
    # Convert surface normals from RGB visualization to actual normals
    if surface_normals.max() > 1.0:
        normals_rgb = surface_normals.astype(np.float32) / 255.0
    else:
        normals_rgb = surface_normals.astype(np.float32)
    
    # Convert from RGB visualization to actual normals [-1, 1]
    normals = normals_rgb * 2.0 - 1.0
    
    # Ensure normals are normalized
    norm_mag = np.linalg.norm(normals, axis=2, keepdims=True)
    norm_mag = np.maximum(norm_mag, 1e-6)
    normals = normals / norm_mag
    
    # Ensure image and normals have same spatial dimensions
    if image.shape[:2] != normals.shape[:2]:
        h, w = image.shape[:2]
        normals = cv2.resize(normals, (w, h), interpolation=cv2.INTER_LINEAR)
        # Renormalize after resize
        norm_mag = np.linalg.norm(normals, axis=2, keepdims=True)
        norm_mag = np.maximum(norm_mag, 1e-6)
        normals = normals / norm_mag
    
    if mask.shape[:2] != image.shape[:2]:
        h, w = image.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Estimate lighting
    sh_coeffs = estimate_lighting(image, normals, mask)
    
    # Compute shading
    shading = compute_shading(normals, sh_coeffs)
    
    # Ensure shading is reasonable and non-zero
    epsilon = 1e-6
    shading_safe = np.maximum(shading, epsilon)
    
    # Check shading statistics
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    valid_shading = shading_safe[mask_3ch > 0.1]
    if len(valid_shading) > 0:
        shading_mean = np.mean(valid_shading)
        if shading_mean < 0.05 or shading_mean > 3.0:
            if shading_mean > 0:
                shading_scale = 0.5 / shading_mean
                shading_safe = shading_safe * min(shading_scale, 2.0)
    
    # Solve for albedo: image = albedo * shading
    albedo_raw = image / shading_safe
    
    # Clamp albedo to reasonable range
    albedo_raw = np.clip(albedo_raw, 0.0, 2.0)
    
    # Additional validation
    valid_albedo = albedo_raw[mask_3ch > 0.1]
    if len(valid_albedo) > 0:
        albedo_median = np.median(valid_albedo)
        if albedo_median > 1.2:
            scale_factor = 0.8 / albedo_median
            albedo_raw = albedo_raw * scale_factor
            albedo_raw = np.clip(albedo_raw, 0.0, 2.0)
    
    # Apply mask to zero out background
    albedo_raw = albedo_raw * mask_3ch
    
    # Normalize to [0, 1] range for display (preserve color balance)
    albedo_display = albedo_raw.copy()
    valid_mask = mask_3ch > 0.1
    
    if np.any(valid_mask):
        valid_albedo = albedo_display[valid_mask]
        if len(valid_albedo) > 0:
            p99 = np.percentile(valid_albedo, 99)
            if p99 > 0:
                albedo_display = np.clip(albedo_display / p99, 0.0, 1.0)
            else:
                max_val = np.max(valid_albedo)
                if max_val > 0:
                    albedo_display = np.clip(albedo_display / max_val, 0.0, 1.0)
    
    # Apply mask to zero out background
    albedo_display = albedo_display * mask_3ch
    
    # Convert to uint8
    albedo_uint8 = (albedo_display * 255).astype(np.uint8)
    
    return albedo_uint8, normals

