"""
Main script for video relighting pipeline.
Clean code flow: MODNet -> DSINE -> Albedo -> DPR -> Guided Filter -> Output
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Import modules
from src.modnet import load_modnet, segment_image, process_frame_modnet
from src.dsine import load_dsine_model, generate_surface_normals
from src.albedo import estimate_albedo
from src.dpr import load_dpr_model, relight_subject_neural
from src.GuidedFilter import apply_guided_filter
from src.lighting import create_lighting_preset


def get_model_paths():
    """Get paths to model checkpoints."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'modnet': os.path.join(script_dir, 'models', 'modnet_photographic_portrait_matting.ckpt'),
        'dsine': os.path.join(script_dir, 'models', 'dsine.pt'),
        'dpr': os.path.join(script_dir, 'models', 'trained_model_03.t7'),
    }


def process_image(input_path, output_dir, modnet, dsine_predictor, dpr_model, dpr_device, 
                 lighting_preset='front', use_gpu=True):
    """Process a single image."""
    print(f'Processing image: {input_path}')
    
    # Read image
    im_orig = Image.open(input_path)
    im_np = np.asarray(im_orig)
    
    # Ensure RGB
    if len(im_np.shape) == 2:
        im_np = im_np[:, :, None]
    if im_np.shape[2] == 1:
        im_np = np.repeat(im_np, 3, axis=2)
    elif im_np.shape[2] == 4:
        im_np = im_np[:, :, 0:3]
    
    frame_rgb = im_np
    
    # 6.2 MODNet: Create alpha matte and segmentation
    print('  Step 1: Generating alpha matte with MODNet...')
    matte = process_frame_modnet(frame_rgb, modnet, use_gpu=use_gpu)
    segmented = segment_image(frame_rgb, matte)
    
    # Save matte and segmented
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    matte_path = os.path.join(output_dir, f'{base_name}_matte.png')
    segmented_path = os.path.join(output_dir, f'{base_name}_segmented.png')
    Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(matte_path)
    Image.fromarray(segmented).save(segmented_path)
    print(f'  Saved matte: {matte_path}')
    print(f'  Saved segmented: {segmented_path}')
    
    # 6.3 DSINE: Create surface normals
    print('  Step 2: Generating surface normals with DSINE...')
    normals = generate_surface_normals(segmented, dsine_predictor)
    normals_path = os.path.join(output_dir, f'{base_name}_normals.png')
    Image.fromarray(normals).save(normals_path)
    print(f'  Saved normals: {normals_path}')
    
    # 6.4 Albedo estimation
    print('  Step 3: Estimating albedo...')
    matte_uint8 = (matte * 255).astype(np.uint8)
    albedo, normals_for_albedo = estimate_albedo(segmented, normals, matte_uint8)
    albedo_path = os.path.join(output_dir, f'{base_name}_albedo.png')
    Image.fromarray(albedo).save(albedo_path)
    print(f'  Saved albedo: {albedo_path}')
    
    # Convert albedo to BGR for DPR
    albedo_bgr = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
    
    # 6.5 DPR: Relighting
    print('  Step 4: Applying relighting with DPR...')
    dpr_sh_coeffs = create_lighting_preset(lighting_preset)
    dpr_output = relight_subject_neural(
        albedo_bgr, dpr_model, dpr_device, dpr_sh_coeffs, 
        mask=matte_uint8, target_size=1024
    )
    # Convert back to RGB
    dpr_output_rgb = cv2.cvtColor(dpr_output, cv2.COLOR_BGR2RGB)
    dpr_output_path = os.path.join(output_dir, f'{base_name}_dpr_output.png')
    Image.fromarray(dpr_output_rgb).save(dpr_output_path)
    print(f'  Saved DPR output: {dpr_output_path}')
    
    # 6.6 Guided filter
    print('  Step 5: Applying guided filter...')
    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
    guided_output = apply_guided_filter(segmented_bgr, dpr_output, radius=8, eps=0.01)
    
    # 6.7 Average and save
    print('  Step 6: Averaging guided filter output with segmented...')
    avg_bgr = ((guided_output.astype(np.float32) + segmented_bgr.astype(np.float32)) / 2.0).astype(np.uint8)
    avg_rgb = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2RGB)
    output_path = os.path.join(output_dir, f'{base_name}_relit.png')
    Image.fromarray(avg_rgb).save(output_path)
    print(f'  Saved final output: {output_path}')
    
    print('  Done!')


def process_video(input_path, output_dir, modnet, dsine_predictor, dpr_model, dpr_device,
                 lighting_preset='front', use_gpu=True, max_frames=10):
    """Process a video."""
    print(f'Processing video: {input_path}')
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {input_path}')
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'  Video info: {width}x{height} @ {fps} fps, {total_frames} frames')
    
    # Limit frames if specified
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    
    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    matte_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'output_matte.mp4'), fourcc, fps, (width, height)
    )
    segmented_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'output_segmented.mp4'), fourcc, fps, (width, height)
    )
    normals_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'output_normals.mp4'), fourcc, fps, (width, height)
    )
    albedo_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'output_albedo.mp4'), fourcc, fps, (width, height)
    )
    dpr_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'output_dpr.mp4'), fourcc, fps, (width, height)
    )
    relit_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'output_relit.mp4'), fourcc, fps, (width, height)
    )
    
    # Load lighting preset
    dpr_sh_coeffs = create_lighting_preset(lighting_preset)
    
    # Process frames
    frame_count = 0
    for _ in tqdm(range(total_frames), desc='Processing frames'):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 6.2 MODNet: Create alpha matte and segmentation
        matte = process_frame_modnet(frame_rgb, modnet, use_gpu=use_gpu)
        segmented = segment_image(frame_rgb, matte)
        
        # Convert to BGR and save
        matte_bgr = cv2.cvtColor((matte * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
        matte_writer.write(matte_bgr)
        segmented_writer.write(segmented_bgr)
        
        # 6.3 DSINE: Create surface normals
        normals = generate_surface_normals(segmented, dsine_predictor)
        normals_bgr = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)
        normals_writer.write(normals_bgr)
        
        # 6.4 Albedo estimation
        matte_uint8 = (matte * 255).astype(np.uint8)
        albedo, _ = estimate_albedo(segmented, normals, matte_uint8)
        albedo_bgr = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
        albedo_writer.write(albedo_bgr)
        
        # 6.5 DPR: Relighting
        dpr_output = relight_subject_neural(
            albedo_bgr, dpr_model, dpr_device, dpr_sh_coeffs,
            mask=matte_uint8, target_size=512
        )
        dpr_writer.write(dpr_output)
        
        # 6.6 Guided filter
        guided_output = apply_guided_filter(segmented_bgr, dpr_output, radius=8, eps=0.01)
        
        # 6.7 Average and save
        avg_bgr = ((guided_output.astype(np.float32) + segmented_bgr.astype(np.float32)) / 2.0).astype(np.uint8)
        relit_writer.write(avg_bgr)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    matte_writer.release()
    segmented_writer.release()
    normals_writer.release()
    albedo_writer.release()
    dpr_writer.release()
    relit_writer.release()
    
    print(f'  Processed {frame_count} frames')
    print('  Done!')


def main():
    parser = argparse.ArgumentParser(description='Video Relighting Pipeline')
    parser.add_argument('input', type=str, help='Input image or video path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--lighting', type=str, default='front', help='Lighting preset name')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--max-frames', type=int, default=10, help='Maximum frames to process (0 = all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model paths
    model_paths = get_model_paths()
    
    # Check if input is image or video
    is_image = args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if not (is_image or is_video):
        raise ValueError(f'Unsupported file format: {args.input}')
    
    use_gpu = not args.cpu
    
    # Load models
    print('Loading models...')
    modnet, _ = load_modnet(model_paths['modnet'], use_gpu=use_gpu)
    dsine_predictor, _ = load_dsine_model(model_paths['dsine'], use_gpu=use_gpu)
    dpr_model, dpr_device = load_dpr_model(model_paths['dpr'], use_gpu=use_gpu)
    print('All models loaded successfully!')
    
    # Process
    if is_image:
        process_image(
            args.input, args.output_dir, modnet, dsine_predictor, dpr_model, dpr_device,
            lighting_preset=args.lighting, use_gpu=use_gpu
        )
    else:
        process_video(
            args.input, args.output_dir, modnet, dsine_predictor, dpr_model, dpr_device,
            lighting_preset=args.lighting, use_gpu=use_gpu, max_frames=args.max_frames
        )
    
    print('Processing complete!')


if __name__ == '__main__':
    main()
