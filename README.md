# Video Relighting Pipeline

A complete video relighting solution that combines MODNet for portrait matting, DSINE for surface normal estimation, and DPR (Deep Portrait Relighting) for neural relighting. This pipeline processes images and videos to generate relit outputs with customizable lighting conditions.

## Features

- **Portrait Matting**: Generate alpha mattes from portrait images/videos using MODNet
- **Image Segmentation**: Extract foreground subjects using the generated alpha mattes
- **Surface Normal Estimation**: Generate surface normals from segmented images using DSINE
- **Albedo Map Generation**: Physics-based albedo estimation using spherical harmonics lighting model
- **Neural Relighting**: Apply new lighting conditions using DPR (Deep Portrait Relighting) neural network
- **Guided Filter**: Advanced guided filter implementation for style transfer to preserve quality while applying relighting effects
- **Lighting Presets**: Pre-configured lighting presets (front, side, top, red, purple, etc.)
- **Image & Video Processing**: Supports both images and videos with automatic format detection
- **GPU Support**: Full GPU acceleration for all models

## Project Structure

```
Video Relighting/
├── main.py              # Main entry point
├── models/              # Model checkpoints (download separately)
│   ├── modnet_photographic_portrait_matting.ckpt
│   ├── dsine.pt
│   └── trained_model_03.t7
├── src/                 # Source code (self-contained)
│   ├── modnet.py        # MODNet matting
│   ├── dsine.py         # DSINE surface normals
│   ├── dpr.py           # DPR relighting
│   ├── albedo.py        # Albedo estimation
│   ├── lighting.py      # Lighting presets
│   ├── GuidedFilter/    # Guided filter implementation
│   │   ├── __init__.py  # Python implementation
│   │   └── *.cpp        # C++ reference implementation
│   ├── models/          # Model definitions
│   │   ├── modnet/      # MODNet architecture
│   │   ├── dsine/       # DSINE architecture
│   │   └── dpr/         # DPR architecture
│   ├── utils/           # Utility functions
│   │   └── dsine/       # DSINE utilities
│   └── data/            # Data files
│       └── dpr/         # DPR lighting presets
├── inputs/              # Input files
└── outputs/             # Output files
```

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for DSINE and DPR)
- PyTorch with CUDA support

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch** (based on your CUDA version):
   ```bash
   # For CUDA 11.8:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only (not recommended, very slow):
   pip install torch torchvision
   ```

4. **Download model checkpoints:**
   
   - **MODNet**: Download `modnet_photographic_portrait_matting.ckpt` from [Google Drive](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing)
   - **DSINE**: Download `dsine.pt` from [DSINE repository](https://github.com/baegwangbin/DSINE)
   - **DPR**: Download `trained_model_03.t7` from [DPR repository](https://github.com/zhhoper/DPR)
   
   Place all checkpoints in the `models/` folder:
   ```bash
   models/
   ├── modnet_photographic_portrait_matting.ckpt
   ├── dsine.pt
   └── trained_model_03.t7
   ```

## Usage

### Basic Usage

**Process an image:**
```bash
python main.py inputs/image.jpg --output-dir outputs --lighting front
```

**Process a video:**
```bash
python main.py inputs/video.mp4 --output-dir outputs --lighting purple
```

### Command Line Arguments

- `input`: Input image or video file path (required)
- `--output-dir`: Output directory (default: `outputs`)
- `--lighting`: Lighting preset name (default: `front`)
  - Available presets: `front`, `side-left`, `side-right`, `top`, `back`, `red`, `purple`, `blue`, `green`, `orange`, `yellow`, `warm`, `cool`, `cinematic`, etc.
- `--cpu`: Force CPU usage (not recommended, very slow)
- `--max-frames`: Maximum frames to process for videos (default: 10, set to 0 for all frames)

### Examples

**Process an image with red lighting:**
```bash
python main.py inputs/portrait.jpg --output-dir outputs --lighting red
```

**Process a video with purple lighting:**
```bash
python main.py inputs/video.mp4 --output-dir outputs --lighting purple
```

**Process all frames in a video:**
```bash
python main.py inputs/video.mp4 --output-dir outputs --lighting front --max-frames 0
```

**Force CPU usage (slow):**
```bash
python main.py inputs/image.jpg --output-dir outputs --lighting front --cpu
```

## Output Files

### Images
For each input image, the pipeline generates:
- `{basename}_matte.png`: Alpha matte (grayscale)
- `{basename}_segmented.png`: Segmented foreground
- `{basename}_normals.png`: Surface normals (RGB visualization)
- `{basename}_albedo.png`: Albedo map
- `{basename}_dpr_output.png`: Raw DPR relighting output
- `{basename}_relit.png`: Final relit output (guided filter + segmented average)

### Videos
For each input video, the pipeline generates:
- `output_matte.mp4`: Alpha matte video
- `output_segmented.mp4`: Segmented foreground video
- `output_normals.mp4`: Surface normals video
- `output_albedo.mp4`: Albedo map video
- `output_dpr.mp4`: Raw DPR relighting output
- `output_relit.mp4`: Final relit video (guided filter + segmented average)

## Pipeline Flow

1. **MODNet**: Generate alpha matte and segment foreground
2. **DSINE**: Generate surface normals from segmented image
3. **Albedo Estimation**: Estimate albedo map from segmented image, normals, and matte
4. **DPR**: Apply neural relighting using DPR model
5. **Guided Filter**: Transfer relighting style to segmented image using advanced guided filter
6. **Average**: Blend guided filter output with segmented image
7. **Output**: Save final relit result

## Lighting Presets

The pipeline includes pre-configured lighting presets:

- **Directional**: `front`, `side-left`, `side-right`, `top`, `back`
- **Colored**: `red`, `purple`, `blue`, `green`, `orange`, `yellow`
- **Atmospheric**: `warm`, `cool`, `cinematic`, `dramatic`, `soft`

Presets are loaded from `src/data/dpr/` directory. You can add custom presets by creating new `.txt` files with 9 SH coefficients.

## Guided Filter

The pipeline uses an advanced guided filter implementation (`src/GuidedFilter/`) that:
- Supports both grayscale and color (BGR) guidance images
- Uses efficient box filtering for fast computation
- Solves 3x3 linear systems for color guidance (per-pixel)
- Preserves edge details while transferring lighting style
- Based on the original Guided Filter algorithm

## Supported Formats

**Images:**
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

**Videos:**
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`

## Notes

- **GPU Required**: DSINE and DPR require CUDA-capable GPU for reasonable performance
- **Processing Time**: Video processing can be slow depending on resolution and number of frames
- **Memory**: Large videos may require significant GPU memory
- **Quality**: Best results with portrait images/videos with clear subject separation
- **Self-Contained**: All source code is in `src/` - no external dependencies on original repositories

## Troubleshooting

**Model not found:**
- Ensure all model checkpoints are in the `models/` folder with correct filenames
- Check that file paths are correct

**CUDA/GPU issues:**
- Verify PyTorch is installed with CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure GPU drivers are properly installed
- Use `--cpu` flag as fallback (very slow)

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version is 3.7+
- Restart Python kernel/runtime if using Jupyter/Colab

**Video codec issues:**
- Output videos use `mp4v` codec
- If playback issues occur, convert with ffmpeg:
  ```bash
  ffmpeg -i output_relit.mp4 -c:v libx264 -crf 23 output_relit_h264.mp4
  ```

## License

- **MODNet**: Apache License 2.0
- **DSINE**: See [DSINE repository](https://github.com/baegwangbin/DSINE) for license
- **DPR**: See [DPR repository](https://github.com/zhhoper/DPR) for license
- **Guided Filter**: Based on original Guided Filter algorithm

## Credits

- **MODNet**: [MODNet repository](https://github.com/ZHKKKe/MODNet)
- **DSINE**: [DSINE repository](https://github.com/baegwangbin/DSINE)
- **DPR**: [DPR repository](https://github.com/zhhoper/DPR)
- **Guided Filter**: Based on the Guided Filter algorithm by Kaiming He et al.
