# Notebooks

This directory contains Jupyter notebooks for testing and experimenting with image-to-3D pipelines.

## 00_end_to_end_smoketest.ipynb

A comprehensive end-to-end notebook that validates the environment and runs the complete image-to-3D pipeline using **Hunyuan3D-2**.

### Features

- ✅ Environment verification (GPU, CUDA, PyTorch)
- ✅ Image loading and preprocessing
- ✅ Quality filtering (blur detection, duplicate removal)
- ✅ Best image selection
- ✅ Hunyuan3D-2 inference (shape + optional texture)
- ✅ Mesh export (OBJ/GLB formats)
- ✅ Turntable video generation
- ℹ️ Nerfstudio integration notes (optional)

### Prerequisites

#### System Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum 8GB VRAM for shape generation
  - 12GB+ VRAM recommended for texture generation
- **CPU fallback**: Quality checks will run, but 3D generation will be skipped

#### Software Dependencies

Install required packages from the root directory:

```bash
pip install -r requirements.txt
```

Core dependencies include:
- `torch` (with CUDA support)
- `numpy`
- `opencv-python`
- `pillow`
- `matplotlib`
- `transformers`
- `huggingface-hub`
- `trimesh`
- `pyrender` (for turntable rendering)
- `imageio[ffmpeg]` (for video creation)

### Setup

#### 1. Prepare Your Data

Create a directory for your image case and add images:

```bash
# Create case directory
mkdir -p data/raw/dog_test_01

# Copy your images
cp /path/to/your/images/*.jpg data/raw/dog_test_01/
```

**Supported formats**: JPG, JPEG, PNG

**Image requirements**:
- Minimum resolution: 256x256 pixels
- Clear, sharp images (blur detection will filter out blurry images)
- Good lighting and exposure
- For single-image 3D: 1 high-quality hero image is enough
- For NeRF: 10-100+ images from different angles

#### 2. Configure the Notebook

Open the notebook and edit the **Configuration Cell**:

```python
# Case/experiment name
CASE_NAME = "dog_test_01"  # ← Change this to match your data folder

# Image processing
MAX_IMAGES = 50            # Maximum images to process
IMAGE_SIZE = 512           # Target size for processing
MIN_IMAGE_SIZE = 256       # Minimum acceptable size

# Quality thresholds
BLUR_THRESHOLD = 100.0     # Lower = more permissive
PERCEPTUAL_HASH_THRESHOLD = 5  # For duplicate detection

# Device
DEVICE = "cuda"            # or "cpu" (GPU strongly recommended)

# Model settings
ENABLE_TEXTURE = True      # Set False to save VRAM
```

#### 3. Run the Notebook

Launch Jupyter:

```bash
jupyter notebook notebooks/00_end_to_end_smoketest.ipynb
```

Or use VS Code's Jupyter extension.

Run all cells in order (`Cell > Run All`) or execute step-by-step.

### Output Structure

Results are saved to `outputs/<CASE_NAME>/`:

```
outputs/dog_test_01/
├── metadata.json              # Run metadata and stats
├── images/
│   ├── input_grid.png        # Grid of input images
│   └── hero_image.png        # Selected best image
├── meshes/
│   ├── dog_test_01_mesh.obj  # 3D mesh (Wavefront OBJ)
│   └── dog_test_01_mesh.glb  # 3D mesh (GLB/glTF)
├── renders/
│   └── frame_*.png           # Individual turntable frames
└── turntable.mp4             # Turntable animation video
```

### Workflow Steps

The notebook runs these steps automatically:

1. **Setup** - Verify Python, PyTorch, CUDA
2. **Configuration** - Set parameters for your case
3. **Load Images** - Find and load all images from input directory
4. **Quality Filtering** - Apply filters:
   - Size check (minimum resolution)
   - Blur detection (Laplacian variance)
   - Duplicate detection (perceptual hashing)
5. **Best Image Selection** - Choose hero image based on quality score
6. **Hunyuan3D-2 Inference** - Generate 3D shape (+ optional texture)
7. **Export** - Save mesh as OBJ/GLB
8. **Turntable Rendering** - Generate 360° orbit video
9. **Summary** - Display results and output locations

### Troubleshooting

#### No GPU / CUDA Not Available

**Symptom**: `CUDA available: False`

**Solutions**:
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check GPU drivers: `nvidia-smi`
- Verify CUDA toolkit installation
- For CPU-only: Notebook will run quality checks but skip 3D generation

#### Out of Memory (OOM) Error

**Symptom**: `CUDA out of memory` or similar

**Solutions**:
- Reduce `IMAGE_SIZE` (try 256 or 384)
- Set `ENABLE_TEXTURE = False` to skip texture generation
- Close other GPU-intensive applications
- Reduce batch size if applicable
- Use a GPU with more VRAM (12GB+ recommended)

#### No Images Found

**Symptom**: `No images found in data/raw/...`

**Solutions**:
- Check that `CASE_NAME` matches your folder name exactly
- Verify image files are in the correct directory
- Ensure images have supported extensions (`.jpg`, `.jpeg`, `.png`)
- Check file permissions

#### Model Download Fails

**Symptom**: Connection errors when loading `tencent/Hunyuan3D-2`

**Solutions**:
- Check internet connection
- Verify Hugging Face access (some models require authentication)
- Try: `huggingface-cli login`
- Check if model ID is correct and model is available
- Use HF mirrors if in restricted regions

#### All Images Filtered Out

**Symptom**: `No images passed quality filters!`

**Solutions**:
- Lower `BLUR_THRESHOLD` (e.g., to 50.0)
- Lower `MIN_IMAGE_SIZE` (e.g., to 128)
- Increase `PERCEPTUAL_HASH_THRESHOLD` for more lenient duplicate detection
- Check input image quality

#### Import Errors

**Symptom**: `ModuleNotFoundError: No module named '...'`

**Solutions**:
- Install missing package: `pip install <package-name>`
- Reinstall all requirements: `pip install -r requirements.txt --upgrade`
- Check Python environment is activated
- For PyRender issues on headless servers: `pip install pyrender osmesa`

#### Turntable Rendering Fails

**Symptom**: PyRender errors or blank frames

**Solutions**:
- Install PyRender properly: `pip install pyrender`
- For headless systems: Install OSMesa (`apt-get install libosmesa6-dev`)
- Check mesh was exported successfully
- Verify mesh has valid geometry (vertices and faces)

### Advanced Usage

#### Custom Quality Metrics

Modify the `compute_blur_score()` function to use different sharpness metrics:

```python
def compute_blur_score(image):
    # Alternative: Use FFT-based metric
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return magnitude[...].mean()
```

#### Batch Processing Multiple Cases

Create a wrapper script:

```python
cases = ["dog_test_01", "cat_test_01", "bird_test_01"]

for case in cases:
    # Update CASE_NAME
    # Run notebook programmatically using nbconvert
    os.system(f"jupyter nbconvert --execute --to notebook \
               --output {case}_results.ipynb \
               00_end_to_end_smoketest.ipynb")
```

#### Using NeRF Instead

For multi-view datasets, use **Nerfstudio**:

```bash
# Process images with COLMAP
ns-process-data images --data data/raw/dog_test_01 \
                        --output-dir data/processed/dog_test_01

# Train NeRF
ns-train nerfacto --data data/processed/dog_test_01

# Export mesh
ns-export poisson --load-config outputs/.../config.yml \
                   --output-dir outputs/dog_test_01/nerf_mesh
```

See the **Nerfstudio Notes** section in the notebook for more details.

### Performance Tips

1. **Image preprocessing**: Resize images before loading to save memory
2. **GPU memory**: Monitor with `nvidia-smi -l 1` during inference
3. **Batch processing**: Process multiple images if model supports it
4. **Caching**: Hugging Face models are cached after first download
5. **Checkpointing**: Save intermediate results to avoid re-running expensive steps

### Model Notes

#### Hunyuan3D-2

- **Source**: Tencent AI Lab
- **Type**: Single-image to 3D
- **Input**: RGB image
- **Output**: 3D mesh (textured or untextured)
- **VRAM**: 6-8GB for shape, 10-12GB with texture
- **Quality**: Good for hero shots, clear objects

**Limitations**:
- Single viewpoint (may hallucinate hidden sides)
- Best with centered objects on clean backgrounds
- Texture quality depends on input image resolution

#### Alternative Models

Consider these if Hunyuan3D-2 doesn't meet your needs:

- **Zero123**: Diffusion-based novel view synthesis → NeRF
- **Shap-E** (OpenAI): Text/image to 3D
- **Point-E** (OpenAI): Fast 3D point cloud generation
- **TripoSR**: Real-time single-image 3D reconstruction

### Contributing

To add new features to this notebook:

1. Add clear markdown headers for new sections
2. Include error handling and graceful degradation
3. Provide template code with clear TODOs
4. Update this README with new requirements/instructions
5. Test on both GPU and CPU configurations

### References

- [Hunyuan3D-2 Paper](https://arxiv.org/abs/...)
- [Nerfstudio Documentation](https://docs.nerf.studio/)
- [PyTorch3D](https://pytorch3d.org/)
- [Trimesh Documentation](https://trimsh.org/)

### License

See repository root for license information.
