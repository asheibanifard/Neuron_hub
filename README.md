# 🧠 Neuron Hub - 3D Gaussian Splatting Neuron Viewer

Interactive web-based 3D volume renderer for reconstructed neuron microscopy data using 3D Gaussian Splatting.

## 🔗 Live Demo

**[View Online →](https://asheibanifard.github.io/Neuron_hub/)**

## 📸 Features

- **Real-time 3D volume rendering** using WebGL2 ray-casting
- **MIP (Maximum Intensity Projection)** mode for clear neuron visualization
- **Alpha blending** mode for semi-transparent volume rendering
- **Interactive controls**: mouse drag to rotate, scroll to zoom
- **Adjustable brightness and threshold** for optimal visualization
- **Auto-rotate** for hands-free viewing
- **Mobile touch support**

## 🔬 Data

The volume data represents a neuron reconstructed from microscopy images using **3D Gaussian Splatting** with skeleton constraints:

- **Original volume**: 100 × 650 × 820 voxels (Z × Y × X)
- **Reconstruction**: 11,477 weighted 3D Gaussians
- **Quality**: 36.25 dB PSNR
- **Skeleton source**: SWC neuron morphology file

## 🛠️ Technical Details

### Rendering Pipeline

1. **Volume data**: Raw 8-bit grayscale (53 MB)
2. **WebGL2 3D texture**: Hardware-accelerated ray-casting
3. **GLSL fragment shader**: Real-time MIP/alpha compositing
4. **512 ray steps** per pixel for high quality

### Controls

| Control | Action |
|---------|--------|
| 🖱️ Drag | Rotate view |
| 🔄 Scroll | Zoom in/out |
| ☀️ Brightness | Adjust intensity |
| 🎚️ Threshold | Filter background |
| MIP/Alpha | Switch render mode |
| Auto-Rotate | Toggle rotation |

## 📁 Files

```
├── index.html        # WebGL volume viewer
├── volume.raw        # 3D volume data (53 MB)
├── volume_dims.json  # Volume dimensions & aspect ratio
└── README.md
```

## 🚀 Local Development

```bash
# Clone the repository
git clone https://github.com/asheibanifard/Neuron_hub.git
cd Neuron_hub

# Start a local server
python -m http.server 8000

# Open in browser
# http://localhost:8000
```

## 📖 Method

This visualization is part of research on applying **3D Gaussian Splatting** to neuron reconstruction from microscopy data. The method uses:

1. **SWC skeleton file** to initialize Gaussian positions along neuron branches
2. **Differentiable volume rendering** to optimize Gaussian parameters
3. **Skeleton-constrained optimization** to maintain neuron topology

## 📜 License

MIT License

## 👤 Author

**Armin Sheibanifard**  
Bournemouth University

---

*Built with WebGL2 and 3D Gaussian Splatting*
