# ONNX Conversion for TransNetV2 with WebGPU Support

This document outlines the steps to convert the TransNetV2 PyTorch model to ONNX format with WebGPU support for the web application.

## Prerequisites

- Python 3.x with PyTorch 2.9.1
- ONNX Runtime (`pip install onnxruntime onnx`)
- Node.js and npm for the web application
- Chrome with WebGPU enabled

## Quick Start

1. **Export the ONNX model**:
   ```bash
   cd inference-pytorch
   /usr/local/bin/python3 export_onnx.py --weights transnetv2-pytorch-weights.pth --output ../web/public/transnetv2_fixed/onnx/model.onnx
   ```

2. **Run the web app**:
   ```bash
   cd web
   npm install
   npm run dev
   ```
   - Open `http://localhost:5173`
   - Select a video, set duration, click "Process"
   - Click on detected shot segments to play them

## Detailed Setup

### Model Export

1. **Navigate to the inference directory**:
   ```
   cd inference-pytorch
   ```

2. **Run the export script** (using specific Python path for PyTorch 2.9.1):
   ```
   /usr/local/bin/python3 export_onnx.py --weights transnetv2-pytorch-weights.pth --output ../web/public/transnetv2_fixed/onnx/model.onnx
   ```
   - Uses opset 12 for ONNX Runtime Web compatibility
   - Legacy TorchScript export (`dynamo=False`)
   - Dynamic batch and time dimensions
   - **Key modifications for WebGPU**:
     - Explicit `ConstantPad3d` modules instead of `Conv` padding attributes
     - Custom `Pool3dAs2d` module replacing `AvgPool3d` with 2D operations
     - Prevents graph optimization issues in WebGPU backend

3. **Verify the export**:
   ```
   /usr/local/bin/python3 -c "import onnx; model = onnx.load('../web/public/transnetv2_fixed/model.onnx'); onnx.checker.check_model(model); print('Model is valid')"
   ```

### Web Application Setup

1. **Install dependencies**:
   ```bash
   cd web
   npm install
   ```

2. **Configure Vite for SharedArrayBuffer** (already done in `vite.config.js`):
   - `Cross-Origin-Opener-Policy: same-origin`
   - `Cross-Origin-Embedder-Policy: require-corp`
   - WASM MIME type configuration

3. **Run development server**:
   ```bash
   npm run dev
   ```
   - Serves on `http://localhost:5173`
   - Model loads from `/transnetv2_fixed/model.onnx`

## Web App Features

- **Video Upload**: Select any video file
- **Duration Control**: Process first N seconds (default: 6)
- **Manual Processing**: Click "Process" button to start analysis
- **Shot Detection**: Automatic shot boundary detection using TransNetV2
- **Interactive Playback**: Click detected shots to play specific segments
- **Backend Selection**: WebGPU prioritized, automatic fallback to WASM

## Technical Details

### Model Architecture Changes

**For WebGPU Compatibility:**

1. **Padding Operations**: Replaced `Conv3d` padding with explicit `nn.ConstantPad3d` modules
2. **Pooling Operations**: Custom `Pool3dAs2d` class reshapes 3D pooling to 2D operations
3. **Graph Optimizations**: Disabled in `worker.js` to prevent Pad+Conv fusion

### Web App Architecture

- **Worker-based Processing**: Model inference runs in Web Worker
- **Transformers.js**: v3.7.6 with ONNX Runtime Web backend
- **Video Processing**: MediaBunny for frame extraction
- **Playback Handling**: Proper Promise-based video play/pause management

## Troubleshooting

### Model Loading Issues
- **Protobuf parsing failed**: Ensure opset 12, not 20
- **Backend not available**: Check WebGPU flags in Chrome
- **Session not initialized**: Wait for model to load completely

### Video Processing Issues
- **AbortError on play**: Fixed with proper Promise handling
- **Process button stuck**: Check console for worker errors
- **No shots detected**: Try different video or adjust threshold

### Performance Issues
- **Slow processing**: Reduce duration or use WebGPU
- **Memory errors**: Process shorter video segments
- **WebGPU fallback**: Check browser compatibility

## Development Notes

### Key Files Modified
- `inference-pytorch/transnetv2_pytorch.py`: Added WebGPU-compatible operations
- `inference-pytorch/export_onnx.py`: Opset 12, dynamo=False
- `web/src/worker.js`: Model loading, session management
- `web/src/main.js`: UI, video processing, playback handling
- `web/src/transnet_model.js`: ONNX session access
- `web/vite.config.js`: SharedArrayBuffer headers

### Version Compatibility
- **PyTorch**: 2.9.1 (specific installation path: `/usr/local/bin/python3`)
- **ONNX Opset**: 12 (not 20)
- **transformers.js**: 3.7.6
- **Node.js**: Latest LTS

### Future Development
- Model changes require re-export with same parameters
- Web app improvements can be made without model changes
- Test with different video formats and sizes

## Development Workflow

### Making Model Changes
1. Modify `inference-pytorch/transnetv2_pytorch.py`
2. Test changes locally with Python inference
3. Re-export ONNX model: `/usr/local/bin/python3 export_onnx.py --weights transnetv2-pytorch-weights.pth --output ../web/public/transnetv2_fixed/model.onnx`
4. Test in web app

### Making Web App Changes
1. Modify files in `web/src/`
2. Test with `npm run dev`
3. Build with `npm run build` for production

### Debugging Tips
- Use browser DevTools to inspect ONNX Runtime Web errors
- Check WebGPU tab in Chrome DevTools for backend issues
- Verify model loads correctly before processing videos
- Test with small video clips first</content>
<parameter name="filePath">/Users/herbst/git/TransNetV2/README_ONNX.md