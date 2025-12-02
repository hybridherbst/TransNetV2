import { env, Tensor } from '@huggingface/transformers';
import { TransNetV2Model } from './transnet_model.js';

// Configure environment
env.allowLocalModels = true; 
env.useBrowserCache = false;
env.allowRemoteModels = true;

const PER_DEVICE_CONFIG = {
    webgpu: {
        dtype: 'fp32',
        device: 'webgpu',
    },
    wasm: {
        dtype: 'fp32',
        device: 'wasm',
    },
};

class ModelSingleton {
    static instance = null;
    static device = null;

    static async getInstance(device = 'webgpu') {
        if (!this.instance || this.device !== device) {
            const config = PER_DEVICE_CONFIG[device];
            if (!config) throw new Error(`Unsupported device: ${device}`);
            
            console.log(`[Worker] Loading model with device: ${device}`);
            
            try {
                // We use /transnetv2_fixed path which contains model.onnx and config.json
                const modelPath = '/transnetv2_fixed';
                console.log(`[Worker] Loading model from: ${modelPath}`);
                
                this.model = await TransNetV2Model.from_pretrained(modelPath, {
                    quantized: false,
                    dtype: config.dtype,
                    device: config.device,
                    local_files_only: false,
                    session_options: {
                        executionProviders: ['webgpu', 'wasm'],
                        graphOptimizationLevel: 'disabled', // Prevent fusion of Pad+Conv that breaks WebGPU
                    },
                });
                this.instance = this.model; // Set the instance
                this.device = device;
                console.log(`[Worker] Model loaded successfully with ${device}`);
            } catch (e) {
                console.warn(`[Worker] Failed to load with ${device}`, e);
                if (device === 'webgpu') {
                    console.log('[Worker] Falling back to wasm');
                    return this.getInstance('wasm');
                }
                throw e;
            }
        }
        return this.instance;
    }
}

self.onmessage = async function(e) {
    const { type, data } = e.data;

    if (type === 'init') {
        try {
            // Try WebGPU first (default in getInstance)
            await ModelSingleton.getInstance('webgpu');
            self.postMessage({ type: 'init-done', provider: ModelSingleton.device });
        } catch (error) {
            console.error('[Worker] Init error:', error);
            self.postMessage({ type: 'error', error: error.message });
        }
    } else if (type === 'process') {
        try {
            const model = await ModelSingleton.getInstance(ModelSingleton.device);
            if (!model) {
                throw new Error('Session not initialized');
            }

            const { frames, height, width } = data;

            // Sliding window parameters
            const paddedFrames = [];

            // Pad start with 25 copies of first frame
            for (let i = 0; i < 25; i++) paddedFrames.push(frames[0]);
            paddedFrames.push(...frames);

            // Pad end
            const padEnd = 25 + 50 - (frames.length % 50 === 0 ? 50 : frames.length % 50);
            for (let i = 0; i < padEnd; i++) paddedFrames.push(frames[frames.length - 1]);

            const singleFramePreds = [];

            // Process in sliding windows
            for (let ptr = 0; ptr + 100 <= paddedFrames.length; ptr += 50) {
                const windowFrames = paddedFrames.slice(ptr, ptr + 100);

                // Create tensor: [Batch=1, Time=100, H=27, W=48, C=3]
                const inputData = new Uint8Array(1 * 100 * height * width * 3);
                let offset = 0;

                for (const frame of windowFrames) {
                    const frameData = frame.data; // RGBA from ImageData
                    for (let i = 0; i < frameData.length; i += 4) {
                        inputData[offset++] = frameData[i];     // R
                        inputData[offset++] = frameData[i + 1]; // G
                        inputData[offset++] = frameData[i + 2]; // B
                    }
                }

                // Create Tensor
                const inputTensor = new Tensor('uint8', inputData, [1, 100, height, width, 3]);

                // Run inference
                // TransNetV2Model._call calls session.run(inputs)
                // inputs should be { input: tensor }
                const results = await model({ input: inputTensor });

                // The output name is 'single_frame_pred'
                const single = results.single_frame_pred.data;

                // Extract middle 50 frames (indices 25 to 75)
                for (let i = 25; i < 75; i++) {
                    singleFramePreds.push(single[i]);
                }

                // Send progress
                self.postMessage({ type: 'progress', progress: Math.min(singleFramePreds.length, frames.length), total: frames.length });
            }

            // Trim to actual length
            const finalPreds = singleFramePreds.slice(0, frames.length);

            self.postMessage({ type: 'done', predictions: finalPreds });
        } catch (error) {
            console.error('[Worker] Process error:', error);
            self.postMessage({ type: 'error', error: error.message });
        }
    }
};
