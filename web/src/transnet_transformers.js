
import { PreTrainedModel, Tensor, AutoModel, env } from '@huggingface/transformers';

// Configure environment to allow loading local models (via fetch)
// In a browser environment, this usually means fetching from relative URLs.
env.allowLocalModels = false; // Browser uses fetch, not FS
env.useBrowserCache = false; // Optional: disable cache if developing

/**
 * Custom TransNetV2 Model class extending PreTrainedModel.
 * This allows us to use the transformers.js API (loading, device management)
 * with a custom ONNX model.
 */
export class TransNetV2Model extends PreTrainedModel {
    constructor(config, session) {
        super(config, session);
    }

    /**
     * The main forward pass of the model.
     * @param {Object} inputs - Dictionary of input tensors.
     * @returns {Promise<Object>} - Dictionary of output tensors.
     */
    async _call(inputs) {
        // The input name for TransNetV2 is 'input'
        // Ensure the inputs object has the correct key
        if (!inputs.input && inputs.pixel_values) {
             // Map standard transformers image input to our model's input if needed
             inputs.input = inputs.pixel_values;
             delete inputs.pixel_values;
        }
        
        // Run the ONNX session
        return await this.session.run(inputs);
    }
}

/**
 * Configuration for different devices/backends.
 * Adapted for a single model (TransNetV2).
 */
export const PER_DEVICE_CONFIG = {
    webgpu: {
        dtype: 'fp32', // TransNetV2 is likely fp32. Use 'q4' etc if quantized.
        device: 'webgpu',
    },
    wasm: {
        dtype: 'fp32', // or 'q8' if available
        device: 'wasm',
    },
};

/**
 * Loads the TransNetV2 model using transformers.js
 * @param {string} device - 'webgpu' or 'wasm'
 * @returns {Promise<TransNetV2Model>}
 */
export async function loadTransNetV2(device = 'webgpu') {
    const config = PER_DEVICE_CONFIG[device];
    if (!config) throw new Error(`Unsupported device: ${device}`);

    console.log(`Loading TransNetV2 on ${config.device} with dtype ${config.dtype}...`);

    // We use the custom class's from_pretrained method.
    // The path '/transnetv2' should point to the folder containing config.json and model.onnx
    // in the public directory.
    const model = await TransNetV2Model.from_pretrained('/transnetv2', {
        dtype: config.dtype,
        device: config.device,
        // If the model file is named 'model.onnx', it's found automatically.
        // If it has a different name, specify it:
        // file: 'transnetv2.onnx', 
    });

    return model;
}

/**
 * Helper to create the input tensor.
 * @param {Uint8Array} data - Raw video data [Batch, Time, 27, 48, 3]
 * @param {number[]} dims - Dimensions [Batch, Time, 27, 48, 3]
 * @returns {Tensor}
 */
export function createInputTensor(data, dims) {
    return new Tensor('uint8', data, dims);
}
