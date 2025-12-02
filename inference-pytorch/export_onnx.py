import torch
import transnetv2_pytorch
import argparse

def export_onnx(weights_path, output_path):
    model = transnetv2_pytorch.TransNetV2()
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Dummy input: [Batch, Time, Height, Width, Channels]
    # Height=27, Width=48, Channels=3
    dummy_input = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)

    # Dynamic axes: Batch and Time can vary
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'time'},
        'single_frame_pred': {0: 'batch_size', 1: 'time'},
        'all_frame_pred': {0: 'batch_size', 1: 'time'} # This output is a dict in forward, need to handle it
    }

    # Wrapper to handle dict output for ONNX
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            single, many = self.model(x)
            return single, many["many_hot"]

    wrapped_model = Wrapper(model)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['single_frame_pred', 'all_frame_pred'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'time'},
            'single_frame_pred': {0: 'batch_size', 1: 'time'},
            'all_frame_pred': {0: 'batch_size', 1: 'time'}
        },
        opset_version=12,
        dynamo=False
    )
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./transnetv2-pytorch-weights.pth")
    parser.add_argument("--output", type=str, default="./transnetv2.onnx")
    args = parser.parse_args()

    export_onnx(args.weights, args.output)
