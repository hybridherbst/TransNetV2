import onnxruntime
import numpy as np
import argparse

def verify_onnx(onnx_path):
    print(f"Loading {onnx_path}...")
    session = onnxruntime.InferenceSession(onnx_path)

    # Input: [Batch, Time, Height, Width, Channels]
    # Height=27, Width=48, Channels=3
    # Let's try a dynamic shape
    batch_size = 1
    time_steps = 100
    input_shape = (batch_size, time_steps, 27, 48, 3)
    dummy_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)

    input_name = session.get_inputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")

    print("Running inference...")
    outputs = session.run(None, {input_name: dummy_input})

    single_frame_pred = outputs[0]
    all_frame_pred = outputs[1]

    print(f"Output 'single_frame_pred' shape: {single_frame_pred.shape}")
    print(f"Output 'all_frame_pred' shape: {all_frame_pred.shape}")

    assert single_frame_pred.shape == (batch_size, time_steps, 1)
    assert all_frame_pred.shape == (batch_size, time_steps, 1)

    print("Verification successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="./transnetv2.onnx")
    args = parser.parse_args()

    verify_onnx(args.onnx)
