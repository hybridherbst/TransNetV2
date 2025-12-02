import onnx

model_path = "/Users/herbst/git/TransNetV2/web/public/transnetv2_fixed/model.onnx"
print(f"Loading model from {model_path}")
model = onnx.load(model_path)

found_pad = False
for node in model.graph.node:
    if node.op_type == "Pad":
        found_pad = True
        print(f"Found Pad node: {node.name}")
        for attr in node.attribute:
            if attr.name == "pads":
                print(f"  Pads: {attr.ints}")

    if node.op_type == "Conv":
        for attr in node.attribute:
            if attr.name == "pads":
                if list(attr.ints) == [0, 1, 1, 0, 1, 1]:
                    print(f"Found BAD Conv node: {node.name} with pads {attr.ints}")

if not found_pad:
    print("No Pad nodes found!")
