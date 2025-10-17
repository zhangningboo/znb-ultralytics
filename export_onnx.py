from ultralytics import YOLO

model = YOLO("./yolov8n_coco_v8.2.0.pt")

options = {
    "format": "onnx",
    "imgsz": (640, 640), # (640, 480)  # (H, W)
    "optimize": False,
    "dynamic": False,
    "simplify": False,
    "opset": 18,
    "nms": False,
}
# Export the model
model.export(**options)