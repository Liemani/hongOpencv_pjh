from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on COCO8
results = model.train(data="mouse_training.yaml", epochs=100, imgsz=320)
