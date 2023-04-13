from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(data = "data_custom.yaml", batch=4, imgsz=640, epochs=10)