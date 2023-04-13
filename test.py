from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source="D:\\Desktop\\yolo\\test", show=True, conf=0.6, iou=0.5, line_thickness=1, save_txt=True) 


