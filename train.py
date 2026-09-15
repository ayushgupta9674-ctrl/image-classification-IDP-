from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo11n-cls.pt")

print("Starting training...")

model.train(
    data=r"C:\Users\AYUSH\.cache\kagglehub\datasets\puneet6060\intel-image-classification\versions\2\seg_train\seg_train",
    epochs=20,
    imgsz=224,
    batch=32,
    project="runs",
    name="image_classifier"
)

print("Training completed!")