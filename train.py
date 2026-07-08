import torch
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        device=device,
        workers=0,
    )

if __name__ == "__main__":
    main()
