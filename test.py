from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    weights = Path("models/best.pt")
    if not weights.exists():
        weight_files = sorted(
            Path("runs/detect").glob("train*/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        weights = weight_files[0] if weight_files else Path("runs/detect/train/weights/best.pt")
    sample = Path("dataset/images/val/2026_03_19_08_56_IMG_1884.JPG")
    device = 0 if torch.cuda.is_available() else "cpu"

    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")
    if not sample.exists():
        raise FileNotFoundError(f"Missing sample image: {sample}")

    model = YOLO(str(weights))
    print(f"using weights: {weights}")
    metrics = model.val(
        data="data.yaml",
        imgsz=640,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )

    results = model.predict(
        str(sample),
        imgsz=640,
        device=device,
        conf=0.25,
        save=True,
        project=str(Path.cwd() / "runs" / "detect"),
        name="check_predict",
        exist_ok=True,
        verbose=False,
    )

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall: {metrics.box.mr:.4f}")
    print(f"detections on sample: {len(results[0].boxes)}")
    print(f"prediction output: {results[0].save_dir}")


if __name__ == "__main__":
    main()
