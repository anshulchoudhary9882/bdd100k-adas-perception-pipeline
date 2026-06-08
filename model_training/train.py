from ultralytics import YOLO
import torch

def main():

    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Load pretrained YOLO11m
    model = YOLO("yolo11m.pt")

    # Train
    results = model.train(
        data="bdd_yolo/bdd_yolo.yaml",

        # Training
        epochs=1,
        batch=8,
        imgsz=480,

        # Hardware
        device=0,
        workers=6,

        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,

        freeze=15,         # Freeze backbone for trasnfer learning

        # Save
        project="runs/bdd100k",
        name="yolo11m_bdd",
        exist_ok=True,

        # Validation
        val=True,
        plots=True,

        # Reproducibility
        seed=42
    )

    print("Training Complete")


if __name__ == "__main__":
    main()