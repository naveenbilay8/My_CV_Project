from rfdetr import RFDETRNano

if __name__ == "__main__":
    # Initialize model: Small variant pretrained on COCO
    model = RFDETRNano(pretrained=True, num_classes=2)  # 1 class: emergency_vehicle

    # Training parameters
    train_params = {
        "dataset_dir": "data3",  # Root folder of your dataset
        "train_json": "train/_annotations.coco.json",
        "val_json": "valid/_annotations.coco.json",
        "epochs": 50,
        "batch_size": 4,
        "output_dir": "checkpoints3/"
    }

    # Train
    model.train(
        dataset_dir=train_params["dataset_dir"],
        train_json=train_params["train_json"],
        val_json=train_params["val_json"],
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        output_dir=train_params["output_dir"]
    )
