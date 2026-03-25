import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from dotenv import load_dotenv

# Load Environment Variables (Kaggle credentials & MLflow URI)
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("MLOps Assignment 5 - Full CI/CD pipeline")

# IMPORTANT: Enable system metrics monitoring
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

# Candidate dataset paths for local and CI layouts.
DATASET_CANDIDATES = [
    "data/realwaste-main/RealWaste",
    "RealWaste",
]


def resolve_dataset_path() -> str:
    repo_root = Path(__file__).resolve().parent
    for relative_path in DATASET_CANDIDATES:
        candidate = repo_root / relative_path
        if candidate.is_dir():
            return str(candidate)

    checked = ", ".join(str((repo_root / p)) for p in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Dataset folder not found. Checked: "
        f"{checked}. Run 'dvc pull' and ensure one of these paths exists."
    )


def main():

    # Start MLflow tracking
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Started MLflow Run: {run_id}")

        # Resolve DVC-downloaded dataset location.
        data_path = resolve_dataset_path()
        print(f"Using DVC-downloaded dataset at: {data_path}")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        full_dataset = datasets.ImageFolder(data_path, transform=transform)

        # 80/20 train-test split
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Build Model (EfficientNet B0)
        print("Initializing EfficientNet...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modify final layer for our 9 classes
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(full_dataset.classes))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 5
        mlflow.log_param("epochs", num_epochs)

        # Train Model
        print(f"Training on {device} for {num_epochs} epoch(s)...")
        accuracy = 0.0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_train_loss = running_loss / len(train_dataset)

            # Evaluate Model each epoch
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_accuracy = correct / total if total > 0 else 0.0
            accuracy = epoch_accuracy
            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"train_loss: {epoch_train_loss:.4f} - "
                f"val_accuracy: {epoch_accuracy:.4f}"
            )
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch + 1)
            mlflow.log_metric("val_accuracy", epoch_accuracy, step=epoch + 1)

        print(f"Final Accuracy: {accuracy:.4f}")

        # Log to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.pytorch.log_model(model, "model")

        #  Write model_info.txt
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        print(f"Wrote Run ID to model_info.txt: {run_id}")

        # log it as an MLflow artifact
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_info_path = os.path.join(tmp_dir, "model_info.txt")
            with open(model_info_path, "w") as f:
                f.write(run_id)
            mlflow.log_artifact(model_info_path)
        print("Saved Run ID artifact to MLflow")


if __name__ == "__main__":
    main()
