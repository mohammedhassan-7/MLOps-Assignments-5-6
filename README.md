# MLflow CI/CD Pipeline Assignment


**Dataset:** [RealWaste on Kaggle](https://www.kaggle.com/datasets/joebeachcapital/realwaste/data)

This repository contains a complete CI/CD pipeline for training, validating, and (mock) deploying a PyTorch image classifier using MLflow, DVC, Docker, and GitHub Actions.

## Features
- **MLflow Tracking**: Logs metrics and models to a remote MLflow server
- **DVC**: Manages and pulls the dataset from Google Drive (via DVC remote)
- **GitHub Actions**: Two-stage workflow (validate, deploy) with linting, artifact passing, and threshold checks
- **Docker**: Minimal Dockerfile for mock deployment

## Usage

### 1. Prerequisites
- Python 3.10+
- MLflow tracking server

### 2. Environment Setup
```sh
pip install -r requirements.txt
```


### 3. Training Locally
```sh
export MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
dvc pull  # Download the dataset from Google Drive
python train.py
```



### 4. Running the Pipeline on GitHub Actions
- Push to `main` branch triggers the workflow
- Set the following repository secrets:
  - `MLFLOW_TRACKING_URI`
  - `GDRIVE_CREDENTIALS_JSON` (contents of your Google service account JSON for DVC Google Drive remote)

### 5. Files
- `train.py` — Trains and logs the model
- `check_threshold.py` — Checks MLflow accuracy for deployment gating
- `.github/workflows/pipeline.yml` — CI/CD workflow
- `Dockerfile` — Minimal mock deployment

---
