
# MLflow Medical Project Documentation

## Overview

This project implements an end-to-end machine learning pipeline for classifying kidney CT scan images using CNNs. It utilizes MLflow for experiment tracking and model management, DVC for data versioning and pipeline orchestration, and integrates with DagsHub for remote storage and collaboration.

## Key Components

### Data Ingestion
- Downloads and extracts kidney CT scan image dataset
- Implemented in `src/cnnClassifer/components/data_ingestion.py`

### Base Model Preparation  
- Loads VGG16 model and prepares it for transfer learning
- Implemented in `src/cnnClassifer/components/prepare_base_model.py`

### Model Training
- Trains the model on the kidney CT scan dataset
- Implements data augmentation and transfer learning
- Implemented in `src/cnnClassifer/components/model_training.py`

### Model Evaluation
- Evaluates trained model performance
- Logs metrics and model to MLflow
- Implemented in `src/cnnClassifer/components/model_evaluation_mlflow.py`

### Pipeline Orchestration
- Manages overall workflow using DVC
- Pipeline defined in `dvc.yaml`

### Configuration  
- Configurations managed in `config/config.yaml` and `params.yaml`

### Utilities
- Helper functions in `src/cnnClassifer/utils/Common.py`

### Web App
- Flask app for serving predictions (`app.py`)
- HTML template for frontend (`templates/index.html`)

## Inputs

- Kidney CT scan images dataset
- Configuration parameters in YAML files

## Outputs  

- Trained CNN model for kidney CT scan classification
- Performance metrics logged to MLflow
- Prediction capability via web interface

## Usage

1. Set up environment and install dependencies
2. Run DVC pipeline: `dvc repro`  
3. Launch web app: `python app.py`

## Key Files

- `main.py`: Entry point to run full pipeline
- `app.py`: Flask web application
- `dvc.yaml`: Defines DVC pipeline stages
- `src/cnnClassifer/pipeline/*`: Individual pipeline stage implementations
