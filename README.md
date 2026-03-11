
# Kidney CT Scan Classification Project

## Overview

This project implements an end-to-end deep learning pipeline for classifying kidney CT scan images as either "Tumor" or "Normal" using Convolutional Neural Networks (CNNs). It utilizes MLflow for experiment tracking and model management, DVC for data versioning and pipeline orchestration, and Flask for web-based prediction serving.

## Key Features

- **CNN-based Classification**: Uses VGG16 pre-trained model with transfer learning for kidney CT scan classification
- **MLflow Integration**: Tracks experiments, metrics, and model artifacts
- **DVC Pipeline**: Orchestrates the complete ML pipeline from data ingestion to model evaluation
- **Flask Web App**: Provides REST API for making predictions on new CT scan images

## Project Structure

```
├── app.py                          # Flask web application for predictions
├── main.py                         # Entry point to run full pipeline
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Configuration parameters
├── config/config.yaml              # Application configuration
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── model/
│   └── model.h5                    # Trained model (after training)
├── src/cnnClassifer/
│   ├── components/
│   │   ├── data_ingestion.py      # Data download and extraction
│   │   ├── prepare_base_model.py  # VGG16 base model preparation
│   │   ├── model_training.py      # Model training with augmentation
│   │   └── model_evaluation_mlflow.py  # Model evaluation with MLflow
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py          # Prediction pipeline
│   ├── config/
│   │   └── configuration.py       # Configuration management
│   ├── utils/
│   │   └── Common.py              # Utility functions
│   └── entity/
│       └── config_entity.py       # Configuration entities
├── templates/
│   └── index.html                  # Frontend HTML template
└── logs/                           # Application logs
```

## Key Components

### Data Ingestion
- Downloads kidney CT scan image dataset from Google Drive
- Extracts and organizes data for training
- Implemented in `src/cnnClassifer/components/data_ingestion.py`

### Base Model Preparation
- Loads VGG16 model pre-trained on ImageNet
- Removes top layers for transfer learning
- Implemented in `src/cnnClassifer/components/prepare_base_model.py`

### Model Training
- Trains the model on kidney CT scan dataset
- Implements data augmentation for better generalization
- Uses transfer learning with custom classification head
- Implemented in `src/cnnClassifer/components/model_training.py`

### Model Evaluation
- Evaluates trained model performance
- Logs metrics (loss, accuracy) to MLflow
- Saves evaluation scores to `scores.json`
- Implemented in `src/cnnClassifer/components/model_evaluation_mlflow.py`

### Prediction Pipeline
- Lazy loads the trained model (only when needed)
- Processes input images and returns classification results
- Returns "Tumor" or "Normal" prediction
- Implemented in `src/cnnClassifer/pipeline/prediction.py`

### Web Application
- Flask app for serving predictions via REST API
- Endpoints: `/` (home), `/train` (trigger training), `/predict` (make prediction)
- HTML frontend in `templates/index.html`

## Configuration

### params.yaml
- `IMAGE_SIZE`: [224, 224, 3] (VGG16 standard input size)
- `BATCH_SIZE`: 16
- `EPOCHS`: 10
- `CLASSES`: 2 (Tumor, Normal)
- `LEARNING_RATE`: 0.01
- `WEIGHTS`: imagenet
- `INCLUDE_TOP`: False
- `AUGMENTATION`: True

### config/config.yaml
- Data ingestion settings (source URL, paths)
- Base model configuration
- Training parameters

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Run full ML pipeline with DVC
dvc repro

# Or run via main.py
python main.py
```

### Starting the Web Application
```bash
# Start Flask app on port 8080
python app.py
```

### Making Predictions
```bash
# Send POST request with base64 encoded image
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'
```

## Model Output

The model classifies kidney CT scans into two categories:
- **Tumor**: Indicates presence of tumor in the CT scan
- **Normal**: Indicates healthy kidney (no tumor)

## Dependencies

- TensorFlow 2.12.0
- Flask & Flask-CORS
- MLflow 2.2.2
- DVC
- Python-Box
- PyYAML
- NumPy, Matplotlib, Seaborn
- Google Drive download utility (gdown)

## License

See LICENSE file for details.

