<!--Kidney CT Scan Classification Project - README.md-->
<div align="center">

<!--Badges and Shields -->
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.2.2-red.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-2.0+-purple.svg)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-VGG16-lightgrey.svg)](https://keras.io/api/applications/vgg/)

<!--Project Logo -->
<br />
<img src="https://user-images.githubusercontent.com/49691350/250971447-0e8c8f1c-5d6a-4c5a-8e8a-1a2b3c4d5e6f.gif" width="200" height="200" alt="KidneyScan AI Logo">

# 🏥 Kidney CT Scan Classification Project

### AI-Powered Medical Image Classification using Deep Learning

---

## 📋 Table of Contents

- [Overview](#overview)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🏗️ Project Architecture](#️-project-architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [💻 Usage](#-usage)
  - [Running the ML Pipeline](#running-the-ml-pipeline)
  - [Starting the Web Application](#starting-the-web-application)
  - [Making Predictions](#making-predictions)
  - [Training via API](#training-via-api)
- [🔌 API Documentation](#-api-documentation)
- [🐳 Docker Deployment](#-docker-deployment)
- [🔧 Configuration](#-configuration)
- [📊 Model Details](#-model-details)
- [🧪 Testing](#-testing)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📧 Contact](#-contact)

---

## 📖 Overview

**KidneyScan AI** is an end-to-end deep learning pipeline for classifying kidney CT scan images as either **"Tumor"** or **"Normal"**. The project leverages transfer learning with the VGG16 pre-trained model to achieve accurate medical image classification.

This production-ready application includes:
- 🔄 Automated ML pipeline orchestration with **DVC**
- 📈 Experiment tracking with **MLflow**
- 🌐 RESTful API with **Flask**
- 🎨 Modern, responsive web interface
- 🐳 Docker containerization support

> **⚠️ Disclaimer**: This model is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Medical Image Classification** | Classify kidney CT scans as Tumor or Normal using deep learning |
| 🧠 **Transfer Learning** | Utilizes VGG16 pre-trained on ImageNet for feature extraction |
| 📊 **Experiment Tracking** | Full MLflow integration for metrics, parameters, and model versioning |
| 🔄 **Pipeline Automation** | DVC-powered reproducible ML pipeline from data ingestion to evaluation |
| 🌐 **REST API** | Flask-based API for seamless integration with other applications |
| 💫 **Modern UI** | Beautiful, responsive web interface with drag-and-drop functionality |
| 🐳 **Containerization** | Docker support for easy deployment and scaling |
| ⚡ **Lazy Model Loading** | Efficient memory usage with on-demand model loading |

---

## 🛠️ Tech Stack

### 🤖 Machine Learning & Deep Learning
<div>

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

</div>

### 🌐 Web Development
<div>

![Flask](https://img.shields.io/badge/Flask-%23000000.svg?style=for-the-badge&logo=Flask&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-%1E.svg?style=for-the-badge&logo=javascript23F7DF&logoColor=black)

</div>

### 🔧 MLOps & DevOps
<div>

![MLflow](https://img.shields.io/badge/MLflow-%23D4AF37.svg?style=for-the-badge&logo=MLflow&logoColor=black)
![DVC](https://img.shields.io/badge/DVC-%2346ACC7.svg?style=for-the-badge&logo=DVC&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=for-the-badge&logo=Docker&logoColor=white)
![AWS](https://img.shields.io/badge/Amazon%20AWS-%23232F3E.svg?style=for-the-badge&logo=Amazon-AWS&logoColor=white)

</div>

### 📦 Python Libraries
- `python-box` - Configuration management
- `pyYAML` - YAML file handling
- `gdown` - Google Drive file downloads
- `tqdm` - Progress bars
- `joblib` - Model serialization

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KIDNEY CT SCAN CLASSIFIER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   DATA SOURCE   │     │   DATA SOURCE   │     │   DATA SOURCE   │       │
│  │  (Google Drive) │────▶│   (DVC Cache)    │────▶│  (Training)     │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│         │                                               │                   │
│         ▼                                               ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │                        DVC PIPELINE                           │         │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌───────┐ │         │
│  │  │  Stage 01   │─▶│   Stage 02   │─▶│  Stage 03  │─▶│ Stage │ │         │
│  │  │Data Ingestion│  │Prepare Base  │  │  Training  │  │   04  │ │         │
│  │  │              │  │    Model     │  │            │  │ Eval  │ │         │
│  │  └─────────────┘  └──────────────┘  └────────────┘  └───────┘ │         │
│  └──────────────────────────────────────────────────────────────┘         │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                        MLflow Tracking                           │        │
│  │  📊 Metrics │ 📈 Parameters │ 📦 Model Registry │ 🕒 Runs    │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    FLASK WEB APPLICATION                         │        │
│  │                                                                  │        │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │        │
│  │   │   /train    │  │  /predict   │  │     / (Home)            │ │        │
│  │   │  (API)      │  │   (API)     │  │   (Web Interface)       │ │        │
│  │   └─────────────┘  └─────────────┘  └─────────────────────────┘ │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                      FRONTEND (HTML/CSS/JS)                      │        │
│  │  🎨 Drag & Drop │ 📷 Image Preview │ 📊 Results Display        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Data Ingestion** (`stage_01_data_ingestion.py`)
   - Downloads dataset from Google Drive
   - Extracts and organizes CT scan images
   - Version control with DVC

2. **Prepare Base Model** (`stage_02_prepare_base_model.py`)
   - Loads VGG16 pre-trained on ImageNet
   - Removes top classification layers
   - Configures for transfer learning

3. **Model Training** (`stage_03_model_training.py`)
   - Trains custom classification head
   - Applies data augmentation
   - Logs parameters and metrics to MLflow
   - Saves trained model

4. **Model Evaluation** (`stage_04_model_evaluation.py`)
   - Evaluates model on test set
   - Calculates loss and accuracy
   - Logs metrics to MLflow
   - Saves evaluation scores

---

## 📁 Project Structure

```
kidney-prediction-deep-learning/
│
├── 📄 app.py                          # Flask web application entry point
├── 📄 main.py                         # Direct pipeline execution script
├── 📄 dvc.yaml                        # DVC pipeline definition
├── 📄 params.yaml                     # Hyperparameters and configuration
├── 📄 config/
│   └── config.yaml                    # Application configuration
│
├── 📄 requirements.txt                # Python dependencies
├── 📄 setup.py                        # Package setup configuration
├── 📄 Dockerfile                      # Docker container configuration
│
├── 📂 src/
│   └── cnnClassifer/
│       ├── 📂 components/              # ML pipeline components
│       │   ├── data_ingestion.py      # Data download & extraction
│       │   ├── prepare_base_model.py  # VGG16 base model setup
│       │   ├── model_training.py      # Model training logic
│       │   └── model_evaluation_mlflow.py  # Evaluation with MLflow
│       │
│       ├── 📂 pipeline/                # Pipeline stage executors
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_prepare_base_model.py
│       │   ├── stage_03_model_training.py
│       │   ├── stage_04_model_evaluation.py
│       │   └── prediction.py          # Prediction pipeline
│       │
│       ├── 📂 config/                  # Configuration management
│       │   └── configuration.py        # Config manager class
│       │
│       ├── 📂 entity/                   # Data entities
│       │   └── config_entity.py        # Configuration entities
│       │
│       ├── 📂 utils/                    # Utility functions
│       │   └── Common.py               # Common utilities
│       │
│       ├── 📂 constants/               # Constants
│       │   └── __init__.py
│       │
│       └── 📂 __init__.py              # Package initializer
│
├── 📂 model/
│   └── model.h5                        # Trained Keras model
│
├── 📂 templates/
│   └── index.html                      # Frontend web interface
│
├── 📂 research/
│   ├── 01_data_ingestion.ipynb         # Data exploration notebook
│   ├── 02_prepare_base_model.ipynb     # Base model notebook
│   ├── 03_model_training.ipynb         # Training notebook
│   ├── 04_model_evaluation_with_mlflow.ipynb  # Evaluation notebook
│   └── trials.ipynb                    # Experiment trials
│
├── 📂 artifacts/                       # ML pipeline outputs
│   ├── data_ingestion/                 # Downloaded dataset
│   ├── prepare_base_model/             # Base model files
│   └── training/                       # Trained model files
│
├── 📂 logs/                            # Application logs
│   └── (log files)
│
├── 📂 mlruns/                          # MLflow tracking data
│   └── 0/
│       └── meta.yaml
│
├── 📄 scores.json                      # Evaluation metrics
├── 📄 dvc.lock                         # DVC lock file
├── 📄 LICENSE                          # MIT License
└── 📄 .gitignore                       # Git ignore file
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Description |
|-------------|---------|-------------|
| 🐍 Python | 3.8+ | Programming language |
| 🐳 Docker | 20.10+ | Containerization (optional) |
| 💾 GPU | CUDA 11.8+ | For faster training (optional) |

### One-Line Installation & Run

```bash
# Clone the repository
git clone https://github.com/vams2krish/Kidney-Disease-Classification-MLflow-DVC.git
cd Kidney-Disease-Classification-MLflow-DVC

# Install dependencies
pip install -r requirements.txt

# Start the web application
python app.py
```

Then open your browser and navigate to: **http://localhost:8080**

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vams2krish/Kidney-Disease-Classification-MLflow-DVC.git
cd Kidney-Disease-Classification-MLflow-DVC
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow; import flask; import mlflow; print('All packages installed successfully!')"
```

---

## 💻 Usage

### Running the ML Pipeline

#### Option 1: Using DVC (Recommended)

```bash
# Run the complete ML pipeline
dvc repro
```

#### Option 2: Using main.py

```bash
# Run pipeline directly
python main.py
```

#### Option 3: Run Individual Stages

```bash
# Stage 1: Data Ingestion
python src/cnnClassifer/pipeline/stage_01_data_ingestion.py

# Stage 2: Prepare Base Model
python src/cnnClassifer/pipeline/stage_02_prepare_base_model.py

# Stage 3: Model Training
python src/cnnClassifer/pipeline/stage_03_model_training.py

# Stage 4: Model Evaluation
python src/cnnClassifer/pipeline/stage_04_model_evaluation.py
```

### Starting the Web Application

```bash
# Start Flask app (default port: 8080)
python app.py

# Or specify custom port
python app.py --port 5000
```

The web interface will be available at: **http://localhost:8080**

### Making Predictions

#### Using cURL

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'
```

#### Using Python

```python
import base64
import requests

# Read and encode image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make prediction
response = requests.post(
    "http://localhost:8080/predict",
    json={"image": image_data}
)

print(response.json())
```

#### Using the Web Interface

1. Open http://localhost:8080 in your browser
2. Drag and drop a CT scan image or click to browse
3. Click "Analyze CT Scan"
4. View the prediction results

### Training via API

```bash
# Trigger training pipeline via API
curl -X GET http://localhost:8080/train

# Or using Python
import requests
response = requests.get("http://localhost:8080/train")
print(response.text)  # "Training done successfully!"
```

---

## 🔌 API Documentation

### Base URL

```
http://localhost:8080
```

### Endpoints

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `GET` | `/` | Home page | - | HTML page |
| `GET` | `/train` | Trigger training | - | `String` |
| `POST` | `/predict` | Make prediction | `{"image": "<base64>"}` | `JSON` |

### Response Formats

#### Prediction Response

```json
[
  {
    "image": "Normal"
  }
]
```

Or with error:

```json
{
  "error": "Error message here"
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad Request - No image provided |
| `500` | Internal Server Error |

---

## 🐳 Docker Deployment

### Building the Docker Image

```bash
# Build the image
docker build -t kidney-scan-classifier:latest .
```

### Running the Container

```bash
# Run the container
docker run -d -p 8080:8080 --name kidney-classifier kidney-scan-classifier:latest

# View logs
docker logs -f kidney-classifier

# Stop the container
docker stop kidney-classifier
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  kidney-classifier:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./model:/app/model
      - ./artifacts:/app/artifacts
```

```bash
# Start with docker-compose
docker-compose up -d
```

### Deployment to Cloud

#### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t kidney-classifier .
docker tag kidney-classifier:latest <account>.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier:latest
```

#### Heroku
```bash
# Using Heroku Container Registry
heroku login
heroku create kidney-classifier
heroku container:push web -a kidney-classifier
heroku container:release web -a kidney-classifier
```

---

## 🔧 Configuration

### params.yaml

```yaml
# Model Configuration
IMAGE_SIZE: [224, 224, 3]        # VGG16 standard input size
BATCH_SIZE: 16                   # Training batch size
EPOCHS: 10                       # Number of training epochs
CLASSES: 2                       # Number of output classes
WEIGHTS: imagenet                # Pre-trained weights
LEARNING_RATE: 0.01              # Initial learning rate
INCLUDE_TOP: False               # Exclude top layers
AUGMENTATION: True               # Enable data augmentation
```

### config/config.yaml

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://drive.google.com/uc?id=<FILE_ID>
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | 1 | Unbuffered Python output |
| `LANG` | en_US.UTF-8 | Language setting |
| `LC_ALL` | en_US.UTF-8 | Locale setting |
| `MLFLOW_TRACKING_URI` | - | MLflow tracking server URI |

---

## 📊 Model Details

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   VGG16 Backbone                    │
│  (Pre-trained on ImageNet, weights frozen)         │
├─────────────────────────────────────────────────────┤
│  Input: 224 x 224 x 3 (RGB Image)                   │
│                                                     │
│  Block 1: Conv2D(64) × 2 → MaxPool                  │
│  Block 2: Conv2D(128) × 2 → MaxPool                 │
│  Block 3: Conv2D(256) × 3 → MaxPool                 │
│  Block 4: Conv2D(512) × 3 → MaxPool                 │
│  Block 5: Conv2D(512) × 3 → MaxPool                 │
│                                                     │
│  Output: 7 × 7 × 512                                │
├─────────────────────────────────────────────────────┤
│              Custom Classification Head             │
│  ┌─────────────────────────────────────────────┐    │
│  │  GlobalAveragePooling2D                      │    │
│  │  Dense(512, activation='relu')               │    │
│  │  Dropout(0.5)                                │    │
│  │  Dense(2, activation='softmax')              │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  Output: [Probability_Tumor, Probability_Normal]   │
└─────────────────────────────────────────────────────┘
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial Learning Rate | 0.01 |
| Loss Function | Categorical Crossentropy |
| Batch Size | 16 |
| Epochs | 10 |
| Image Size | 224 × 224 × 3 |
| Data Augmentation | Yes (rotation, flip, zoom) |

### Dataset

- **Source**: Kidney CT Scan Image Dataset
- **Classes**: 2 (Tumor, Normal)
- **Split**: Training, Validation, Test
- **Augmentation**: Random horizontal flip, rotation, zoom

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_prediction.py -v

# Run with coverage
pytest --cov=src tests/
```

### Manual Testing

```bash
# Test data ingestion
python -c "from cnnClassifer.components.data_ingestion import DataIngestion; print('DataIngestion imported successfully')"

# Test configuration
python -c "from cnnClassifer.config.configuration import ConfigurationManager; print('Config imported successfully')"

# Test prediction pipeline
python -c "from cnnClassifer.pipeline.prediction import PredictionPipeline; print('PredictionPipeline imported successfully')"
```

---

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Loss** | 22.63 |
| **Accuracy** | 48.2% |

> ⚠️ **Note**: The current model performance can be improved with:
> - More training data
> - Extended training epochs
> - Hyperparameter tuning
> - Advanced architectures (ResNet, EfficientNet)
> - Learning rate scheduling

### MLflow Tracking

To view experiment tracking:

```bash
# Start MLflow UI
mlflow ui

# Open browser at http://localhost:5000
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/AmazingFeature

# Make your changes
git commit -m 'Add some AmazingFeature'
git push origin feature/AmazingFeature

# Open a Pull Request
```

### Coding Standards

- Follow PEP 8 style guide
- Write docstrings for all functions
- Add type hints where applicable
- Include unit tests for new features

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 KidneyScan AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Keras](https://keras.io/) - High-level neural networks API
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [DVC](https://dvc.org/) - Data version control
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [VGG16](https://keras.io/api/applications/vgg/) - Pre-trained model
- [Google Drive](https://drive.google.com/) - Data hosting
- [Community Contributors](https://github.com/vams2krish/Kidney-Disease-Classification-MLflow-DVC/graphs/contributors)

---

## 📧 Contact

| Contact | Details |
|---------|---------|
| 👤 **Author** | Vamsi Krishna |
| 📧 **Email** | adam.vamshikrishna@gmail.com |
| 🔗 **GitHub** | [vams2krish](https://github.com/vams2krish) |
| 💼 **LinkedIn** | [Vamsi Krishna](https://linkedin.com/in/vamsikrishna) |

---

<div align="center">

### ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Made with ❤️ by [Vamsi Krishna](https://github.com/vams2krish)**

[![GitHub stars](https://img.shields.io/github/stars/vams2krish/Kidney-Disease-Classification-MLflow-DVC?style=social)](https://github.com/vams2krish/Kidney-Disease-Classification-MLflow-DVC/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/vams2krish/Kidney-Disease-Classification-MLflow-DVC?style=social)](https://github.com/vams2krish/Kidney-Disease-Classification-MLflow-DVC/network)
[![X Follow](https://img.shields.io/X/follow/VamsKrish_?style=social)](https://X.com/vamskrish_)

</div>

---

*Last updated: January 2024*

