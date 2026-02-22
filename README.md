# ML Classification System - MLOps Best Practices

A complete, production-ready machine learning classification system demonstrating modern MLOps practices from data versioning to deployment.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Tests](https://img.shields.io/badge/Tests-pytest-orange)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)

## 🎯 Overview

This project implements a complete ML classification system following **all MLOps best practices**:

### Key Components:
1. **Data Layer**: PostgreSQL with Neon (serverless)
2. **ML Pipeline**: Scikit-learn with hyperparameter tuning
3. **Experiment Tracking**: Weights & Biases (W&B)
4. **API**: FastAPI with Prometheus metrics
5. **Frontend**: Streamlit interactive UI
6. **Deployment**: Docker + Render
7. **Monitoring**: Prometheus + Grafana
8. **CI/CD**: GitHub Actions workflows

## ✨ Features

- ✅ PostgreSQL database integration
- ✅ Hyperparameter tuning (Grid/Random/Bayesian Search)
- ✅ Weights & Biases experiment tracking
- ✅ FastAPI with Pydantic validation
- ✅ Streamlit frontend
- ✅ Docker containerization
- ✅ Prometheus + Grafana monitoring
- ✅ 15+ comprehensive pytest tests
- ✅ Flake8 + Pylint code quality
- ✅ GitHub Actions CI/CD
- ✅ Render deployment support

## ⚡ Quick Start

### Without Docker
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model
python scripts/train.py

# Terminal 1: API
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
streamlit run frontend/app.py
```

### With Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up -d
```

Access:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:8501
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 📁 Project Structure

```
ml-classification-system/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── ml_pipeline.py          # ML training & inference
│   ├── data_manager.py         # Database operations
│   └── __init__.py
├── frontend/
│   └── app.py                  # Streamlit application
├── tests/
│   ├── test_all.py             # 15+ comprehensive tests
│   └── __init__.py
├── scripts/
│   └── train.py                # Training script
├── data/
│   └── dataset.csv             # Training dataset
├── model_artifacts/
│   ├── model.joblib
│   └── preprocessor.joblib
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── monitoring/
│   └── prometheus.yml
├── .github/workflows/
│   ├── backend.yml             # Backend CI/CD
│   └── frontend.yml            # Frontend CI/CD
├── requirements.txt
├── .env.example
└── README.md
```

## 📚 API Documentation

### Key Endpoints

**Health Check**
```bash
GET /health
```

**Single Prediction**
```bash
POST /predict
{
  "feature1": 1.2,
  "feature2": 3.4,
  "feature3": 5.6,
  "feature4": 2.1,
  "feature5": 1.5
}
```

**Batch Predictions**
```bash
POST /predict-batch
[
  {"feature1": 1.2, ...},
  {"feature1": 2.3, ...}
]
```

**Metrics**
```bash
GET /metrics  # Prometheus metrics
GET /info     # Model information
```

## 🧪 Testing

```bash
# All tests
pytest tests/ -v --cov=backend

# With coverage report
pytest tests/ --cov=backend --cov-report=html

# Specific test
pytest tests/test_all.py::TestMLPipeline::test_prediction -v
```

**Test Coverage**: 15+ tests covering:
- Data loading and validation
- ML pipeline training and evaluation
- API endpoints and error handling
- Batch predictions
- Integration tests

## 🚀 Deployment

### Render Deployment
1. Push to GitHub
2. Create Render services for backend and frontend
3. Set environment variables
4. Deploy

### Docker Local
```bash
docker-compose -f docker/docker-compose.yml up --build
```

## 📊 Monitoring

### Prometheus
- Request counts per endpoint
- Prediction latency
- Model accuracy metrics
- System health

### Grafana
- Pre-built dashboards
- Custom metrics visualization
- Alerts configuration

## 🎯 Business Value

This system enables:
- **Automated Classification**: ML-powered predictions
- **Real-time Inference**: Sub-100ms predictions
- **Scalability**: Handles batch and single requests
- **Transparency**: Full experiment tracking with W&B
- **Reliability**: Comprehensive testing and CI/CD
- **Monitoring**: Production-grade observability

## 📝 W&B Integration

Experiment tracking automatically logs:
- Hyperparameters
- Performance metrics (Accuracy, F1, ROC-AUC, Precision, Recall)
- Model artifacts
- Cross-validation results
- Training curves

## 🔐 Environment Setup

```bash
cp .env.example .env
# Edit .env with:
DATABASE_URL=postgresql://user:password@host/db
WANDB_API_KEY=your_api_key
MODEL_PATH=model_artifacts/model.joblib
API_URL=http://localhost:8000
```

## 📈 Code Quality

```bash
# Format
black backend frontend tests

# Lint
flake8 backend frontend tests

# Type checking
mypy backend/**/*.py
```

All code follows PEP8 standards with Flake8 and Pylint checks in CI/CD.

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Docker Reference](https://docs.docker.com/)
- [Weights & Biases](https://wandb.ai/)

## 📄 License

MIT License

---

**Built with ❤️ for MLOps Excellence** | v1.0.0
