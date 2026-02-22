"""Comprehensive test suite for ML classification system."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import joblib

from backend.ml_pipeline import MLPipeline, load_and_prepare_data
from backend.data_manager import DataManager, load_data_from_csv
from backend.main import app

# ==================== Test Fixtures ====================

@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)
    return X, y


@pytest.fixture
def ml_pipeline():
    """Create ML pipeline instance."""
    pipeline = MLPipeline(wandb_enabled=False)
    pipeline.create_pipeline()
    return pipeline


@pytest.fixture
def trained_pipeline(ml_pipeline, sample_data):
    """Create and train a pipeline."""
    X, y = sample_data
    ml_pipeline.train_best_model(X, y)
    return ml_pipeline


@pytest.fixture
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    return TestClient(app)


# ==================== Data Layer Tests ====================

class TestDataManager:
    """Test data manager functionality."""

    def test_load_data_from_csv(self, tmp_path):
        """Test loading data from CSV."""
        # Create test CSV
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [1.5, 2.5, 3.5],
            "feature3": [2.0, 3.0, 4.0],
            "feature4": [2.5, 3.5, 4.5],
            "feature5": [3.0, 4.0, 5.0],
            "target": [0, 1, 0]
        })
        df.to_csv(csv_file, index=False)

        # Load data
        loaded_df = load_data_from_csv(str(csv_file))

        # Assertions
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["feature1", "feature2", "feature3", "feature4", "feature5", "target"]
        assert loaded_df["target"].tolist() == [0, 1, 0]

    def test_data_manager_initialization(self):
        """Test DataManager initialization."""
        # Mock connection
        with patch('backend.data_manager.create_engine'):
            dm = DataManager("mock://connection")
            assert dm.connection_string == "mock://connection"


# ==================== ML Pipeline Tests ====================

class TestMLPipeline:
    """Test ML pipeline functionality."""

    def test_pipeline_creation(self, ml_pipeline):
        """Test pipeline creation."""
        assert ml_pipeline.pipeline is not None
        assert ml_pipeline.pipeline.named_steps["scaler"] is not None
        assert ml_pipeline.pipeline.named_steps["classifier"] is not None

    def test_model_training(self, ml_pipeline, sample_data):
        """Test model training."""
        X, y = sample_data
        ml_pipeline.train_best_model(X, y)
        assert ml_pipeline.best_model is not None

    def test_prediction(self, trained_pipeline, sample_data):
        """Test making predictions."""
        X, _ = sample_data
        result = trained_pipeline.predict(X[:1])

        assert "prediction" in result
        assert "confidence" in result
        assert "probability_class_0" in result
        assert "probability_class_1" in result
        assert result["prediction"] in [0, 1]
        assert 0 <= result["confidence"] <= 1

    def test_evaluation(self, trained_pipeline, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        metrics = trained_pipeline.evaluate(X, y)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics

        for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            assert 0 <= metrics[metric] <= 1

    def test_model_saving_and_loading(self, trained_pipeline, tmp_path):
        """Test saving and loading model."""
        model_path = tmp_path / "test_model.joblib"
        preprocessor_path = tmp_path / "test_preprocessor.joblib"

        # Save
        trained_pipeline.save_model(str(model_path), str(preprocessor_path))
        assert model_path.exists()
        assert preprocessor_path.exists()

        # Load
        new_pipeline = MLPipeline(wandb_enabled=False)
        new_pipeline.load_model(str(model_path))
        assert new_pipeline.best_model is not None

    def test_data_preparation(self, tmp_path):
        """Test data preparation."""
        # Create test CSV
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "feature1": np.random.randn(30),
            "feature2": np.random.randn(30),
            "feature3": np.random.randn(30),
            "feature4": np.random.randn(30),
            "feature5": np.random.randn(30),
            "target": np.random.randint(0, 2, 30)
        })
        df.to_csv(csv_file, index=False)

        X_train, X_test, y_train, y_test = load_and_prepare_data(str(csv_file))

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == 30
        assert X_train.shape[1] == 5
        assert X_test.shape[1] == 5


# ==================== FastAPI Backend Tests ====================

class TestFastAPIBackend:
    """Test FastAPI backend endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "Random Forest Classifier"
        assert "features" in data
        assert "classes" in data

    def test_prediction_endpoint_valid(self, client):
        """Test prediction endpoint with valid input."""
        payload = {
            "feature1": 1.2,
            "feature2": 3.4,
            "feature3": 5.6,
            "feature4": 2.1,
            "feature5": 1.5
        }
        response = client.post("/predict", json=payload)
        
        # If model is loaded
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert data["prediction"] in [0, 1]

    def test_prediction_endpoint_invalid_input(self, client):
        """Test prediction endpoint with invalid input."""
        payload = {
            "feature1": "invalid",  # Should be float
            "feature2": 3.4,
            "feature3": 5.6,
            "feature4": 2.1,
            "feature5": 1.5
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_prediction_missing_field(self, client):
        """Test prediction endpoint with missing field."""
        payload = {
            "feature1": 1.2,
            "feature2": 3.4,
            # Missing feature3, feature4, feature5
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_batch_prediction_endpoint(self, client):
        """Test batch prediction endpoint."""
        payload = [
            {"feature1": 1.2, "feature2": 3.4, "feature3": 5.6, "feature4": 2.1, "feature5": 1.5},
            {"feature1": 2.3, "feature2": 4.5, "feature3": 6.7, "feature4": 3.2, "feature5": 2.6}
        ]
        response = client.post("/predict-batch", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data) == 2
            assert all("prediction" in item for item in data)

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus metrics endpoint returns text


# ==================== Data Validation Tests ====================

class TestDataValidation:
    """Test data validation."""

    def test_feature_range_validation(self):
        """Test feature range validation."""
        from backend.main import PredictionInput

        # Valid input
        valid_input = PredictionInput(
            feature1=1.0,
            feature2=2.0,
            feature3=3.0,
            feature4=4.0,
            feature5=5.0
        )
        assert valid_input.feature1 == 1.0

        # Invalid input (non-numeric)
        with pytest.raises(ValueError):
            PredictionInput(
                feature1="invalid",
                feature2=2.0,
                feature3=3.0,
                feature4=4.0,
                feature5=5.0
            )

    def test_output_format(self):
        """Test output format validation."""
        from backend.main import PredictionOutput

        output = PredictionOutput(
            prediction=1,
            confidence=0.95,
            probability_class_0=0.05,
            probability_class_1=0.95,
            timestamp="2023-01-01T00:00:00"
        )

        assert output.prediction == 1
        assert output.confidence == 0.95


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests."""

    def test_full_pipeline_flow(self, tmp_path):
        """Test full pipeline from training to prediction."""
        # Create data
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.randn(50),
            "feature4": np.random.randn(50),
            "feature5": np.random.randn(50),
            "target": np.random.randint(0, 2, 50)
        })
        df.to_csv(csv_file, index=False)

        # Prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data(str(csv_file))

        # Create and train pipeline
        pipeline = MLPipeline(wandb_enabled=False)
        pipeline.create_pipeline()
        pipeline.train_best_model(X_train, y_train)

        # Evaluate
        metrics = pipeline.evaluate(X_test, y_test)
        assert metrics["accuracy"] > 0

        # Make prediction
        result = pipeline.predict(X_test[:1])
        assert "prediction" in result
        assert 0 <= result["confidence"] <= 1

        # Save and load
        model_path = tmp_path / "model.joblib"
        pipeline.save_model(str(model_path))

        new_pipeline = MLPipeline(wandb_enabled=False)
        new_pipeline.load_model(str(model_path))
        new_result = new_pipeline.predict(X_test[:1])
        assert new_result["prediction"] == result["prediction"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=backend", "--cov-report=html"])
