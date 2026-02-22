"""ML training pipeline with hyperparameter tuning and W&B tracking."""

import os
import joblib
import pandas as pd
import numpy as np
import wandb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, BayesianSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class MLPipeline:
    """Machine Learning pipeline for classification with hyperparameter tuning."""

    def __init__(self, project_name: str = "ml-classification", wandb_enabled: bool = True):
        """Initialize ML pipeline.
        
        Args:
            project_name: W&B project name
            wandb_enabled: Enable W&B tracking
        """
        self.project_name = project_name
        self.wandb_enabled = wandb_enabled
        self.pipeline = None
        self.best_model = None
        self.best_params = None
        self.test_metrics = {}
        
        if wandb_enabled:
            wandb.login(key=os.getenv("WANDB_API_KEY", ""))

    def create_pipeline(self, use_polynomial_features: bool = False) -> Pipeline:
        """Create sklearn pipeline.
        
        Args:
            use_polynomial_features: Whether to add polynomial features
            
        Returns:
            Configured sklearn Pipeline
        """
        steps = [("scaler", StandardScaler())]
        
        if use_polynomial_features:
            steps.append(("poly_features", PolynomialFeatures(degree=2, include_bias=False)))
        
        steps.append(("classifier", RandomForestClassifier(random_state=42)))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search_type: str = "grid",
        cv_folds: int = 5,
        wandb_run_name: str = "hyperparameter_tuning"
    ) -> dict:
        """Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            search_type: "grid", "random", or "bayesian"
            cv_folds: Number of cross-validation folds
            wandb_run_name: W&B run name
            
        Returns:
            Dictionary with best parameters and metrics
        """
        if self.wandb_enabled:
            wandb.init(project=self.project_name, name=wandb_run_name)

        param_grid = {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }

        logger.info(f"Starting {search_type} hyperparameter search...")

        if search_type == "grid":
            search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=cv_folds,
                n_jobs=-1,
                verbose=1,
                scoring="f1_weighted"
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                self.pipeline,
                param_grid,
                n_iter=20,
                cv=cv_folds,
                n_jobs=-1,
                verbose=1,
                scoring="f1_weighted",
                random_state=42
            )
        else:
            search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=cv_folds,
                n_jobs=-1,
                verbose=1,
                scoring="f1_weighted"
            )

        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_

        results = {
            "best_params": self.best_params,
            "best_score": search.best_score_,
            "all_results": search.cv_results_
        }

        if self.wandb_enabled:
            wandb.log({
                "best_params": str(self.best_params),
                "best_cv_score": search.best_score_,
                "search_type": search_type
            })
            
            # Log all trials
            for i, params in enumerate(search.cv_results_["params"]):
                wandb.log({
                    f"trial_{i}_params": str(params),
                    f"trial_{i}_mean_score": search.cv_results_["mean_test_score"][i]
                })

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {search.best_score_}")

        return results

    def train_best_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the best model on full training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training best model on full training data...")
        self.best_model.fit(X_train, y_train)
        logger.info("Best model training complete")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, dataset_name: str = "test"):
        """Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of dataset (for logging)
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        self.test_metrics = metrics

        if self.wandb_enabled:
            wandb.log({
                f"{dataset_name}_accuracy": metrics["accuracy"],
                f"{dataset_name}_f1": metrics["f1"],
                f"{dataset_name}_precision": metrics["precision"],
                f"{dataset_name}_recall": metrics["recall"],
                f"{dataset_name}_roc_auc": metrics["roc_auc"]
            })

        logger.info(f"Evaluation metrics ({dataset_name}):")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def save_model(self, model_path: str, preprocessor_path: str = None):
        """Save trained model and preprocessor.
        
        Args:
            model_path: Path to save model
            preprocessor_path: Path to save preprocessor
        """
        try:
            joblib.dump(self.best_model, model_path)
            logger.info(f"Model saved to {model_path}")

            if preprocessor_path:
                scaler = self.best_model.named_steps["scaler"]
                joblib.dump(scaler, preprocessor_path)
                logger.info(f"Preprocessor saved to {preprocessor_path}")

            if self.wandb_enabled:
                wandb.save(model_path)
                if preprocessor_path:
                    wandb.save(preprocessor_path)
                
                # Log model as artifact
                artifact = wandb.Artifact("trained-model", type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: str):
        """Load trained model.
        
        Args:
            model_path: Path to model file
        """
        try:
            self.best_model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, features: np.ndarray) -> dict:
        """Make predictions on new data.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.best_model is None:
            raise ValueError("Model not trained or loaded")

        prediction = self.best_model.predict(features)[0]
        probabilities = self.best_model.predict_proba(features)[0]

        return {
            "prediction": int(prediction),
            "probability_class_0": float(probabilities[0]),
            "probability_class_1": float(probabilities[1]),
            "confidence": float(max(probabilities))
        }

    def close_wandb(self):
        """Close W&B run."""
        if self.wandb_enabled:
            wandb.finish()


def load_and_prepare_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load and prepare data for training.
    
    Args:
        csv_path: Path to CSV file
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test
