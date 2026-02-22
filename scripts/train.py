"""Training script for ML model with W&B tracking."""

import os
import sys
import logging
from pathlib import Path
from backend.ml_pipeline import MLPipeline, load_and_prepare_data
from backend.data_manager import DataManager, load_data_from_csv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "model_artifacts/model.joblib"
PREPROCESSOR_PATH = "model_artifacts/preprocessor.joblib"
SEARCH_TYPE = "random"  # "grid", "random", or "bayesian"


def main():
    """Main training function."""
    
    logger.info("="*80)
    logger.info("ML CLASSIFICATION TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load data
    logger.info(f"Loading data from {DATA_PATH}...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_PATH)
    
    # Initialize pipeline
    logger.info("Initializing ML pipeline...")
    pipeline = MLPipeline(project_name="ml-classification", wandb_enabled=True)
    
    # Create pipeline
    pipeline.create_pipeline(use_polynomial_features=False)
    
    # Hyperparameter tuning
    logger.info(f"Starting hyperparameter tuning ({SEARCH_TYPE} search)...")
    tune_results = pipeline.tune_hyperparameters(
        X_train, y_train,
        search_type=SEARCH_TYPE,
        cv_folds=5,
        wandb_run_name=f"hpo-{SEARCH_TYPE}"
    )
    
    # Train best model
    logger.info("Training best model...")
    pipeline.train_best_model(X_train, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = pipeline.evaluate(X_test, y_test, dataset_name="test")
    
    # Also evaluate on training set
    train_metrics = pipeline.evaluate(X_train, y_train, dataset_name="train")
    
    # Save model
    logger.info("Saving model artifacts...")
    os.makedirs("model_artifacts", exist_ok=True)
    pipeline.save_model(MODEL_PATH, PREPROCESSOR_PATH)
    
    # Log summary
    logger.info("="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Best parameters: {pipeline.best_params}")
    logger.info("\nTest Metrics:")
    for metric, value in test_metrics.items():
        if metric != "confusion_matrix":
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nModel saved to:")
    logger.info(f"  - Model: {MODEL_PATH}")
    logger.info(f"  - Preprocessor: {PREPROCESSOR_PATH}")
    
    # Close W&B
    pipeline.close_wandb()
    
    logger.info("="*80)
    logger.info("Training complete!")
    logger.info("="*80)
    
    return pipeline, test_metrics


if __name__ == "__main__":
    main()
