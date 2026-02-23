"""Training script for Hepatitis Classification with W&B tracking."""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/hepatitis.csv"
MODEL_PATH = "model_artifacts/hepatitis_model.joblib"
PREPROCESSOR_PATH = "model_artifacts/hepatitis_preprocessor.joblib"


def load_hepatitis_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load and prepare hepatitis data for training.
    
    Args:
        csv_path: Path to CSV file
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, label_mapping, feature_names)
    """
    logger.info(f"Loading hepatitis data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)
    
    # Clean data - remove rows with missing target
    df = df.dropna(subset=['Category'])
    
    # Handle missing values - fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Encode target variable
    categories = df['Category'].unique()
    label_mapping = {cat: idx for idx, cat in enumerate(sorted(categories))}
    y = df['Category'].map(label_mapping)
    
    # Prepare features
    X = df.drop('Category', axis=1)
    
    # Encode Sex (m/f)
    X['Sex'] = X['Sex'].map({'m': 0, 'f': 1}).fillna(0)
    
    feature_names = X.columns.tolist()
    X = X.values
    y = y.values
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Classes: {label_mapping}")
    
    return X_train, X_test, y_train, y_test, label_mapping, feature_names


def train_hepatitis_model(X_train, y_train, X_test, y_test, label_mapping):
    """Train model for hepatitis classification.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        label_mapping: Category to index mapping
        
    Returns:
        Tuple of (model, scaler, metrics)
    """
    logger.info("="*80)
    logger.info("HEPATITIS CLASSIFICATION MODEL TRAINING")
    logger.info("="*80)
    
    # Preprocessing
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training
    logger.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_pred_train),
        'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
        'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
        'f1': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    }
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING METRICS")
    logger.info("="*80)
    for metric, value in train_metrics.items():
        logger.info(f"  Train {metric.upper():10s}: {value:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("TEST METRICS")
    logger.info("="*80)
    for metric, value in test_metrics.items():
        logger.info(f"  Test {metric.upper():10s}: {value:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': range(len(model.feature_importances_)),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n" + "="*80)
    logger.info("TOP 10 IMPORTANT FEATURES")
    logger.info("="*80)
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  Feature {int(row['feature']):2d}: {row['importance']:.4f}")
    
    return model, scaler, {'train': train_metrics, 'test': test_metrics}


def save_hepatitis_model(model, scaler, model_path, preprocessor_path):
    """Save trained model and preprocessor.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        model_path: Path to save model
        preprocessor_path: Path to save preprocessor
    """
    logger.info(f"Saving model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    logger.info(f"Saving preprocessor to {preprocessor_path}...")
    joblib.dump(scaler, preprocessor_path)
    
    logger.info("✅ Model and preprocessor saved successfully!")


def main():
    """Main training function."""
    try:
        # Load data
        X_train, X_test, y_train, y_test, label_mapping, feature_names = load_hepatitis_data(DATA_PATH)
        
        # Train model
        model, scaler, metrics = train_hepatitis_model(X_train, y_train, X_test, y_test, label_mapping)
        
        # Save model
        save_hepatitis_model(model, scaler, MODEL_PATH, PREPROCESSOR_PATH)
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        
        return model, scaler, metrics
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
