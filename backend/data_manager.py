"""Data layer module for loading and managing data from Neon Postgres."""

import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class DataManager:
    """Manages data operations with Neon Postgres."""

    def __init__(self, connection_string: str = None):
        """Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string.
                If None, uses DATABASE_URL from environment.
        """
        if connection_string is None:
            connection_string = os.getenv(
                "DATABASE_URL",
                "postgresql://user:password@localhost/mlops_db"
            )
        self.connection_string = connection_string
        self.engine = None
        self._connect()

    def _connect(self):
        """Create database connection."""
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def create_tables(self):
        """Create necessary database tables."""
        try:
            with self.engine.connect() as conn:
                # Create datasets table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        version VARCHAR(50)
                    )
                """))
                
                # Create training_data table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id SERIAL PRIMARY KEY,
                        dataset_id INTEGER REFERENCES datasets(id),
                        feature1 FLOAT,
                        feature2 FLOAT,
                        feature3 FLOAT,
                        feature4 FLOAT,
                        feature5 FLOAT,
                        target INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create predictions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        feature1 FLOAT,
                        feature2 FLOAT,
                        feature3 FLOAT,
                        feature4 FLOAT,
                        feature5 FLOAT,
                        prediction INTEGER,
                        confidence FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def load_csv_to_db(self, csv_path: str, dataset_name: str = "default"):
        """Load CSV data into the database.
        
        Args:
            csv_path: Path to CSV file
            dataset_name: Name of the dataset
        """
        try:
            df = pd.read_csv(csv_path)
            
            with self.engine.connect() as conn:
                # Insert dataset metadata
                result = conn.execute(text("""
                    INSERT INTO datasets (name, version)
                    VALUES (:name, :version)
                    RETURNING id
                """), {"name": dataset_name, "version": "1.0"})
                
                dataset_id = result.scalar()
                
                # Insert training data
                for _, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO training_data 
                        (dataset_id, feature1, feature2, feature3, feature4, feature5, target)
                        VALUES (:dataset_id, :f1, :f2, :f3, :f4, :f5, :target)
                    """), {
                        "dataset_id": dataset_id,
                        "f1": row["feature1"],
                        "f2": row["feature2"],
                        "f3": row["feature3"],
                        "f4": row["feature4"],
                        "f5": row["feature5"],
                        "target": row["target"]
                    })
                
                conn.commit()
                logger.info(f"Loaded {len(df)} records from {csv_path}")
                return dataset_id
        except Exception as e:
            logger.error(f"Error loading CSV to database: {e}")
            raise

    def get_training_data(self, dataset_id: int = None) -> pd.DataFrame:
        """Retrieve training data from database.
        
        Args:
            dataset_id: Optional specific dataset ID
            
        Returns:
            DataFrame with training data
        """
        try:
            query = "SELECT * FROM training_data"
            if dataset_id:
                query += f" WHERE dataset_id = {dataset_id}"
            
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"Retrieved {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error retrieving training data: {e}")
            raise

    def save_prediction(self, features: list, prediction: int, confidence: float = None):
        """Save prediction to database.
        
        Args:
            features: List of feature values [f1, f2, f3, f4, f5]
            prediction: Predicted class
            confidence: Confidence score (optional)
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO predictions 
                    (feature1, feature2, feature3, feature4, feature5, prediction, confidence)
                    VALUES (:f1, :f2, :f3, :f4, :f5, :pred, :conf)
                """), {
                    "f1": features[0],
                    "f2": features[1],
                    "f3": features[2],
                    "f4": features[3],
                    "f5": features[4],
                    "pred": prediction,
                    "conf": confidence
                })
                conn.commit()
                logger.info("Prediction saved to database")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """Load data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with the data
    """
    return pd.read_csv(csv_path)
