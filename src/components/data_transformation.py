import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    transformed_train_path: str = os.path.join("artifacts", "transformed_train.csv")
    transformed_test_path: str = os.path.join("artifacts", "transformed_test.csv")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Entered the data transformation component")

            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data successfully")

            # Verify expected columns
            expected_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            for col in expected_cols:
                if col not in train_df.columns:
                    raise CustomException(f"Column '{col}' missing in training data", sys)

            # Removed negative or zero monetary values (if any)
            train_df = train_df[train_df['Monetary'] > 0]
            test_df = test_df[test_df['Monetary'] > 0]

            # Saved transformed versions
            os.makedirs(os.path.dirname(self.config.transformed_train_path), exist_ok=True)
            train_df.to_csv(self.config.transformed_train_path, index=False)
            test_df.to_csv(self.config.transformed_test_path, index=False)

            logging.info("Data transformation completed and files saved")

            return (
                self.config.transformed_train_path,
                self.config.transformed_test_path
            )

        except Exception as e:
            raise CustomException(e, sys)
