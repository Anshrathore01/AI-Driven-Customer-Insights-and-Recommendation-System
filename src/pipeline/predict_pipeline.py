import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import os


class PredictPipeline:
    def __init__(self):
        # Path to the trained model only
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict(self, features):
        try:
            logging.info("Starting prediction pipeline")

            # Load trained model
            model = load_object(self.model_path)
            logging.info("Model loaded successfully for prediction")

            # Convert incoming data to DataFrame (if it's dict or list)
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, list):
                features = pd.DataFrame(features)

            # Predict directly (no preprocessing required)
            preds = model.predict(features)
            logging.info("Prediction completed successfully")

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, Recency, Frequency, Monetary):
        self.Recency = Recency
        self.Frequency = Frequency
        self.Monetary = Monetary

    def get_data_as_dataframe(self):
        """Convert user input into pandas DataFrame"""
        try:
            custom_data_input_dict = {
                "Recency": [self.Recency],
                "Frequency": [self.Frequency],
                "Monetary": [self.Monetary]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data converted to DataFrame successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example input (you can change these values)
    input_data = CustomData(
        Recency=30,
        Frequency=10,
        Monetary=500
    )

    df = input_data.get_data_as_dataframe()
    print("Input DataFrame:")
    print(df)

    # Make prediction
    pipeline = PredictPipeline()
    preds = pipeline.predict(df)
    print("\nPredicted Value:")
    print(preds)
