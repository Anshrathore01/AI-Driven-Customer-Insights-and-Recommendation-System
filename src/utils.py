import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Save any Python object using pickle"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a pickled Python object"""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(model, X_test, y_test):
    """Evaluate a regression model and return performance metrics"""
    try:
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return {"r2_score": r2, "mae": mae, "rmse": rmse}
    except Exception as e:
        raise CustomException(e, sys)
