import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model  # type: ignore
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import numpy as np

class ModelHandler:
    @staticmethod
    def save_model(model, file_path):
        if isinstance(model, xgb.XGBModel):
            model.save_model(file_path)
        elif isinstance(model, lgb.LGBMModel):
            model.booster_.save_model(file_path)
        elif hasattr(model, 'save'):
            model.save(file_path)
        else:
            joblib.dump(model, file_path)

    @staticmethod
    def load_model(file_path, model_type='sklearn'):
        if model_type == 'xgboost':
            model = xgb.Booster()
            model.load_model(file_path)
        elif model_type == 'xgbclassifier':
            model = xgb.XGBClassifier()
            model.load_model(file_path)
        elif model_type == 'lightgbm':
            model = lgb.Booster(model_file=file_path)
        elif model_type == 'keras':
            model = load_model(file_path)
        else:
            model = joblib.load(file_path)
        return model

    @staticmethod
    def predict(model, data, threshold = None):
        if isinstance(model, (xgb.XGBModel, lgb.LGBMModel, LogisticRegression)):
            y_pred = model.predict(data)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(data)
        else:
            raise ValueError("Unsupported model type for prediction.")

        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        if threshold and isinstance(model, (xgb.XGBModel, lgb.LGBMModel, LogisticRegression)) or hasattr(model, 'predict_proba'):
            y_pred = [1 if x > threshold else 0 for x in y_pred]
        return np.array(y_pred)
