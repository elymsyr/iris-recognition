import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model # type: ignore
import tensorflow.core
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

class ModelHandler:
    @staticmethod
    def save_model(model, file_path):
        if isinstance(model, xgb.XGBModel):  # Check for any XGBoost model
            model.save_model(file_path)
        elif isinstance(model, lgb.LGBMModel):  # Check for any LightGBM model
            model.booster_.save_model(file_path)
        elif hasattr(model, 'save'):  # Check for Keras/TensorFlow models
            model.save(file_path)
        else:  # Assume Scikit-Learn or other models
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
        else:  # Assume Scikit-Learn or other models
            model = joblib.load(file_path)
        return model
