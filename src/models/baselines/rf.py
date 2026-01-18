import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .registry import register

@register("rf")
def train_predict_rf(X_train, y_train, X_test, rf_params: dict):
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    return model, y_pred_test
