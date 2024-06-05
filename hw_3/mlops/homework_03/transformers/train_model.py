if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import pickle

@transformer
def transform(df, *args, **kwargs):
    # Ensure the relevant columns are in string format
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # Create a dictionary representation of the DataFrame
    train_dicts = df[categorical].to_dict(orient='records')
    
    # Fit the DictVectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print(f'Feature matrix size: {X_train.shape}')
    
    # Train the Linear Regression model
    target = 'duration'
    y_train = df[target].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict and calculate RMSE
    y_pred = lr.predict(X_train)
    print(f'Train RMSE: {mean_squared_error(y_train, y_pred, squared=False)}')
    
    # Print the intercept
    print(f'Model intercept: {lr.intercept_}')
    
    # Set the MLflow tracking URI (adjust if necessary)
    mlflow.set_tracking_uri('http://mlflow:5000')
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log the linear regression model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Save and log the DictVectorizer as an artifact
        with open("dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dict_vectorizer.pkl")
    
    print(f"Model and DictVectorizer logged to MLflow with run_id: {run.info.run_id}")
    
    # Return the vectorizer and the model
    return {
        'dict_vectorizer': dv,
        'linear_regression_model': lr
    }