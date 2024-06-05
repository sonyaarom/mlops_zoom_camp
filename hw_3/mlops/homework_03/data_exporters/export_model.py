if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
import mlflow
import mlflow.sklearn
import pickle


@data_exporter
def export_data(*args, **kwargs):
    # Retrieve the model and vectorizer from the previous block's output
    model_data = kwargs['block_output']['train_model']
    dv = model_data['dict_vectorizer']
    lr = model_data['linear_regression_model']
    
    # Set the MLflow tracking URI (adjust if necessary)
    mlflow.set_tracking_uri('http://mlflow:8888')
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log the linear regression model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Save and log the DictVectorizer as an artifact
        with open("dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dict_vectorizer.pkl")
    
    print(f"Model and DictVectorizer logged to MLflow with run_id: {run.info.run_id}")


