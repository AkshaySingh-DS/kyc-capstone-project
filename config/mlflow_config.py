import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MLFLOW_EXPERIMENT = "KYC-Agent-Workflow"


def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)