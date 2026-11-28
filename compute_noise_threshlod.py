import dagshub
import mlflow
from mlflow_utils import experiment_id, get_metrics_from_runs
import numpy as np
from pathlib import Path
import json


ROOT_DIR = Path(__file__).parent
THRESH_PATH = ROOT_DIR/ "thresholds.json"

def save_thresholds(json_path, stds):
    if json_path.exist():
        with open (json_path, "r") as file:
            noise_thresholds = json.load(file).get("noise_thresholds")
            regression_thresholds = json.load(file).get("regression_thresholds")

    else:
        thresholds = {}
        thresholds["noise_threshold"] = stds
        thresholds["regression_thresholds"] = regression_thresholds

    with open(json_path, "w") as file:
        json.dump(thresholds,  file, indent=4)



# calcualte noise threshold using all data 
def calculate_noise_threshold(metrics: list[dict[str, float]]):
    all_metrics = []
    for metric in metrics:
        metric_name = list(metric.keys())
        metric_values = list(metric.values())
        all_metrics.append(metric_values)

    matrix = np.array(all_metrics)
    stds = np.std(matrix, axis=0, ddof=1)
    return {key: val for key, val in zip(metric_name, stds)}



dagshub.init(repo_owner='vermadhanraj015', repo_name='LLMOps', mlflow=True)

    # set the tracking server
mlflow.set_tracking_uri("https://dagshub.com/vermadhanraj015/LLMOps.mlflow")

# get experiment ID using experiment name -> through experiment id we extract all metrics
experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id

# get the metrics across noise thresh runs
all_metrics = get_metrics_from_runs(tag_name="noise_threshold",
                    experiment_id=experiment_id)