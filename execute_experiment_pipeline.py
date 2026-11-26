from data.generate_eval_dataset import generate_evaluation_dataset
from eval.application_evals.evaluate_rag_app import evaluate_app
from config.paramenter_config import params_config
from pathlib import Path
from langfuse import get_client
from typing import Literal
import json
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).parent

# flatten params which is in nested dictionary and convert into single dictionary
def flatten_params(params_dict: dict) -> dict:
    config_dict = {}

    for key, value in params_dict.items():
        if isinstance(value,dict):
            config_dict.update(flatten_params(value))

        else:
            if key in config_dict:
                raise ValueError(f"Duplicate key found: {key}")

            config_dict[key] = value

    return config_dict

# get latest report file name
def get_latest_result(path: Path | str):

    filenames = []

    files = path.glob("*.json")
    for file in files:
        filenames.append(file.stem)

    return max(filenames)

# get metrics from report file
def get_merics_from_results(results_json) -> dict:
    metrics = {}

    with open(results_json, "r") as file:
        metrics_results = json.load(file)["metricsScore"]

    # if getting error in strip() can edit
    for result in metrics_results:
        metric_name = result["metric"].removesuffix("[GEval]").strip() if "[GEval]" in result["metric"] else result["metric"].strip()
        scores = result["scores"]
        avg_scores = round((sum(scores)/len(scores)),2)

        metrics[metric_name] = avg_scores

    return metrics

# get staging label prompt
def import_system_prompt(label="staging") -> str:

    langfuse = get_client()

    prompt = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type = "text",
        labele = label
    )

    return prompt.prompt

def artifact_name(artifact_type: Literal["eval_dataset", "golden_datset"], artifact_dir):
    DATA_PATH = ROOT_DIR / "data" / "evaluation"

    if artifact_type == "eval_dataset":
        artifact_path = (DATA_PATH / "eval_dataset" / params_config.evaluation_dataset.evaluation_dataset_filename).as_posix() + ".json"
        return (artifact_path, artifact_dir)

    elif artifact_type == "golden_dataset":
        artifact_path = artifact_path = (DATA_PATH / "eval_dataset" / params_config.golden_dataset.golden_dataset_filename).as_posix() + ".json"
        return (artifact_path, artifact_dir)

def return_code_files():
    code_file_paths = ROOT_DIR / "src"
    app_code_path = code_file_paths / "app" / "rag_workflow.py"
    eval_data_path = code_file_paths / "data" / "generate_eval_dataset.py"
    evaluation_path = code_file_paths / "evals" / "application_evals" / "evaluate_rag_app.py"
    golden_dataset_path = code_file_paths / "data"/ "generate_goldens.py"

    return [file.as_posix() for file in [app_code_path, eval_data_path, evaluation_path, golden_dataset_path]]