from config.paramenter_config import params_config
import pprint
from pathlib import Path

pprint.pprint(params_config.model_dump())


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

def get_latest_result(path: Path | str):

    filenames = []

    files = path.glob("*.json")
    for file in files:
        filenames.append(file.stem)

    recent_file = max(filenames, key = file.name)
    return recent_file

ROOT_DIR = Path(__file__).parent
evaluation_results_dir = ROOT_DIR / "reports"/ "evaluation_results"
recent_file = get_latest_result(evaluation_results_dir)