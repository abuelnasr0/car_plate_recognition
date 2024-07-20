import os, argparse
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "offscreen"


WORKING_DIR = os.getcwd()

BENCHMARK_PATH = os.path.join(WORKING_DIR, "benchmark/")
TRAN_RESULTS_PATH = os.path.join(WORKING_DIR, "train_results/")

DATASET_PATH = os.path.join(WORKING_DIR, "dataset/")
YAML_FILE_PATH=os.path.join(DATASET_PATH, "data.yaml")

def get_base_model_name_len(models):
    for model in models:
        if model.endswith(".pt"):
            return len(model[:-3]) + 1
        else:
            return len(model.split("_")[0]) + 1

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
        help=(
            "Project name."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ALL",
        help=(
            "train run name."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "device to run benchmark at."
        ),
    )


    args = parser.parse_args()

    project = args.project
    name = args.name
    device = args.device

    project_path = os.path.join(TRAN_RESULTS_PATH, project)

    names = [name]
    if name == "ALL":
        names = os.listdir(project_path)
    
    failed_models = {}

    for name in names:
        failed_models[name] = []
        models_path = os.path.join(project_path, name, "weights")
        project_path = os.path.join(BENCHMARK_PATH, project, name+"_"+device)
        models = os.listdir(models_path)
        metrics_list = []
        speed_list = []
        base_name_len = get_base_model_name_len(models)
        for model_name in models:
            model = YOLO(os.path.join(models_path, model_name))
            model_format = model_name[base_name_len:]
            model_format = model_format.replace(".", "_")

            kwargs={
                "data": YAML_FILE_PATH,
                "split": "test",
                "project": project_path,
                "name": model_format,
                "device": device
            }
            try:
                metrics = model.val(**kwargs)  # no arguments needed, dataset and settings remembered

                speed_dict = metrics.speed
                speed_dict["format"] = model_format
                speed_list.append(speed_dict)

                metrics_dict = metrics.results_dict
                metrics_dict["format"] = model_format
                metrics_list.append(metrics_dict)
            except:
                failed_models[name].append(model_format)

        speed_df = pd.DataFrame(speed_list)
        speed_df.set_index("format")
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.set_index("format")

        speed_df.to_csv(os.path.join(project_path, "speed.csv"), index=False)
        metrics_df.to_csv(os.path.join(project_path, "metrics.csv"), index=False)

        columns = set(speed_df.columns)
        columns.discard("format")
        columns.discard("loss")
        fig, axis = plt.subplots(nrows=len(columns), ncols=1, figsize=(6, len(columns)*2))
        fig.autofmt_xdate(rotation=45)
        for i, column_name in enumerate(columns):
            axis[i].bar(speed_df["format"], speed_df[column_name], width=0.3)
            axis[i].set_ylabel(column_name + " (ms)")
        fig.savefig(os.path.join(project_path, "speed.png"))

        columns = set(metrics_df.columns)
        columns.discard("format")
        columns.discard("fitness")
        fig, axis = plt.subplots(nrows=len(columns), ncols=1, figsize=(6, len(columns)*2))
        fig.autofmt_xdate(rotation=45)
        for i, column_name in enumerate(columns):
            axis[i].bar(metrics_df["format"], metrics_df[column_name], width=0.3)
            axis[i].set_ylabel(column_name)
        fig.savefig(os.path.join(project_path, "metrics.png"))
    
    for key in failed_models:
        print(f"for {key}:")
        for failed_model in failed_models[key]:
            print(f"{failed_model} failed")
        
        