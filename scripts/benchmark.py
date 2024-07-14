import os, argparse
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt


WORKING_DIR = os.getcwd()

BENCHMARK_PATH = os.path.join(WORKING_DIR, "benchmark/")
TRAN_RESULTS_PATH = os.path.join(WORKING_DIR, "train_results/")

DATASET_PATH = os.path.join(WORKING_DIR, "dataset/")
YAML_FILE_PATH=os.path.join(DATASET_PATH, "data.yaml")

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
        help=(
            "train run name."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        help=(
            "device to run benchmark at."
        ),
    )


    args = parser.parse_args()

    project = args.project
    name = args.name
    device = args.device



    models_path = os.path.join(TRAN_RESULTS_PATH, project, name, "weights")
    project_path = os.path.join(BENCHMARK_PATH, project, name+"_"+device)
    models = os.listdir(models_path)
    metrics_list = []
    speed_list = []
    for model_name in models:
        model = YOLO(os.path.join(models_path, model_name))
        model_format = model_name.split(".")[-1]
        kwargs={
            "data": YAML_FILE_PATH,
            "split": "test",
            "project": project_path,
            "name": model_format,
            "device": device
        }
        metrics = model.val(**kwargs)  # no arguments needed, dataset and settings remembered

        speed_dict = metrics.speed
        speed_dict["format"] = model_format
        speed_list.append(speed_dict)

        metrics_dict = metrics.results_dict
        metrics_dict["format"] = model_format
        metrics_list.append(metrics_dict)

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
    for i, column_name in enumerate(columns):
        axis[i].bar(speed_df["format"], speed_df[column_name], width=0.3)
        axis[i].set_ylabel(column_name + " (ms)")
        axis[i].set_xlabel("runtime type")
    fig.savefig(os.path.join(project_path, "speed.png"))

    columns = set(metrics_df.columns)
    columns.discard("format")
    columns.discard("fitness")
    fig, axis = plt.subplots(nrows=len(columns), ncols=1, figsize=(6, len(columns)*2))
    for i, column_name in enumerate(columns):
        axis[i].bar(metrics_df["format"], metrics_df[column_name], width=0.3)
        axis[i].set_ylabel(column_name)
        axis[i].set_xlabel("runtime type")
    fig.savefig(os.path.join(project_path, "metrics.png"))
    
    