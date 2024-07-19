from ultralytics import YOLO

import argparse
import os

import yaml


WORKING_DIR = os.getcwd()

TRAINING_ARGS_PATH = os.path.join(WORKING_DIR, "training_args.yaml")
MULTI_TRAINING_ARGS_PATH = os.path.join(WORKING_DIR, "multi_training_args.yaml")
TMP_DIR = os.path.join(WORKING_DIR, "tmp")
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)
TMP_TRAINING_ARGS_PATH = os.path.join(TMP_DIR, "tmp_training_args.yaml")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_args_path",
        default=TRAINING_ARGS_PATH,
        type=str,
        help=(
            "The path to training_args yaml file"
        ),
    )
    parser.add_argument(
        "--multi_training_args_path",
        default=MULTI_TRAINING_ARGS_PATH,
        type=str,
        help=(
            "The path to training_args yaml file"
        ),
    )

    args = parser.parse_args()
    training_args_path = args.training_args_path
    multi_training_args_path = args.multi_training_args_path

    with open(training_args_path) as stream:
        try:
            training_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open(multi_training_args_path) as stream:
        try:
            multi_training_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    for curr_training_args in multi_training_args:
        training_args.update(curr_training_args)
        with open(TMP_TRAINING_ARGS_PATH, "w") as tmp_yaml:
            yaml.dump(training_args, tmp_yaml, default_flow_style=False)

        print(f"starting {training_args["model"]} training...\n")
        print(f"*"*20)
        print(f"\n")
        os.system(f"python ./scripts/train.py --training_args_path {TMP_TRAINING_ARGS_PATH}")

    
    os.remove(TMP_TRAINING_ARGS_PATH)
    os.rmdir(TMP_DIR)