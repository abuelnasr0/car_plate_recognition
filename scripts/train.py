from ultralytics import YOLO

import argparse
import os

import yaml


WORKING_DIR = os.getcwd()

PRETRAINED_MODELS_PATH = os.path.join(WORKING_DIR, "models/pretrained/")
TRAN_RESULTS_PATH = os.path.join(WORKING_DIR, "train_results/")
TRAINING_ARGS_PATH = os.path.join(WORKING_DIR, "training_args.yaml")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cont",
        default=False,
        type=bool,
        help=(
            "Whether to cintinue training from a checkpoint or not."
        ),
    )
    parser.add_argument(
        "--training_args_path",
        default=TRAINING_ARGS_PATH,
        type=str,
        help=(
            "The path to training_args yaml file"
        ),
    )

    args = parser.parse_args()
    continue_training = args.cont
    training_args_path = args.training_args_path

    with open(training_args_path) as stream:
        try:
            training_args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    model_name = training_args["model"]

    if continue_training:
        models_path = os.path.join(TRAN_RESULTS_PATH, training_args["model_dir"], "weights")
        accepted_models = os.listdir(models_path)
        if model_name not in accepted_models:
            raise ValueError("model not found")
        model = YOLO(os.path.join(models_path, model_name))
    else:
        accepted_models = os.listdir(PRETRAINED_MODELS_PATH)
        if model_name not in accepted_models:
            raise ValueError("model not found")
        model = YOLO(os.path.join(PRETRAINED_MODELS_PATH, model_name))
    model.info()


    project = training_args["project"]
    project_path = os.path.join(TRAN_RESULTS_PATH, project)
    if not os.path.exists(project_path):
        os.mkdir(project_path)
    name = training_args["name"]

    training_args["data"] = os.path.join(WORKING_DIR, training_args["data"])
    training_args["project"] = project_path
    training_args["name"] = name

    # Model Name is not necessary. it is contained in the `model` object
    del training_args['model']
    # Only used to continue training from a checkpoint
    del training_args['model_dir']

    model.train(**training_args)
