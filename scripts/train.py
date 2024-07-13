from ultralytics import YOLO

import argparse
import os

import yaml


WORKING_DIR = os.getcwd()

PRETRAINED_MODELS_PATH = os.path.join(WORKING_DIR, "models/pretrained/")
CHECKPOINT_MODELS_PATH = os.path.join(WORKING_DIR, "train_results/")


if __name__ == "__main__":
    with open(os.path.join(WORKING_DIR, "training_args.yaml")) as stream:
        try:
            training_args = yaml.safe_load(stream)
            print(training_args)
        except yaml.YAMLError as exc:
            print(exc)

    model_name = training_args["model"]
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cont",
        default=False,
        type=bool,
        help=(
            "Whether to cintinue training from a preset or not"
        ),
    )

    args = parser.parse_args()
    continue_training = args.cont

    if continue_training:
        models_path = os.path.join(CHECKPOINT_MODELS_PATH, training_args["train_dir"], "weights")
        accepted_models = os.listdir(models_path)
        if model_name not in accepted_models:
            raise ValueError("model not found")
        model = YOLO(os.path.join(models_path, model_name))
    else:
        accepted_models = os.listdir(PRETRAINED_MODELS_PATH)
        if model_name not in accepted_models:
            raise ValueError("model not found")
        print(os.path.join(PRETRAINED_MODELS_PATH, model_name))
        model = YOLO(os.path.join(PRETRAINED_MODELS_PATH, model_name))
    model.info()

    training_args["data"] = os.path.join(WORKING_DIR, training_args["data"])
    training_args["project"] = os.path.join(WORKING_DIR, "train_results")

    del training_args['model']
    del training_args['train_dir']

    model.train(**training_args)
