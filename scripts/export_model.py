import os, argparse
from ultralytics import YOLO
from sklearn.model_selection import ParameterGrid


WORKING_DIR = os.getcwd()

TRAN_RESULTS_PATH = os.path.join(WORKING_DIR, "train_results/")


export_args = [
    {
        "format": "torchscript",
        "kwargs_list": [
            { 
                "optimize": [True, False]
            },
        ]
    },
    {
        "format": "engine",
        "kwargs_list": [
            { 
                "half": [True, False],
                "simplify": [True, False]
            },
            { 
                "int8": [True],
                "simplify": [True, False]
            },
        ]
    },
    {
        "format": "tflite",
        "kwargs_list": [
            { 
                "half": [True, False],
            },
            { 
                "int8": [True],
            },
        ]
    },
    {
        "format": "onnx",
        "kwargs_list": [
            { 
                "half": [True, False],
                "simplify": [True, False]
            },
        ]
    },
]

if __name__ == "__main__":
   
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
        help=(
            "Project name"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        help=(
            "Training run name"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help=(
            "Which model to export."
            " can be found inside `train_results/<PROJECT>/<NAME>/weights"
        ),
    )

    args = parser.parse_args()

    project = args.project
    name = args.name
    model_name = args.model

    model = YOLO(os.path.join(TRAN_RESULTS_PATH, project, name, "weights", model_name))

    for export_arg in export_args:
        format = export_arg["format"]
        for export_args in export_arg["kwargs_list"]:
            kwargs_list = list(ParameterGrid(export_args))
            for kwargs in kwargs_list:
                kwargs["format"] = format
                model.export(**kwargs)

