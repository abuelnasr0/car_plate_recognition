import os 
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATASET_DIR = os.getenv("DATASET_DIR")

CLASSES = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "a": "أ",
    "b": "ب",
    "car plate": "P",
    "d": "د",
    "en": "ع",
    "f": "ف",
    "g": "ج",
    "h": "ه",
    "k": "ك",
    "l": "ل",
    "mem": "م",
    "non": "ن",
    "q": "ق",
    "r": "ر",
    "sad": "ص",
    "sen": "س",
    "t": "ط",
    "w": "و",
    "y": "ي",
}
CLASSES_AR = list(CLASSES.values())
CLASSES_EN = list(CLASSES.keys())

NUM_EXAMPLES=5

IMAGES="images"
LABELS="labels"

IMAGES_EXT=".jpg"
TEXT_EXT=".txt"

"""
Visualize random car plates and thier numbers from train, test, valid.
"""

def get_images_labels_files(dir):
    labels_dir = os.path.join(dir, LABELS)
    images_names = os.listdir(labels_dir)
    return images_names

def get_image_classes(labels_file):
    img_clss = []
    with open(os.path.join(dir, LABELS, labels_file), "r") as f:
        for line in f.readlines():
            cls = line.split(" ")[0]
            img_clss.append(cls)
    return img_clss

if __name__ == "__main__":
    train_dir = DATASET_DIR+"/train"
    test_dir = DATASET_DIR+"/test"
    valid_dir = DATASET_DIR+"/valid"

    images_dirs = [train_dir, test_dir, valid_dir]

    fig, axis = plt.subplots(4, 1, figsize=(10, 10))
    for i, ax in enumerate(axis[:-1]):
        text = "Train" if i==0 else ""
        text = "Test" if i==1 else text
        text = "Validation" if i==2 else text
        ax.set_ylabel(text)
    axis[-1].set_ylabel("ALL")

    labels_counts = [dict() for _ in range(3)]
    for i, dir in enumerate(images_dirs):
        labels_files = get_images_labels_files(dir)
        for j, labels_file in enumerate(labels_files):
            img_clss = get_image_classes(labels_file)
            for img_cls in img_clss:
                label_count = labels_counts[i].get(img_cls, None)
                labels_counts[i][img_cls] = 0 if label_count == None else label_count+1

        labels_counts[i] = dict(sorted(labels_counts[i].items(), key=lambda x: x[1], reverse=True))
        labels_counts_keys = [CLASSES_AR[int(x)] for x in labels_counts[i].keys()]
        axis[i].bar(labels_counts_keys, labels_counts[i].values())
    
    # All data histogram
    all_data_dict = pd.DataFrame.from_records(labels_counts).sum().to_dict()
    labels_counts_keys = [CLASSES_AR[int(x)] for x in all_data_dict.keys()]
    axis[3].bar(labels_counts_keys, all_data_dict.values())

    plt.savefig("./images/labels_count.png", bbox_inches='tight')