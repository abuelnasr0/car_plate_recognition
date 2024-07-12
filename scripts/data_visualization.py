import os, random, dataclasses, PIL
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATASET_DIR = os.getenv("DATASET_DIR")

@dataclasses.dataclass
class BoundingBox:
    cls: int
    x: float
    y: float
    w: float
    h:float


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
    "car plate": "car plate",
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

def select_images(dir):
    labels_dir = os.path.join(dir, LABELS)
    images_names = random.choices(os.listdir(labels_dir), k=NUM_EXAMPLES) 
    images_names = [name[:-4] for name in images_names]
    return images_names

def get_image_bbxs(image_name):
    img_clss = []
    with open(os.path.join(dir, LABELS, image_name+TEXT_EXT), "r") as f:
        for line in f.readlines():
            cls = line[:-1].split(" ")
            cls = cls[:1] + [float(num)*640 for num in cls[1:]]
            cls[0] = int(cls[0])
            cls[1] = cls[1] - cls[3]/2
            cls[2] = cls[2] - cls[4]/2
            img_clss.append(BoundingBox(*cls))
    return img_clss

def get_plate_num(bbxs):
    num=""
    bbxs = sorted(bbxs, key=lambda x : x.x)
    for bbx in bbxs:
        if bbx.cls == 11:
            continue
        num+=CLASSES_AR[bbx.cls]
    return num

if __name__ == "__main__":
    train_dir = DATASET_DIR+"/train"
    test_dir = DATASET_DIR+"/test"
    valid_dir = DATASET_DIR+"/valid"

    images_dirs = [train_dir, test_dir, valid_dir]

    fig, axis = plt.subplots(3, NUM_EXAMPLES, figsize=(10, 7))
    for i, ax in enumerate(axis[:,0]):
        text = "Train" if i==0 else ""
        text = "Test" if i==1 else text
        text = "Validation" if i==2 else text
        ax.set_ylabel(text)

    for i, dir in enumerate(images_dirs):
        images_names = select_images(dir)
        for j, image_name in enumerate(images_names):
            img = np.asarray(PIL.Image.open(os.path.join(dir, IMAGES, image_name+IMAGES_EXT)))
            img_bbxs = get_image_bbxs(image_name)
            
            axis[i, j].axes.get_xaxis().set_ticks([])
            axis[i, j].axes.get_yaxis().set_ticks([])

            axis[i, j].imshow(img)
            for bbx in img_bbxs:
                rect = patches.Rectangle((bbx.x, bbx.y), bbx.w, bbx.h, linewidth=1, edgecolor='r', facecolor='none')
                axis[i, j].add_patch(rect)
            axis[i, j].set_title(get_plate_num(img_bbxs))


    plt.savefig("./images/data_examples.png", bbox_inches='tight')