# Data exploration of imagenette dataset, and convert all jpegs to npy for faster training.
# Available 10 classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

train_path = "imagenette2/train"
train_npy_path = "imagenette2/train_npy"

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
])

def load_img(path, dim=64):
    img = Image.open(path)
    img = transform(img)
    return np.array(img, dtype=np.uint8)

def save_train_npy():
    """Preprocess Imagenette train images and save them as numpy arrays.

    Walks every class folder under ``train_path``, loads each JPEG with PIL,
    applies ``Resize(64)`` + ``CenterCrop(64)``, and writes the result to
    ``train_npy/{class_name}_npy/{filename}.npy``.

    Each saved array has shape ``(64, 64, 3)`` and dtype ``uint8``.
    """
    os.makedirs(train_npy_path, exist_ok=True)

    for class_name in sorted(os.listdir(train_path)):
        class_dir = os.path.join(train_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        out_dir = os.path.join(train_npy_path, f"{class_name}_npy")
        os.makedirs(out_dir, exist_ok=True)

        filenames = [
            filename for filename in sorted(os.listdir(class_dir))
            if filename.lower().endswith((".jpeg", ".jpg"))
        ]

        for filename in tqdm(filenames, desc=class_name):
            img_path = os.path.join(class_dir, filename)
            npy_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".npy")
            np.save(npy_path, load_img(img_path))

if __name__ == "__main__":
    # visualize some samples
    img1_path = os.path.join(train_path, "n02102040/ILSVRC2012_val_00008334.JPEG") # dog
    img2_path = os.path.join(train_path, "n03445777/ILSVRC2012_val_00002314.JPEG") # golf ball
    img3_path = os.path.join(train_path, "n01440764/ILSVRC2012_val_00009346.JPEG") # tench (fish)
    img1 = load_img(img1_path)
    img2 = load_img(img2_path)
    img3 = load_img(img3_path)

    print("img1 shape: ", img1.shape)
    print("img2 shape: ", img2.shape)
    print("img3 shape: ", img3.shape)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img in zip(axes, [img1, img2, img3]):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # convert jpegs to augmented npys 
    save_train_npy()

