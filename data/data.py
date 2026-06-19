# Data exploration of imagenette dataset
# Available 10 classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

train_path = "imagenette2/train"
img1_path = os.path.join(train_path, "n02102040/ILSVRC2012_val_00008334.JPEG") # dog
img2_path = os.path.join(train_path, "n03445777/ILSVRC2012_val_00002314.JPEG") # golf ball
img3_path = os.path.join(train_path, "n01440764/ILSVRC2012_val_00009346.JPEG") # tench (fish)

def load_img(path, dim=64):
    img = Image.open(path)
    img = img.resize((dim, dim))
    img = np.array(img)
    return img

if __name__ == "__main__":
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

