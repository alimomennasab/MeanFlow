# Data exploration of imagenette dataset, and convert all jpegs to npy for faster training.
# Available 10 classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
])

def load_img(path, dim=64):
    img = Image.open(path)
    img = transform(img)
    return np.array(img, dtype=np.uint8)

def convert_jpeg_to_npy(dir, output_dir):
    """Preprocess Imagenette images and save them as numpy arrays.

    Walks every class folder under ``train_path``, loads each JPEG with PIL,
    applies ``Resize(64)`` + ``CenterCrop(64)``, and writes the result to
    ``train_npy/{class_name}_npy/{filename}.npy``.

    Each saved array has shape ``(64, 64, 3)`` and dtype ``uint8``.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for class_name in sorted(os.listdir(dir)):
        class_dir = os.path.join(dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        out_dir = os.path.join(output_dir, f"{class_name}_npy")
        os.makedirs(out_dir, exist_ok=True)

        filenames = [
            filename for filename in sorted(os.listdir(class_dir))
            if filename.lower().endswith((".jpeg", ".jpg"))
        ]

        for filename in tqdm(filenames, desc=class_name):
            img_path = os.path.join(class_dir, filename)
            npy_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".npy")
            np.save(npy_path, load_img(img_path))
            saved_paths.append(npy_path)

    # plot some converted samples
    samples = saved_paths[:3]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, path in zip(axes, samples):
        ax.imshow(np.load(path))
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # visualize some samples'
    train_path = "imagenette2/train"
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
    train_path = "imagenette2/train"
    val_path = "imagenette2/val"
    train_npy_path = "imagenette2/train_npy"
    val_npy_path = "imagenette2/val_npy"

    #convert_jpeg_to_npy(train_path, train_npy_path)
    convert_jpeg_to_npy(val_path, val_npy_path)

