import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from unet import UNet

run = 0
run_dir = os.path.join("experiments", f"run{run}")
checkpoint_path = os.path.join(run_dir, "meanflow.pt")

gt_paths = [
    "data/imagenette2/train_npy/n02102040_npy/ILSVRC2012_val_00008334.npy",  # dog
    "data/imagenette2/train_npy/n03445777_npy/ILSVRC2012_val_00002314.npy",  # golf ball
    "data/imagenette2/train_npy/n01440764_npy/ILSVRC2012_val_00009346.npy",  # tench
]
titles = ["dog", "golf ball", "tench"]

model = UNet(in_ch=3, ch=(64, 128, 256, 512), d_emb=256)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
model.eval()

def load_gt(path):
    img = np.load(path).astype(np.float32) / 255.0
    return img

def sample(model, num_samples=3):
    with torch.no_grad():
        e = torch.randn(num_samples, 3, 64, 64)
        r = torch.zeros(num_samples)
        t = torch.ones(num_samples)
        x = e - model(e, r, t)
    return x.clamp(0, 1).permute(0, 2, 3, 1).numpy()

gt_images = [load_gt(path) for path in gt_paths]
sampled_images = sample(model, num_samples=3)

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for i, title in enumerate(titles):
    axes[0, i].imshow(gt_images[i])
    axes[0, i].set_title(f"GT: {title}")
    axes[0, i].axis("off")

    axes[1, i].imshow(sampled_images[i])
    axes[1, i].set_title(f"Sample: {title}")
    axes[1, i].axis("off")

plt.tight_layout()
save_path = os.path.join(run_dir, "samples.png")
plt.savefig(save_path)
plt.show()
print(f"Saved samples to {save_path}")
