"""Run one-step MeanFlow sampling repeatedly, compute their MSE difference from 3 training samples, and display the 3
closest generations. 
"""

import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from train import MeanFlowDataset
from unet import UNet

run = 7
batch_size = 3
seed = 42
num_generated = 300
run_dir = os.path.join("experiments", f"run{run}")
checkpoint_path = os.path.join(run_dir, "meanflow.pt")


def set_seed(seed_value: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_generator = torch.Generator(device=device).manual_seed(seed + 1)

train_dataset = MeanFlowDataset("data/imagenette2/train_npy/")
train_subset = Subset(train_dataset, [0, 1, 2])
small_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)

model = UNet(in_ch=3, ch=(64, 128, 256, 512), d_emb=256)
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

x, labels = next(iter(small_train_loader))
x = x.to(device)

with torch.no_grad():
    generated_batches = []
    for start in tqdm(range(0, num_generated, batch_size)):
        current_batch_size = min(batch_size, num_generated - start)

        # One-step generation from MeanFlow: x_gen = e - u(e, r=0, t=1).
        e = torch.randn(current_batch_size, device=device)
        r = torch.zeros(current_batch_size, device=device)
        t = torch.ones(current_batch_size, device=device)
        x_gen = e - model(e, r, t) # r = 0, t = 1
        x_gen.clamp(0.0, 1.0) # normalize to 0, 1
        generated_batches.append(x_gen.cpu())

gt = x.detach().cpu()
all_generated = torch.cat(generated_batches, dim=0)

# For each ground-truth image, find the closest generated sample by pixel MSE
mse_matrix = ((gt[:, None] - all_generated[None, :]) ** 2).mean(dim=(2, 3, 4))
best_mse, best_idx = mse_matrix.min(dim=1)

gt_images = gt.permute(0, 2, 3, 1).numpy()
best_images = all_generated[best_idx].permute(0, 2, 3, 1).numpy()

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for i, label in enumerate(labels.tolist()):
    axes[0, i].imshow(gt_images[i])
    axes[0, i].set_title(f"GT (label {label})")
    axes[0, i].axis("off")

    axes[1, i].imshow(best_images[i])
    axes[1, i].set_title(f"Best sample MSE={best_mse[i].item():.5f}")
    axes[1, i].axis("off")

plt.tight_layout()
save_path = os.path.join(run_dir, "samples.png")
plt.savefig(save_path)
plt.show()
print(f"Saved samples to {save_path}")

# Print nearest reconstruction quality for each target image
for i, mse in enumerate(best_mse.tolist()):
    print(f"Image {i} best reconstruction MSE over {num_generated} samples: {mse:.8f}")
 