import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def sample_t_r(batch_size, percent_unequal=0.25):
    # percent_unequal: fraction of batch where r != t
    t = torch.rand(batch_size)
    r = torch.rand(batch_size) * t  # r in [0, t], so r <= t

    num_unequal = int(percent_unequal * batch_size)

    unequal_mask = torch.zeros(batch_size, dtype=torch.bool)
    unequal_indices = torch.randperm(batch_size)[:num_unequal] # randomly choose percent_unequal% of (r, t) to be unequal
    unequal_mask[unequal_indices] = True

    r = torch.where(unequal_mask, r, t)

    return t, r

class MeanFlowDataset(Dataset):
    """Load preprocessed 64x64x3 images from ``train_npy/{class}_npy/*.npy``."""

    def __init__(self, data_dir: str):
        class_dirs = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.endswith("_npy")
        )
        self.samples = [
            (os.path.join(data_dir, class_dir, filename), class_idx)
            for class_idx, class_dir in enumerate(class_dirs)
            for filename in sorted(os.listdir(os.path.join(data_dir, class_dir)))
            if filename.endswith(".npy")
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = np.load(path)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, label


if __name__ == "__main__":
    dataset_path = "data/imagenette2/train_npy/"
    train_dataset = MeanFlowDataset(dataset_path)
    print(f"num samples: {len(train_dataset)}")

    image, label = train_dataset[0]
    print(f"sample shape: {image.shape}, label: {label}")

    # subset of 3 samples
    #train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    subset = torch.utils.data.Subset(train_dataset, [0, 1, 2])
    small_train_loader = DataLoader(subset, batch_size=3, shuffle=False)

    # training loop
    t, r = sample_t_r(batch_size=8)
    print(t)
    print(r)
