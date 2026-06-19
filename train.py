import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from unet import UNet, sample_t_r
from utils import plot_loss, create_experiment_dir

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
    # hyperparams & config
    run = 0
    epochs = 20
    batch_size = 3
    loss_fn = nn.MSELoss()
    optimizer = 

    # dataset, dataloader, save path
    save_path = create_experiment_dir(run)
    train_dataset_path = "data/imagenette2/train_npy/"
    val_dataset_path = "data/imagenette2/val_npy/"
    train_dataset = MeanFlowDataset(train_dataset_path)
    val_data = MeanFlowDataset(val_dataset_path)
    print(f"num train samples: {len(train_dataset)}")
    print(f"num val samples: " {len(val_dataset)})

    image, label = train_dataset[0]
    print(f"sample shape: {image.shape}, label: {label}")

    # subset of 3 samples
    #train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    train_subset = torch.utils.data.Subset(train_dataset, [0, 1, 2])
    val_subset = torch.utils.data.Subset(val_dataset, [0, 1, 2])
    small_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    small_val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # initialize model & optimizer
    model = UNet(in_ch=3, ch=(64, 128, 256, 512), d_emb=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)


    # training loop
    train_start_time = time.Time()
    train_losses, val_losses = [], []

    for i in range(epochs):
        model.train()
        epoch_start_time = time.Time()
        epoch_train_loss = 0

        for images, labels in small_train_loader: 
            optimizer.zero_grad()
            t, r = sample_t_r(batch_size=8)
            print("t: ", t)
            print("r: ", r)

            e = torch.randn_like(x)
            z = (1 - t) * x + t * e
            v = e - x
            fn = model(images, r, t)
            u, dudt = torch.func.jvp(fn, (z, r, t), (v, 0, 1))
            u_tgt = v - (t - r) * dudt
            train_loss = loss_fn(u, u_tgt.detatch()) #  error = u - stopgrad(u_tgt)
            train_loss.backwards()
            optimizer.step()
            epoch_train_loss += train_loss.item()

        train_losses.append(epoch_train_loss / len(small_train_loader))

        model.eval()
        epoch_val_loss = 0
        for images, labels in small_val_loader:
            t, r = sample_t_r(batch_size=8)
            print("t: ", t)
            print("r: ", r)

            e = torch.randn_like(x)
            z = (1 - t) * x + t * e
            v = e - x
            fn = model(images, r, t)
            u, dudt = torch.func.jvp(fn, (z, r, t), (v, 0, 1))
            u_tgt = v - (t - r) * dudt
            val_loss = loss_fn(u, u_tgt.detatch()) #  error = u - stopgrad(u_tgt)
            epoch_val_loss += val_loss.item()
        
        val_losses.append(epoch_val_loss / len(small_val_loader))

        plot_loss(train_loss, val_loss, save_path)
        print(f"Epoch {i+1}: {time.Time() - epoch_start_time}. Train loss: {train_losses[-1]}, Val loss: {val_losses[-1]}") 

    torch.save(model.state_dict(), os.path.join(save_path, "meanflow.pt"))
    print("Total training time: ", time.Time() - train_start_time)
