import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # hyperparams & config
    run = 6
    seed = 42
    full_set = False
    epochs = 10000
    lr = 1e-3
    batch_size = 3
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # dataset, dataloader, save path
    save_path = create_experiment_dir(run)
    train_dataset_path = "data/imagenette2/train_npy/"
    val_dataset_path = "data/imagenette2/val_npy/"
    train_dataset = MeanFlowDataset(train_dataset_path)
    val_dataset = MeanFlowDataset(val_dataset_path)
    val_dataset = MeanFlowDataset(val_dataset_path)
    print(f"num train samples: {len(train_dataset)}")
    print(f"num val samples: {len(val_dataset)}")

    image, label = train_dataset[0]
    print(f"sample shape: {image.shape}, label: {label}")

    # choose either small dataset (3 samples) or full dataset
    if full_set:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = torch.utils.data.Subset(train_dataset, [0, 1, 2])
        val_subset = torch.utils.data.Subset(val_dataset, [0, 1, 2])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # initialize model, optimizer, and scheduler
    model = UNet(in_ch=3, ch=(64, 128, 256, 512), d_emb=256)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # training loop
    train_start_time = time.time()
    train_losses, val_losses = [], []
    best_train_loss = float('inf')

    print(f"Training experiment {run}")
    for i in range(epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_train_loss = 0

        for x, y in train_loader: 
            x = x.to(device)
            optimizer.zero_grad()
            t, r = sample_t_r(x.shape[0])
            t = t.to(device)
            r = r.to(device)

            # reshape t and r for image operations
            t_reshape = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1) 
            r_reshape = r.view(-1, 1, 1, 1)  # (B, 1, 1, 1) 

            e = torch.randn_like(x)
            z = (1 - t_reshape) * x + t_reshape * e
            v = e - x

            u, dudt = torch.func.jvp(
                model.u_fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t))
            )
            u_tgt = v - (t_reshape - r_reshape) * dudt
            train_loss = loss_fn(u, u_tgt.detach())
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        for x, y in val_loader:
            x = x.to(device)
            t, r = sample_t_r(x.shape[0])
            t = t.to(device)
            r = r.to(device)
            t_reshape = t.view(-1, 1, 1, 1)
            r_reshape = r.view(-1, 1, 1, 1)

            e = torch.randn_like(x)
            z = (1 - t_reshape) * x + t_reshape * e
            v = e - x

            with torch.no_grad():
                u, dudt = torch.func.jvp(
                    model.u_fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t))
                )
            u_tgt = v - (t_reshape - r_reshape) * dudt
            val_loss = loss_fn(u, u_tgt.detach())
            epoch_val_loss += val_loss.item()
        
        val_losses.append(epoch_val_loss / len(val_loader))
        scheduler.step()

        plot_loss(train_losses, val_losses, save_path)
        print(f"Epoch {i+1}: {(time.time() - epoch_start_time) / 60:.2f} min. Train loss: {train_losses[-1]}, Val loss: {val_losses[-1]}")

        # save best checkpoiint (based on train loss for overfit task)
        if train_losses[-1] < best_train_loss:
            best_train_loss = train_losses[-1]
            torch.save(model.state_dict(), os.path.join(save_path, "meanflow_best.pt"))

    torch.save(model.state_dict(), os.path.join(save_path, "meanflow.pt"))
    print(f"Total training time: {(time.time() - train_start_time) / 60:.2f} min")
