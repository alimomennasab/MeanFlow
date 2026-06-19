import os
import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close()


def create_experiment_dir(experiment_num: int) -> str:
    save_path = os.path.join("experiments", f"run{experiment_num}")
    os.makedirs(save_path, exist_ok=True)
    return save_path