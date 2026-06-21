# MeanFlow

This repository contains the code for a PyTorch implementation of ["Mean Flows for One-step Generative Modeling"](https://arxiv.org/abs/2505.13447) (Geng et al.).

There are two workflows for training and generating samples:
- locally (`train.py` + `inference.py`)
- Google Colab notebook (`meanflow.ipynb`)

Both workflows have two training options:
- 3-sample dataset: quickly overfit on three Imagenette2 training samples to ensure the MeanFlow objective/sampler work.
- Full dataset (in progress): train with the entire Imagenette2 dataset. 

## Data Preprocessing

1. Download Imagenette2 from the official fast.ai page:  
   [https://github.com/fastai/imagenette](https://github.com/fastai/imagenette)
2. Place the extracted `imagenette2` directory inside this repo's `data/` folder.
   Expected layout:
   - `data/imagenette2/train/...`
   - `data/imagenette2/val/...`
3. Run the preprocessing script to convert each raw jpeg to a cropped 64x64 tensor:
   ```bash
   python data/data.py
   ```
4. After preprocessing, training expects:
   - `data/imagenette2/train_npy/...`
   - `data/imagenette2/val_npy/...`

## Training

### Local script workflow (`train.py`)

`train.py`:

- loads train/val `.npy` datasets with `MeanFlowDataset`
- optionally switches between:
  - 3-sample overfit mode (`full_set = False`)
  - full dataset training (`full_set = True`)
- trains the UNet using the MeanFlow target constructed with JVP
- logs and saves loss curves every epoch
- writes best and final checkpoints

Important config values:

- `run`: selects output folder name `experiments/run{run}` for the current training run.
- `full_set`: toggles full-data vs tiny overfit sanity-check mode
- `epochs`, `lr`, `batch_size`: training hyperparameters
- `train_dataset_path`, `val_dataset_path`: dataset locations

Run:

```bash
python train.py
```

### Colab workflow (`meanflow.ipynb`)

`meanflow.ipynb` mirrors the local script logic, but is designed for Colab GPU usage in case you don't have local GPU access.

To use:

1. Upload the preprocessed Imagenette2 dataset to Google Drive.
2. Update dataset paths in the setup/config cells.
3. Execute cells in order.

## Inference

If you use the local training workflow, use `inference.py` to evaluate one-step generation quality against the 3 reference images. If you are using the Colab workflow, this inference step is in the very last cell. 



1. Set `run` in `inference.py` to match the trained run folder.
2. Run:
   ```bash
   python inference.py
   ```
3. Check visualization output `samples.png` in `experiments/run{run}`.

## Experiment Saving and Folder Structure

All run loss curves/checkpoints are grouped by run id:

- `experiments/run0`
- `experiments/run1`
- `experiments/run2`
- etc


Files in each run folder:

- `meanflow_best.pt`: best checkpoint by training loss
- `meanflow.pt`: last checkpoint from end of run
- `loss.png`: training/evaluation loss curves
- `samples.png`: image comparison between generated images & training images. 

## Directory Reference

- `train.py`: local training script and dataset class used by local runs
- `inference.py`: local one-step sample generation and nearest-match MSE evaluation
- `meanflow.ipynb`: Colab notebook pipeline for training and evaluation
- `unet.py`: model architecture, timestep embeddings, and `(t, r)` sampling helpers
- `utils.py`: shared utility helpers (seed, plotting, experiment directory creation)
- `data/data.py`: data preprocessing/conversion helpers for dataset preparation
- `experiments/`: generated artifacts organized by run id
