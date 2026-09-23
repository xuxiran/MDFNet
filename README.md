# MDFNet

Multi-Dimension Fusion of EEG Features for Auditory Spatial Attention Decoding.

This repository contains the MDFNet implementation and its three in-repository ablation variants (`Nofre`, `Notem`, and `Nocat`). Third-party baseline source code is not bundled. Official repositories for excluded baselines: [ListenNet](https://github.com/fchest/ListenNet), [MHANet](https://github.com/fchest/MHANet), [DARNet](https://github.com/fchest/DARNet), and [DBPNet](https://github.com/fchest/DBPNet). This release contains none of their source code. The source for CNN, STANet, XANet, and DenseNet is also not bundled.

## Scope

The training entry point implements the source package's four-fold trial-disjoint protocol: for each subject it stably alternates class trials, assigns one quarter to test, the next quarter to validation, and the remaining half to training. This is the LTO path in this code. It does not implement LOSO. It also does not implement region-specific frequency-band variants. Do not interpret these files as the complete code for paper results that use those settings. This release implements LTO only; reported results for unsupported protocols require their corresponding implementation and provenance.

## Installation

Use Python 3.10 or later, install a PyTorch build suitable for your device, then:

```bash
pip install -r requirements.txt
python main.py --help
```

## Data

Data are not included. Obtain KUL/DTU from their original distribution channels and prepare MATLAB v7.3 (HDF5) files named `KUL_1D.mat` and `DTU_1D.mat` in the data directory. Inputs are preprocessed EEG sampled at 128 Hz; this package does not convert raw recordings or perform upstream 1–50 Hz preprocessing.

After the loader reverses MATLAB/HDF5 dimensions, `EEG` must have shape `[subjects, trials, time_samples, 64]` and finite values. `ENV` must have shape `[subjects, trials, time_samples]`, contain binary labels constant within each trial, and have balanced classes per subject. Trial counts must be divisible by four. Channel order must match the 64-channel mapping in `AADdataset.py`.

The loader filters complete trials into five bands (1–4, 4–7, 8–13, 14–29, and 30–47 Hz), maps channels onto a 10×11 electrode grid, and discards trailing samples that do not form a complete decision window. Frequency processing here is the MDFNet input pipeline; there are no region-specific frequency-band model variants.

## Train and evaluate

```bash
python main.py --dataset KUL --model MDFNet --data-dir /path/to/data --decision-window 64 --sbfold 0 --seed 2025
```

Supported models are `MDFNet`, `Nofre`, `Notem`, and `Nocat`. Decision windows of 64, 128, and 256 samples correspond to 0.5, 1, and 2 seconds at 128 Hz. Run folds 0–3 separately with one seed; the default seed is 2025. The default optimizer is Adam with learning rate 0.001, weight decay 0.01, 50 epochs, and batch size 128.

All feature tensors are prepared and resident on the selected device before training. There is no batch-wise or memory-mapped loading fallback. `--device auto` chooses `cuda:0` when available; selected data and training activations must fit. Every epoch prints its validation accuracy. Outputs are isolated by dataset, model, window, seed, and fold and existing nonempty run directories are not overwritten.

To combine four folds:

```bash
python get_final_res.py --run-root outputs/KUL/MDFNet/dw64/seed2025
```

## Slurm

On Huairou, submit training through Slurm:

```bash
sbatch scripts/train.slurm --dataset KUL --model MDFNet --data-dir /path/to/data --decision-window 64 --sbfold 0 --seed 2025
```

The example requests one A800 and seven CPU cores. Adapt the allocation directives to the target cluster. The training implementation does not distribute resident inputs across multiple GPUs.

## Synthetic smoke check

`tests/smoke.py` performs CPU-only import, model construction, and synthetic forward checks. It does not load real EEG data, run training, or validate paper accuracy. No completed validation record is bundled.

See `CHANGES.md` for the source-scope summary. No repository license is included. Confirm redistribution rights for the retained source and dependencies before redistributing this repository.
