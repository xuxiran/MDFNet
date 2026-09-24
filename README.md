# MDFNet

Multi-Dimension Fusion of EEG Features for Auditory Spatial Attention Decoding.

This release provides MDFNet, the in-repository ablations `Nofre`, `Notem`, and `Nocat`, and seven paper baselines: `CNN`, `XANet`, `DenseNet`, `DARNet`, `DBPNet`, `ListenNet`, and `MHANet`. All seven baselines are selectable through lazy model imports in the LTO/LOSO runner. DARNet and DBPNet use MNE CSP preprocessing fitted on training data only. The bundled implementations have received synthetic or structural validation; this release has not rerun the real-data experiments or established equivalence with historical baseline scores. It documents three protocols: four-fold trial-disjoint LTO (`main.py`), 0.5-second subject-holdout LOSO (`run_loso.py`), and the 1-second Fig. 2 conditions (`analysis/run_fig2_experiment.py`). The paper's reported numbers are frozen historical results. The current cuDNN settings (`deterministic=False`, `benchmark=True`) do not provide exact bitwise reproduction of exp041/042.

## Installation and data

Use Python 3.10 or later and install a PyTorch build suitable for the target device:

```bash
pip install -r requirements.txt
python main.py --help
python run_loso.py --help
python analysis/run_fig2_experiment.py --help
```

Data are not included. Obtain KUL/DTU through their original distribution channels and place MATLAB v7.3 (HDF5) files named `KUL_1D.mat` and `DTU_1D.mat` in a data directory. After reversing MATLAB/HDF5 dimensions, `EEG` must have shape `[subjects, trials, time_samples, 64]`; `ENV` must have shape `[subjects, trials, time_samples]`. EEG values must be finite, and each trial must have one binary label. The input contract is preprocessed EEG sampled at 128 Hz; this package does not convert raw recordings or perform upstream 1–50 Hz preprocessing. Channel order must match the 64-channel mapping in `AADdataset.py`.

All required input/features are loaded into memory and made resident on the selected device before training; there is no memory-mapped or streaming fallback. Check available host and GPU memory before submitting. Training uses one seed, 2025. Run outputs include a manifest with data/configuration provenance, per-epoch history, and the best validation checkpoint (`best.ckpt`). Existing nonempty run directories are not overwritten.

## LTO: trial-disjoint

The `--model` choices supported by the LTO/LOSO runner are `MDFNet`, `Nofre`, `Notem`, `Nocat`, `CNN`, `XANet`, `DenseNet`, `DARNet`, `DBPNet`, `ListenNet`, and `MHANet`. `MDFNet` is the proposed model; `Nofre`, `Notem`, and `Nocat` are its ablations. The `BASE` alias is also accepted for MDFNet. DARNet and DBPNet fit their CSP spatial filters using training data only; the fitted filters are then applied to validation and test data.

At 128 Hz, windows of 64, 128, and 256 samples are 0.5, 1, and 2 seconds. The runner alternates balanced classes in stable order and assigns one quarter of trials to test, the next quarter to validation, and half to training for the selected fold.

```bash
python main.py --dataset KUL --model MDFNet --data-dir /path/to/data --decision-window 64 --sbfold 0 --seed 2025
```

On Huairou, submit the same arguments with `sbatch scripts/train.slurm --dataset KUL --model MDFNet --data-dir /path/to/data --decision-window 64 --sbfold 0 --seed 2025` instead of running Python directly.

Run folds `0` through `3` separately. Aggregate a completed set of four fold runs with:

```bash
python get_final_res.py --run-root outputs/KUL/MDFNet/dw64/seed2025
```

## LOSO: 0.5-second subject holdout

LOSO uses 64-sample windows. KUL requires 16 subjects with 8 trials each; DTU requires 21 subjects with 32 trials each. For each held-out subject, training/validation use the other subjects' fixed source trial split; all trials of the held-out subject are tested. Submit every held-out subject as a separate Slurm job on Huairou:

```bash
sbatch scripts/loso.slurm --dataset KUL --model MDFNet --data-dir /path/to/data --test-subject 0 --output-dir outputs_loso --seed 2025
```

Repeat with `--test-subject 0` through `15` for KUL or `0` through `20` for DTU. After all jobs finish, aggregate offline; aggregation requires the complete subject set and reports the mean and sample standard deviation across subject accuracies:

```bash
python run_loso.py --dataset KUL --model MDFNet --output-dir outputs_loso --seed 2025 --aggregate
```

## Fig. 2 conditions

`analysis/run_fig2_experiment.py` retrains 1-second (128-sample) models for four trial-disjoint folds. Conditions are `All`, five overlapping region masks (`region_F`, `region_C`, `region_P`, `region_T`, `region_O`), and five single-band inputs (`band_delta`, `band_theta`, `band_alpha`, `band_beta`, `band_gamma`). `All` uses all 64 channels and all five bands; each region uses its listed channels across all bands, and each single-band condition uses all channels. Masks are in `configs/regions.json`.

Submit one condition/fold job per invocation on Huairou. For example:

```bash
sbatch --export=ALL,DATASET=KUL,CONDITION=All,FOLD=0,DATA_DIR=/path/to/data,OUTPUT_DIR=outputs_fig2 scripts/fig2.slurm
```

Run all 11 conditions for folds `0` through `3` (44 jobs per dataset). Outputs include condition masks, split/order information and hashes in `manifest.json`, epoch history, best checkpoint, predictions, and test metrics. Huairou training must use Slurm; `--allow-local` is for non-Huairou hosts. Diagnostic `--smoke --epochs 2` runs are not paper-protocol results.

After all 44 jobs for each dataset complete (88 total), aggregate the new run metrics and write a CSV plus a regenerated two-panel figure to a separate output directory:

```bash
python analysis/aggregate_fig2.py --input-dir outputs_fig2 --output-dir reports/fig2_rerun
```

Aggregation requires all four folds for all 11 conditions on both KUL and DTU, with matching manifests for the full 50-epoch protocol; diagnostic `--smoke` manifests are rejected. The generated figure summarizes these new reruns; it is not the frozen historical Fig. 2 or its underlying historical CSV. Fig. 2 uses 128-sample trial windows, so each trial length must be divisible by 128; masked channels and bands are zero-filled while retaining the same 30,534-parameter architecture. On Huairou, submit every training run through Slurm.



### Baseline implementations and attribution

The bundled baselines draw on the [KU Leuven auditory-attention CNN](https://github.com/exporl/locus-of-auditory-attention-cnn), [ASAD-DenseNet](https://github.com/xuxiran/ASAD_DenseNet), XANet, [DARNet](https://github.com/fchest/DARNet), [DBPNet](https://github.com/fchest/DBPNet), [ListenNet](https://github.com/fchest/ListenNet), and [MHANet](https://github.com/fchest/MHANet). `BASEmodels/DenseNet.py` adapts the ASAD-DenseNet 3D model to this release's API and uses the broadband topographic waveform `X[2]` with shape `[batch, time, 10, 11]`; its upstream copyright and MIT terms are retained in `LICENSES/ASAD_DenseNet_LICENSE`. Consult each upstream project's license and attribution terms before reusing its code; the DenseNet license does not apply to other baselines.

No historical baseline score equivalence is claimed for any bundled baseline implementation.

See `CHANGES.md` for the release summary.
