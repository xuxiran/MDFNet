# Change summary

- Document the runnable four-fold trial-disjoint LTO path, the 0.5-second KUL/DTU subject-holdout LOSO protocol with complete subject-macro aggregation, and the 1-second Fig. 2 conditions across four folds.
- Add Huairou Slurm examples for LOSO and Fig. 2 training, plus the offline LOSO aggregation command.
- Record the 128 Hz data contract, full in-memory/device residency requirement, single seed (2025), and per-run manifest, epoch history, and best-checkpoint outputs.
- Move each MDFNet/ablation inner model to the selected device before creating its optimizer, so GPU training updates the model's active parameters.
- State that paper results remain frozen historical numbers: no real-data rerun or exact bitwise reproduction of exp041/042 under the current cuDNN settings is claimed. No baseline reproduction is claimed.
- Keep third-party ListenNet, MHANet, DARNet, and DBPNet out of the release because redistribution permission is unconfirmed; link to official upstreams and direct readers to check licenses and adapt to matching splits. CNN, XANet, and DenseNet are not bundled; STANet is not in the paper's seven-baseline set.
- Add no blanket repository license.
