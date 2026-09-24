# Change summary

- Document the runnable four-fold trial-disjoint LTO path, the 0.5-second KUL/DTU subject-holdout LOSO protocol with complete subject-macro aggregation, and the 1-second Fig. 2 conditions across four folds.
- Add Huairou Slurm examples for LOSO and Fig. 2 training, plus the offline LOSO aggregation command.
- Add strict offline Fig. 2 aggregation for complete four-fold results across all 11 conditions and both datasets, with CSV and regenerated plot outputs.
- Record the 128 Hz data contract, full in-memory/device residency requirement, single seed (2025), and per-run manifest, epoch history, and best-checkpoint outputs.
- Move each MDFNet/ablation inner model to the selected device before creating its optimizer, so GPU training updates the model's active parameters.
- State that paper results remain frozen historical numbers: no real-data rerun or exact bitwise reproduction of exp041/042 under the current cuDNN settings is claimed. No baseline reproduction is claimed.
- Document the supported LTO/LOSO model choices: MDFNet, its `Nofre`/`Notem`/`Nocat` ablations, and all seven paper baselines (CNN, XANet, DenseNet, DARNet, DBPNet, ListenNet, and MHANet); clarify that `BASE` aliases MDFNet and that model imports are lazy.
- Include provenance links and attribution guidance for all seven bundled baseline implementations; retain the upstream MIT terms for the ASAD-DenseNet adapter.
- Document MNE CSP fitting on training data only for DARNet and DBPNet. Record that baselines received synthetic or structural validation only, with no real-data rerun or historical score equivalence claimed.
