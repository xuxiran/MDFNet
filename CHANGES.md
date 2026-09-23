# Change summary

- Retain the MDFNet model, its three in-repository ablation variants, their data pipeline, the training/evaluation entry point, fold aggregation, and a configurable Slurm training example.
- Exclude third-party baseline source files, including ListenNet, MHANet, DARNet, and DBPNet. The README links to their official repositories; none of their source code is included. CNN, STANet, XANet, and DenseNet baseline source is also not included.
- Limit the documented protocol to the implemented four-fold trial-disjoint (LTO) path. This release does not implement LOSO or region-specific frequency-band variants.
- Set the default seed to 2025 and use the requested PyTorch seed and cuDNN configuration.
- Retain all loaded feature tensors on the chosen device before training, without memory mapping or a batch-loading fallback. Print validation accuracy every epoch.
- Remove the old source manifest and validation report. This package has not run a real-data experiment and contains no synthetic-check PASS record.

No repository license is included. Confirm that retained source and bundled dependencies may be redistributed.
