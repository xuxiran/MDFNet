# Historical result evidence

These files are selected exports from archived seed-2025 experiments, not fresh outputs of the current GitHub runner. Accuracy columns ending in `_percent` are percentages; `validation_accuracy` is a fraction between zero and one. Dataset subject indices are zero-based.

## Coverage

| Paper result | Retained matching records | What is not established |
| --- | --- | --- |
| LTO Table 1: 48 dataset/model/window cells | 14 cells, each with four folds (56 rows): ListenNet and MHANet on both datasets at all three windows, plus MDFNet at 1 s on both datasets | Matching per-fold records were not located for the other 34 cells: 30 cells for the other five baseline families and MDFNet at 0.5/2 s on both datasets. Older fold records for the five baselines exist, but their aggregates differ from the current table. |
| LOSO: 16 dataset/model cells at 0.5 s | All 296 subject results: eight paper models × (21 DTU + 16 KUL subjects) | Historical score equivalence of the current public runner has not been established. |

`lto_matched_folds.csv` contains only the 56 fold results that reproduce the current LTO table to its displayed precision. `lto_run_metadata.jsonl` exports selected fields from each retained run's metrics and split manifest, including the original artifact SHA-256 hashes. `lto_validation_history.csv` contains all 50 validation accuracies for each of those 56 runs. `loso_0p5s_subjects.csv` is the archived subject CSV with machine-specific path prefixes removed; `loso_0p5s_summary.csv` retains the archived summary values. `loso_run_metadata.jsonl` and `loso_validation_history.csv` provide the same per-run metadata and 50-epoch validation histories for all 296 LOSO subjects. These histories are exports, not original Slurm stdout. `loso_protocol.json` contains selected configuration fields and original configuration hashes. `source_provenance.json` maps LTO source-run IDs to retained experiments and hashes of source files.

The historical run directories also retain configurations, Slurm logs, predictions, and checkpoints, but those bulky original directories are not mirrored here. The exported run metadata and epoch histories omit machine-specific paths and environment variables. Original private-directory paths are not needed to recompute the released summaries.

SHA-256 values identify the retained original files; this repository cannot independently verify those hashes without the original archives. The released CSVs and metadata are sufficient to recalculate the displayed groups but are not a complete copy of every raw experiment artifact.

The KUL ListenNet/MHANet historical LTO runs and the MDFNet 1-second run use the same test quarters, but their training/validation quarters differ in folds 0, 1, and 3. They should not be treated as fully matched development splits for paired uncertainty analysis. The current public code was edited after these runs and has not been shown to regenerate the frozen accuracies.

Run `python evidence/verify.py` from the repository root to check row coverage, source consistency, 50-epoch histories, and the LOSO mean/sample-standard-deviation calculations. No raw EEG or predictions are included.
