"""Aggregate one seed across four folds without mixing decision windows."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, type=Path, help='outputs/KUL/MDFNet/dw64/seed2025')
    args = p.parse_args()
    records = [json.loads((args.run_root/f'fold{i}'/'metrics.json').read_text()) for i in range(4)]
    identity = ('dataset', 'model', 'window_samples', 'seed')
    for fold, record in enumerate(records):
        if record['fold'] != fold or any(record[k] != records[0][k] for k in identity):
            raise ValueError('Fold identities or experimental configurations do not match')
    values = np.array([r['accuracy'] for r in records])
    result = {k:records[0][k] for k in identity}
    result.update(mean_percent=float(100*values.mean()), std_percent=float(100*values.std(ddof=1)),
                  folds_percent=(100*values).tolist())
    (args.run_root/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
