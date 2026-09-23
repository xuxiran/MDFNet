"""Trial-disjoint training using the original AAAI MDFNet implementation."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from AADdataset import AADdataset
from get_model import get_model, MODEL_NAMES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['KUL', 'DTU'], default='KUL')
    p.add_argument('--model', choices=MODEL_NAMES, default='MDFNet')
    p.add_argument('--data-dir', type=Path, default=Path(cfg.process_data_dir))
    p.add_argument('--output-dir', type=Path, default=Path('outputs'))
    p.add_argument('--sbfold', type=int, choices=range(4), default=0)
    p.add_argument('--only_evaluate', '--only-evaluate', type=int, choices=[0, 1], default=0)
    p.add_argument('--epoch', type=int, default=50)
    p.add_argument('--batch_size', '--batch-size', type=int, default=128)
    p.add_argument('--decision_window', '--decision-window', type=int, choices=[64,128,256], default=64,
                   help='Samples at 128 Hz: 64=0.5 s, 128=1 s, 256=2 s')
    p.add_argument('--seed', type=int, default=2025, help='Random seed (default: 2025)')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda:0; independent of fold index')
    return p


def load_trials(path):
    # HDF5-backed MATLAB v7.3 arrays are transposed exactly as in the source.
    with h5py.File(path, 'r') as handle:
        data = np.asarray(handle['EEG']).transpose().copy()
        labels = np.asarray(handle['ENV']).transpose().copy()
    if data.ndim != 4 or data.shape[-1] != 64 or labels.shape != data.shape[:3]:
        raise ValueError('After transpose, EEG must be [subjects,trials,time,64], ENV [subjects,trials,time]')
    if not np.isfinite(data).all() or not np.isin(labels, [0,1]).all():
        raise ValueError('EEG must be finite and ENV must contain binary labels')
    if not np.all(labels == labels[..., :1]):
        raise ValueError('This protocol requires a constant attention label within each trial')
    return data, labels


def reorder_trials(data, labels):
    """Preserve the legacy stable alternating class order; record original indices."""
    count = data.shape[1]
    if count % 4:
        raise ValueError('The four-fold source protocol requires a trial count divisible by four')
    order = []
    for subject_labels in labels:
        zero = np.flatnonzero(subject_labels[:, 0] == 0)
        one = np.flatnonzero(subject_labels[:, 0] == 1)
        if len(zero) != len(one):
            raise ValueError('The source alternating-trial protocol requires balanced classes per subject')
        order.append(np.stack([zero, one], axis=1).reshape(-1))
    order = np.asarray(order)
    subjects = np.arange(len(data))[:,None]
    return data[subjects, order], labels[subjects, order], order


def split_trials(trials, fold):
    test = np.arange(fold * (trials//4), (fold+1) * (trials//4))
    valid = (test + trials//4) % trials
    train = np.setdiff1d(np.arange(trials), np.concatenate([test,valid]))
    return {'train':train, 'valid':valid, 'test':test}


def resident_dataset(dataset, device):
    # Every precomputed feature and label is resident before the first batch.
    try:
        for name, value in list(vars(dataset).items()):
            if torch.is_tensor(value):
                setattr(dataset, name, value.to(device))
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError('Full input residency exceeded GPU memory. Request more GPU resources; no streaming fallback is used.') from error
    return dataset


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def run(args):
    if args.epoch < 1 or args.batch_size < 1:
        raise ValueError('epoch and batch size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(min(torch.get_num_threads(), 7))
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    run_dir = args.output_dir / args.dataset / args.model / f'dw{args.decision_window}' / f'seed{args.seed}' / f'fold{args.sbfold}'
    checkpoint = run_dir / 'best.ckpt'
    if not args.only_evaluate and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {run_dir}; choose another output directory')
    if args.only_evaluate and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir.mkdir(parents=True, exist_ok=True)
    data, labels = load_trials(args.data_dir / f'{args.dataset}_1D.mat')
    data, labels, order = reorder_trials(data, labels)
    # Preserve full-trial filtering, but allow a trailing incomplete window.
    usable = data.shape[2] // args.decision_window * args.decision_window
    if usable == 0:
        raise ValueError('Each trial must contain at least one decision window')
    splits = split_trials(data.shape[1], args.sbfold)
    arrays = {}
    for name, ids in splits.items():
        arrays[name] = (data[:, ids].reshape(-1, data.shape[2], 64), labels[:,ids].reshape(-1,labels.shape[2]))
    datasets = {n: resident_dataset(AADdataset(x, y, 128, args.decision_window, args.dataset), device)
                for n,(x,y) in arrays.items()}
    loaders = {n:DataLoader(d, batch_size=args.batch_size, shuffle=n=='train', num_workers=0) for n,d in datasets.items()}
    model, _ = get_model(args.model, args.decision_window, len(arrays['train'][0]), device)
    manifest = {'dataset':args.dataset, 'model':args.model, 'window_samples':args.decision_window,
                'sample_rate':128, 'seed':args.seed, 'fold':args.sbfold, 'epochs':args.epoch,
                'batch_size':args.batch_size, 'learning_rate':cfg.lr, 'weight_decay':cfg.weight_decay,
                'device':str(device),
                'resident_before_training':True, 'original_trial_order':order.tolist(),
                'trial_splits':{n:ids.tolist() for n,ids in splits.items()},
                'windows':{n:len(d) for n,d in datasets.items()},
                'parameter_count':sum(p.numel() for p in model.parameters()),
                'source_hashes':{str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.rglob('*.py') if '__pycache__' not in str(p)}}
    if not args.only_evaluate:
        save_json(run_dir/'manifest.json', manifest)
        best, history = -float('inf'), []
        for epoch in range(args.epoch):
            model.train(loaders['train'], device, epoch, args.epoch)
            pred, truth = model.test(loaders['valid'], device)
            accuracy = float(np.mean(pred == truth))
            if not np.isfinite(accuracy):
                raise RuntimeError('Validation accuracy is non-finite')
            history.append({'epoch':epoch,'validation_accuracy':accuracy})
            if accuracy > best:
                best = accuracy
                torch.save(model.state_dict(), checkpoint)
            print(f'epoch={epoch} validation_accuracy={accuracy:.6f}', flush=True)
        save_json(run_dir/'history.json', history)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    prediction, truth = model.test(loaders['test'], device)
    np.savez_compressed(run_dir/'predictions.npz', prediction=prediction, truth=truth)
    metrics = {'accuracy':float(np.mean(prediction==truth)), 'n_windows':len(truth),
               'dataset':args.dataset, 'model':args.model, 'window_samples':args.decision_window,
               'seed':args.seed, 'fold':args.sbfold}
    save_json(run_dir/'metrics.json', metrics)
    print(json.dumps(metrics), flush=True)
    return metrics


if __name__ == '__main__':
    run(parser().parse_args())
