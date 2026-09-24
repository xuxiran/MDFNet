"""CSP preprocessing used by the public DARNet and DBPNet baselines."""

import numpy as np


CSP_MODEL_NAMES = frozenset({"DARNet", "DBPNet"})
CSP_COMPONENTS = 64


def fit_baseline_csp(model_name, train_eeg, train_labels):
    """Fit the archived CSP parameterization on the supplied training trials.

    Args:
        model_name: Selected registry model name.
        train_eeg: Full training trials in ``[trials, time, channels]`` order.
        train_labels: Per-sample labels in ``[trials, time]`` order.

    Returns:
        A fitted MNE CSP transformer for DARNet/DBPNet, otherwise ``None``.
    """
    if model_name not in CSP_MODEL_NAMES:
        return None

    eeg = np.asarray(train_eeg)
    labels = np.asarray(train_labels)
    if eeg.ndim != 3:
        raise ValueError("CSP training EEG must have shape [trials, time, channels]")
    if labels.ndim != 2 or labels.shape != eeg.shape[:2]:
        raise ValueError("CSP training labels must have shape [trials, time]")
    if eeg.shape[-1] != CSP_COMPONENTS:
        raise ValueError(f"CSP requires {CSP_COMPONENTS} input channels; got {eeg.shape[-1]}")
    if not np.all(labels == labels[:, :1]):
        raise ValueError("CSP requires one constant class label per training trial")

    from mne.decoding import CSP

    transformer = CSP(
        n_components=CSP_COMPONENTS,
        transform_into="csp_space",
        cov_est="concat",
        norm_trace=True,
        log=None,
        reg=None,
    )
    transformer.fit(eeg.transpose(0, 2, 1), labels[:, 0])
    return transformer
