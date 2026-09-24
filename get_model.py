"""Model registry for the public MDFNet comparison package."""

from importlib import import_module


# MDFNet/BASE are aliases for the proposed five-band model.  The remaining
# entries are the three supported ablations and seven paper baselines.
MODEL_NAMES = (
    "MDFNet", "BASE", "CNN", "XANet", "DenseNet", "DARNet",
    "DBPNet", "ListenNet", "MHANet", "Nofre", "Notem", "Nocat",
)

_MODEL_MODULES = {
    "MDFNet": ("OURmodels.Base", "Base", "OURmodels"),
    "BASE": ("OURmodels.Base", "Base", "OURmodels"),
    "Nofre": ("OURmodels.Nofre", "Nofre", "OURmodels"),
    "Notem": ("OURmodels.Notem", "Notem", "OURmodels"),
    "Nocat": ("OURmodels.Nocat", "Nocat", "OURmodels"),
    "CNN": ("BASEmodels.CNN", "CNN", "BASEmodels"),
    "XANet": ("BASEmodels.XANet", "XANet", "BASEmodels"),
    "DenseNet": ("BASEmodels.DenseNet", "DenseNet", "BASEmodels"),
    "DARNet": ("BASEmodels.DARNet", "DARNet", "BASEmodels"),
    "DBPNet": ("BASEmodels.DBPNet", "DBPNet", "BASEmodels"),
    "ListenNet": ("BASEmodels.ListenNet", "ListenNet", "BASEmodels"),
    "MHANet": ("BASEmodels.MHANet", "MHANet", "BASEmodels"),
}


def get_model(model_name, decision_window, sbnum, device):
    del sbnum  # kept for compatibility with the historical caller
    try:
        module_name, class_name, model_type = _MODEL_MODULES[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}; choose from MODEL_NAMES") from exc
    constructor = getattr(import_module(module_name), class_name)
    return constructor(device, decision_window).to(device), model_type
