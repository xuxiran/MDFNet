"""Model registry for the public MDFNet comparison package."""

from OURmodels.Base import Base
from OURmodels.Nofre import Nofre
from OURmodels.Notem import Notem
from OURmodels.Nocat import Nocat
from BASEmodels.STANet import STANet
from BASEmodels.CNN import CNN
from BASEmodels.XANet import XANet
from BASEmodels.DenseNet import DenseNet
from BASEmodels.DARNet import DARNet
from BASEmodels.DBPNet import DBPNet
from BASEmodels.ListenNet import ListenNet
from BASEmodels.MHANet import MHANet


# MDFNet/BASE are aliases for the proposed five-band model.  The remaining
# entries are the eight baselines and three supported ablations.
MODEL_NAMES = (
    "MDFNet", "BASE", "STANet", "CNN", "XANet", "DenseNet", "DARNet",
    "DBPNet", "ListenNet", "MHANet", "Nofre", "Notem", "Nocat",
)


def get_model(model_name, decision_window, sbnum, device):
    del sbnum  # kept for compatibility with the historical caller
    constructors = {
        "MDFNet": (Base, "OURmodels"),
        "BASE": (Base, "OURmodels"),
        "Nofre": (Nofre, "OURmodels"),
        "Notem": (Notem, "OURmodels"),
        "Nocat": (Nocat, "OURmodels"),
        "STANet": (STANet, "BASEmodels"),
        "CNN": (CNN, "BASEmodels"),
        "XANet": (XANet, "BASEmodels"),
        "DenseNet": (DenseNet, "BASEmodels"),
        "DARNet": (DARNet, "BASEmodels"),
        "DBPNet": (DBPNet, "BASEmodels"),
        "ListenNet": (ListenNet, "BASEmodels"),
        "MHANet": (MHANet, "BASEmodels"),
    }
    try:
        constructor, model_type = constructors[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}; choose from MODEL_NAMES") from exc
    return constructor(device, decision_window).to(device), model_type
