"""Model registry for the public MDFNet comparison package."""

from OURmodels.Base import Base
from OURmodels.Nofre import Nofre
from OURmodels.Notem import Notem
from OURmodels.Nocat import Nocat


MODEL_NAMES = (
    "MDFNet", "Nofre", "Notem", "Nocat",
)


def get_model(model_name, decision_window, sbnum, device):
    del sbnum  # kept for compatibility with the historical caller
    constructors = {
        "MDFNet": (Base, "OURmodels"),
        "Nofre": (Nofre, "OURmodels"),
        "Notem": (Notem, "OURmodels"),
        "Nocat": (Nocat, "OURmodels"),
    }
    try:
        constructor, model_type = constructors[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}; choose from MODEL_NAMES") from exc
    return constructor(device, decision_window).to(device), model_type
