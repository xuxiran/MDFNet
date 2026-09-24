"""CPU-only synthetic imports, model construction, and forward checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from get_model import MODEL_NAMES, get_model


def main():
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cpu")
    batch_size = 1
    window = 64
    features = [torch.randn(batch_size, window, 10, 11) for _ in range(5)]
    frequency_features = torch.randn(batch_size, 5, 10, 11)

    for model_name in MODEL_NAMES:
        model, _ = get_model(model_name, window, 1, device)
        model.model.eval()
        with torch.inference_mode():
            if model_name in ("Nofre", "DenseNet"):
                output = model.model(features[2])
            elif model_name == "Notem":
                output = model.model(frequency_features)
            else:
                output = model.model(*features)
        logits = output[1] if isinstance(output, tuple) else output
        if logits.shape != (batch_size, 2) or not torch.isfinite(logits).all():
            raise AssertionError(f"Unexpected output for {model_name}: {tuple(logits.shape)}")
        print(f"{model_name}: constructed and forwarded {tuple(logits.shape)}", flush=True)


if __name__ == "__main__":
    main()
