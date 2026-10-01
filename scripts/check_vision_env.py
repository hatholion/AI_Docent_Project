"""Validate the Python environment required for EfficientNet training."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    try:
        import PIL
        import matplotlib
        import numpy
        import sklearn
        import torch
        import torchvision
        from torchvision.models import efficientnet_b0
    except ImportError as error:
        package = error.name or "unknown"
        raise SystemExit(
            f"Missing dependency: {package}. Follow the Vision setup in README.md."
        ) from error

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"pillow={PIL.__version__}")
    print(f"matplotlib={matplotlib.__version__}")
    print(f"scikit_learn={sklearn.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"selected_device={device.type}")

    if device.type == "cuda":
        print(f"cuda_device_count={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")

    model = efficientnet_b0(weights=None).to(device)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224, device=device))
    if output.shape != (1, 1000):
        raise RuntimeError(f"Unexpected EfficientNet-B0 output shape: {output.shape}")
    print("efficientnet_b0_smoke_test=ok")


if __name__ == "__main__":
    main()
