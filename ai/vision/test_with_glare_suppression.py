"""bon002789_002/003 같은 글레어 사진에 억제 전처리를 적용했을 때 예측이 바뀌는지 확인."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ai.vision.inference import ArtifactPredictor


def suppress_glare(image: Image.Image, brightness_threshold: int = 235, saturation_threshold: int = 40) -> Image.Image:
    rgb = image.convert("RGB")
    hsv = np.array(rgb.convert("HSV"))
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    glare_mask = (value > brightness_threshold) & (saturation < saturation_threshold)
    if not glare_mask.any():
        return rgb
    blurred = np.array(rgb.filter(ImageFilter.GaussianBlur(radius=25)))
    array = np.array(rgb).copy()
    array[glare_mask] = blurred[glare_mask]
    return Image.fromarray(array)


predictor = ArtifactPredictor.from_paths(
    checkpoint_path=Path("runs/vision/classifier_final/best_model.pth"),
    metadata_path=Path("data/metadata.csv"),
)

for name in ["bon002789_002.JPG", "bon002789_003.JPG", "bon002789_001.webp", "bon002789_004.JPG"]:
    path = Path("data/webcam_test") / name
    original = Image.open(path).convert("RGB")
    cleaned = suppress_glare(original)

    before = predictor.predict_pil(original, top_k=1)[0]
    after = predictor.predict_pil(cleaned, top_k=1)[0]
    print(f"{name}: before={before.artifact_id}({before.confidence:.3f})  after={after.artifact_id}({after.confidence:.3f})")
