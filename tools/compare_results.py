import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two image result directories")
    parser.add_argument("--result", required=True)
    parser.add_argument("--reference", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result)
    reference_dir = Path(args.reference)
    result_paths = sorted(result_dir.glob("*.png"))
    if not result_paths:
        raise ValueError(f"No PNG files found in {result_dir}")
    rows = []
    exact = 0
    for result_path in result_paths:
        reference_path = reference_dir / result_path.name
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference: {reference_path}")
        result = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if result.shape != reference.shape:
            raise ValueError(f"Shape mismatch for {result_path.name}: {result.shape} vs {reference.shape}")
        difference = np.abs(result.astype(np.int16) - reference.astype(np.int16))
        exact += int(np.array_equal(result, reference))
        rows.append((
            float(difference.mean()),
            int(difference.max()),
            peak_signal_noise_ratio(reference, result, data_range=255),
            structural_similarity(reference, result, channel_axis=2, data_range=255),
        ))
    values = np.asarray(rows)
    print(f"Images: {len(rows)}")
    print(f"Pixel-identical: {exact}/{len(rows)}")
    print(f"MAE: {values[:, 0].mean():.6f}")
    print(f"Maximum absolute error: {int(values[:, 1].max())}")
    print(f"PSNR: {values[:, 2].mean():.6f} dB")
    print(f"SSIM: {values[:, 3].mean():.6f}")


if __name__ == "__main__":
    main()
