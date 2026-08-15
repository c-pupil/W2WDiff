from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class LabReferenceMatcher:
    def __init__(self, statistics_path):
        statistics = np.load(statistics_path)
        self.histograms = tuple(statistics[name] for name in ("hist_l", "hist_a", "hist_b"))

    def __call__(self, image_bgr):
        channels = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
        matched_channels = []
        for channel, reference_histogram in zip(channels, self.histograms):
            histogram, edges = np.histogram(channel, bins=256, range=(0, 256), density=True)
            mapping = np.interp(
                np.cumsum(histogram),
                np.cumsum(reference_histogram),
                np.linspace(0, 255, 256),
            )
            matched = np.interp(channel.ravel(), edges[:-1], mapping).reshape(channel.shape)
            matched_channels.append(matched.astype(np.uint8))
        matched_lab = cv2.merge(matched_channels)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


class InferenceDataset(Dataset):
    def __init__(self, input_path, lab_statistics=None, recursive=True):
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input does not exist: {input_path}")
        if input_path.is_file():
            self.paths = [input_path]
        else:
            iterator = input_path.rglob("*") if recursive else input_path.glob("*")
            self.paths = sorted(path for path in iterator if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No supported images found in {input_path}")
        self.matcher = LabReferenceMatcher(lab_statistics) if lab_statistics else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {path}")
        if self.matcher is not None:
            image_bgr = self.matcher(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float()
        tensor = tensor.div_(255.0).sub_(0.5).div_(0.5)
        return {"lq": tensor, "path": str(path)}
