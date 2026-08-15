import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import InferenceDataset
from utils import util_common, util_image, util_net
from utils.util_image import ImageSpliterTh


class W2WDiffSampler:
    def __init__(self, config, checkpoint, vae_checkpoint, chop_size=2048,
                 chop_stride=1792, seed=12345, use_amp=True):
        if not torch.cuda.is_available():
            raise RuntimeError("W2WDiff inference requires a CUDA-capable GPU")
        self.device = torch.device("cuda")
        self.config = config
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = config.model.params.get("lq_size", 256)
        self._set_seed()
        self.diffusion = util_common.instantiate_from_config(config.diffusion)
        self.model = util_common.instantiate_from_config(config.model).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        util_net.reload_model(self.model, state.get("state_dict", state))
        self.model.requires_grad_(False).eval()
        self.autoencoder = util_common.instantiate_from_config(config.autoencoder).to(self.device)
        self.autoencoder.load_checkpoint(vae_checkpoint)
        self.autoencoder.requires_grad_(False).eval()

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _sample(self, image):
        original_height, original_width = image.shape[-2:]
        pad_height = math.ceil(original_height / self.padding_offset) * self.padding_offset - original_height
        pad_width = math.ceil(original_width / self.padding_offset) * self.padding_offset - original_width
        if pad_height or pad_width:
            image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
        result = self.diffusion.p_sample_loop(
            y=image,
            model=self.model,
            first_stage_model=self.autoencoder,
            noise=None,
            noise_repeat=False,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs={"lq": image},
            progress=False,
            guidance=None,
        )
        return result[:, :, :original_height, :original_width].clamp_(-1.0, 1.0)

    def _process(self, image):
        amp = torch.amp.autocast if hasattr(torch, "amp") else torch.cuda.amp.autocast
        amp_kwargs = {"device_type": "cuda"} if hasattr(torch, "amp") else {}
        if image.shape[-2] <= self.chop_size and image.shape[-1] <= self.chop_size:
            with amp(**amp_kwargs, enabled=self.use_amp):
                return self._sample(image)
        splitter = ImageSpliterTh(
            image, self.chop_size, stride=self.chop_stride, sf=1, extra_bs=1
        )
        for patch, index_info in splitter:
            with amp(**amp_kwargs, enabled=self.use_amp):
                splitter.update(self._sample(patch), index_info)
        return splitter.gather()

    @torch.inference_mode()
    def infer(self, input_path, output_path, lab_statistics, batch_size=1, num_workers=0):
        dataset = InferenceDataset(input_path, lab_statistics=lab_statistics)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        processed = 0
        for batch in loader:
            results = self._process(batch["lq"].to(self.device))
            results = results.mul(0.5).add(0.5)
            for result, source_path in zip(results, batch["path"]):
                image = util_image.tensor2img(result, rgb2bgr=True, min_max=(0.0, 1.0))
                destination = output_path / f"{Path(source_path).stem}.png"
                util_image.imwrite(image, destination, chn="bgr", dtype_in="uint8")
                processed += 1
        elapsed = time.perf_counter() - started
        print(f"Processed {processed} image(s) in {elapsed:.2f}s ({elapsed / processed:.3f}s/image)")
