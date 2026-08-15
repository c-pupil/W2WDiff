# W2WDiff

Official inference code for **W2WDiff: Generalizing Underwater Diffusion Model via Unsupervised Underwater Conversion**, IEEE Transactions on Geoscience and Remote Sensing, 2025.

[[Paper](https://ieeexplore.ieee.org/document/11230813)] [[Project page](https://c-pupil.github.io/projects/W2WDiff/index.html)]

This release focuses on the final inference pipeline. It does not include the original three-stage training code. The water-to-water conversion and the 15-step latent diffusion model are integrated into one command: input images are matched to the reference LAB distribution in memory, so no intermediate images are written to disk.

## Installation

The verified environment is Python 3.10 with PyTorch 2.1.x and CUDA 11.8.

```bash
conda env create -f environment.yml
conda activate w2wdiff
```

The original machine also has an environment named `resshift`; `resshift2` is incomplete and does not contain PyTorch.

## Pretrained models

Download all ten parts from the public [W2WDiff Google Drive folder](https://drive.google.com/open?id=17N__82LpPlOPn8xiLw2No9Avbq63HosD). The Drive API quota did not permit a reliable single-file upload, so each checkpoint is split into five lossless parts. Reassemble them in `weights/`:

```bash
mkdir -p weights
cat /path/to/downloads/ema_model_130000.pth.part-* > weights/ema_model_130000.pth
cat /path/to/downloads/w2wdiff_vae.ckpt.part-* > weights/w2wdiff_vae.ckpt
```

The resulting layout is:

```text
weights/
├── ema_model_130000.pth
└── w2wdiff_vae.ckpt
```

Verify the files before inference:

| File | SHA-256 |
| --- | --- |
| `ema_model_130000.pth` | `021d7934d8adb5b0b17c4f1d754992e16bc61275fa09243e003d1a91051d3890` |
| `w2wdiff_vae.ckpt` | `f2d38350aa618f3de6405319830fdf670d83c5a355f11759df3f7806295ff3be` |

## Inference

Run inference on one image or a directory:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_cfm.py \
  -i path/to/input \
  -o results/output \
  --task uie \
  --scale 1
```

The command used for the T90 experiment is now equivalent to:

```bash
CUDA_VISIBLE_DEVICES=1 python inference_cfm.py \
  -i uie_test/T90 \
  -o result/rebuttal_config/step_20 \
  --task uie \
  --scale 1
```

Previously, `hol_match_lab.py` generated `T90_lab_histogram_255` before diffusion. That step is now performed by `InferenceDataset` using `assets/uieb_gt_lab_histograms.npz`. Use `--no-lab-match` only when the input has already been converted by the old script.

Useful options:

- `--seed 12345`: sampling seed.
- `--chop-size {64,256,512}`: patch size before the internal scale multiplier.
- `--chop-stride N`: explicit patch stride.
- `--checkpoint PATH` and `--vae-checkpoint PATH`: custom weight locations.
- `--lab-statistics PATH`: custom LAB reference statistics.

## Reproducibility check

The integrated LAB preprocessing was checked against the former file-based pipeline and is pixel-identical. On the preserved U45 inputs, compare a new output folder with an archived result using:

```bash
python tools/compare_results.py \
  --result results/U45 \
  --reference /path/to/archived/U45
```

Diffusion inference is stochastic and can vary across checkpoints, seeds, and PyTorch/CUDA kernels. Keep the seed, both checkpoints, dependency versions, and command line fixed when reproducing archived images.

For the release validation, all 45 U45 images completed successfully. Compared with the preserved historical output, the new run produced MAE 3.079808/255, PSNR 36.182500 dB, and SSIM 0.986841; none of the 45 files were pixel-identical. The integrated LAB conversion was independently pixel-identical to the old intermediate files, so the remaining difference is attributable to the historical diffusion run state rather than preprocessing.

## Citation

```bibtex
@ARTICLE{11230813,
  author={Zhang, Yuanlin and Yuan, Jieyu and Chen, Xiao and Tang, Xiongxin and Chen, Qiao and Wang, Yiquan and Li, Chongyi},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={W2WDiff: Generalizing Underwater Diffusion Model via Unsupervised Underwater Conversion},
  year={2025},
  doi={10.1109/TGRS.2025.3629979}
}
```

## Acknowledgements

This code is built on [ResShift](https://github.com/zsyOAOA/ResShift). We sincerely thank Zongsheng Yue, Jianyi Wang, and Chen Change Loy for releasing ResShift and enabling this work. The VAE components also build on Latent Diffusion and StableSR, and the RRDB blocks originate from BasicSR/ESRGAN.

## License

This repository follows the included [NTU S-Lab License 1.0](LICENSE), inherited from ResShift. Please review its terms before redistribution or commercial use.
