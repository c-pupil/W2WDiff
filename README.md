# W2WDiff

Official inference code for **W2WDiff: Generalizing Underwater Diffusion Model via Unsupervised Underwater Conversion**, IEEE Transactions on Geoscience and Remote Sensing, 2025.

[[Paper](https://ieeexplore.ieee.org/document/11230813)] [[Project Page](https://c-pupil.github.io/projects/W2WDiff/index.html)]

## Installation

```bash
conda env create -f environment.yml
conda activate w2wdiff
```

## Pretrained Models

Download the pretrained models from [Google Drive](https://drive.google.com/open?id=17N__82LpPlOPn8xiLw2No9Avbq63HosD).

The checkpoints are provided in split parts. Merge them before inference:

```bash
mkdir -p weights
cat /path/to/downloads/ema_model_130000.pth.part-* > weights/ema_model_130000.pth
cat /path/to/downloads/w2wdiff_vae.ckpt.part-* > weights/w2wdiff_vae.ckpt
```

The resulting directory should be:

```text
weights/
├── ema_model_130000.pth
└── w2wdiff_vae.ckpt
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference_cfm.py \
  -i /path/to/input \
  -o /path/to/output \
  --task uie \
  --scale 1
```

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

This project is built on [ResShift](https://github.com/zsyOAOA/ResShift). We thank the authors for their work.

## License

This project follows the [NTU S-Lab License 1.0](LICENSE).
