import argparse
from pathlib import Path

from omegaconf import OmegaConf

from sampler import W2WDiffSampler


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="W2WDiff underwater image enhancement")
    parser.add_argument("-i", "--in_path", required=True, help="Input image or directory")
    parser.add_argument("-o", "--out_path", required=True, help="Output directory")
    parser.add_argument("--task", default="uie", choices=["uie"])
    parser.add_argument("--scale", type=int, default=1, choices=[1])
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "ema_model_130000.pth"))
    parser.add_argument("--vae-checkpoint", default=str(ROOT / "weights" / "w2wdiff_vae.ckpt"))
    parser.add_argument("--lab-statistics", default=str(ROOT / "assets" / "uieb_gt_lab_histograms.npz"))
    parser.add_argument("--no-lab-match", action="store_true", help="Skip water-to-water LAB matching")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chop-size", type=int, default=512, choices=[64, 256, 512])
    parser.add_argument("--chop-stride", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(ROOT / "configs" / "uie.yaml")
    chop_size = args.chop_size
    if args.chop_stride < 0:
        overlap = {512: 64, 256: 32, 64: 16}[args.chop_size]
        chop_stride = args.chop_size - overlap
    else:
        chop_stride = args.chop_stride
    sampler = W2WDiffSampler(
        config,
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        chop_size=chop_size,
        chop_stride=chop_stride,
        seed=args.seed,
        use_amp=True,
    )
    statistics = None if args.no_lab_match else args.lab_statistics
    sampler.infer(args.in_path, args.out_path, statistics,
                  batch_size=args.bs, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
