import torch

from ldm.modules.diffusionmodules.model import Decoder_Mix, Encoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class AutoencoderKLResi(torch.nn.Module):
    def __init__(self, ddconfig, embed_dim, fusion_w=1.0, **kwargs):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        self.decoder.fusion_w = fusion_w
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("first_stage_model."):
                key = key[len("first_stage_model."):]
            if key.startswith("loss."):
                continue
            remapped[key] = value
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"VAE checkpoint mismatch: {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys"
            )

    def encode(self, image):
        features, encoder_features = self.encoder(image, return_fea=True)
        posterior = DiagonalGaussianDistribution(self.quant_conv(features))
        return posterior, encoder_features

    def decode(self, latent, encoder_features):
        return self.decoder(self.post_quant_conv(latent), encoder_features)
