"""Copied and modified from,
- https://github.com/CompVis/latent-diffusion
- https://github.com/CompVis/taming-transformers
"""

from typing import List

import lightning as pl
import numpy as np
import torch
from torch.functional import F

from vqcompress.ldm.distributions import DiagonalGaussianDistribution
from vqcompress.ldm.model import Encoder, Decoder
from vqcompress.ldm.quantize import GumbelQuantize
from vqcompress.ldm.quantize import VectorQuantizer


class AutoencoderKL(pl.LightningModule):
    def __init__(
            self,
            ddconfig,
            # lossconfig,
            embed_dim,
            ckpt_path=None,
            ignore_keys=[],
            image_key="image",
            colorize_nlabels=None,
            monitor=None,
            custom_load_type: List = None,
    ):
        super().__init__()
        self.image_key = image_key

        assert ddconfig["double_z"]        

        if custom_load_type is None:
            custom_load_type = ['encoder', 'decoder']

        if "encoder" in custom_load_type:
            self.encoder = Encoder(**ddconfig)
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            print('LOADED ENCODER')
        if "decoder" in custom_load_type:
            self.decoder = Decoder(**ddconfig)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
            print('LOADED DECODER')

        # self.embed_dim = embed_dim
        # if colorize_nlabels is not None:
        #     assert type(colorize_nlabels) == int
        #     self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        # if monitor is not None:
        #     self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class VQModel(pl.LightningModule):
    def __init__(
            self,
            ddconfig,
            # lossconfig,
            n_embed,
            embed_dim,
            ckpt_path=None,
            ignore_keys=[],
            image_key="image",
            colorize_nlabels=None,
            monitor=None,
            batch_resize_range=None,
            scheduler_config=None,
            lr_g_factor=1.0,
            remap=None,
            sane_index_shape=False,  # tell vector quantizer to return indices as bhw
            use_ema=False,
            custom_load_type: List = None,
    ):
        super().__init__()
        if custom_load_type is None:
            custom_load_type = ['encoder', 'decoder']

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key

        if "encoder" in custom_load_type:
            self.encoder = Encoder(**ddconfig)
            self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
            print('LOADED ENCODER')
        if "decoder" in custom_load_type:
            self.decoder = Decoder(**ddconfig)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
            print('LOADED DECODER')

        # self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)

        # if colorize_nlabels is not None:
        #     assert type(colorize_nlabels) == int
        #     self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        # if monitor is not None:
        #     self.monitor = monitor

        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x, force_not_quantize=False, grad_forward=False, return_emb_loss=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if force_not_quantize:
            return h
        quant, emb_loss, info = self.quantize(h)
        return quant #, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, force_not_quantize=False, grad_forward=False):
        if force_not_quantize:
            quant = self.post_quant_conv(quant)
            dec = self.decoder(quant)
            return dec
        else:
            quant, emb_loss, info = self.quantize(quant)
            quant = self.post_quant_conv(quant)
            dec = self.decoder(quant)
            return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class GumbelVQ(VQModel):
    def __init__(
            self,
            ddconfig,
            # lossconfig,
            n_embed,
            embed_dim,
            # temperature_scheduler_config,
            ckpt_path=None,
            ignore_keys=[],
            image_key="image",
            colorize_nlabels=None,
            monitor=None,
            kl_weight=1e-8,
            remap=None,
    ):
        z_channels = ddconfig["z_channels"]
        super().__init__(
            ddconfig,
            # lossconfig,
            n_embed,
            embed_dim,
            ckpt_path=None,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
        )

        # self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(
            z_channels, embed_dim,
            n_embed=n_embed,
            kl_weight=kl_weight, temp_init=1.0,
            remap=remap
        )

        # self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError
