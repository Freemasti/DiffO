"""
BUG: 
1. Current code loads all layers despite trying to delete. Skip, `instantiate_from_config`.
2. Check if gradient disabled for inference.

Solution 1:
Need to do this for all models replacing `AutoencoderKL`.

```
cfg_path = r"PATH_TO_CFG"
ckpt_path = r"PATH_TO_CKPT"
config = OmegaConf.load(cfg_path)
self.vae_model = AutoencoderKL(
    ddconfig=config.model.params.ddconfig,
    embed_dim=config.model.params.embed_dim,
    ckpt_path=ckpt_path,
    ignore_keys=['post_quant_conv', 'decoder', 'loss', 'model_ema'],
    # ignore_keys=['quant_conv', 'encoder', 'post_quant_conv', 'decoder', 'loss', 'model_ema'],
    custom_load_type=['encoder'],
)
self.vae_model = self.vae_model.requires_grad_(False)
self.vae_model.eval()
self.vae_model.to(device)
```
"""

import argparse
import json
import os
import pathlib
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from vqcompress.core.config import CompressionConfig
from vqcompress.ldm.util import instantiate_from_config

try:
    import xformers.ops
    CompressionConfig.available_xformers = True
except ModuleNotFoundError as err:
    print(err)

torch.set_grad_enabled(False)


def get_xformers_flash_attention_op(q, k, v):
    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        # flash_attention_op = xformers.ops.MemoryEfficientAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        print(e, "enabling flash attention")

    return None


def patch_xformers_attn_forward(self, x):
    """Can replace LDM vqgan attention method forward with xformers attention. Should also work when replacing other
    attention code. Code copied from,
    https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py.

    Examples:
        >>> import vqcompress.ldm.model
        >>> vqcompress.ldm.model.AttnBlock.forward = patch_xformers_attn_forward
    """
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    # dtype = q.dtype
    # if True:
    #     q, k = q.float(), k.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
    out = xformers.ops.memory_efficient_attention(q, k, v)
    # out = out.to(dtype)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out


class CustomEncodingDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            image_size: int,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.image_size = image_size
        encoding_exts = ['json']
        self.files_list = [p for enc_ext in encoding_exts for p in Path(f'{input_path}').glob(f'*.{enc_ext}')]

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, index) -> Tuple[dict, dict]:
        path = self.files_list[index]

        with open(path, "r") as jsonfile:
            data = json.load(jsonfile)
            tensors = {}
            with safe_open(
                    os.path.join(path.parent.absolute(), data[CompressionConfig.output_filename_key]),
                    framework="pt",
                    device="cuda"
                    # device="cpu"
            ) as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        return tensors, data


class CustomImageDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            image_size: int,
            keep_aspect_ratio: bool,
    ) -> None:
        super().__init__()

        self.input_path = input_path
        self.image_size = image_size
        self.files_list = [
            p for ext in CompressionConfig.dataset_image_exts for p in Path(f'{input_path}').glob(f'*.{ext}')
        ]

        resize_type = transforms.Resize(image_size) if keep_aspect_ratio else transforms.Resize(
            (image_size, image_size))

        self.transform = transforms.Compose([
            resize_type,
            transforms.ToTensor(),
            transforms.Lambda(preprocess_vqgan),
        ])

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str, str]:
        path = self.files_list[index]
        img = Image.open(path).convert('RGB')
        transformed_img = self.transform(img)
        return transformed_img, path.stem, path.name


def custom_to_pil(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0)
    x = (255 * x).type(torch.uint8)
    x = x.detach().cpu().numpy()
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def preprocess_vqgan(x):
    return 2. * x - 1.


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ImageCompression:
    def __init__(
            self,
            input_path: str,
            output_path: str,
            cfg_path: str,
            ckpt_path: str,
            image_size: int = 256,
            batch_size: int = 2,
            num_workers: int = 0,
            device: str = 'cuda',
            use_kl: bool = True,
            vq_ind: bool = True,
            use_decompress: bool = True,
            keep_aspect_ratio: bool = False,
            saved_indices_bits: int = 16,
            use_float16_precision: bool = True,
            use_xformers: bool = False,
    ):
        self.saved_indices_bits = saved_indices_bits
        self.use_xformers = use_xformers
        self.use_decompress = use_decompress
        self.vq_ind = vq_ind
        self.use_kl = use_kl
        self.device = device
        self.input_path = input_path
        self.output_path = output_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.saved_indices_bits_torch = torch.int16 if saved_indices_bits == 16 else torch.uint8

        self.precision_type = torch.float16 if use_float16_precision else torch.float32

        self.output_path = self.output_path or self.input_path
        # dest_path = f'{args.dest}/output'
        pathlib.Path(input_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        img_dataset = CustomImageDataset(
            input_path=self.input_path,
            image_size=self.image_size,
            keep_aspect_ratio=keep_aspect_ratio,
        )
        self.img_data_loader = DataLoader(
            img_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

        encoding_dataset = CustomEncodingDataset(
            input_path=self.input_path,
            image_size=self.image_size,
        )
        # For encoding to image processinge single item batch used.
        self.encoding_data_loader = DataLoader(
            encoding_dataset,
            batch_size=1,
            shuffle=False,
            # pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Only load weights for inference and delete unnecessary values to save vram. Works on kl-f8 config.
        config = OmegaConf.load(cfg_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        sd_keys = sd.keys()

        def delete_model_layers(layer_initial_list: List[str]):
            for layer_initial in layer_initial_list:
                key_delete_list = []
                for dkey in sd_keys:
                    if dkey.startswith(layer_initial):
                        key_delete_list.append(dkey)

                for k in key_delete_list:
                    del sd[f'{k}']

        delete_model_layers(['loss', 'model_ema'])

        if use_decompress:
            delete_model_layers(['quant_conv', 'encoder'])
        else:
            delete_model_layers(['post_quant_conv', 'decoder'])

        if use_xformers:
            import vqcompress.ldm.model
            vqcompress.ldm.model.AttnBlock.forward = patch_xformers_attn_forward

        # print(sd.keys())
        self.ldm_model = instantiate_from_config(config.model)
        self.ldm_model.load_state_dict(sd, strict=False)
        if use_float16_precision:
            self.ldm_model.half()
        self.ldm_model = self.ldm_model.to(device)
        self.ldm_model.eval()
        # for param in self.ldm_model.parameters():
        #     param.requires_grad = False

        del sd
        del pl_sd

    def process(self) -> None:
        if self.use_decompress:
            self.decompress()
        else:
            self.compress()

    def decompress(self) -> None:
        with tqdm(self.encoding_data_loader) as pbar:
            for batch_idx, (input_data_dict, loaded_json_data) in enumerate(pbar):
                input_data = input_data_dict['weight']
                input_data = input_data.squeeze(0)
                input_data = input_data.to(self.device)

                full_filename_list = loaded_json_data['file_name_list']

                if self.use_kl:
                    with torch.autocast(device_type=self.device, dtype=self.precision_type):
                        xrec = self.ldm_model.decode(input_data)

                    for idx, fname in enumerate(full_filename_list):
                        x0 = custom_to_pil(xrec[idx])
                        x0.save(os.path.join(self.output_path, f"{fname[0]}"))
                else:
                    if self.vq_ind:
                        if CompressionConfig.saved_indices_precision_key in loaded_json_data.keys():
                            input_data = input_data.type(torch.int64)

                        if 'z_shape' in loaded_json_data.keys():
                            z_shaped_loaded = loaded_json_data['z_shape']

                        out = self.ldm_model.quantize.get_codebook_entry(
                            input_data, tuple(z_shaped_loaded)
                        )

                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
                            xrec = self.ldm_model.decode(out)

                        del out
                        del input_data
                        # gc.collect()
                        torch.cuda.empty_cache()

                        for idx, fname in enumerate(full_filename_list):
                            x0 = custom_to_pil(xrec[idx])
                            x0.save(os.path.join(self.output_path, f"{fname[0]}"))

                    else:
                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
                            xrec = self.ldm_model.decode(input_data)

                        for idx, fname in enumerate(full_filename_list):
                            x0 = custom_to_pil(xrec[idx])
                            x0.save(os.path.join(self.output_path, f"{fname[0]}"))

    def compress(self) -> None:
        with tqdm(self.img_data_loader) as pbar:
            for batch_idx, (input_data, filename, full_filename) in enumerate(pbar):
                input_data = input_data.to(self.device)

                tensors = {}
                json_data_obj = {}

                if self.use_kl:
                    with torch.autocast(device_type=self.device, dtype=self.precision_type):
                        posterior = self.ldm_model.encode(input_data)
                        compressed_tensor = posterior.sample()
                else:
                    if self.vq_ind:
                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
                            z, _, [_, _, indices] = self.ldm_model.encode(input_data)

                        z_shape = (z.shape[0], z.shape[2], z.shape[3], z.shape[1])

                        compressed_tensor = indices
                        json_data_obj['z_shape'] = z_shape

                        compressed_tensor = compressed_tensor.type(self.saved_indices_bits_torch)
                        json_data_obj[CompressionConfig.saved_indices_precision_key] = self.saved_indices_bits

                    else:
                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
                            z, _, _ = self.ldm_model.encode(input_data)

                        compressed_tensor = z

                tensors['weight'] = compressed_tensor
                output_file_name = f"batch_{batch_idx}.safetensors"

                save_file(tensors, os.path.join(self.output_path, output_file_name))

                json_data_obj[CompressionConfig.output_filename_key] = output_file_name
                json_data_obj["file_name_list"] = list(full_filename)
                json_object = json.dumps(json_data_obj, indent=4)

                with open(os.path.join(self.output_path, f"batch_{batch_idx}.json"), "w") as jsonfile:
                    jsonfile.write(json_object)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compress and decompress images.')
    parser.add_argument('-s', '--src', required=True, help="path to source folder", type=pathlib.Path)
    parser.add_argument('-d', '--dest', help="path to destination folder", type=pathlib.Path)
    parser.add_argument('--device', help="use cpu or cuda gpu", default='cuda', type=str)
    parser.add_argument('--cfg', help="model config file path", type=pathlib.Path)
    parser.add_argument('--ckpt', help="pretrained encode decode model path for config file", type=pathlib.Path)
    parser.add_argument('--img_size', metavar='--img-size', help="image size for processing", default=512, type=int)
    parser.add_argument('--batch', help="process image count", default=1, type=int)
    parser.add_argument('--kl', help="generates output from kl autoencoder", action='store_true')
    parser.add_argument('--dc', help="decompresses encoding", action='store_true')
    parser.add_argument('--vq_ind', help="generates vqgan encoding instead of indices", action='store_true')
    parser.add_argument('--aspect', help="preserves aspect ratio resize by given image size", action='store_true')
    parser.add_argument('--ind_bit', help='saves vq indices to selected bits', default=16, type=int, choices=[8, 16])
    parser.add_argument('--float16', help="process in half precision", action='store_true')
    parser.add_argument('--xformers', help="use xformers to save memory and speedup in some cases", action='store_true')

    args = parser.parse_args()

    image_compression = ImageCompression(
        input_path=args.src,
        output_path=args.dest,
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        image_size=args.img_size,
        use_kl=args.kl,
        vq_ind=args.vq_ind,
        use_decompress=args.dc,
        batch_size=args.batch,
        keep_aspect_ratio=args.aspect,
        saved_indices_bits=args.ind_bit,
        use_float16_precision=args.float16,
        use_xformers=args.xformers,
    )
    image_compression.process()


if __name__ == '__main__':
    main()
