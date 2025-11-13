import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import Sampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url
import torch
import time
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-r", "--ref_path", type=str, default=None, help="reference image")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length. (The number of steps that the model trained on.)")
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-is", "--infer_steps", type=int, default=None, help="Diffusion length for inference")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--one_step", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
            "--chop_size",
            type=int,
            default=768, #512,
            choices=[1024,768 ,512, 256],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="Single",
            choices=["Single"],
            help="Chopping forward.",
            )
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--vqckpt", type=str, default=None, help="VQVAE checkpoint path.")
    parser.add_argument("--GT_vqckpt", type=str, default=None, help="GT_VQVAE checkpoint path.")
    args = parser.parse_args()
    if args.infer_steps is None:
        args.infer_steps = args.steps
    print(f"[INFO] Using the inference step: {args.steps}")
    return args

def get_configs(args):
    if args.config is None:
        if args.task == "Single":
            configs = OmegaConf.load('./configs/config.yaml')
    else:
        configs = OmegaConf.load(args.config)
    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    print(f"[INFO] Using the checkpoint {ckpt_path}")
    
    if args.vqckpt is not None:
        vqgan_path = Path(args.vqckpt)
    else:
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if args.GT_vqckpt is not None:
        GT_vqgan_path = Path(args.GT_vqckpt)
    else:
        GT_vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
         load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.timestep_respacing = args.infer_steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)
    configs.GT_encoder.ckpt_path = str(GT_vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 1024:
        chop_stride = 896
    elif args.chop_size == 768:
        chop_stride = 672
    elif args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 256
    else:
        raise ValueError("Chop size must be in [512, 384, 256]")

    return configs, chop_stride

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    resshift_sampler = Sampler(
            configs,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_fp16=False, #True
            seed=args.seed,
            ddim=args.ddim
            )

    # Measure inference time
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Synchronize before timing starts
    start_time = time.time()  # Start the timer

    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False, one_step=args.one_step)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Synchronize after inference ends
    end_time = time.time()  # End the timer

    # Log the inference time
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.2f} seconds")
    import evaluate
    evaluate.evaluate(args.out_path, args.ref_path, None)
    
    
if __name__ == '__main__':
    main()
    
