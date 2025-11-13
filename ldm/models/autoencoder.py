import torch
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.vqvae.quantize import EMAVectorQuantizer

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

import torchac
from einops import rearrange, repeat, reduce, pack, unpack

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def calculate_bpp(compressed_data, num_pixels, bytes=True, num_bytes=None):
    """Calculate bpp given the compressed text and number of pixels."""
    scaling_factor = 8 if bytes else 1
    if num_bytes:
        return num_bytes * scaling_factor / num_pixels
    return len(compressed_data) * scaling_factor / num_pixels

def compute_cdf_uniform_prob(codebook_size, target_shape):
    """Obtain CDF from uniform distribution, cast to target_shape"""
    #print(target_shape)
    b, = target_shape #h,w
    prob_per_entry = 1.0 / codebook_size

    # Compute the cumulative sum starting from 0
    cdf = torch.cumsum(torch.full((codebook_size,), prob_per_entry), dim=0)

    cdf = torch.cat([torch.zeros(1), cdf])
    cdf = cdf.view(1, -1).expand(b, -1) #(1, 1, 1, -1)  (b,h,w,-1)
    cdf = cdf.clone()
    cdf[..., -1] = 1.0
    return cdf

def compute_cdf_from_prob(z_hat_indices, codebook_size, target_shape):
    """Obtain CDF from a given probability distribution, cast to target_shape"""
    prob_dist = torch.bincount(z_hat_indices, minlength=codebook_size)#.float()

    # Ensure the probabilities sum to 1
    prob_dist = prob_dist / prob_dist.sum()

    # Compute the cumulative distribution function (CDF)
    cdf = torch.cumsum(prob_dist, dim=0).to(prob_dist.device) 
    cdf = torch.cat([torch.zeros(1, device=prob_dist.device), cdf]) 

    # Reshape and expand to match the target shape
    b, = target_shape
    cdf = cdf.view(1, -1).expand(b, -1)

    # Ensure the last value is exactly 1.0 (to handle precision issues)
    cdf = cdf.clone()
    cdf[..., -1] = 1.0
    cdf = cdf.to('cpu')
    #print(cdf)
    return cdf

def compress_hyper_latent(z_hat_indices):
    """Compress hyper-latent to bytes using torchac."""
    #_, cfg_cs = cfg_perco.rate_cfg[cfg_perco.target_rate]
    cfg_cs = 8192
    #print(z_hat_indices.shape)
    flatten = z_hat_indices
    #flatten, ps = pack_one(z_hat_indices, 'h * d')
    #print(ps)
    #flatten = unpack_one(flatten_pack, ps, 'h *')
    cdf = compute_cdf_from_prob(z_hat_indices, cfg_cs, flatten.shape) #compute_cdf_uniform_prob(cfg_cs, flatten.shape) 
    cdf = torch.clamp(cdf, max=1.0 - 1e-6)
    #print(flatten.shape)
    z_hat_indices = flatten.to(torch.int16).to('cpu')
    with open("/root/workspace/z_hat_indices.txt", "w") as file:
        for index in z_hat_indices:
            file.write(f"{index}\n")
    #print("z_hat_indices: ",z_hat_indices)
    #print("Bitstream: ",torchac.encode_float_cdf(cdf, z_hat_indices, check_input_bounds=True))
    return torchac.encode_float_cdf(cdf, z_hat_indices, check_input_bounds=True)

def decompress_hyper_latent(compressed_hyper_latent, shape):
    """Decompress hyper-latent using torchac."""
    #cfg_ss, cfg_cs = cfg_perco.rate_cfg[cfg_perco.target_rate]
    cfg_cs = 8192
    H, W = shape
    factor = 512 // cfg_ss
    h, w = H // factor, W // factor
    cdf = compute_cdf_uniform_prob(cfg_cs, (1, int(h), int(w)))
    return torchac.decode_float_cdf(cdf, compressed_hyper_latent)


# class VQModelTorch(torch.nn.Module):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  remap=None,
#                  sane_index_shape=False,  # tell vector quantizer to return indices as bhw
#                  ):
#         super().__init__()
#         self.encoder = Encoder(**ddconfig)
#         self.decoder = Decoder(**ddconfig)

#         self.loss = instantiate_from_config(lossconfig) #discriminator loss
        
#         self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
#                                         remap=remap, sane_index_shape=sane_index_shape)
#         self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
#         self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

#     def encode(self, x, encode=True, quant_conv=True, force_not_quantize=False, grad_forward=False, return_emb_loss=False):
#         if encode:
#             x = self.encoder(x)

#         if quant_conv:
#             x = self.quant_conv(x)

#         if force_not_quantize:
#             return x
#         else: 
#             if grad_forward:
#                 with torch.no_grad():
#                     quant, emb_loss, info = self.quantize(x)

#                 quant = (quant - x).detach()+x
#             else:
#                 quant, emb_loss, info = self.quantize(x)
#         #print(quant.shape)

#         # if quant.shape[2] == 64:
#         #     print('Compressing hyper-latent...')
#         #     byte_stream_hyper_latent = compress_hyper_latent(info[2])
#         #     bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 512 * 768)
#         #     print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))

#         if return_emb_loss:
#             return quant, emb_loss
#         else:
#             return quant

#     def decode(self, h, grad_forward=False, post_quant_conv=True, decode=True):
#         if post_quant_conv:
#             h = self.post_quant_conv(h)

#         # print('Compressing hyper-latent...')
#         # byte_stream_hyper_latent = compress_hyper_latent(quant)
#         # bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 512 * 768)
#         # print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))

#         if decode:
#             h = self.decoder(h)

#         return h

#     def decode_code(self, code_b):
#         quant_b = self.quantize.embed_code(code_b)
#         dec = self.decode(quant_b, force_not_quantize=True)
#         return dec

#     def forward(self, input, force_not_quantize=False):
#         h = self.encode(input)
#         dec = self.decode(h, force_not_quantize)
#         return dec

# class EMAVQModelTorch(torch.nn.Module):
#     def __init__(self,
#                  ddconfig,
#                  n_embed,
#                  embed_dim,
#                  remap=None,
#                  #sane_index_shape=False,  # tell vector quantizer to return indices as bhw
#                  ):
#         super().__init__()
#         self.encoder = Encoder(**ddconfig)
#         self.decoder = Decoder(**ddconfig)
#         self.quantize = EMAVectorQuantizer(n_embed, embed_dim, beta=0.25,
#                                         remap=remap, )#sane_index_shape=sane_index_shape)
#         self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
#         self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

#     def encode(self, x, force_not_conv=False, force_not_quantize=False, grad_forward=False, return_emb_loss=False):
#         if force_not_conv:
#             h = x
#         else:
#             h = self.encoder(x)
#             h = self.quant_conv(h)
#         if force_not_quantize:
#             return h
#         else: 
#             if grad_forward:
#                 with torch.no_grad():
#                     quant, emb_loss, info = self.quantize(h)

#                 quant = (quant - h).detach()+h
#             else:
#                 quant, emb_loss, info = self.quantize(h)
#         #print(quant.shape)

#         # print('Compressing hyper-latent...')
#         # byte_stream_hyper_latent = compress_hyper_latent(info[2])
#         # bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 768 * 512)
#         # print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))

#         if return_emb_loss:
#             return quant, emb_loss
#         else:
#             return quant

#         # from PIL import Image
#         # tensor = (quant)/2 + 0.5
#         #         # Ensure the tensor is in (C, H, W) format
#         # if tensor.ndim == 4:  # (B, C, H, W)
#         #     tensor = tensor[0]  # Select the first image in the batch

#         # # Normalize to [0, 1] if not already
#         # #tensor = tensor / 2 + 0.5  # Assume input is in [-1, 1]

#         # # Clamp to [0, 1]
#         # tensor = torch.clamp(tensor, 0, 1)

#         # # Convert to [0, 255]
#         # tensor = (tensor * 255).byte()

#         # # Convert to NumPy array (H, W, C)
#         # image_array = tensor.permute(1, 2, 0).cpu().numpy()

#         # # Save the image using PIL
#         # image = Image.fromarray(image_array)
#         # image.save("/root/workspace/enc_250131.png")

        


            

#     def decode(self, h, force_not_quantize=False, grad_forward=False):
#         if not force_not_quantize:
#             if grad_forward:
#                 with torch.no_grad():
#                     quant, emb_loss, info = self.quantize(h)

#                 quant = (quant - h).detach()+h
#             else:
#                 quant, emb_loss, info = self.quantize(h)
#         else:
#             quant = h
#         #quant = h
#         quant = self.post_quant_conv(quant)
#         # print('Compressing hyper-latent...')
#         # byte_stream_hyper_latent = compress_hyper_latent(quant)
#         # bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 512 * 768)
#         # print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))



#         # from PIL import Image
#         # tensor = (quant)/2 + 0.5
#         #         # Ensure the tensor is in (C, H, W) format
#         # if tensor.ndim == 4:  # (B, C, H, W)
#         #     tensor = tensor[0]  # Select the first image in the batch

#         # #Normalize to [0, 1] if not already
#         # tensor = tensor / 2 + 0.5  # Assume input is in [-1, 1]

#         # # Clamp to [0, 1]
#         # tensor = torch.clamp(tensor, 0, 1)

#         # # Convert to [0, 255]
#         # tensor = (tensor * 255).byte()

#         # # Convert to NumPy array (H, W, C)
#         # image_array = tensor.permute(1, 2, 0).cpu().numpy()

#         # # Save the image using PIL
#         # image = Image.fromarray(image_array)
#         # image.save("/root/workspace/dec_250131.png")


        
#         dec = self.decoder(quant)
#         return dec

#     def decode_code(self, code_b):
#         quant_b = self.quantize.embed_code(code_b)
#         dec = self.decode(quant_b, force_not_quantize=True)
#         return dec

#     def forward(self, input, force_not_quantize=False):
#         h = self.encode(input)
#         dec = self.decode(h, force_not_quantize)
#         return dec
        
class VQModelTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x, force_not_conv=False, force_not_quantize=False, grad_forward=False, return_emb_loss=False):
        if force_not_conv:
            h = x
        else:
            h = self.encoder(x)
            h = self.quant_conv(h)
        if force_not_quantize:
            return h
        else: 
            if grad_forward:
                with torch.no_grad():
                    quant, emb_loss, info = self.quantize(h)

                quant = (quant - h).detach()+h
            else:
                quant, emb_loss, info = self.quantize(h)
        #print(quant.shape)

        # if quant.shape[2] == 64:
        #     print('Compressing hyper-latent...')
        #     byte_stream_hyper_latent = compress_hyper_latent(info[2])
        #     bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 512 * 768)
        #     print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))

        if return_emb_loss:
            return quant, emb_loss
        else:
            return quant

        # from PIL import Image
        # tensor = (quant)/2 + 0.5
        #         # Ensure the tensor is in (C, H, W) format
        # if tensor.ndim == 4:  # (B, C, H, W)
        #     tensor = tensor[0]  # Select the first image in the batch

        # # Normalize to [0, 1] if not already
        # #tensor = tensor / 2 + 0.5  # Assume input is in [-1, 1]

        # # Clamp to [0, 1]
        # tensor = torch.clamp(tensor, 0, 1)

        # # Convert to [0, 255]
        # tensor = (tensor * 255).byte()

        # # Convert to NumPy array (H, W, C)
        # image_array = tensor.permute(1, 2, 0).cpu().numpy()

        # # Save the image using PIL
        # image = Image.fromarray(image_array)
        # image.save("/root/workspace/enc_250131.png")
    def decode(self, h, force_not_quantize=False, grad_forward=False):
            if not force_not_quantize:
                if grad_forward:
                    with torch.no_grad():
                        quant, emb_loss, info = self.quantize(h)

                    quant = (quant - h).detach()+h
                else:
                    quant, emb_loss, info = self.quantize(h)
            else:
                quant = h
            #quant = h
            quant = self.post_quant_conv(quant)
            # print('Compressing hyper-latent...')
            # byte_stream_hyper_latent = compress_hyper_latent(quant)
            # bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 512 * 768)
            # print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))



            # from PIL import Image
            # tensor = (quant)/2 + 0.5
            #         # Ensure the tensor is in (C, H, W) format
            # if tensor.ndim == 4:  # (B, C, H, W)
            #     tensor = tensor[0]  # Select the first image in the batch

            # #Normalize to [0, 1] if not already
            # tensor = tensor / 2 + 0.5  # Assume input is in [-1, 1]

            # # Clamp to [0, 1]
            # tensor = torch.clamp(tensor, 0, 1)

            # # Convert to [0, 255]
            # tensor = (tensor * 255).byte()

            # # Convert to NumPy array (H, W, C)
            # image_array = tensor.permute(1, 2, 0).cpu().numpy()

            # # Save the image using PIL
            # image = Image.fromarray(image_array)
            # image.save("/root/workspace/dec_250131.png")


            
            dec = self.decoder(quant)
            return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec
        

class EMAVQModelTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 #sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = EMAVectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, )#sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x, force_not_conv=False, force_not_quantize=False, grad_forward=False, return_emb_loss=False):
        if force_not_conv:
            h = x
        else:
            h = self.encoder(x)

            #h = x
            h = self.quant_conv(h)
        if force_not_quantize:
            return h
        else: 
            if grad_forward:
                with torch.no_grad():
                    quant, emb_loss, info = self.quantize(h)

                quant = (quant - h).detach()+h
            else:
                quant, emb_loss, info = self.quantize(h)

        #print(quant.shape)
            #print(h.shape)
        #if quant.shape[2] == 64:
        # print('Compressing hyper-latent...')
        # print(info[2].shape)
        byte_stream_hyper_latent = compress_hyper_latent(info[2])
        bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 768 * 512)
        print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))
        # # === BPP 값을 텍스트 파일로 기록 ===
        # with open('bpp_hyper_latent.txt', 'a') as f:
        #     f.write('BPP hyper-latent: {:.4f}\n'.format(bpp_hyper_latent))

        if return_emb_loss:
            return quant, emb_loss
        else:
            return quant

        # from PIL import Image
        # tensor = (quant)/2 + 0.5
        #         # Ensure the tensor is in (C, H, W) format
        # if tensor.ndim == 4:  # (B, C, H, W)
        #     tensor = tensor[0]  # Select the first image in the batch

        # # Normalize to [0, 1] if not already
        # #tensor = tensor / 2 + 0.5  # Assume input is in [-1, 1]

        # # Clamp to [0, 1]
        # tensor = torch.clamp(tensor, 0, 1)

        # # Convert to [0, 255]
        # tensor = (tensor * 255).byte()

        # # Convert to NumPy array (H, W, C)
        # image_array = tensor.permute(1, 2, 0).cpu().numpy()

        # # Save the image using PIL
        # image = Image.fromarray(image_array)
        # image.save("/root/workspace/enc_250131.png")

        


            

    def decode(self, h, force_not_quantize=False, grad_forward=False):
        if not force_not_quantize:
            if grad_forward:
                with torch.no_grad():
                    quant, emb_loss, info = self.quantize(h)

                quant = (quant - h).detach()+h
            else:
                quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        #quant = h
        quant = self.post_quant_conv(quant)
        # print('Compressing hyper-latent...')
        # byte_stream_hyper_latent = compress_hyper_latent(quant)
        # bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, 512 * 768)
        # print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))



        # from PIL import Image
        # tensor = (quant)/2 + 0.5
        #         # Ensure the tensor is in (C, H, W) format
        # if tensor.ndim == 4:  # (B, C, H, W)
        #     tensor = tensor[0]  # Select the first image in the batch

        # #Normalize to [0, 1] if not already
        # tensor = tensor / 2 + 0.5  # Assume input is in [-1, 1]

        # # Clamp to [0, 1]
        # tensor = torch.clamp(tensor, 0, 1)

        # # Convert to [0, 255]
        # tensor = (tensor * 255).byte()

        # # Convert to NumPy array (H, W, C)
        # image_array = tensor.permute(1, 2, 0).cpu().numpy()

        # # Save the image using PIL
        # image = Image.fromarray(image_array)
        # image.save("/root/workspace/dec_250131.png")


        
        dec = self.decoder(quant)

        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec



            





class AutoencoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input, sample_posterior, return_moments=False)
        dec = self.decode(z)
        return dec

class EncoderKLTorch(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z
    def forward(self, x, sample_posterior=True, return_moments=False):
        return self.encode(x, sample_posterior, return_moments)

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x



import torch
import csv
import numpy as np

class VQModelTorch_save_tensor(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def save_tensor_to_csv(self, tensor, filename):
        # Move tensor to CPU and convert to numpy
        tensor_np = tensor.detach().cpu().numpy()
        # Flatten the tensor to 2D (if necessary)
        tensor_np = tensor_np.reshape(tensor_np.shape[0], -1)
        # Save to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in tensor_np:
                writer.writerow(row)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # Save latent tensor after encoding
        self.save_tensor_to_csv(h, '/root/workspace/latent_after_encode.csv')
        return h

    def decode(self, h, force_not_quantize=False, grad_forward=False):
        if not force_not_quantize:
            if grad_forward:
                with torch.no_grad():
                    quant, emb_loss, info = self.quantize(h)
                quant = (quant - h).detach() + h
            else:
                quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        # Save latent tensor before decoding
        self.save_tensor_to_csv(quant, '/root/workspace/latent_before_decode.csv')
        
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec



