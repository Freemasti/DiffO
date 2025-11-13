import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

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
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ , _= self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class DualVQModel(VQModel):
    """
    DualVQModel은 VQModel을 상속받아 아래와 같은 파이프라인을 구현합니다.
    
      입력 이미지 x
        └──> 외부 모델의 Encoder: x_ext = external.encoder(x)
                └──> 내부 모델의 Encoder: x_int = internal.encoder(x_ext)
                        └──> 내부 VQ 단계: 
                                h_int = internal.quant_conv(x_int)
                                quant_int, qloss_int, _ = internal.quantize(h_int)
                                h_int_rec = internal.post_quant_conv(quant_int)
                                x_internal = internal.decoder(h_int_rec)
                                        └──> adapter: adapted = adapter(x_internal)
                                                └──> 외부 VQ 단계:
                                                        h_ext = external.quant_conv(adapted)
                                                        quant_ext, qloss_ext, _ = external.quantize(h_ext)
                                                        h_ext_rec = external.post_quant_conv(quant_ext)
                                                        x_rec = external.decoder(h_ext_rec)
    
    최종적으로 x_rec와 두 VQ 단계의 양자화 손실(qloss_int와 qloss_ext의 합)을 반환합니다.
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 # 내부(학습 대상) VQModel 관련 인자는 부모(VQModel)로 전달됨.
                 external_ddconfig,
                 external_lossconfig,
                 external_n_embed,
                 external_embed_dim,
                 external_ckpt_path=None,
                 teacher_ignore_keys=[],
                 adapter_in_channels=None,
                 adapter_out_channels=None,
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,
                 **kwargs):
        # 부모(VQModel)를 호출하여 내부 모델을 초기화 (internal encoder/decoder 등)
        super().__init__(ddconfig, lossconfig, n_embed, embed_dim,
                         image_key=image_key, colorize_nlabels=colorize_nlabels,
                         monitor=monitor, remap=remap, sane_index_shape=sane_index_shape,
                         **kwargs)
        
        # 외부 VQModel 생성 (동일한 VQModel 클래스를 사용)
        external_config = {
            "target": "taming.models.vqgan.VQModel",
            "params": {
                "ddconfig": external_ddconfig,
                "lossconfig": external_lossconfig,
                "n_embed": external_n_embed,
                "embed_dim": external_embed_dim,
                "image_key": image_key,
                "colorize_nlabels": colorize_nlabels,
                "monitor": monitor,
                "remap": remap,
                "sane_index_shape": sane_index_shape,
            }
        }
        self.external = instantiate_from_config(external_config)
        if external_ckpt_path is not None:
            self.external.init_from_ckpt(external_ckpt_path, ignore_keys=teacher_ignore_keys)
        # freeze external 모델
        for param in self.external.parameters():
            param.requires_grad = False
        self.external.train()
        
        # adapter: 내부 디코더의 출력 채널을 외부 모델의 quant_conv 입력 채널에 맞춤.
        # 기본값: 내부 출력 채널 = ddconfig["out_ch"] (없으면 3), 외부 입력 채널 = external_ddconfig["z_channels"] (없으면 16)
        if adapter_in_channels is None:
            adapter_in_channels = ddconfig.get("out_ch", 3)
        if adapter_out_channels is None:
            adapter_out_channels = external_ddconfig.get("z_channels", 16)
        self.adapter = torch.nn.Conv2d(adapter_in_channels, adapter_out_channels, kernel_size=1)

    def forward(self, x):
        # 1. 외부 Encoder
        x_ext = self.external.encoder(x)
        # 2. 내부 Encoder (부모 VQModel의 encoder)
        x_int = self.encoder(x_ext)
        # 3. 내부 VQ: quant_conv → quantize → post_quant_conv → internal decoder
        h_int = self.quant_conv(x_int)
        quant_int, qloss_int, _ = self.quantize(h_int)
        h_int_rec = self.post_quant_conv(quant_int)
        x_internal = self.decoder(h_int_rec)
        # 4. adapter: 내부 디코더 출력 조정
        #adapted = self.adapter(x_internal)
        # 5. 외부 VQ: 외부 모델의 quant_conv → quantize → post_quant_conv → external decoder
        h_ext = self.external.quant_conv(x_internal)
        quant_ext, qloss_ext, _ = self.external.quantize(h_ext)
        h_ext_rec = self.external.post_quant_conv(quant_ext)
        x_rec = self.external.decoder(h_ext_rec)

        int_rec_loss = torch.mean(torch.abs(x_ext-x_internal))

        return x_rec, qloss_int, int_rec_loss #total_qloss




    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        x_rec, qloss, int_rec_loss = self(x)
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(qloss, x, x_rec, optimizer_idx, self.global_step,
                                            last_layer=self.decoder.conv_out.weight, split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True,
                          on_step=True, on_epoch=True)
            return aeloss + int_rec_loss
        elif optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(qloss, x, x_rec, optimizer_idx, self.global_step,
                                                last_layer=self.decoder.conv_out.weight, split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True,
                          on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        x_rec, qloss, int_rec_loss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, x_rec, 0, self.global_step,
                                        last_layer=self.decoder.conv_out.weight, split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        discloss, log_dict_disc = self.loss(qloss, x, x_rec, 1, self.global_step,
                                            last_layer=self.decoder.conv_out.weight, split="val")
        self.log_dict(log_dict_disc)
        return {"aeloss": aeloss+int_rec_loss, "discloss": discloss}

    def configure_optimizers(self):
        if self.learning_rate is None:
            raise ValueError("learning_rate must be set before training.")
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.quantize.parameters()) +
            list(self.post_quant_conv.parameters()) +
            list(self.decoder.parameters()), #+
            #list(self.adapter.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        # 내부 VQModel의 log_images를 그대로 사용 (최종 재구성은 forward에서 외부 모델 체인을 통해 생성됨)
        return super().log_images(batch, **kwargs)