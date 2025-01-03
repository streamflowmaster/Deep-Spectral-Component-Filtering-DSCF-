#
import torch.nn as nn
import torch
import math
from typing import List

import numpy as np
import torch
from einops import rearrange
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx
from torch.nn.init import trunc_normal_
from DSCF_Submit.pretrain.DSCF_models_pe_ import *

@dataclass
class DSCFConfig:
    inplanes: int  = 1,
    outplanes: int  = 1,
    encoder_name: str = 'SiT',
    embed_dim: int  = 128,
    decoder_name: str = 'PPS',
    layers: list = [3, 2, 2, 2],
    d_layers: list = [1, 1, 1, 1],
    sig_len: int  = 512,
    device: str = 'cuda:0',
    mask: bool  = True,
    patch_size: int  = 16,
    epoches: int  = 800,
    lr: float = 1e-4,
    batch_size:int = 128,
    snr:float = 25,
    dict_size:int = 1000,
    val_steps:int = 10,


class Hierarchical_1d_model(nn.Module):

    def  __init__(self,
                  config:DSCFConfig):

        super(Hierarchical_1d_model, self).__init__()
        self.inplanes = config.inplanes
        self.inputs = config.inplanes
        self.sig_len = config.sig_len
        self.encoder_name = config.encoder_name
        self.decoder_name = config.decoder_name
        self.device = config.device
        self.config = config
        if self.encoder_name == 'SiT':
            self.patch_embed = PatchEmbed1D(in_chans=config.inplanes,patch_size=config.patch_size,embed_dim=config.embed_dim,device=config.device)
            self.unpatchy = UnPatchEmbed1D(out_chans=config.outplanes, embed_dim=config.embed_dim, patch_size=config.patch_size,
                                           device=config.device)

            self.inplanes = config.embed_dim
        layers = config.layers
        device = config.device
        encoder_name = config.encoder_name
        decoder_name = config.decoder_name
        embed_dim = config.embed_dim
        d_layers = config.d_layers

        self.enc1 = self._make_encoder(block_name=encoder_name, planes=embed_dim*2, blocks=layers[0]).to(device)
        self.enc2 = self._make_encoder(block_name=encoder_name, planes=embed_dim*4, blocks=layers[1]).to(device)
        self.enc3 = self._make_encoder(block_name=encoder_name, planes=embed_dim*8, blocks=layers[2]).to(device)
        self.enc4 = self._make_encoder(block_name=encoder_name, planes=embed_dim*16, blocks=layers[3]).to(device)

        self.dec3 = self._make_decoder(block_name=decoder_name, dim=embed_dim*32, blocks=d_layers[0]).to(device)
        self.dec2 = self._make_decoder(block_name=decoder_name, dim=embed_dim*16, blocks=d_layers[1]).to(device)
        self.dec1= self._make_decoder(block_name=decoder_name, dim=embed_dim*8, blocks=d_layers[2]).to(device)
        self.upsample = self._make_upsample(block_name=decoder_name,dim=embed_dim*4).to(device)
        if self.encoder_name == 'SiT':
            self.dec0 = self.conv1x1 = nn.ConvTranspose1d(embed_dim*2, out_channels=embed_dim, kernel_size=1, stride=1,
                                                          bias=False).to(device)
        else:
            self.dec0 = self.conv1x1 = nn.ConvTranspose1d(embed_dim*2, out_channels=config.outplanes, kernel_size=1, stride=1,
                                                          bias=False).to(device)
        in_features = self._test_feature_size(config.inplanes)
        self.fc1 = nn.Linear(in_features=in_features,out_features=2).to(device)
        self.mask = config.mask

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, E, L], sequence
        """
        N, E, L = x.shape  # batch, embed, length
        len_keep = int(L * (1 - mask_ratio))
        len_remove = L - len_keep
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # batch, length_k
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask.unsqueeze(1).repeat(1,E,1).bool()

    def mask_operation(self,x,mask):

        x_unmask = torch.masked_fill(x,mask,0)
        aver_power = x.mean()
        noise = torch.rand(x.shape, device=x.device)*aver_power*0.01
        x_mask = torch.masked_fill(noise,~mask,0)
        # print('mask:', mask)
        # print('x_mask', x_mask)
        # print('x_unmask',x_unmask)
        return x_unmask+x_mask

    def _test_feature_size(self,inplanes):
        self.enc = nn.Sequential(
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
        ).to(self.device)
        test_input = torch.rand((1, inplanes, self.sig_len)).to(self.device)
        if self.encoder_name == 'SiT':
            test_input = self.patch_embed(test_input)
        test_output = self.enc(test_input).reshape(-1)
        return test_output.shape[0]

    def _make_encoder(self, block_name, planes, blocks):

        if block_name == 'SiT':
            layer = SwinBasicEncoder(dim=planes,depth=blocks,num_heads=16,downsample=PatchMerging,drop=0.3)

        elif block_name == 'ResUnet':
            layer = ResUnetBasicEncoder(dim=planes,depth=blocks,)

        else: layer = None
        if self.inplanes != planes:
            downsample = nn.Conv1d(self.inplanes, planes, stride=1, kernel_size=1, bias=False)
            layers = [downsample, layer]

        else:
            layers = [layer]
        self.inplanes = planes*2

        return nn.Sequential(*layers)

    def _make_upsample(se,block_name,dim):
        if block_name in ['PatchPixelShuffle', 'PPS']:
            layer = PatchPxielShuffle(dim = dim)

        elif block_name in ['TransposeConv', 'TConv']:
            layer = nn.ConvTranspose1d(in_channels=dim, out_channels=dim//2,
                                           padding=2, kernel_size=6, stride=2, bias=False)

        else:
            layer = None
        return layer

    def _make_decoder(self,block_name,dim,blocks):


        if block_name in ['PatchPixelShuffle', 'PPS']:
            layer = SiT1dDecoder(inplanes=dim,outplanes=dim//2,depth=blocks)

        elif block_name in ['TransposeConv', 'TConv']:
            layer = ConvTranspose1dDecoder(inplanes=dim,outplanes=dim//2,depth=blocks)
        else: layer = None
        return layer

    def linear_probe(self,x):
        B,C,L = x.shape
        down1 = self.enc1(x)
        down2 = self.enc2(down1)
        down3 = self.enc3(down2)
        down4 = self.enc4(down3).reshape(B,-1)
        return self.fc1(down4)

    def forward(self,x):
        if self.encoder_name == 'SiT':
            x = self.patch_embed(x)  # B,1,L -> B,E,N
            # print(x.shape)
            if self.mask:
                mask = self.random_masking(x,mask_ratio=self.mask)
                x = self.mask_operation(x,mask)
        down1 = self.enc1(x)
        # print('x:',x)
        # print('down1:',down1)
        down2 = self.enc2(down1)
        down3 = self.enc3(down2)
        down4 = self.enc4(down3)

        # print(down1.shape,down2.shape,down3.shape,down4.shape)
        up3 = self.dec3(down4,down3)
        up2 = self.dec2(up3, down2)
        up1 = self.dec1(up2, down1)
        if self.decoder_name in ['PatchPixelShuffle', 'PPS']:
            up1 = up1.permute(0,2,1)
        up0 = self.upsample(up1)
        up = self.dec0(up0)
        if self.encoder_name == 'SiT':
            up = self.unpatchy(up)
        return up

    def load_encoder_from_pretrain(self,pretrain_path):
        pretrain = torch.load(pretrain_path,map_location=self.device)['model']


        PRETRAIN = MultiDec_1d_model(config=self.config)
        encs_dicts = PRETRAIN.load_pretrained_encoder(pretrain)

        self.enc1.load_state_dict(encs_dicts[0])
        self.enc2.load_state_dict(encs_dicts[1])
        self.enc3.load_state_dict(encs_dicts[2])
        self.enc4.load_state_dict(encs_dicts[3])

        print("LOADED FROM PRETRAINED MODEL")
        del PRETRAIN

