import torch
from DSCF_Submit.pretrain.DSCF_models_pe_ import MultiDec_1d_model,DSCFConfig as MultiDecConfig
from DSCF_models_pe_ import Hierarchical_1d_model,DSCFConfig
import DSCF_Submit.pretrain.DSCF_models_pe_
import os
from forward import forward


def run_protocol(
        encoder_name,
        decoder_name,
        save_head_path='models/',
        pretrain_pth = '../pretrain/models/',
        batch_size=128,
        epoch = 800,
        device = 'cuda:1',
        lr = 1e-5,
        scale = 'large',
        sig_len = 512,
        output_channels = 1,
        snr = 15
):
    os.makedirs(save_head_path,exist_ok=True)
    if scale == 'tiny':
        layers = [3, 6, 8, 3]
        d_layers = [3, 6, 8, 3]
    elif scale == 'large':
        layers = [5, 10, 16, 5]
        d_layers = [3, 6, 8, 3]
    elif scale == 'huge':
        layers = [10, 20, 32, 10]
        d_layers = [5, 10, 16, 5]
    elif scale == 'ultra':
        layers = [40, 80, 128, 40]
        d_layers = [5, 10, 16, 5]
    elif scale == 'ultra_pro':
        layers = [80, 160, 256, 80]
        d_layers = [10, 20, 30, 10]

    model_args = dict(
        inplanes=1,
        outplanes=output_channels,
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        layers=layers,
        d_layers=d_layers,
        device=device,
        mask=0.01,
        patch_size=4,
        embed_dim=64,
        sig_len=sig_len,
        epoches=epoch,
        lr = lr,
        batch_size = batch_size,
        snr = 15,
        val_steps = 10
    )
    config = DSCFConfig(**model_args)
    model = Hierarchical_1d_model(config)

    save_name = encoder_name+'_'+decoder_name +'_'+str(scale)+'.pt'
    model_save_path = os.path.join(save_head_path,save_name)


    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        forward(model,device, batch_size)


if __name__ == '__main__':
   run_protocol(encoder_name='SiT',decoder_name='PPS',
                save_head_path='models/',
                pretrain_pth = '../pretrain/model_512/',
                batch_size=32,
                epoch = 800,
                device = 'cpu',
                lr = 1e-5,
                scale = 'tiny',
                sig_len = 512,
                output_channels = 1,
                snr = 15
                )


