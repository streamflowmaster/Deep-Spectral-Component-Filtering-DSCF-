import torch
from DSCF_models_pe_ import Hierarchical_1d_model
import os
from train_models import train
def run_protocol(
        encoder_name,decoder_name,
        save_head_path='../WorkSpace_15_ComFilter/',
        batch_size=2048,
        epoch = 500, device = 'cuda:1',lr = 5e-6, snr = 35, scale = 'tiny'
):
    if not os.path.exists(save_head_path):
        os.makedirs(save_head_path)
    if scale == 'tiny':
        layers = [2, 3, 6, 2]
    elif scale == 'large':
        layers = [3, 6, 8, 3]
    elif scale == 'huge':
        layers = [5, 10, 16, 5]

    model = Hierarchical_1d_model(inplanes=1,outplanes=7,
                          encoder_name=encoder_name,
                          decoder_name=decoder_name,layers=layers,mask=0)

    model_pth = '[Enc_Dec_'+encoder_name+'_'+decoder_name+'][SNR_'+str(snr)+']'+\
                scale+'.pt'
    model_save_path = os.path.join(save_head_path,model_pth)

    if os.path.exists(model_save_path):
        print(model_save_path)
        model.load_state_dict(torch.load(model_save_path))

    train(epoches=epoch, model=model, lr=lr, device=device, save_path=model_save_path, batch_size=batch_size,
          snr=snr)


if __name__ == '__main__':
    # encoder_names = ['ResUnet','SiT',]
    # decoder_names= ['TConv','PPS',]
    encoder_names = ['SiT','ResUnet',]
    decoder_names= ['PPS','TConv',]
    scales = ['large','huge','tiny',]
    # snrs = [10,15,20,25,30,35]
    snr = 25
    for encoder_name in encoder_names:
        for decoder_name in decoder_names:
            for scale in scales:
                save_head_path = '../Quantification_SNR' + str(snr)
                run_protocol(encoder_name=encoder_name,
                             decoder_name=decoder_name,
                             save_head_path=save_head_path, scale=scale,
                             snr = snr)
