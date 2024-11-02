import torch
from DSCF_models_pe_ import Hierarchical_1d_model
import os
from forward import forward
def run_protocol(
        encoder_name,decoder_name,
        save_head_path='../WorkSpace_15_ComFilter/',
        batch_size=2048,
        epoch = 500, device = 'cuda:0',lr = 5e-4,snr = 25, scale = 'tiny'
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
    print(model_save_path)
    if os.path.exists(model_save_path):
        print(model_save_path)
        model.load_state_dict(torch.load(model_save_path))

    forward(model=model,device=device, batch_size=batch_size)


def fetch_model(scale='large',snr=25,encoder_name='SiT',decoder_name='PPS'):
    save_head_path = '../Quantification_SNR'+str(snr)
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
    print(model_save_path + ' exist!')
    if os.path.exists(model_save_path):
        print(model_save_path+' exist!')
        model.load_state_dict(torch.load(model_save_path))
    return model.to('cpu')

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
