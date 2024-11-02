import torch
from DSCF_models_pe_ import Hierarchical_1d_model
import os
from ComponentFiltering.BackgroundRemoval_SERS_NP.bg_learning.train_models import train

def run_protocol(
        encoder_name,decoder_name,
        save_head_path='../SERS_bg_removal/',
        batch_size=128,dict_size=15,
        epoch = 2000, device = 'cuda:1',lr = 1e-4,snr = 25,scale = 'large',
        patch_size = 4,embed_dim = 64
):
    if not os.path.exists(save_head_path):
        os.makedirs(save_head_path)
    print(save_head_path)
    if scale == 'tiny':
        layers = [2, 3, 6, 2]
    elif scale == 'large':
        layers = [3, 6, 8, 3]
    elif scale == 'huge':
        layers = [5, 10, 16, 5]

    model = Hierarchical_1d_model(inplanes=1,outplanes=1,
                                  encoder_name=encoder_name,
                                  decoder_name=decoder_name,
                                  layers=layers,d_layers=[2,3,6,2],
                                  mask=0.2,patch_size=patch_size,embed_dim=embed_dim,
                                  sig_len=1799)

    save_name = encoder_name+'_'+decoder_name +'_'+str(dict_size)+'.pt'
    model_pth = '[Enc_Dec:'+encoder_name+'_'+decoder_name+'][SNR:'+str(snr)+'][PatchSize'+str(patch_size)+']'+scale+'_.pt'
    model_save_path = os.path.join(save_head_path,model_pth)
    print(model_save_path)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        train(epoch, model, lr, device, save_path = model_save_path, batch_size=batch_size,
              )

    else:
        train(epoch, model, lr, device, save_path = model_save_path, batch_size=batch_size,
              )
if __name__ == '__main__':

    encoder_names = ['SiT','ResUnet',]
    decoder_names= ['PPS','TConv',]
    snr = 25
    for encoder_name in encoder_names:
        for decoder_name in decoder_names:
            save_head_path = 'Background_Removal_SERS_workspace'
            run_protocol(encoder_name = encoder_name,
                         decoder_name = decoder_name,
                         save_head_path = save_head_path,
                         snr=snr,device='cuda:0')