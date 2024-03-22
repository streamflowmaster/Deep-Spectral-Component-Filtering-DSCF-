import torch

from ComponentFiltering.utils.DSCF_models import Hierarchical_1d_model
import os
from ComponentFiltering.utils.train_models import train
def run_protocol(
        encoder_name,decoder_name,
        save_head_path='../WorkSpace_15_ComFilter/',
        batch_size=256,dict_size=15,data_head_path='../ConstructedData/Training_15',
        epoch = 50, device = 'cuda:1',lr = 1e-4,
):
    model = Hierarchical_1d_model(inplanes=1,outplanes=dict_size,
                                  encoder_name=encoder_name,
                          decoder_name=decoder_name,layers=[2,3,6,2])
    save_name = encoder_name+'_'+decoder_name +'_'+str(dict_size)+'.pt'
    model_save_path = os.path.join(save_head_path,save_name)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    train(epoches=epoch, model=model, lr=lr, device=device, save_path=model_save_path, batch_size=batch_size,
          dict_size=dict_size, head_path=data_head_path)


if __name__ == '__main__':
    # encoder_names = ['ResUnet','SiT',]
    # decoder_names= ['TConv','PPS',]
    encoder_names = ['SiT','ResUnet',]
    decoder_names= ['PPS','TConv',]
    for encoder_name in encoder_names:
        for decoder_name in decoder_names:
            run_protocol(encoder_name = encoder_name,
                         decoder_name = decoder_name)