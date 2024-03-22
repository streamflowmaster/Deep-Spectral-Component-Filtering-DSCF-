from EfficientSpatialSpectralEmbedding.SpectralRemoveBackground_AD.utils.models import Uresnet
from EfficientSpatialSpectralEmbedding.SpectralRemoveBackground_AD.utils.dataset import train_dataloader,test_dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
test_data = test_dataloader(batch_size=10,head_pth='../../../ADSERS/scaled_data_pt/')
train_data = train_dataloader(batch_size=10)
def visual(model,train_data, device= 'cpu',train=True):
    for idx,data in enumerate(train_data):

        if train:
            inputs,target = data
            target = target.float().to(device).unsqueeze(1)
            inputs = inputs.float().to(device).unsqueeze(1)
        else:
            inputs = data
            inputs = inputs.float().to(device).unsqueeze(1)
        outputs = model(inputs)
        plt.plot(inputs.squeeze(1).detach().numpy().T,c = 'b',label = 'raw_data')
        if train: plt.plot(target.squeeze(1).detach().numpy().T,c = 'cyan',label = 'GT')
        plt.plot(outputs.squeeze(1).detach().numpy().T,c = 'r',label = 'predicted_bg')
        # plt.plot((outputs-inputs).squeeze(1).detach().numpy().T, c='g', label='predicted_bg')
        # plt.legend()
        plt.show()

model = Uresnet(inplanes=1)
model.load_state_dict(torch.load('../WorkSpace/rbg.pt'))
visual(model=model,train_data=test_data,train=False)
# visual(model=model,train_data=train_data,train=True)