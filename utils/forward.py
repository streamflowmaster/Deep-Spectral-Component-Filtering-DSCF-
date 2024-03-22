import os.path
from collections import defaultdict
import torch.nn.functional as F
from EfficientSpatialSpectralEmbedding.SemanticSegmentation.utils.loss import dice_loss
import time
import copy
from ComponentFiltering.utils.dataset import train_dataloader,test_dataloader,return_dict
from ComponentFiltering.utils.models import Uresnet
import torch
import torch.optim as optim
from ComponentFiltering.utils.metrics import HQI,MSE,idf_acc
import matplotlib.pyplot as plt

def test_one_epoch(model,epoch,optimizer,metric,device,train_data):
    num_sum = 0
    hqi_sum,mse_sum = 0,0
    dict = return_dict().to(device)
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.float().to(device)
        optimizer.zero_grad()
        # print(inputs.shape)
        outputs = model(inputs)
        # print('input:', inputs.shape)
        # print('output:',outputs.shape)
        # print('target:', target.shape)

        B,C,Lt = target.shape
        B,C,Lo = outputs.shape
        # print(Lt,Lo)
        if Lt<Lo:
            outputs = outputs[:,:,:Lt]
        else:
            target = target[:,:,:Lo]

        plt.plot(outputs[1,3,:].T.cpu().detach().numpy(),c = 'b')
        plt.plot(target[1,3,:].T.cpu().detach().numpy(),c = 'r')
        plt.show()
        hqi,mse = HQI(target,outputs),MSE(target,outputs)
        hqi_sum += hqi
        mse_sum += mse
        num_sum += 1

        identify_acc = idf_acc(target,outputs,dict)
        print(identify_acc)
        print('[%d]-th hqi:%.3f' % (epoch + 1, hqi_sum / num_sum))
        print('[%d]-th mse:%.3f' % (epoch + 1, mse_sum / num_sum))
    print('[%d]-th hqi:%.3f' % (epoch + 1, hqi_sum / num_sum))
    print('[%d]-th mse:%.3f' % (epoch + 1, mse_sum / num_sum))

def forward(epoches, model,lr, device, save_path, batch_size = 1,head_path= '../ConstructedData/Training_15',dict_size = 15):
    train_data = train_dataloader(batch_size=batch_size,head_path=head_path,dict_size=dict_size)
    metric = defaultdict(float)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoches):
        test_one_epoch(model, epoch,
                        optimizer, metric, device, train_data)
        torch.cuda.empty_cache()