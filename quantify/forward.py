from collections import defaultdict
from dataset import fintuning_dataloader
import torch
import torch.optim as optim
from ComponentFiltering.utils.metrics import HQI,MSE,idf_acc
import matplotlib.pyplot as plt

def test_one_epoch(model,epoch,device,train_data):
    num_sum = 0
    hqi_sum,mse_sum = 0,0
    # dict = return_dict().to(device)
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device)
        target = target.float().to(device)
        outputs = model(inputs)
        B,C,Lt = target.shape
        B,C,Lo = outputs.shape
        # print(Lt,Lo)
        if Lt<Lo:
            outputs = outputs[:,:,:Lt]
        else:
            target = target[:,:,:Lo]

        plt.subplot(C+1,1,C+1)
        plt.plot(inputs[0, 0,:].T.cpu().detach().numpy(),c = 'r')

        for i in range(C):
            plt.subplot(C+1,1,i+1)
            plt.plot(target[1, i,:].T.cpu().detach().numpy(),c = 'r')
            plt.plot(outputs[1, i, :].T.cpu().detach().numpy(), c='b')
        plt.show()

def forward( model,device, batch_size=1):
    train_data = fintuning_dataloader(batch_size=batch_size,device=device,snr=15)
    metric = defaultdict(float)
    model = model.to(device)
    epoch = 0
    with torch.no_grad():
        test_one_epoch(model, epoch,
                        device, train_data)
        torch.cuda.empty_cache()