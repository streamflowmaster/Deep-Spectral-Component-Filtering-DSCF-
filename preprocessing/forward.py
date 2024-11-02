from collections import defaultdict
from ComponentFiltering.BackgroundRemoval_SERS_NP.dataset import train_dataloader,test_dataloader
import torch
import torch.optim as optim
from ComponentFiltering.utils.metrics import HQI,MSE,idf_acc
import matplotlib.pyplot as plt

def test_one_epoch(model,epoch,device,train_data):
    num_sum = 0
    hqi_sum,mse_sum = 0,0
    # dict = return_dict().to(device)
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.float().to(device).unsqueeze(1)
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

        plt.plot(inputs[1, 0, :].T.cpu().detach().numpy(), c='g')
        plt.plot(target[1,0,:].T.cpu().detach().numpy(),c = 'r')
        plt.plot(outputs[1, 0, :].T.cpu().detach().numpy(), c='b')
        plt.show()
        # hqi,mse = HQI(target,outputs),MSE(target,outputs)
        # hqi_sum += hqi
        # mse_sum += mse
        # num_sum += 1
        #
        # identify_acc = idf_acc(target,outputs,dict)
        # print(identify_acc)
    #     print('[%d]-th hqi:%.3f' % (epoch + 1, hqi_sum / num_sum))
    #     print('[%d]-th mse:%.3f' % (epoch + 1, mse_sum / num_sum))
    # print('[%d]-th hqi:%.3f' % (epoch + 1, hqi_sum / num_sum))
    # print('[%d]-th mse:%.3f' % (epoch + 1, mse_sum / num_sum))

def forward( model,device, batch_size=1):
    train_data = test_dataloader(batch_size=batch_size)
    metric = defaultdict(float)
    model = model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch = 0
    with torch.no_grad():
        test_one_epoch(model, epoch,
                        device, train_data)
        torch.cuda.empty_cache()