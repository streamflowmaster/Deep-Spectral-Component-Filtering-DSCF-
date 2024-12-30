import os.path
from collections import defaultdict
from dataset import fintuning_dataloader,length_adopt
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
mse = torch.nn.MSELoss()
l1loss = torch.nn.L1Loss()

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [...,M,D]
    :param x2: [...,N,D]
    :return: similarity matrix [...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    # print(x1.shape,x2.shape)
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    # print(sim.shape)
    sim = torch.diag(sim)
    return sim.mean()

sad = pairwise_cos_sim

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.shape[0]
        h_x = x.shape[2]
        count_h =  x.shape[2]-1
        h_tv = torch.pow((x[:,:,1:]-x[:,:,:h_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h)/batch_size

tvloss = TVLoss()


def calc_loss(pred, target, metrics, bce_weight=0.2):
    # loss = mse(pred,target) - bce_weight*sad(pred,target).log()
    loss = mse(pred, target)
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_one_epoch(model,epoch,optimizer,metric,device,train_data):
    loss_sum = 0
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print('output:',outputs.shape,
        #       'target:',target.shape)
        B,C,Lt = target.shape
        B,C,Lo = outputs.shape
        # print(Lt,Lo)
        if Lt>Lo:
            outputs = F.pad(outputs,(0,Lt-Lo))
        elif Lt<Lo:
            target = F.pad(target,(0,Lo-Lt))

        loss = calc_loss(outputs.squeeze(1),target.squeeze(1),metrics=metric)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    print('[%d]-th train loss:%.3f' % (epoch + 1, loss_sum))


def finetune(epoches, model,lr, device, save_path, batch_size = 1,snr=15):
    train_data = fintuning_dataloader(batch_size=batch_size,device=device,snr=snr)
    metric = defaultdict(float)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epoches):

        train_one_epoch(model, epoch,optimizer, metric, device, train_data)
        torch.save(model.state_dict(), save_path)
        torch.cuda.empty_cache()
        lr_scheduler.step()

    torch.save(model.state_dict(), save_path)



