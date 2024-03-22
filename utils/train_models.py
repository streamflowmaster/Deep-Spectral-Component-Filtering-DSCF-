import os.path
from collections import defaultdict

import matplotlib.pyplot as plt
import torch.nn.functional as F
from EfficientSpatialSpectralEmbedding.SemanticSegmentation.utils.loss import dice_loss
import time
import copy
from ComponentFiltering.utils.dataset import train_dataloader,test_dataloader
from ComponentFiltering.utils.models import Uresnet
import torch
import torch.optim as optim
from ComponentFiltering.utils.Interp import interp1d

mse = torch.nn.MSELoss()
# class SAD(torch.nn.Module):
#
#     def __init__(self):
#         super(SAD, self).__init__()
#
#     def forward(self,pred,target):
#         if len(pred.shape) == 3:
#             B,C,L = pred.shape
#             pred = pred.reshape(B*C,L)
#             target = target.reshape(B*C,L)
#         return torch.mm(pred,target.T).mean()/(pred.norm()*target.norm())

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
    sim = torch.diag(sim)
    # print(sim)
    return sim.mean()
# sad = SAD()
sad = pairwise_cos_sim

def calc_loss(pred, target, metrics, weight=0.7):
    # print(-sad(pred,target).log())
    # print(mse(pred,target))
    # loss = mse(pred,target) - weight*sad(pred,target).log()
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
        # plt.plot(target[0,0].cpu().numpy())
        if Lt!=Lo:
            coor_t = torch.linspace(0,1,Lt).unsqueeze(0).unsqueeze(0).repeat(B,C,1).reshape(B*C,Lt).to(device)
            coor_o = torch.linspace(0,1,Lo).unsqueeze(0).unsqueeze(0).repeat(B,C,1).reshape(B*C,Lo).to(device)
            target = interp1d(coor_t,target.reshape(B*C,Lt),coor_o).reshape(B,C,Lo)
        # plt.plot(target[0, 0].cpu().numpy())
        # plt.show()
        loss = calc_loss(outputs,target,metrics=metric)
        # print(loss)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    print('[%d]-th train loss:%.3f' % (epoch + 1, loss_sum))


def train(model, save_path, batch_size = 1,epoches = 100,lr= 1e-4,device = 'cuda:0',
          snr=None, head_path= '../ConstructedData/Training_15',dict_size = 15):

    train_data = train_dataloader(batch_size=batch_size,snr=snr,head_path=head_path,
                                  dict_size=dict_size,device = device)
    metric = defaultdict(float)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoches):
        lr /= 1.02
        train_one_epoch(model, epoch,
                        optimizer, metric, device, train_data)
        # test(model=model, embed_model=embed_model,metric = metric_test, test_path_head=head_path,device=device)
        if epoch % 1 == 0: torch.save(model.state_dict(), save_path)
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), save_path)







if __name__ == '__main__':
    model = Uresnet(inplanes=1,outplanes=50)
    if not os.path.exists('../WorkSpace/'):
        os.makedirs('../WorkSpace/')
    if os.path.exists('../WorkSpace/rbg_50.pt'):
        model.load_state_dict(torch.load('../WorkSpace/rbg_50.pt'))
    train(epoches=500,model=model,lr=5e-5,device='cuda:0',save_path='../WorkSpace/rbg.pt',batch_size=128,dict_size=50,head_path='../ConstructedData/Training_50')

