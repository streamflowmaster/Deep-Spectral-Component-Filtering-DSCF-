from collections import defaultdict

import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from dataset import train_loader
from DSCF_models_pe_ import Hierarchical_1d_model
import torch
import torch.optim as optim
from Interp import interp1d

def save_log(epoch,loss,lr,train_log_filename = "train_log.txt"):
    if not 'txt' in train_log_filename:
        train_log_filepath = train_log_filename[:-3]+'_log.txt'
    else:
        train_log_filepath = train_log_filename[:-4]+'_log.txt'

    train_log_txt_formatter = "Test:{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                              epoch=epoch,
                                              # lr = " ".join(["{}".format(lr)]),
                                              loss_str=" ".join(["{}".format(loss)]))

    with open(train_log_filepath, "a") as f:
        f.write(to_write)

def save_test_log(acc,lr,epoch = 999,train_log_filename = "train_log.txt"):

    if not 'txt' in train_log_filename:
        train_log_filepath = train_log_filename[:-3]+'_log.txt'
    else:
        train_log_filepath = train_log_filename[:-4]+'_log.txt'

    train_log_txt_formatter = "Test:{time_str} [Epoch] {epoch:03d} [lr] {lr:07d} [Acc] {loss_str}\n"
    to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                              epoch=epoch,
                                              lr = " ".join(["{}".format(lr)]),
                                              loss_str=" ".join(["{}".format(acc)]))
    with open(train_log_filepath, "a") as f:
        f.write(to_write)

mse = torch.nn.MSELoss()

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
    return sim.mean()

sad = pairwise_cos_sim

def calc_loss(pred, target, metrics, weight=0.7):
    # print(-sad(pred,target).log())
    # print(mse(pred,target))
    # loss = mse(pred,target) - weight*sad(pred,target).log()
    loss = mse(pred,target)
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def test_one_epoch(model,metric,device,train_data):
    loss_sum = 0
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.float().to(device)
        outputs = model(inputs)
        B,C,Lt = target.shape
        B,C,Lo = outputs.shape
        # print(target.shape,outputs.shape)
        # if Lt<Lo:
        #     outputs = outputs[:,:,:Lt]
        # else:
        #     target = target[:,:,:Lo]

        if Lt!=Lo:
            coor_t = torch.linspace(0,1,Lt).unsqueeze(0).unsqueeze(0).repeat(B,C,1).reshape(B*C,Lt).to(device)
            coor_o = torch.linspace(0,1,Lo).unsqueeze(0).unsqueeze(0).repeat(B,C,1).reshape(B*C,Lo).to(device)
            target = interp1d(coor_t,target.reshape(B*C,Lt),coor_o).reshape(B,C,Lo)
        # plt.plot(target[0,0,:].cpu().numpy())
        # plt.plot(outputs[0,0,:].cpu().numpy())
        # plt.show()
        title_list = [
            'component',
            'particle_bg',
            'fiber_bg',
            'baseline',
            'noise',
            'spike']
        plt.plot(inputs[0,0,:].cpu().numpy())
        plt.show()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.plot(target[0,  i, :].cpu().numpy())
            plt.plot(outputs[0, i, :].cpu().numpy())
            plt.title(title_list[i])
        plt.show()
        loss = calc_loss(outputs[0,0],target[0,0],metrics=metric)
        print(loss.item())
        loss_sum += loss.item()
    # save_log(epoch,loss_sum,lr,train_log_filename=save_pth)
    print('[%d]-th train loss:%.3f' % (1, loss_sum))


def forward(model,batch_size = 1,device = 'cuda:0',
          snr=25):
    train_data = train_loader(batch_size=batch_size,snr=snr,device = device)
    metric = defaultdict(float)
    model = model.to(device).eval()
    with torch.no_grad():
        test_one_epoch(model, metric, device,train_data)
