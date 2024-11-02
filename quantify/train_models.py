from collections import defaultdict
import torch.nn.functional as F
import time
from dataset import train_loader
from DSCF_models_pe_ import Hierarchical_1d_model
import torch
import torch.optim as optim
from Interp import interp1d
import os
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
def SAM(x_true, x_pred):
    "calculate method in PSGAN code"

    assert len(x_true.shape) == 3 and x_true.shape == x_pred.shape
    dot_sum = torch.sum(x_true * x_pred, dim=2)
    norm_true = torch.linalg.norm(x_true, axis=2)
    norm_pred = torch.linalg.norm(x_pred, axis=2)
    res = torch.arccos(dot_sum / norm_pred / norm_true)
    # is_nan = torch.nonzero(torch.isnan(res))
    # for (x, y) in zip(is_nan[0], is_nan[1]):
    #     res[x, y] = 0
    sam = torch.mean(res)
    # print(sam)
    return sam

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


def train_one_epoch(model,epoch,optimizer,metric,device,train_data,save_pth,lr):
    loss_sum = 0
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        B,C,Lt = target.shape
        B,C,Lo = outputs.shape
        # print(target.shape,outputs.shape)
        # if Lt<Lo:
        #     outputs = outputs[:,:,:Lt]
        # else:
        #     target = target[:,:,:Lo]


        # plt.plot(outputs[0,0].cpu().numpy())
        if Lt!=Lo:
            coor_t = torch.linspace(0,1,Lt).unsqueeze(0).unsqueeze(0).repeat(B,C,1).reshape(B*C,Lt).to(device)
            coor_o = torch.linspace(0,1,Lo).unsqueeze(0).unsqueeze(0).repeat(B,C,1).reshape(B*C,Lo).to(device)
            target = interp1d(coor_t,target.reshape(B*C,Lt),coor_o).reshape(B,C,Lo)
        # plt.plot(outputs[0, 0].cpu().numpy())
        # plt.show()
        loss = calc_loss(outputs,target,metrics=metric)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    save_log(epoch,loss_sum,lr,train_log_filename=save_pth)
    print('[%d]-th train loss:%.3f' % (epoch + 1, loss_sum))


def train(model, save_path, batch_size = 1,epoches = 100,lr= 1e-4,device = 'cuda:0',
          snr=None):
    train_data = train_loader(batch_size=batch_size,snr=snr,device = device)
    metric = defaultdict(float)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    log_save_pth_ = save_path[:-3]+'_log.txt'
    log_save_pth = save_path[:-2] + '.txt'
    if os.path.exists(log_save_pth_):
        with open(log_save_pth_, "r", encoding='utf-8') as f:  # 打开文本
            log = f.readlines()  # 读取文本
        idx_epoch = log[-1].find('Epoch')
        idx_loss = log[-1].find('Loss')
        start_epoch = int(log[-1][idx_epoch+7:idx_loss-1])
    else: start_epoch = 0
    for epoch in range(start_epoch,start_epoch+epoches):
        lr /= 1.02
        train_one_epoch(model, epoch,
                        optimizer, metric, device, train_data,save_path,lr)
        # test(model=model, embed_model=embed_model,metric = metric_test, test_path_head=head_path,device=device)
        if epoch % 20 == 0: torch.save(model.state_dict(), save_path)
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), save_path)
