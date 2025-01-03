import os.path
from collections import defaultdict
import torch.nn.functional as F
import time
import copy
from dataset import fast_train_dataloader as train_dataloader
from cls_models import Hierarchical_1d_cls_model as cls_model
from DSCF_Submit.customized_task.DSCF_models_pe_ import Hierarchical_1d_model as filter_model
from evaluation import evaluate
import torch
import torch.optim as optim
import numpy as np

mse = torch.nn.CrossEntropyLoss()
evaluation = evaluate(cls_num=2)
def calc_loss(pred, target, metrics, bce_weight=0.7):
    loss = mse(pred,target)
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def length_adopt(B,Lc,Li,componnet):
    if Lc != Li:
        x = np.linspace(0, Lc - 1, Li)
        xp = np.linspace(0, Lc - 1, Lc)
        data_intp = torch.zeros((B,1,Li))
        for i in range(B):
            data_intp[i] = torch.tensor(np.interp(x, xp, componnet[i,0]))
        return data_intp

def train_one_epoch(model,epoch,optimizer,metric,device,train_data,f_model,component_idx):
    loss_sum = 0
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.long().to(device)
        if component_idx != None:
            componnets = f_model(inputs)
            component = componnets[:,component_idx].unsqueeze(1)
            B,C,Lc = componnets.shape
            B,_,Li = inputs.shape
            if Lc != Li: component = length_adopt(B,Lc,Li,component)
            inputs = inputs - component

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = calc_loss(outputs,target,metrics=metric)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    print('[%d]-th train loss:%.3f' % (epoch + 1, loss_sum))

def test_one_epoch(model,epoch,optimizer,metric,device,train_data,f_model,component_idx):
    loss_sum = 0
    for idx,(inputs,target) in enumerate(train_data):
        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.long().to(device)
        if component_idx != None:
            componnets = f_model(inputs)
            component = componnets[:,component_idx].unsqueeze(1)
            B,C,Lc = componnets.shape
            B,_,Li = inputs.shape
            if Lc != Li: component = length_adopt(B,Lc,Li,component)
            inputs = inputs - component

        optimizer.zero_grad()
        outputs = model(inputs)
        evaluation.calculation(target,outputs)
    return evaluation.eval()

def train(epoches, model,filter_model, lr, device, save_path, batch_size = 1,
          snr=None,head_path= '../ConstructedData/Training_15',
          dict_size = 15, logic_settings = 'AND.yaml',
          component_idx = None,
          val_path= '../ConstructedData/Val_15',):

    train_data = train_dataloader(batch_size=batch_size,snr=snr,head_path=head_path,
                                  dict_size=dict_size,logic_settings=logic_settings,device = device)
    # val_data = train_dataloader(batch_size=batch_size,snr=snr,head_path=val_path,
    #                               dict_size=dict_size,logic_settings=logic_settings,device = device)
    metric = defaultdict(float)
    model = model.to(device)
    f_model = filter_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoches):
        # lr /= 1.02
        train_one_epoch(model, epoch,
                        optimizer, metric, device, train_data,f_model,component_idx = component_idx)
        test_one_epoch(model, epoch,
                        optimizer, metric, device, train_data,f_model,component_idx = component_idx)
        # test(model=model, embed_model=embed_model,metric = metric_test, test_path_head=head_path,device=device)
        if epoch % 1 == 0: torch.save(model.state_dict(), save_path)
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), save_path)
