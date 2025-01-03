import os.path
from collections import defaultdict
import torch.nn.functional as F
from EfficientSpatialSpectralEmbedding.SemanticSegmentation.utils.loss import dice_loss
import time
import copy
from ComponentFiltering.COMFILE.dataset import train_dataloader,test_dataloader
from ComponentFiltering.COMFILE.cls_models import Hierarchical_1d_cls_model
import torch
import torch.optim as optim

mse = torch.nn.CrossEntropyLoss()
def calc_loss(pred, target, metrics, bce_weight=0.7):
    loss = mse(pred,target)
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
        target = target.long().to(device)
        optimizer.zero_grad()
        # print(inputs.shape)
        outputs = model(inputs)
        # print('input:', inputs.shape)
        # print('output:',outputs.shape)
        # print('target:', target.shape)

        # B,C,Lt = target.shape
        # B,C,Lo = outputs.shape
        # # print(Lt,Lo)
        # if Lt<Lo:
        #     outputs = outputs[:,:,:Lt]
        # else:
        #     target = target[:,:,:Lo]

        loss = calc_loss(outputs,target,metrics=metric)
        # print(loss)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    print('[%d]-th train loss:%.3f' % (epoch + 1, loss_sum))


def train(epoches, model,lr, device, save_path, batch_size = 1,
          snr=None,head_path= '../ConstructedData/Training_15',
          dict_size = 15, logic_settings = 'AND.yaml'):
    train_data = train_dataloader(batch_size=batch_size,snr=snr,head_path=head_path,
                                  dict_size=dict_size,logic_settings=logic_settings)
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




def test(model,embed_model,metric,device,test_path_head= '../ConstructedData/Testing_15'):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embed_model = embed_model.to(device)
    test_data = test_dataloader(test_path_head,batch_size=1,device=device)

    with torch.no_grad():
        loss_sum = 0
        for epoch in range(1):
            for idx, (inputs,target) in enumerate(test_data):
                inputs = inputs.float().to(device)
                b, c, h, w = inputs.shape
                # inputs = inputs.reshape(b, c, h * w).permute(0, 2, 1)
                target = target.to(device)
                outputs = embed_model(inputs)
                b, n, e = outputs.shape
                outputs = model(outputs.permute(0, 2, 1).reshape(b, e, h, w))
                loss_sum += dice_loss(outputs,target).item()
        print('[%d]-th Test loss:%.3f' % (epoch + 1, metric['loss']))
        print('Dice on test set: %d' % metric['dice'])
        # with open("test.txt", "a") as f:
         #     f.write('Accuracy on test set: (%d/%d)%d %% \n' % (correct, total, 100 * correct / total))


if __name__ == '__main__':
    model = Hierarchical_1d_cls_model(sig_len=709)
    if not os.path.exists('../WorkSpace/'):
        os.makedirs('../WorkSpace/')
    if os.path.exists('../WorkSpace/rbg_50.pt'):
        model.load_state_dict(torch.load('../WorkSpace/rbg_50.pt'))
    train(epoches=500,model=model,lr=5e-5,device='cuda:0',save_path='../WorkSpace/rbg.pt',batch_size=128,dict_size=50,head_path='../ConstructedData/Training_50')

