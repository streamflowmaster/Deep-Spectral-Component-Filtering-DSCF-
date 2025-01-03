import os.path
from ComponentFiltering.COMFILE.dataset import fast_test_dataloader as test_dataloader
from ComponentFiltering.COMFILE.cls_models import Hierarchical_1d_cls_model
from ComponentFiltering.utils.DSCF_models import Hierarchical_1d_model
from ComponentFiltering.COMFILE.evaluation import evaluate
import torch
import matplotlib.pyplot as plt
from ComponentFiltering.COMFILE.Interp import interp1d





def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def length_adopt(B, Lc, Li, componnet,device):
    B,C,Lc = componnet.shape
    # print(componnet.shape)
    # print(Lc,Li)
    if Li != Lc:
        coor_i = torch.linspace(0, 1, Li).unsqueeze(0).unsqueeze(0).repeat(B, C, 1).reshape(B * C, Li).to(device)
        coor_c = torch.linspace(0, 1, Lc).unsqueeze(0).unsqueeze(0).repeat(B, C, 1).reshape(B * C, Lc).to(device)
        componnet = interp1d(coor_c, componnet.reshape(B * C, Lc), coor_i).reshape(B, C, Li)
    return componnet

# def length_adopt(B, Lc, Li, componnet):
#     if Lc != Li:
#         componnet = componnet.cpu().detach().numpy()
#         x = np.linspace(0, Lc - 1, Li)
#         xp = np.linspace(0, Lc - 1, Lc)
#         data_intp = torch.zeros((B,1,Li))
#         for i in range(B):
#             data_intp[i] = torch.tensor(np.interp(x, xp, componnet[i,0]))
#         return data_intp

def test_one_epoch(model, device, train_data,f_model,component_idx,visual = False):
    evaluation = evaluate(cls_num=2)
    acc_sum = 0
    num_sum = 0
    for idx,(inputs,target) in enumerate(train_data):
        # print(inputs.shape)
        # print(target.shape)
        inputs = inputs.float().to(device).unsqueeze(0).unsqueeze(0)
        target = target.long().to(device).unsqueeze(0)

        if component_idx is not None:
            componnets = f_model(inputs)
            if type(component_idx) == torch.Tensor:
                component = componnets[:,component_idx].sum(1).unsqueeze(1)
            else:
                component = componnets[:,component_idx].unsqueeze(1)
            # print(component.shape)
            B,C,Lc = componnets.shape
            B,_,Li = inputs.shape
            if Lc != Li: component = length_adopt(B,Lc,Li,component,device).to(device)
            filtered_inputs = inputs - component


            if visual and idx ==0:
                # plt.plot(inputs.squeeze(1).cpu().detach().numpy()[0])
                # plt.plot(component.squeeze(1).cpu().detach().numpy()[0],c = 'b', linewidth = 5)
                # plt.plot(filtered_inputs.squeeze(1).cpu().detach().numpy()[0])
                fig = plt.figure(figsize=(4, 10), dpi=600)
                B,C,L = componnets.shape
                componnets = componnets.cpu().detach().numpy()[0]
                for i in range(C):
                    plt.plot(componnets[i]+i*1,c = 'b', linewidth = 2)
                plt.show()

                plt.plot(inputs.squeeze(1).cpu().detach().numpy()[0])
                plt.show()

        else:
            filtered_inputs = inputs
        # print(filtered_inputs.shape)
        # print(target.shape)
        filtered_inputs = filtered_inputs.squeeze(0).detach().cpu().numpy()
        acc = model.score(filtered_inputs,target.cpu())
        acc_sum += acc
        num_sum += 1
        if idx>500: break
    return acc_sum/num_sum




def eva(model,device,filter_model,test_data,component_idx = None):
    f_model = filter_model.to(device)
    return test_one_epoch(model,device, test_data,
                   f_model = f_model,component_idx = component_idx)

if __name__ == '__main__':
    model = Hierarchical_1d_cls_model(sig_len=709)
    f_model = Hierarchical_1d_model(layers=[1,2,2,1],outplanes=8)
    if not os.path.exists('../WorkSpace/'):
        os.makedirs('../WorkSpace/')
    if os.path.exists('../WorkSpace/rbg_50.pt'):
        model.load_state_dict(torch.load('../WorkSpace/rbg_50.pt'))
    eva(model=model,filter_model=f_model,device='cuda:0',batch_size=128,
        dict_size=50,head_path='../ConstructedData/Training_50',component_idx=1)





