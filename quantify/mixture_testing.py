import torch
import os
import matplotlib.pyplot as plt
from forward_protocol import fetch_model

GT_dict = {
    'Mixture_01': [1,1,1,1,1],
    'Mixture_02': [2,2,2,1,2],
    'Mixture_03': [3,3,3,1,3],
    'Mixture_05': [2,3,3,2,1],
    'Mixture_06': [1,3,3,2,2],
    'Mixture_07': [2,2,1,2,3],
    'Mixture_08': [2,1,1,2,3],
    'Mixture_09': [3,2,2,3,1],
    'Mixture_10': [2,1,3,3,2],
    'Mixture_11': [1,1,2,3,3],
    'Mixture_12': [2,3,1,3,1],
}

def test_forward(model,device,mixture_pth = '../SERS_Quantify_Data/mixture_pt/'):
    predict_dict = {
    'Mixture_01': [1,1,1,1,1],
    'Mixture_02': [2,2,2,1,2],
    'Mixture_03': [3,3,3,1,3],
    'Mixture_05': [2,3,3,2,1],
    'Mixture_06': [1,3,3,2,2],
    'Mixture_07': [2,2,1,2,3],
    'Mixture_08': [2,1,1,2,3],
    'Mixture_09': [3,2,2,3,1],
    'Mixture_10': [2,1,3,3,2],
    'Mixture_11': [1,1,2,3,3],
    'Mixture_12': [2,3,1,3,1]
    }
    mixture_names = os.listdir(mixture_pth)
    model = model.to(device)
    for mixture_name in mixture_names:
        if '1000ms' in mixture_name:
            mixture = torch.load(os.path.join(mixture_pth,mixture_name))
            max_ = mixture.max(1)
            min_ = mixture.min(1)
            # plt.plot(mixture[0].cpu().numpy())
            # plt.title(mixture_name)
            # plt.show()
            mixture = (mixture - min_[0].reshape(-1,1)) / (max_[0] - min_[0]).reshape(-1,1)
            print(mixture_name)

            with torch.no_grad():
                mixture = mixture.to(device).unsqueeze(1).float()
                output = model(mixture)
                # print((max_[0] - min_[0]).shape)
                # output = output * (max_[0] - min_[0]).unsqueeze(1).unsqueeze(1).to(device)
                print(mixture.mean(0).shape)
                # for i in range(7):
                #     plt.subplot(7,1,i+1)
                #     plt.plot(output.mean(0)[i].cpu().numpy())
                #     plt.plot(mixture.mean(0)[0].cpu().numpy(),c = 'r')
                # plt.title(mixture_name)
                # plt.show()

                plt.plot(mixture.mean(0)[0].cpu().numpy(),c = 'r')
                plt.plot(output.mean(0).sum(0).cpu().numpy())
                plt.show()
            if 'Mixture' in mixture_name:
                print(mixture_name[:10])
                # if mixture_name[:10] in ['Mixture_01','Mixture_02',
                #                          'Mixture_06',
                #                          'Mixture_09','Mixture_10',
                #                          'Mixture_12']:
                predict_dict[mixture_name[:10]] = output.mean(0).cpu().numpy()
                output = output.max(0)[0].max(1)[0][:5]
                predict_dict[mixture_name[:10]] = output.cpu().numpy()
        print('---------------------')
    print(predict_dict)

    # for key in GT_dict.keys():
    #     for i in range(4):
    #         plt.subplot(4, 1, i + 1)
    #         plt.title(key)
    #         plt.scatter([1,3],[0,300],c='r')
    #         plt.scatter(GT_dict[key][i],predict_dict[key][i])
    #     plt.show()


    for i in range(5):
        plt.subplot(5, 1, i + 1)
        for key in GT_dict.keys():
            plt.title(key)
            # plt.scatter([1,3],[0,300],c='r')
            plt.scatter(GT_dict[key][i],predict_dict[key][i])
    plt.show()


model = fetch_model(snr=25,encoder_name='SiT',decoder_name='PPS',scale='large')
test_forward(model,'cuda:0')



