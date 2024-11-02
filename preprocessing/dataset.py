import os.path
import random
import torch
import torch.utils.data as tud
import matplotlib.pyplot as plt
import math
from Interp import interp1d
from scipy.signal import savgol_filter
# background = torch.load('../../../LiverFTIR/pt_data81.pt')
# C,H,W = background.shape
# background= background.reshape(C,H*W)
# print(background)
# plt.plot(background[:800,:200])
# plt.show()

def length_adopt(signal,background):
    L = background.shape[-1]
    S = signal.shape[-1]
    device = signal.device
    # print(Lt,Lo)
    if L!=S:
        if len(signal.shape) == 3:
            B,C,S = signal.shape
            coor_t = torch.linspace(0,1,S).unsqueeze(0).unsqueeze(0).repeat(1,1,1).reshape( B*C,S).to(device)
            coor_o = torch.linspace(0,1,L).unsqueeze(0).unsqueeze(0).repeat(1,1,1).reshape( B*C,S).to(device)
            signal = interp1d(coor_t,signal.reshape(B*C,S),coor_o).reshape( B,C,L).to(device)
        elif len(signal.shape) == 2:
            C,S = signal.shape
            coor_t = torch.linspace(0,1,S).unsqueeze(0).repeat(C,1).reshape(C,S).to(device)
            coor_o = torch.linspace(0,1,L).unsqueeze(0).repeat(C,1).reshape(C,L).to(device)
            signal = interp1d(coor_t,signal,coor_o).reshape(C,L).to(device)
        elif len(signal.shape) == 1:
            S = signal.shape[0]
            coor_t = torch.linspace(0,1,S).to(device)
            coor_o = torch.linspace(0,1,L).to(device)
            signal = interp1d(coor_t,signal,coor_o).to(device)
    return signal

class Generation(tud.Dataset):

    def __init__(self,device = 'cuda:1',snr = None):
        self.device = device
        self.AD_head_pth= '../SERS_Background_Removal_AD/pure_data/'
        self.pure_data_list_AD = os.listdir(self.AD_head_pth)
        # self.pure_data_list_AD = []
        self.pure_data_list_AD.sort()
        self.pure_data_list_AD = [os.path.join(self.AD_head_pth,_) for _ in self.pure_data_list_AD]
        self.PCA_head_pth = '../SERS_Background_Removal_PCa/pure_aligned_data/'
        self.pure_data_list_PCA = os.listdir(self.PCA_head_pth)
        self.pure_data_list_PCA.sort()
        self.pure_data_list_PCA = [os.path.join(self.PCA_head_pth,_) for _ in self.pure_data_list_PCA]
        self.background_pth = '../SERS_Background_Removal_PCa/Int-Interp-Citrate-AgNPs.pt'
        self.dict_pth= '../SERS_Dictionary/'
        self.Channel = 1799

        self.background = torch.load(self.background_pth)[:,:self.Channel]
        self.background = self.max_min_norm(self.background).to(self.device)
        print(self.background.shape)
        self.load_dictionary()
        self.load_pure_data()


    def __len__(self):
        return self.dim*10


    def load_pure_data(self):
        self.pure_data_list = self.pure_data_list_PCA+self.pure_data_list_AD

        self.dim = 0
        for data_path in self.pure_data_list:
            spec = torch.load(data_path)
            if len(spec.shape) == 3:
                C,H,W = spec.shape
                spec = spec.reshape(C,H*W)
                N = H*W
            elif len(spec.shape) == 2:
                spec = spec.T
                C, N = spec.shape
                # print(spec.shape)
            else: N = 1
            self.dim += N
        self.pure_data = torch.zeros((self.Channel,self.dim))

        print(self.dim)
        dim = 0
        for data_path in self.pure_data_list:
            spec = torch.load(data_path)
            if len(spec.shape) == 3:
                C, H, W = spec.shape
                spec = spec.reshape(C, H * W)
                N = H * W
            elif len(spec.shape) == 2:
                spec = spec.T
                C, N = spec.shape
            else:
                N = 1
            if C<self.Channel:spec= length_adopt(spec.T,self.background).T
            self.pure_data[:,dim:dim+N] = self.max_min_norm(spec[:self.Channel].T).T
            dim += N
        self.pure_data = self.pure_data.to(self.device)
        return self.pure_data

    def max_min_norm(self,spec):
        if type(spec) != torch.Tensor:
            spec = torch.tensor(spec)

        if len(spec.shape) == 1:
            max_ = spec.max(0)[0]
            min_ = spec.min(0)[0]
            spec = (spec-min_)/(max_-min_)
            # print(spec.shape)
        if len(spec.shape) == 2:
            max_ = spec.max(1)[0].unsqueeze(1)
            min_ = spec.min(1)[0].unsqueeze(1)
            spec = (spec-min_)/(max_-min_)
        return spec

    def load_dictionary(self):
        '''

        Returns:  dictionary shape is (Dict_len, N, L)

        '''
        self.dict_list = os.listdir(self.dict_pth)
        dict_pth = os.path.join(self.dict_pth, self.dict_list[0])
        data = torch.load(dict_pth)
        self.signal_compnents = torch.zeros((len(self.dict_list),self.Channel))
        for idx,pth in enumerate(self.dict_list):
            dict_pth = os.path.join(self.dict_pth,pth)
            data = torch.load(dict_pth)
            data = length_adopt(data,self.background)
            # print(data.shape)
            self.signal_compnents[idx,:] = data[:,:self.Channel].mean(0)

        self.component_num = len(self.dict_list)
        # self.signal_compnents = self.signal_compnents.mean(1).to(self.device)
        com_max = self.signal_compnents.max(1)[0].unsqueeze(1).repeat(1,self.Channel)
        com_min = self.signal_compnents.min(1)[0].unsqueeze(1).repeat(1,self.Channel)
        self.signal_compnents = (self.signal_compnents-com_min)/(com_max-com_min)
        self.signal_compnents = self.signal_compnents.to(self.device)
        return self.signal_compnents


    def get_mixture(self):
        label_ = torch.rand([1,self.component_num],device=self.device)+1e-5
        idx__ = torch.linspace(0,self.component_num-1,steps=self.component_num).long()
        signal = torch.mm(label_,self.signal_compnents[idx__,:]).mean(0)
        signal[1650:] = 0.2 * signal[1650:]
        # print(signal)
        if_shift = random.randint(0,4)
        if if_shift == 0:
            signal = self.signal_shift(signal)
        return self.max_min_norm(signal)

    def get_pure(self):
        seed = random.randint(0,self.dim-1)
        pure = self.pure_data[:,seed]
        pure[1650:] = 0.2 * pure[1650:]
        return self.max_min_norm(pure)

    def signal_shift(self,signal):
        shift = random.randint(-40,60)
        if shift == 0: return signal
        else:
            pad_signal = torch.ones(abs(shift)).to(self.device)
            if shift>0:
                signal = torch.cat([signal[shift:],signal[-1]*pad_signal],dim=0)
            else:
                signal = torch.cat([signal[0]*pad_signal,signal[:shift]],dim=0)
        return signal

    def get_background(self):
        bg_selector = random.randint(0,self.background.shape[0]-1)
        bg_ampor = random.uniform(0,8)
        back_select = self.background[bg_selector]
        back_select = self.max_min_norm(back_select)
        back_select[800:1100] = random.uniform(0.6,1.2)*back_select[800:1100]
        back_select[1100:1400] = random.uniform(0.6,2.5)*back_select[1100:1400]
        back_select[1550:] = random.uniform(0.6,1.2)*back_select[1550:]

        if_shift = random.randint(0,4)
        if if_shift == 0:
            back_select = self.signal_shift(back_select)

        return back_select*bg_ampor

    def add_noise(self,snr,signal):
        # print(signal)
        if snr == None: return torch.zeros(signal.shape).to(self.device)
        else:
            signal_power = torch.square(signal).mean(0)
            noise_power = math.pow(10,(-snr/10))*signal_power
            sigma = math.sqrt(noise_power)
            noise = torch.normal(0.0,std=sigma,size=signal.shape).to(self.device)
            return noise

    def __getitem__(self, item):
        seed = random.randint(0,1)
        if seed == 0 or seed==1: signal = self.get_pure().to(self.device)
        elif seed == 2: signal = self.get_mixture().to(self.device)+self.get_pure().to(self.device)
        else: signal = self.get_mixture().to(self.device)
        noise = self.add_noise(snr=None,signal=signal).to(self.device)
        background = self.get_background().to(self.device)
        # print(signal.shape,background.shape,noise.shape)
        signal = signal+background+noise
        signal_max = signal.max()
        signal = signal/signal_max
        background = background/signal_max
        return signal,background


def train_dataloader(batch_size,head_pth='../../FTIR_Background_Removal/',device = 'cuda:1'):
    dataset = Generation(device=device)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

def test_dataloader(batch_size,head_pth='../../FTIR_Background_Removal/'):
    dataset = Generation()
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

if __name__ == '__main__':
    dataset = Generation()
    signal_compnents = dataset.signal_compnents
    # pure_data = dataset.pure_data
    # background = dataset.background
    # plt.plot(pure_data.mean(1).cpu().detach().numpy(),c='k',alpha=1)
    # plt.plot(background.T.cpu().detach().numpy(),c='r',alpha=0.1)
    # plt.show()

    print(signal_compnents)
    for idx in range(signal_compnents.shape[0]):
        plt.plot(signal_compnents[idx].cpu().detach().numpy())
    plt.show()
    loader = train_dataloader(5)
    for idx,data in enumerate(loader):
        signal,background = data
        print(signal.shape)
        print(background.shape)
        plt.plot(signal[0].cpu().detach().numpy())
        plt.plot(background[0].cpu().detach().numpy())
        plt.show()