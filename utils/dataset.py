import os.path
import random
import torch
import torch.utils.data as tud
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks as fp
import torch.nn.functional as F

def return_dict(dict_size=1000):
    if 0 < dict_size <= 100:
        scale = '100'
    elif 100 < dict_size <= 1000:
        scale = '1k'
    elif 1000 < dict_size <= 10000:
        scale = '10k'
    else:
        scale = 'all'
    head_pth = '../IR_Dictionary'
    spectra_pth = os.path.join(head_pth, 'ir_dict_' + scale + '.pt')
    spectra_dict = torch.load(spectra_pth)[:dict_size]
    return spectra_dict

# def collate_fn(batch):
#     batch = list(zip(*batch))
#     batch[0] = nested_tensor_from_tensor_list(batch[0])
#     return tuple(batch)

def find_peaks(data:torch.Tensor):
    """
    实现AMPD算法
    :param data: 1-D numpy.ndarray
    :return: 波峰所在索引值的列表
    """
    if type(data) == torch.Tensor:
        data = data.cpu().numpy()

    print(data.shape)
    if len(data.shape) == 1:
        peaks, _ = fp(data, distance=10)
        print(len(peaks))
        return peaks
    else:
        peaks_list = []
        for i in range(data.shape[0]):
            peak = find_peaks(data[i])
            peaks_list.append(peak)
        return peaks_list

class Generation(tud.Dataset):

    def __init__(self,dict_size=1000,ir_head_pth = '../IR_Dictionary',
                 raman_head_pth = '../Raman_Dictionary',
                 uv_head_pth = '../UV_Dictionary',
                 sers_head_pth = '../SERS_Dictionary_1',
                 snr=None,
                 select_channel_num = 100,
                 device='cuda:0',if_train = True,prompt_select = None,
                 peak_num = None,multi_output_num = 16,
                 signal_length = 512):
        '''

        Args:
            dict_size:
            head_pth:
            snr:
            device:
            if_train:
            prompt_select: [0：'peaks',1：'bands',2：'query_spec_inputs']
        '''
        self.snr = snr
        self.device = device
        self.signal_length = signal_length
        self.if_train = if_train
        self.ir_head_pth = ir_head_pth
        self.raman_head_pth = raman_head_pth
        self.uv_head_pth = uv_head_pth
        self.sers_head_pth = sers_head_pth
        self.load_ir_dict(dict_size,if_train)
        self.load_raman_dict(dict_size,if_train)
        self.load_uv_dict(dict_size,if_train)
        self.load_sers_dict_(dict_size,if_train)
        self.prompt_select = prompt_select
        self.peak_num = peak_num
        self.dicts_name = ['ir','raman','sers','uv',]

        if self.prompt_select !=0:
            self.peak_num = None
        self.multi_output_num = multi_output_num

        self.select_channel_num = select_channel_num

        print(self.ir_spectra_dict.shape,
              self.raman_spectra_dict.shape,
              self.uv_spectra_dict.shape,
              self.sers_spectra_dict.shape)

    def load_ir_dict(self,dict_size,if_train):
        if if_train:
            if 0< dict_size <= 100:
                scale = '100'
            elif 100 < dict_size <= 1000:
                scale = '1k'
            elif 1000 < dict_size <= 10000:
                scale = '10k'
            else:
                scale = 'all'

            self.dict_size = dict_size
            spectra_pth = os.path.join(self.ir_head_pth,'ir_dict_'+scale+'.pt')
            self.ir_spectra_dict = torch.load(spectra_pth)[:dict_size].to(self.device)
            print('Spectra Dictionary Loaded on %s'%self.ir_spectra_dict.device)

        else:
            scale = 'all'
            self.dict_size = 30000
            spectra_pth = os.path.join(self.ir_head_pth,'ir_dict_'+scale+'.pt')
            self.ir_spectra_dict = torch.load(spectra_pth).to(self.device)


            print('Spectra Dictionary Loaded on %s'%self.ir_spectra_dict.device)
        # print(self.spectra_dict.shape)
        self.ir_spectra_dict = F.interpolate(self.ir_spectra_dict.unsqueeze(0),size=(self.signal_length),mode='linear').squeeze(0)
        self.ir_spectra_dict = self.ir_spectra_dict.to(self.device).float()
        # print(self.spectra_dict.shape)

    def load_raman_dict(self,dict_size,if_train):
        if if_train:
            if 0< dict_size <= 100:
                scale = '100'
            elif 100 < dict_size <= 1000:
                scale = '1k'
            elif 1000 < dict_size <= 10000:
                scale = '10k'
            else:
                scale = 'all'

            self.dict_size = dict_size
            spectra_pth = os.path.join(self.raman_head_pth,'raman_dict_'+scale+'.pt')
            self.raman_spectra_dict = torch.load(spectra_pth)[:dict_size].to(self.device)
            print('Spectra Dictionary Loaded on %s'%self.raman_spectra_dict.device)

        else:
            scale = 'all'
            self.dict_size = 30000
            spectra_pth = os.path.join(self.raman_head_pth,'raman_dict_'+scale+'.pt')
            self.raman_spectra_dict = torch.load(spectra_pth).to(self.device)
        self.raman_spectra_dict = F.interpolate(self.raman_spectra_dict.unsqueeze(0), size=(self.signal_length),
                                             mode='linear').squeeze(0)
        self.raman_spectra_dict = self.raman_spectra_dict.to(self.device).float()

    def load_uv_dict(self,dict_size,if_train):
        '''

        Args:
            dict_size:
            if_train:

        Returns:
            uv_spectra_dict:  dictionary shape is (Dict_len, L)
        '''
        if if_train:
            if 0< dict_size <= 100:
                scale = '100'
            elif 100 < dict_size <= 1000:
                scale = '1k'
            elif 1000 < dict_size <= 10000:
                scale = '10k'
            else:
                scale = 'all'

            self.dict_size = dict_size
            spectra_pth = os.path.join(self.uv_head_pth,'uv_dict_'+scale+'.pt')
            self.uv_spectra_dict = torch.load(spectra_pth)[:dict_size].to(self.device)
            print('Spectra Dictionary Loaded on %s'%self.uv_spectra_dict.device)

        else:
            scale = 'all'
            self.dict_size = 30000
            spectra_pth = os.path.join(self.uv_head_pth,'uv_dict_'+scale+'.pt')
            self.uv_spectra_dict = torch.load(spectra_pth).to(self.device)

        self.uv_spectra_dict = F.interpolate(self.uv_spectra_dict.unsqueeze(0), size=(self.signal_length),
                                                mode='linear').squeeze(0)
        self.uv_spectra_dict = self.uv_spectra_dict.to(self.device).float()

    def load_sers_dict(self,dict_size,if_train):

        '''

        Returns:  dictionary shape is (Dict_len, N, L)

        '''
        self.dict_list = os.listdir(self.sers_head_pth)
        dict_pth = os.path.join(self.sers_head_pth, self.dict_list[0])
        data = torch.load(dict_pth)
        N, L = data.shape
        self.sers_spectra_dict = torch.zeros((len(self.dict_list), 50, L))
        for idx, pth in enumerate(self.dict_list):
            dict_pth = os.path.join(self.sers_head_pth, pth)
            data = torch.load(dict_pth)
            N, L = data.shape
            if L == 723:
                self.sers_spectra_dict[idx, :, :L] = data[:50, :L]
                self.sers_spectra_dict[idx, :, -1] = data[:50, -1]
            else:
                self.sers_spectra_dict[idx, :, :] = data[:50]
        # self.component_num = len(self.dict_list)
        # self.component_dim = 50
        self.sers_spectra_dict = self.sers_spectra_dict.to(self.device)
        self.sers_spectra_dict = self.sers_spectra_dict.float().mean(dim=1)
        self.sers_spectra_dict = F.interpolate(self.sers_spectra_dict.unsqueeze(0), size=(self.signal_length),
                                                mode='linear').squeeze(0)
        self.sers_spectra_dict = self.sers_spectra_dict.to(self.device).float()

    def load_sers_dict_(self,dict_size,if_train):
        '''

        Args:
            dict_size:
            if_train:

        Returns:

        '''
        sers_pth = os.path.join(self.sers_head_pth,'sers_dict.pt')
        self.sers_spectra_dict = torch.load(sers_pth)[1:,1:]

        self.sers_spectra_dict = F.interpolate(self.sers_spectra_dict.unsqueeze(0), size=(self.signal_length),
                                                mode='linear').squeeze(0)
        self.sers_spectra_dict = self.sers_spectra_dict.to(self.device).float()

        print('Spectra Dictionary Loaded on %s' % self.sers_spectra_dict.device)

    def sparse_dense_mul(self,s, d):
        i = s._indices()
        v = s._values().unsqueeze(1)
        dv = d[i[0, :], :]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, (s.size(0),d.size(1)))

    def mix(self,spectra_dict):
        '''

        Args:
            spectra_dict:

        Returns:

        '''
        dict_size = spectra_dict.shape[0]
        min_mix_num = int(dict_size*0.25)
        max_mix_num = int(dict_size*0.75)
        mix_num = random.randint(min_mix_num,max_mix_num)
        # select the mixed spectra idx
        select_idx = torch.tensor(random.sample(range(0, dict_size), mix_num),
                                  device=self.device).unsqueeze(0)
        intensity = torch.randint(1, 1000, (mix_num,), device=self.device)
        # intensity,arg = torch.sort(intensity, descending=True)

        # the abundance of each spectrum
        label = torch.sparse.FloatTensor(select_idx, intensity, (dict_size,)).float()
        # mixing the spectra
        mixture = torch.sparse.mm(label.unsqueeze(0), spectra_dict).squeeze(0)
        seprate = self.sparse_dense_mul(label, spectra_dict).to_dense()
        # seprate = spectra_dict[select_idx[0],:]*intensity.unsqueeze(1)
        # normalize the spectra to 0-1
        mixture_max = mixture.max()
        mixture = mixture / mixture_max
        seprate = seprate / mixture_max
        # label = label / mixture_max
        # intensity = intensity / mixture_max
        return mixture.unsqueeze(0),torch.concat((mixture.unsqueeze(0),seprate), dim=0)

    def __len__(self):
        if self.if_train:
            return 10000
        else:
            return 1000

    def add_spike(self,spec):
        prob = random.randint(0,10)
        if self.if_train: threshold = 8
        else: threshold = 0
        if prob < threshold:
            return torch.zeros(spec.shape).to(self.device)
        else:
            max_inten = spec.max()
            spike = torch.zeros(spec.shape)
            spike_idx = random.randint(0,spec.shape[0]-1)
            spike[spike_idx] = max_inten*random.randint(50,200)/100
        return spike.to(self.device)


    def __getitem__(self, item):
        select_idx = random.randint(0,2)
        if select_idx == 0:
            spectra_dict = self.ir_spectra_dict.clone()
        elif select_idx == 1:
            spectra_dict = self.raman_spectra_dict.clone()
        elif select_idx == 2:
            spectra_dict = self.sers_spectra_dict.clone()
        else:
            spectra_dict = self.uv_spectra_dict.clone()

        # spectra_dict = self.uv_spectra_dict
        mixture,spectra_sentence = self.mix(spectra_dict)
        mixture = mixture +  self.add_noise(self.snr,mixture)+\
                              self.add_spike(mixture)
        return mixture, spectra_sentence[:self.select_channel_num],self.dicts_name[select_idx]


    def add_noise(self,snr,signal):
        # print(snr)
        if snr is None:
            return torch.zeros(signal.shape).to(self.device)
        else:
            signal_power = torch.square(torch.abs(signal)).mean()
            # print(signal_power)
            noise_power = math.pow(10,(-snr/10))*signal_power
            sigma = torch.sqrt(noise_power)
            # print(signal_power,noise_power,sigma)
            if torch.isnan(sigma):
                # print(signal_power,noise_power)
                # plt.plot(signal.cpu().numpy())
                # plt.show()
                sigma = 0.01
            noise = torch.normal(0,std=sigma,size=signal.shape)
            return noise.to(self.device)


    def get_batch(self,batch_size,block_size=36):
        batch = []
        for i in range(batch_size):
            spectra_sentence = self.__getitem__(i)[:block_size+1]
            # print(spectra_sentence.shape)
            batch.append(spectra_sentence)
        batch = torch.stack(batch)
        x = batch[:,:block_size]
        y = batch[:,1:block_size+1]
        # print(x.shape,y.shape)
        return x,y

def train_dataloader(sig_len,batch_size=128,dict_size=1000,snr=25,device='cuda:0'):
    dataset = Generation(dict_size=dict_size,snr=snr,if_train=True,device=device,signal_length=sig_len)
    dataloader = tud.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataloader

def test_dataloader(sig_len,batch_size=128,dict_size=1000,snr=25,device='cuda:0'):
    dataset = Generation(dict_size=dict_size,snr=snr,if_train=False,device=device,signal_length=sig_len)
    dataloader = tud.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataloader


if __name__ == '__main__':
    dataset = Generation(dict_size=1000,snr=25,if_train=True,device='cuda:1')
    plt.plot(dataset.sers_spectra_dict.cpu().numpy()[0])
    plt.show()
    for i in range(10):
        mixture, spectra_sentence,dicts_name = dataset.__getitem__(0)
        print(mixture.shape,spectra_sentence.shape,dicts_name)
        plt.plot(mixture.cpu().numpy()[0])
        plt.plot(spectra_sentence.cpu().numpy()[1])
        plt.title(dicts_name)
        plt.show()
