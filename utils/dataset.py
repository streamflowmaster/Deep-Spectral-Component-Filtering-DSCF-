import os.path
import random
import torch
import torch.utils.data as tud
import numpy as np
import matplotlib.pyplot as plt
import math
def return_dict():
    dict_pth = '../SpectraDict/Reference123.txt'
    dict_size = 122
    spectra_dict = torch.tensor(np.loadtxt(dict_pth))[:,1:dict_size+1]
    return spectra_dict.T


class FastGeneration(tud.Dataset):
    def __init__(self,head_path = '../ConstructedData/Testing', spectra_pth = 'data_with_BG',snr = None,
                 labe_path = 'BG_GT',dict_size = 15, batch = 200,
                 spectra_dict_pth = '../SpectraDict/Reference123.txt',
                 logic_settings = 'AND.yaml',device= 'cuda:1'):

        labe_path = os.path.join(head_path,labe_path)
        spectra_pth = os.path.join(head_path,spectra_pth)
        self.labels =  os.listdir(labe_path)
        self.spectra = os.listdir(spectra_pth)
        self.batch = batch
        self.labels_pth = labe_path
        self.spectra_pth = spectra_pth
        self.dict_size = dict_size
        self.snr = snr
        self.device = device
        self.load_all_data()
        self.targets = self.targets.to(device)
        self.specs = self.specs.to(device)
        if spectra_dict_pth != None:
            self.spectra_dict = torch.tensor(np.loadtxt(spectra_dict_pth))[:,1:dict_size+1].to(device)
            self.label_is_abundance_vec = True
        else:
            self.label_is_abundance_vec = False

    def __len__(self):
        return len(self.labels)*self.batch

    def load_all_data(self):
        sepc = torch.load(os.path.join(self.spectra_pth, self.spectra[0]))
        label = torch.load(os.path.join(self.labels_pth, self.spectra[0]))
        C,N = sepc.shape
        C,D = label.shape
        self.specs = torch.zeros((len(self.labels),self.batch,N))
        self.targets = torch.zeros((len(self.labels),self.batch,self.dict_size))
        for idx in range(len(self.labels)):
            self.specs[idx]  = torch.load(os.path.join(self.spectra_pth, self.spectra[idx]))
            self.targets[idx] = torch.load(os.path.join(self.labels_pth, self.spectra[idx]))

    def __getitem__(self, item):
        file_id = item//self.batch
        spec_id = item% self.batch
        spec = self.specs[file_id,spec_id]
        label = self.targets[file_id,spec_id]
        if self.snr != None:
             spec = self.add_noise(self.snr,spec)
        if self.label_is_abundance_vec:
            # print(labels.shape,self.spectra_dict.shape)
            label = torch.multiply(label,self.spectra_dict)  # channel_dim*spectra_dim * 1*spectra_dim -> channel_dim*spectra_dim
        if self.snr != None:
             spec = self.add_noise(self.snr,spec)
        return spec,label.T

    def add_noise(self,snr,signal):
        signal_power = torch.square(signal).mean(0)
        noise_power = math.pow(10,(-snr/10))*signal_power
        sigma = torch.sqrt(noise_power)
        noise = torch.normal(0,std=sigma,size=signal.shape).to(self.device)
        signal = signal+noise
        signal = torch.abs(signal)
        return signal

class Generation(tud.Dataset):

    def __init__(self,device,head_path = '../ConstructedData/Testing', spectra_pth = 'data_with_BG',snr = None,
                 labe_path = 'BG_GT',dict_size = 15, batch = 200, spectra_dict_pth = '../SpectraDict/Reference123.txt'):
        labe_path = os.path.join(head_path,labe_path)
        spectra_pth = os.path.join(head_path,spectra_pth)
        self.labels =  os.listdir(labe_path)
        self.spectra = os.listdir(spectra_pth)
        self.batch = batch
        self.labels_pth = labe_path
        self.spectra_pth = spectra_pth
        self.dict_size = dict_size
        self.snr = snr
        if spectra_dict_pth != None:
            self.spectra_dict = torch.tensor(np.loadtxt(spectra_dict_pth))[:,1:dict_size+1]
            self.label_is_abundance_vec = True
        else:
            self.label_is_abundance_vec = False

    def __len__(self):
        return len(self.labels)*self.batch

    def __getitem__(self, item):
        file_id = item//self.batch
        spec_id = item% self.batch
        specs = torch.load(os.path.join(self.spectra_pth, self.spectra[file_id]))
        labels = torch.load(os.path.join(self.labels_pth, self.spectra[file_id]))
        C,N = specs.shape
        # print(specs.shape)
        spec = specs[spec_id] # channel_dim*batch
        # print(spec.shape)
        label = labels[spec_id,:].unsqueeze(0)  # batch*spectra_dim -> spectra_dim*1
        if self.label_is_abundance_vec:
            # print(labels.shape,self.spectra_dict.shape)
            label = torch.multiply(label,self.spectra_dict)  # channel_dim*spectra_dim * 1*spectra_dim -> channel_dim*spectra_dim
        # plt.plot(label[:,0].cpu().detach().numpy())
        # plt.plot(self.spectra_dict[:,0].cpu().detach().numpy())
        # plt.show()
        if self.snr != None:
             spec = self.add_noise(self.snr,spec)
        # print(label.shape)
        # print(spec.shape)
        return spec,label.T

    def add_noise(self,snr,signal):
        signal_power = torch.square(signal).mean(0)
        noise_power = math.pow(10,(-snr/10))*signal_power
        sigma = torch.sqrt(noise_power)
        noise = torch.normal(0,std=sigma,size=signal.shape)
        signal = signal+noise
        signal = torch.abs(signal)
        return signal



def train_dataloader(batch_size,snr=25, head_path='../ConstructedData/Testing_15', dict_size=15,device = 'cuda:0'):
    dataset = Generation(head_path=head_path, dict_size=dict_size,snr=snr,device=device)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

def test_dataloader(batch_size,head_path = '../ConstructedData/Training_15', dict_size=15,device = 'cuda:0'):
    dataset = Generation(head_path=head_path, dict_size=dict_size,device=device)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

def quan_dataloader(batch_size,head_path = '../ConstructedData/Training_15', dict_size=15,device = 'cuda:0'):
    dataset = Generation(head_path=head_path, dict_size=dict_size,device=device,spectra_dict_pth=None)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

if __name__ == '__main__':
    # print(return_dict().shape)
    loader = train_dataloader(5,snr=30)
    for idx,(spec,label) in enumerate(loader):
        signal,background = spec, label
        print(signal.shape)
        print(background.shape)
        plt.plot(signal[0])
        # plt.plot(background[0])
        plt.show()