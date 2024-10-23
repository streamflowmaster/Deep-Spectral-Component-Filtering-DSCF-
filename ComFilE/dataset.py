import os.path
import random
import torch
import torch.utils.data as tud
import numpy as np
import matplotlib.pyplot as plt
import math
from ComponentFiltering.COMFILE.generating_virtual_label import generate_label
def return_dict():
    dict_pth = '../SpectraDict/Reference123.txt'
    dict_size = 122
    spectra_dict = torch.tensor(np.loadtxt(dict_pth))[:,1:dict_size+1]
    return spectra_dict.T

class FastGeneration(tud.Dataset):
    def __init__(self,head_path = '../ConstructedData/Testing', spectra_pth = 'data_with_BG',snr = None,
                 labe_path = 'BG_GT',dict_size = 15, batch = 200,
                 spectra_dict_pth = '../SpectraDict/Reference123.txt',
                 logic_settings = 'AND.yaml',device= 'cuda:0'):

        labe_path = os.path.join(head_path,labe_path)
        spectra_pth = os.path.join(head_path,spectra_pth)
        self.labels =  os.listdir(labe_path)
        self.spectra = os.listdir(spectra_pth)
        self.batch = batch
        self.labels_pth = labe_path
        self.spectra_pth = spectra_pth
        self.dict_size = dict_size
        self.virtual_label_logic = generate_label(logic_file=logic_settings)
        self.snr = snr
        self.device = device
        self.load_all_data()
        self.targets = self.targets.to(device)
        self.specs = self.specs.to(device)
        if spectra_dict_pth != None:
            self.spectra_dict = torch.tensor(np.loadtxt(spectra_dict_pth))[:,1:dict_size+1]
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
        # print(label.shape)
        # print(spec.shape)
        cls = self.virtual_label_logic.generate(abm=label)
        return spec,cls

    def add_noise(self,snr,signal):
        signal_power = torch.square(signal).mean(0)
        noise_power = math.pow(10,(-snr/10))*signal_power
        sigma = torch.sqrt(noise_power)
        noise = torch.normal(0,std=sigma,size=signal.shape).to(self.device)
        signal = signal+noise
        signal = torch.abs(signal)
        return signal

class Generation(tud.Dataset):

    def __init__(self,head_path = '../ConstructedData/Testing', spectra_pth = 'data_with_BG',snr = None,
                 labe_path = 'BG_GT',dict_size = 15, batch = 200,
                 spectra_dict_pth = '../SpectraDict/Reference123.txt',
                 logic_settings = 'AND.yaml'):

        labe_path = os.path.join(head_path,labe_path)
        spectra_pth = os.path.join(head_path,spectra_pth)
        self.labels =  os.listdir(labe_path)
        self.spectra = os.listdir(spectra_pth)
        self.batch = batch
        self.labels_pth = labe_path
        self.spectra_pth = spectra_pth
        self.dict_size = dict_size
        self.virtual_label_logic = generate_label(logic_file=logic_settings)
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
        label = labels[spec_id,:]  # batch*spectra_dim -> spectra_dim*1
        # if self.label_is_abundance_vec:
        #     # print(labels.shape,self.spectra_dict.shape)
        #     label = torch.multiply(label,self.spectra_dict)  # channel_dim*spectra_dim * 1*spectra_dim -> channel_dim*spectra_dim
        if self.snr != None:
             spec = self.add_noise(self.snr,spec)
        # print(label.shape)
        # print(spec.shape)
        cls = self.virtual_label_logic.generate(abm=label)
        return spec,cls

    def add_noise(self,snr,signal):
        signal_power = torch.square(signal).mean(0)
        noise_power = math.pow(10,(-snr/10))*signal_power
        sigma = torch.sqrt(noise_power)
        noise = torch.normal(0,std=sigma,size=signal.shape)
        signal = signal+noise
        signal = torch.abs(signal)
        return signal

def fast_train_dataloader(batch_size,snr=None, head_path='../ConstructedData/Testing_15', dict_size=15,device = 'cuda:0', logic_settings = 'AND.yaml'):
    dataset = FastGeneration(head_path=head_path, dict_size=dict_size,snr=snr,logic_settings = logic_settings,device=device)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
def fast_test_dataloader(batch_size,head_path = '../ConstructedData/Training_15', dict_size=15,logic_settings = 'AND.yaml',snr = None):
    dataset = FastGeneration(head_path=head_path, dict_size=dict_size,logic_settings = logic_settings, snr = snr)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

def train_dataloader(batch_size,snr=None, head_path='../ConstructedData/Testing_15', dict_size=15,logic_settings = 'AND.yaml'):
    dataset = Generation(head_path=head_path, dict_size=dict_size,snr=snr,logic_settings = logic_settings)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

def test_dataloader(batch_size,head_path = '../ConstructedData/Training_15', dict_size=15,logic_settings = 'AND.yaml',snr = None):
    dataset = Generation(head_path=head_path, dict_size=dict_size,logic_settings = logic_settings, snr = snr)
    return tud.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

if __name__ == '__main__':
    # print(return_dict().shape)
    loader = fast_train_dataloader(1,snr=30,head_path='../ConstructedData/Testing_50',dict_size=50)
    # loader = train_dataloader(1,snr=30,head_path='../ConstructedData/Testing_50',dict_size=50)
    for idx,(spec,label) in enumerate(loader):
        signal,cls = spec, label
        print(cls)