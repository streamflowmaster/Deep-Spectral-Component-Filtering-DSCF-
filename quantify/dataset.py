'''
    different channels output
    channel-0 signal_components
    channel-1 particle background
    channel-2 noise
    channel-3 fiber background
    channel-4 baseline
    channel-5 spike (abnormal single-points)

            component,
            particle_bg,
            fiber_bg,
            baseline,
            noise,
            spike

    Training only uses the constructed data without any real data

    particle background is from pure bg signal
    noise is gaussian white noise
    fiber background is from pure fiber bg signal
    baseline is extracted from the pure data

'''
import os
import random
import torch
import torch.utils.data as tud
import math

class dataset(tud.Dataset):
    def __init__(self,dictionary_pth = '../SERS_Dictionary',particle_bg_pth='../SERS_Particle_bg',
                 baseline_pth='../SERS_Baselines',fiber_bg_pth = '../SERS_Fiber_bg',snr = 25,
                 device = 'cpu',target_pth = '../SERS_Quantify_Data/library_pt/'):
        self.dict_pth = dictionary_pth
        self.particle_bg_pth = particle_bg_pth
        self.baseline_pth = baseline_pth
        self.fiber_bg_pth = fiber_bg_pth
        self.target_pth = target_pth
        self.snr = snr
        self.device = device
        self.load_dictionary()
        self.load_target_component()
        # self.load_baseline()
        # self.load_fiber_bg()
        self.load_particle_bg()

        # self.signal_compnents = self.signal_compnents.to(device)
        # self.baseline_components = self.baseline_components.to(device)
        # self.particle_components = self.particle_components.to(device)
        # self.fiber_components = self.fiber_components.to(device)
    def load_dictionary(self):
        '''

        Returns:  dictionary shape is (Dict_len, N, L)

        '''
        self.dict_list = os.listdir(self.dict_pth)
        dict_pth = os.path.join(self.dict_pth, self.dict_list[0])
        data = torch.load(dict_pth)
        N,L = data.shape
        self.signal_compnents = torch.zeros((len(self.dict_list),50,L))
        for idx,pth in enumerate(self.dict_list):
            dict_pth = os.path.join(self.dict_pth,pth)
            data = torch.load(dict_pth)
            N,L = data.shape
            if L == 723:
                self.signal_compnents[idx, :, :L] = data[:50,:L]
                self.signal_compnents[idx, :, -1] = data[:50,-1]
            else:
                self.signal_compnents[idx,:,:] = data[:50]
        self.component_num = len(self.dict_list)
        self.component_dim = 50
        self.signal_compnents = self.signal_compnents.mean(1).to(self.device)
        com_max = self.signal_compnents.max(1)[0].unsqueeze(1).repeat(1,L)
        com_min = self.signal_compnents.min(1)[0].unsqueeze(1).repeat(1,L)

        self.signal_compnents = (self.signal_compnents-com_min)/(com_max-com_min)


    def load_target_component(self):
        self.target_list = os.listdir(self.target_pth)
        target_pth = os.path.join(self.target_pth, self.target_list[0])
        data = torch.load(target_pth)
        N,L = data.shape
        self.target_compnents = torch.zeros((len(self.target_list),50,L))
        for idx,pth in enumerate(self.target_list):
            target_pth = os.path.join(self.target_pth,pth)
            data = torch.load(target_pth)
            N,L = data.shape
            if L == 723:
                self.target_compnents[idx, :, :L] = data[:50,:L]
                self.target_compnents[idx, :, -1] = data[:50,-1]
            else:
                self.target_compnents[idx,:,:] = data[:50]
        self.target_num = len(self.target_list)
        self.target_dim = 50
        self.target_compnents = self.target_compnents.mean(1).to(self.device)
        com_max = self.target_compnents.max(1)[0].unsqueeze(1).repeat(1,L)
        com_min = self.target_compnents.min(1)[0].unsqueeze(1).repeat(1,L)
        self.target_compnents = (self.target_compnents-com_min)/(com_max-com_min)


        # print('signal_compnents',self.signal_compnents.device)

    def load_baseline(self):
        baseline_list = os.listdir(self.baseline_pth)
        b_pth = os.path.join(self.baseline_pth, baseline_list[0])
        data = torch.load(b_pth)
        N,L = data.shape
        self.baseline_components = torch.zeros((N*len(baseline_list),L))
        for idx,pth in enumerate(baseline_list):
            data = torch.load(os.path.join(
                self.baseline_pth,pth))
            self.baseline_components[idx*N:idx*N+N,:] = data
        self.baseline_num = N*len(baseline_list)
        self.baseline_components = self.baseline_components.to(self.device)

    def load_fiber_bg(self):
        fiber_list = os.listdir(self.fiber_bg_pth)
        b_pth = os.path.join(self.fiber_bg_pth, fiber_list[0])
        data = torch.load(b_pth)
        data = data.unsqueeze(0)
        N, L = data.shape
        self.fiber_components = torch.zeros((N * len(fiber_list), L))
        for idx, pth in enumerate(fiber_list):
            data = torch.load(os.path.join(
                self.fiber_bg_pth, pth))
            self.fiber_components[idx * N:idx * N + N, :] = data
        self.fiber_num = N * len(fiber_list)
        self.fiber_components = self.fiber_components.to(self.device)

    def load_particle_bg(self):

        particle_list = os.listdir(self.particle_bg_pth)
        b_pth = os.path.join(self.particle_bg_pth, particle_list[0])
        data = torch.load(b_pth)[:,0,:724]
        # print(data.shape)
        N, L = data.shape
        self.particle_components = torch.zeros((N * len(particle_list), L))
        for idx, pth in enumerate(particle_list):
            data = torch.load(os.path.join(
                self.particle_bg_pth, pth))
            self.particle_components[idx * N:idx * N + N, :724] = data[:,0,:724]

        self.particle_num = N * len(particle_list)
        self.particle_components = self.particle_components.to(self.device)

    def max_min_norm(self,signal):
        return (signal-torch.min(signal)
                )/(torch.max(signal)-torch.min(signal)+1e-6)

    def __len__(self):
        return 100000

    def get_signal(self):
        label_ = torch.rand([1,self.component_num],device=self.device)
        # vanish = torch.randint(0,2,[self.component_num],device=self.device)
        # label = label_*vanish
        label = label_
        # idx_ = torch.randint(0,self.component_dim-1,size=[self.component_num])
        idx__ = torch.linspace(0,self.component_num-1,steps=self.component_num).long()
        # idx = torch.concat([
        #     idx__.unsqueeze(1),
        #     idx_.unsqueeze(1),
        # ],dim=1)
        # print(idx)
        # print(self.signal_compnents.shape)
        # print(self.signal_compnents[idx__,idx_,:].shape)
        signal = torch.mm(label,self.signal_compnents[idx__,:])
        max_signal = torch.max(signal)
        min_signal = torch.min(signal)
        components = label.T*self.signal_compnents[idx__,:]
        # signal_norm = (signal-min_signal)/(max_signal-min_signal+1e-6)
        # components_norm = (components-min_signal)/(max_signal-min_signal+1e-6)
        # print(self.signal_compnents.device)
        # return signal_norm.squeeze(0),components_norm.squeeze(0).T
        return signal.squeeze(0),components.squeeze(0).T

    def get_target(self):
        label_ = torch.rand([1,self.target_num],device=self.device)
        # vanish = torch.randint(0,2,[self.component_num],device=self.device)
        # label = label_*vanish
        label = label_
        # idx_ = torch.randint(0,self.component_dim-1,size=[self.component_num])
        idx__ = torch.linspace(0,self.target_num-1,steps=self.target_num).long()
        # idx = torch.concat([
        #     idx__.unsqueeze(1),
        #     idx_.unsqueeze(1),
        # ],dim=1)
        # print(idx)
        # print(self.signal_compnents.shape)
        # print(self.signal_compnents[idx__,idx_,:].shape)
        signal = torch.mm(label,self.target_compnents[idx__,:])
        max_signal = torch.max(signal)
        min_signal = torch.min(signal)
        components = label.T*self.target_compnents[idx__,:]
        return signal.squeeze(0),components.squeeze(0).T

    def get_particle(self):
        idx = random.randint(0,self.particle_num-1)
        return self.max_min_norm(self.particle_components[idx])

    def get_baseline(self):
        idx = random.randint(0,self.baseline_num-1)
        return self.max_min_norm(self.baseline_components[idx])

    def get_fiber(self):
        idx = random.randint(0,self.fiber_num-1)
        amp = random.uniform(0,2)
        return amp*self.max_min_norm(self.fiber_components[idx])

    def get_spike(self,signal):
        intensity = signal.sum()
        signal_len = signal.shape[0]
        spike_num = random.randint(0,3)
        spike_seq = torch.randint(0,signal_len,[spike_num])
        spike = torch.zeros([signal_len])
        for idx in spike_seq:
            spike[idx] = random.uniform(0,0.003)*intensity
        return spike.to(self.device)



    def add_noise(self,snr,signal):
        signal_power = torch.square(signal).mean(0)
        noise_power = math.pow(10,(-snr/10))*signal_power
        sigma = torch.sqrt(noise_power)
        noise = torch.normal(torch.tensor(0),std=sigma,size=signal.shape,
                             device=self.device)

        signal = signal+noise
        signal = torch.abs(signal)
        return signal,noise

    def __getitem__(self, item):
        signal,component = self.get_signal()
        particle_bg = self.get_particle()
        target,targe_component = self.get_target()
        L,N = targe_component.shape
        # fiber_bg = self.get_fiber()
        # baseline = self.get_baseline()
        # print(particle_bg.shape,
        #       fiber_bg.shape,
        #       baseline.shape)
        signal = torch.concat([
            targe_component,
            0.03 * component,
            particle_bg.unsqueeze(1),
            # fiber_bg.unsqueeze(1),
            # baseline.unsqueeze(1)
        ],dim=1)

        # print(component.device,
        #       signal.device,
        #       particle_bg.device,
        #       fiber_bg.device,
        #       baseline.device,
        #       self.device)

        amp = torch.rand([1,signal.shape[1]],device=self.device)
        sumation = (signal*amp).sum(1)
        spike = self.get_spike(sumation)
        _,noise = self.add_noise(self.snr, sumation)
        max_intensity = sumation.max()
        sumation = sumation+spike+noise

        signal = torch.concat([
            signal * amp,
            noise.unsqueeze(1),
            spike.unsqueeze(1),
        ],dim=1)
        sumation = sumation/max_intensity
        signal = signal/max_intensity
        return sumation,signal.T[:N]


def train_loader(batch_size = 1, device = 'cuda:0',snr = 25):
    set = dataset(snr=snr,device=device)
    return tud.DataLoader(set,batch_size = batch_size,shuffle=True)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    loader = train_loader(batch_size=1)
    for idx,(sumation,signal) in enumerate(loader):
        print(sumation.shape)
        print(signal.shape)
        plt.plot(sumation[0])
        # plt.plot(signal[0])
        plt.show()