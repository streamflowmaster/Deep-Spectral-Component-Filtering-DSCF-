import os.path
import random
import torch
import torch.utils.data as tud
import matplotlib.pyplot as plt
import math
from Interp import interp1d
import numpy as np

def random_time_warp(signal, max_warp=0.2):
    """
    Randomly time-warp a signal.

    Parameters:
    - signal: The input signal (1D numpy array).
    - max_warp: Maximum warp factor (e.g., 0.1 for up to ±10% time warp).

    Returns:
    - warped_signal: The time-warped signal.
    """

    if len(signal.shape) > 1:
        for idx in range(signal.shape[0]):
            signal[idx] = random_time_warp(signal[idx],max_warp)
        return signal
    else:
        n = signal.shape[-1]
        device = signal.device
        warp_factors = torch.linspace(1 - max_warp, 1 + max_warp, n)
        warp_indices = torch.arange(n) * torch.from_numpy(np.random.choice(warp_factors, n))
        warp_indices[warp_indices < 0 ] = torch.randint(0, n, (1,))
        warp_indices[warp_indices >= n] = torch.randint(0, n, (1,))
        warp_indices = torch.sort(warp_indices)[0].long()
        warped_signal = signal[warp_indices] * torch.from_numpy(
            np.random.uniform(1 - max_warp/2, 1 + max_warp/2, n)).to(device)

        return warped_signal

def length_adopt(signal,background):
    L = background.shape[-1]
    S = signal.shape[-1]
    device = 'cpu'
    device = background.to(device)
    signal = signal.to(device)

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
    '''
    the dataset for customized task,
    Pure Experiment data dir
    Impure substances dir
    Component dictionary dir

    '''
    def __init__(self,
                 snr:float = None,
                 Pure: dict = {'dir':'Pure-spec/',
                               'tensor_dim':3,
                               'spec_tensor_dim':3,},
                 Impure: dict = {'dir':'Impure-spec/',
                                    'tensor_dim':3,
                                    'spec_tensor_dim':3,},
                 Component: dict = {'dir':'Component-spec/',
                                        'tensor_dim':3,
                                        'spec_tensor_dim':3,},
                 mix_protocol:dict = {'max_num_of_component':3,
                                        'min_num_of_component':1,
                                      'abudance_distribution':None,},
                 spec_len:int=512,
                 warp_factor:float=0.1,
                 warp_prob:float=0.5,
                 device:str='cuda:1',
                 ):

        self.device = device
        self.signal_len = spec_len
        self.snr = snr
        self.example = torch.rand((1,self.signal_len)).to('cpu')

        self.Pures = self.load_spec(Pure)
        self.Impures = self.load_spec(Impure)
        self.Components = self.load_dictionary(Component)
        self.mix_protocol = mix_protocol
        self.warp_factor = warp_factor
        self.warp_prob = warp_prob


    def load_spec(self,SPEC:dir={'dir':'Impure-spec/',
                                    'tensor_dim':3,
                                    'spec_tensor_dim':3,},):
        '''
        load the data from the dir_list
        Args:
        - dir_list: the list of the dir path
        - spec_dim: which dim is the spectrum data, default is -1
        for example, the shape of the data is H,W,C, the spec_dim is -1
        if the shape of the data is C,H,W, the spec_dim is 0

        Returns:
        - data: the data of the dir_list
        '''
        if SPEC is None or SPEC is {}:
            return torch.empty((0,self.signal_len))

        else:
            dir_list = os.listdir(SPEC['dir'])
            tensor_dim = SPEC['tensor_dim']
            spec_tensor_dim = SPEC['spec_tensor_dim']
            if spec_tensor_dim < 0:
                spec_tensor_dim = tensor_dim + spec_tensor_dim

            #try to load the first data to get the shape of the data
            data = torch.load(os.path.join(SPEC['dir'],dir_list[0]))
            assert len(data.shape) == tensor_dim, 'The tensor_dim of ' + SPEC['dir'] +' is not correct'
            assert spec_tensor_dim < tensor_dim, 'The spec_tensor_dim  of ' + SPEC['dir'] +' is not correct'
            # permute the spec_tensor_dim to the last dim
            print([i for i in range(tensor_dim) if i!=spec_tensor_dim]+[spec_tensor_dim])
            data = data.permute(*[i for i in range(tensor_dim) if i!=spec_tensor_dim]+[spec_tensor_dim])
            # reshape the data to the shape of (N,C,L)
            data = data.reshape(-1,data.shape[-1])

            total_spec_num = data.shape[0]
            # first scan the data to get the shape of the data
            for idx in range(1,len(dir_list)):
                data = torch.load(os.path.join(SPEC['dir'],dir_list[idx]))
                data = data.permute(*[i for i in range(tensor_dim) if i!=spec_tensor_dim]+[spec_tensor_dim])
                data = data.reshape(-1,data.shape[-1])
                total_spec_num += data.shape[0]

            # load the data
            data = torch.zeros((total_spec_num,self.signal_len))
            dim = 0
            for idx in range(len(dir_list)):
                data_ = torch.load(os.path.join(SPEC['dir'],dir_list[idx]))
                data_ = data_.permute(*[i for i in range(tensor_dim) if i!=spec_tensor_dim]+[spec_tensor_dim])
                data_ = data_.reshape(-1,data_.shape[-1])
                # align the length of the signal
                data_ = length_adopt(data_,self.example)
                data[dim:dim+data_.shape[0],:] = self.max_min_norm(data_)
                dim += data_.shape[0]
        print('Number of the ',SPEC['dir'],' is ',data.shape[0])
        data = data.to(self.device)
        return data

    def load_dictionary(self,Component:dict={'dir':'Component-spec/',
                                        'tensor_dim':3,
                                        'spec_tensor_dim':3,}):
        '''

        Returns:  dictionary shape is (Dict_len, N, L)
        where
        Dict_len is the number of the different component
        N is the different observation of the same component
        L is the length of the signal
        '''
        if Component is None:
            return torch.empty((0,0,self.signal_len))

        else:
            dir_list = os.listdir(Component['dir'])
            print(dir_list)
            tensor_dim = Component['tensor_dim']
            spec_tensor_dim = Component['spec_tensor_dim']
            if spec_tensor_dim < 0:
                spec_tensor_dim = tensor_dim + spec_tensor_dim

            #try to load the first data to get the shape of the data
            data = torch.load(os.path.join(Component['dir'],dir_list[0]))
            assert len(data.shape) == tensor_dim, 'The tensor_dim of ' + Component['dir'] +' is not correct'
            assert spec_tensor_dim < tensor_dim, 'The spec_tensor_dim of ' + Component['dir'] +' is not correct'


            total_spec_num = 0
            max_observation = 0
            # first scan the data to get the shape of the data
            for idx in range(0,len(dir_list)):
                data = torch.load(os.path.join(Component['dir'],dir_list[idx]))
                # permute the spec_tensor_dim to the last dim
                data = data.permute(*[i for i in range(tensor_dim) if i != spec_tensor_dim] + [spec_tensor_dim])
                # reshape the data to the shape of (N,C,L)
                data = data.reshape(-1, data.shape[-1])
                total_spec_num += 1
                max_observation = max(max_observation,data.shape[0])

            # load the data
            data = torch.zeros((total_spec_num,max_observation,self.signal_len))
            dim = 0
            for idx in range(len(dir_list)):
                data_ = torch.load(os.path.join(Component['dir'],dir_list[idx]))
                data_ = data_.permute(*[i for i in range(tensor_dim) if i != spec_tensor_dim] + [spec_tensor_dim])
                data_ = data_.reshape(-1, data_.shape[-1])
                data_ = length_adopt(data_, self.example)
                data[dim,0:data_.shape[0],:] = self.max_min_norm(data_)
                dim += 1
            print('Number of the ',Component['dir'],' is ',data.shape[0])
            data = data.to(self.device)
            return data

    def __len__(self):
        return (self.Pures.shape[0]+self.Components.shape[0]+1)*(self.Impures.shape[0]+1)*100

    def max_min_norm(self,spec):
        if type(spec) != torch.Tensor:
            spec = torch.tensor(spec)

        if len(spec.shape) == 1:
            max_ = spec.max(0)[0]
            min_ = spec.min(0)[0]
            spec = (spec-min_)/(max_-min_+1e-8)
            # print(spec.shape)
        if len(spec.shape) == 2:
            max_ = spec.max(1)[0].unsqueeze(1)
            min_ = spec.min(1)[0].unsqueeze(1)
            spec = (spec-min_)/(max_-min_+1e-8)
        return spec




    def get_mixture(self):
        assert self.Components.shape[0]>0, 'The component is empty'
        assert self.mix_protocol['max_num_of_component']<=self.Components.shape[0], 'The max_num_of_component is larger than the number of the component'

        num_of_component = random.randint(self.mix_protocol['min_num_of_component'],self.mix_protocol['max_num_of_component'])
        component_idx = torch.randint(0,self.Components.shape[0],(num_of_component,))
        component_abudance = torch.rand(num_of_component)
        component_abudance = component_abudance/component_abudance.sum()
        mixture = torch.zeros(self.signal_len).to(self.device)

        components = torch.zeros((self.Components.shape[0],self.signal_len)).to(self.device)
        for idx in range(num_of_component):
            component = self.Components[component_idx[idx],:,:]
            component = component[torch.randint(0,component.shape[0],(1,))[0],:]
            mixture += component_abudance[idx]*self.signal_warp(component)
            components[component_idx[idx],:] = component_abudance[idx]*self.signal_warp(component)
        max_ = mixture.max(0)[0]
        min_ = mixture.min(0)[0]
        components = components/(max_-min_+1e-8)
        return self.max_min_norm(mixture)[None],components


    def get_pure(self):
        if self.Pures.shape[0] == 0:
            return torch.zeros((1,self.signal_len)).to(self.device)
        pure_idx = torch.randint(0,self.Pures.shape[0]-1,(1,))
        return self.signal_warp(self.Pures[pure_idx,:])

    def signal_warp(self,signal):
        # random warp the signal
        prob = random.uniform(0,1)
        if prob<=self.warp_prob:
            return random_time_warp(signal, self.warp_factor)
        else:
            return signal

    def get_Impure(self):
        if self.Impures.shape[0] == 0:
            return torch.zeros((1,self.signal_len)).to(self.device)
        impure_idx = torch.randint(0,self.Impures.shape[0]-1,(1,))
        impure = self.signal_warp(self.Impures[impure_idx,:])
        return impure

    def add_noise(self,snr,signal):
        if snr is None:
            return torch.zeros(signal.shape).to(self.device)
        else:
            noise = torch.randn(signal.shape).to(self.device)
            signal_power = signal.pow(2).mean()+1e-8
            noise_variance = signal_power / (10 ** (snr / 10))
            noise = noise * (noise_variance ** 0.5)
            return noise

    def __getitem__(self, item):
        # dim of components should align with the 2-dim of model
        pure = self.get_pure()
        impure = self.get_Impure()
        mixture,components = self.get_mixture()
        # radom weight the signal
        weight = torch.rand(3)
        signal = weight[0]*pure + weight[1]*impure + weight[2]*mixture
        signal = self.max_min_norm(signal)
        noise = self.add_noise(self.snr,signal)
        signal_n = pure + impure + mixture + noise
        # switch the second return value (signal, impure, pure)
        # for the customized task
        return signal_n,components


def main():
    dataset = Generation(snr=25,
                         Pure={'dir':'Pure-spec/',
                               'tensor_dim':2,
                               'spec_tensor_dim':-1,},
                         Impure={'dir':'Impure-spec/',
                                    'tensor_dim':2,
                                    'spec_tensor_dim':-1,},
                         Component={'dir':'Component-spec/',
                                        'tensor_dim':2,
                                        'spec_tensor_dim':-1,},
                         mix_protocol={'max_num_of_component':23,
                                        'min_num_of_component':4,
                                      'abudance_distribution':None,},
                         spec_len=512,
                         warp_factor=0.1,
                         warp_prob=0.5,
                         device='cpu',
                         )
    signal,impure = dataset[0]
    print(signal.shape,impure.shape)
    plt.plot(signal.cpu().numpy()[0])
    plt.plot(impure.cpu().numpy()[0])
    plt.show()

def fintuning_dataloader(batch_size=1,device='cuda:1',snr=15):
    dataset = Generation(snr=snr,
                         Pure=None,
                         Impure=None,
                         Component={'dir':'Component-spec/',
                                        'tensor_dim':2,
                                        'spec_tensor_dim':-1,},
                         mix_protocol={'max_num_of_component':23,
                                        'min_num_of_component':12,
                                      'abudance_distribution':None,},
                         spec_len=512,
                         warp_factor=0.1,
                         warp_prob=0.5,
                         device=device,
                         )
    return tud.DataLoader(dataset,batch_size=batch_size,shuffle=True)



if __name__ == '__main__':
    main()