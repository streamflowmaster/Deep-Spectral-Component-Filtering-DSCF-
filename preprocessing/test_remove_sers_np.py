import torch
from forward_protocol import fetch_model
from comparas.mcr_als import als_removal
from comparas.scaling import scaling_removal
import matplotlib.pyplot as plt
import os
from Interp import interp1d
from dataset import  Generation
import numpy as np
import pandas as pd

def length_adopt(Lo,target):
    B,  Lt = target.shape
    device = target.device
    if Lt != Lo:
        coor_t = torch.linspace(0, 1, Lt).unsqueeze(0).repeat(B, 1).to(device)
        coor_o = torch.linspace(0, 1, Lo).unsqueeze(0).repeat(B, 1).to(device)
        target = interp1d(coor_t, target.reshape(B, Lt), coor_o).reshape(B, Lo)
    return target

def all_files_path(rootDir,filepaths = []):

    for root, dirs, files in os.walk(rootDir):     # 分别代表根目录、文件夹、文件
        for file in files:                         # 遍历文件
            file_path = os.path.join(root, file)   # 获取文件绝对路径
            filepaths.append(file_path)            # 将文件路径添加进列表
        for dir in dirs:                           # 遍历目录下的子目录
            dir_path = os.path.join(root, dir)     # 获取子目录路径
            all_files_path(dir_path,filepaths)               # 递归调用

class max_min_normalization(object):
    def __init__(self, data):
        self.max = torch.max(data,dim=-1)[0]
        self.min = torch.min(data,dim=-1)[0]
        self.min = self.min.unsqueeze(-1)
        self.max = self.max.unsqueeze(-1)
        print(self.max.shape,self.min.shape)
        # self.max = max
        # self.min = min

    def __call__(self, img):
        return (img - self.min) / (self.max - self.min)

    def back(self,img):
        return img * (self.max - self.min) + self.min

def bg_removal_sers_np(model,device= 'cuda:1',save_img=False,save_score_info=False,save_removed=False,save_removed_individual=False,
                       data_pth='PCa_data/original_aligned',processed_pth='PCa_data/dscf_removed_v3'):
    files = []
    frame = pd.DataFrame(columns=['file_name','idx_dcsf','idx_scl','idx_als'])
    score = pd.DataFrame(columns=['file_name','0','1','2'])
    removed = {'dscf':np.array([]), 'scl':np.array([]), 'als':np.array([]),'raw':np.array([])}

    frame_name = os.path.join(processed_pth,'info.csv')
    score_name = os.path.join(processed_pth,'score.csv')

    os.makedirs(processed_pth,exist_ok=True)
    os.makedirs(processed_pth+'_data',exist_ok=True)
    dataset = Generation(device=device)
    model = model.to(device)
    ref = torch.load('../SERS_Background_Removal_PCa/Int-Interp-Citrate-AgNPs.pt').mean(0).to(device)
    pure = dataset.pure_data.mean(1).to(device)
    patterns = torch.cat([ref.unsqueeze(0),pure.unsqueeze(0)],dim=0)
    all_files_path(data_pth,filepaths=files)
    # files = files.sort()
    print('files',files)

    with torch.no_grad():
        for file in files:
            # print(file)
            spec = torch.tensor(torch.load(file)).to(device).float()
            # [:, 100: 540]
            # plt.plot(spec[0].detach().cpu().numpy())
            # plt.vlines(185,0,1000)
            # plt.vlines(510, 0, 1000)
            # plt.show()
            spec = length_adopt(Lo=1799,target=spec)
            spec_raw = spec.clone()
            normalize = max_min_normalization(spec)
            spec = normalize(spec).unsqueeze(1)
            # print(spec.shape)
            # plt.plot(spec[0, 0].detach().cpu().numpy())
            # plt.show()

            bg_dscf = model(spec).squeeze(1)
            bg_dscf = length_adopt(Lo=1799, target=bg_dscf)
            spec_dscf = (spec.squeeze(1) - bg_dscf)
            spec_dscf = normalize.back(spec_dscf)

            bg_scl = scaling_removal(spec, ref).squeeze(1)
            spec_scl = bg_scl.to(device)
            spec_scl = normalize.back(spec_scl)

            bg_als = als_removal(spec, patterns)[:,0,0]
            bg_als = bg_als.unsqueeze(1).repeat(1,ref.shape[-1]).to(device)
            spec_als = (spec.squeeze(1) - (bg_als*ref.unsqueeze(0)))
            spec_als = normalize.back(spec_als)
            # bg = normalize.back(bg)
            if save_removed:
                if removed['dscf'].shape[0] == 0:
                    removed['dscf'] = spec_dscf.detach().cpu().numpy()
                    removed['scl'] = spec_scl.detach().cpu().numpy()
                    removed['als'] = spec_als.detach().cpu().numpy()
                    removed['raw'] = spec_raw.detach().cpu().numpy()
                else:
                    removed['dscf'] = np.concatenate((removed['dscf'],spec_dscf.detach().cpu().numpy()))
                    removed['scl'] = np.concatenate((removed['scl'],spec_scl.detach().cpu().numpy()))
                    removed['als'] = np.concatenate((removed['als'],spec_als.detach().cpu().numpy()))
                    removed['raw'] = np.concatenate((removed['raw'],spec_raw.detach().cpu().numpy()))
                    print(removed['dscf'].shape,removed['scl'].shape,removed['als'].shape)
                np.save('removed.npy',removed)

            if save_removed_individual:
                removed['dscf'] = spec_dscf.detach().cpu().numpy()
                removed['scl'] = spec_scl.detach().cpu().numpy()
                removed['als'] = spec_als.detach().cpu().numpy()
                removed['raw'] = spec_raw.detach().cpu().numpy()
                np.save(os.path.join(processed_pth+'_data',file.split('/')[-1].split('.')[0]+'.npy'),removed)
            if save_img:
                print(bg_dscf.shape,spec_dscf.shape,
                      bg_scl.shape,spec_scl.shape,
                      bg_als.shape,spec_als.shape)
                relu = torch.nn.LeakyReLU()
                spec_removed= torch.cat([(spec_dscf)[0].unsqueeze(0),
                                         spec_scl[0].unsqueeze(0),
                                         spec_als[0].unsqueeze(0)],dim=0)
                print('spec_removed',spec_removed.shape)
                data = np.random.uniform(0, 1, (3))
                idx = np.argsort(data)
                plt.figure(figsize=(10, 10))
                plt.subplot(4, 1, 1)
                plt.plot(spec_raw[0].detach().cpu().numpy())
                plt.hlines(0,0,spec_raw.shape[-1])
                plt.text(0, spec_raw[0].max().item()-100, 'raw')
                plt.yticks([])

                for i in range(3):
                    plt.subplot(4, 1,  idx[i]+2)
                    # plt.subplot(4, 1,  i+2)
                    plt.plot(spec_removed[i].detach().cpu().numpy())
                    plt.hlines(0,0,spec_raw.shape[-1])
                    # plt.text(0, spec_removed[idx[i]].max().item()-100, 'removed')
                    plt.yticks([])

                # plt.show()
                plt.savefig(os.path.join(processed_pth,file.split('/')[-1].split('.')[0]+'.png'))

            if save_score_info:
                frame = pd.concat([frame,pd.DataFrame({'file_name':file.split('/')[-1],
                                'idx_dcsf':idx[0],'idx_scl':idx[1],'idx_als':idx[2]},index=[0])],
                          ignore_index=True)
                frame.to_csv(frame_name)
                score = pd.concat([score,pd.DataFrame({'file_name':file.split('/')[-1]},
                                                      index=[0])],
                          ignore_index=True)
                score.to_csv(score_name)
    if save_score_info:
        score.to_csv('score.csv')
        frame.to_csv('info.csv')



            # torch.save(bg,os.path.join(processed_pth,file.split('/')[-1]))
            # return img.squeeze().detach().cpu().numpy()

# if __name__ == '__main__':

save_head_path = 'Background_Removal_SERS_workspace'
model = fetch_model(snr=25,encoder_name='SiT',decoder_name='PPS',save_head_path=save_head_path)
bg_removal_sers_np(model = model,save_removed=False,save_img=False,save_score_info=False,save_removed_individual=True,
                   data_pth='PCa_data/original_aligned',processed_pth='PCa_data/dscf_removed_v3')