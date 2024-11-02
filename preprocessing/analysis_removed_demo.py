import os

import matplotlib.pyplot as plt
import numpy as np

def load_removed_data(pth='removed.npy'):
    removed = np.load(pth,allow_pickle=True).item()
    dscf = removed['dscf']
    scl = removed['scl']
    als = removed['als']
    raw = removed['raw']
    return dscf,scl,als,raw

# def draw_spectra(dscf,scl,als,raw):
#     colors = ['#7B468C', '#2F6B5A', '#A8D8CA']
#     plt.figure(figsize=(10, 6),dpi=300)
#     dscf_mean = np.mean(np.array(dscf),axis=0)
#     scl_mean = np.mean(np.array(scl),axis=0)+1000
#     als_mean = np.mean(np.array(als),axis=0)+2000
#     raw_mean = np.mean(np.array(raw),axis=0)+3000
#     L = 3
#     dscf_std = np.std(np.array(dscf))
#     scl_std = np.std(np.array(scl))
#     als_std = np.std(np.array(als))
#     raw_std = np.std(np.array(raw))
#
#     sig_len = dscf.shape[-1]
#     plt.plot(dscf_mean,label='DSCF',color = colors[0],linewidth=L)
#     plt.fill_between(np.arange(sig_len),
#                         (dscf_mean-dscf_std),
#                         (dscf_mean+dscf_std),
#                         alpha=0.1,color = colors[0])
#     plt.plot(scl_mean,label='Scaling',color = colors[1],linewidth=L)
#     plt.fill_between(np.arange(sig_len),
#                         (scl_mean-scl_std),
#                         (scl_mean+scl_std),
#                         alpha=0.1,color = colors[1])
#     plt.plot(als_mean,label='MCR-ALS',color = colors[2],linewidth=L)
#     plt.fill_between(np.arange(sig_len),
#                         (als_mean-als_std),
#                         (als_mean+als_std),
#                         alpha=0.1,color = colors[2],linewidth=L)
#     plt.plot(raw_mean,label='Raw',color = 'r')
#     plt.fill_between(np.arange(sig_len),
#                         (raw_mean-raw_std),
#                         (raw_mean+raw_std),
#                         alpha=0.1,color = 'r',linewidth=L)
#     plt.xlim(0,1600)
#     plt.ylim(-500,6500)
#     plt.yticks([])
#     plt.xticks(np.linspace(0,1600,4),np.linspace(200,2000,4))
#     # plt.legend()
#     plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.01, bottom=0.04, right=0.99, top=0.99)
#     plt.show()


# dscf,scl,als,raw = load_removed_data()
# draw_spectra(dscf,scl,als,raw)

def draw_spectra(dscf,scl,als,raw):
    if_std = False
    colors = ['#7B468C', '#2F6B5A', '#A8D8CA']
    plt.figure(figsize=(10, 5),dpi=300)
    ax = plt.subplot(111)
    dscf = dscf[0:1]
    scl = scl[0:1]
    als = als[0:1]
    raw = raw[0:1]
    dscf_mean = np.mean(np.array(dscf),axis=0)
    scl_mean = np.mean(np.array(scl),axis=0)+1000
    als_mean = np.mean(np.array(als),axis=0)+2000
    raw_mean = np.mean(np.array(raw),axis=0)+3000
    L = 3
    if if_std:
        dscf_std = np.std(np.array(dscf))
        scl_std = np.std(np.array(scl))
        als_std = np.std(np.array(als))
        raw_std = np.std(np.array(raw))
    else:
        dscf_std = 0
        scl_std = 0
        als_std = 0
        raw_std = 0

    sig_len = dscf.shape[-1]
    plt.plot(dscf_mean,label='DSCF',color = colors[0],linewidth=L)
    plt.hlines(0,0,sig_len,linestyles='dashed',linewidth=1,color=colors[0])
    plt.fill_between(np.arange(sig_len),
                        (dscf_mean-dscf_std),
                        (dscf_mean+dscf_std),
                        alpha=0.1,color = colors[0])
    plt.plot(scl_mean,label='Scaling',color = colors[1],linewidth=L)
    plt.hlines(1000, 0, sig_len,linestyles='dashed',linewidth=1,color=colors[1])
    plt.fill_between(np.arange(sig_len),
                        (scl_mean-scl_std),
                        (scl_mean+scl_std),
                        alpha=0.1,color = colors[1])
    plt.plot(als_mean,label='MCR-ALS',color = colors[2],linewidth=L)
    plt.hlines(2000, 0, sig_len,linestyles='dashed',linewidth=1,color=colors[2])
    plt.fill_between(np.arange(sig_len),
                        (als_mean-als_std),
                        (als_mean+als_std),
                        alpha=0.1,color = colors[2],linewidth=L)
    plt.plot(raw_mean,label='Raw',color = 'r')
    plt.hlines(3000, 0, sig_len,linestyles='dashed',linewidth=1,color='r')
    plt.fill_between(np.arange(sig_len),
                        (raw_mean-raw_std),
                        (raw_mean+raw_std),
                        alpha=0.1,color = 'r',linewidth=L)
    plt.vlines(960, -500, 4600,linestyles='dashed',linewidth=2)
    plt.vlines(1375, -500, 4600,linestyles='dashed',linewidth=2)
    plt.fill_betweenx(np.arange(-500,4600),
                      945,975,
                      alpha=0.1,color = 'grey')
    plt.fill_betweenx(np.arange(-500,4600),
                      1350,1400,
                      alpha=0.1,color = 'grey')
    plt.xlim(0,1600)
    plt.ylim(-500,4600)
    plt.yticks([])
    idx = [960,1375]
    print(np.linspace(200, 2000, 1600)[idx])
    plt.xticks(np.linspace(0,1600,1600)[idx],['1280.6','1747.8'],fontsize=20)
    # plt.legend()
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.01, bottom=0.08, right=0.99, top=0.99)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.savefig('fig-3b.svg', dpi=300)
    plt.show()


file_list = os.listdir('PCa_data/dscf_removed_v3_data/')
file_list = [os.path.join('PCa_data/dscf_removed_v3_data/',file) for file in file_list]
file_list = [file for file in file_list if 'npy' in file]
# file = file_list[10]
file = file_list[13]
dscf,scl,als,raw = load_removed_data(pth=file)
draw_spectra(dscf,scl,als,raw)
# dscf,scl,als,raw = load_removed_data(pth=)
# draw_spectra(dscf,scl,als,raw)