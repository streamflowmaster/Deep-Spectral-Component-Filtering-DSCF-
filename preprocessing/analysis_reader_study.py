import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
info = pd.read_csv('info-3.csv')
score = pd.read_csv('score-3-re.csv')
info_list = info.values.tolist()
score_list = score.values.tolist()
print(info_list)
print(score_list)
dscf = []
scl = []
als = []

for i in range(len(info_list)):
    dscf_idx = info_list[i][2]
    scl_idx = info_list[i][3]
    als_idx = info_list[i][4]
    dscf.append(score_list[i][dscf_idx+2])
    scl.append(score_list[i][scl_idx+2])
    als.append(score_list[i][als_idx+2])


# plt.figure(figsize=(5, 5),dpi=300)
# plt.boxplot([dscf,scl,als],labels=['DSCF','Scaling','MCR-ALS'],meanline=True,showmeans=True)
# # plt.violinplot([dscf,scl,als],showmeans=True,showmedians=True)
# plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, bottom=0.05, right=0.99, top=0.99)
# data = [dscf,scl,als]
# for i in range(len(data)):
#     to_scatter = data[i]
#     x = np.random.normal(i+1, 0.04, size=len(to_scatter))
#     plt.scatter(x, to_scatter, alpha=0.5, s=1, color='black')
#
# plt.yticks([1,2,3],['1','2','3'],rotation=90)
# plt.show()

def draw_pie(dscf,scl,als):
    dscf = np.array(dscf)
    scl = np.array(scl)
    als = np.array(als)

    dscf_3 = dscf[dscf==3]
    dscf_2 = dscf[dscf==2]
    dscf_1 = dscf[dscf==1]
    scl_3 = scl[scl==3]
    scl_2 = scl[scl==2]
    scl_1 = scl[scl==1]
    als_3 = als[als==3]
    als_2 = als[als==2]
    als_1 = als[als==1]

    plt.figure(figsize=(5, 5),dpi=300)

    patches, l_text, p_text = \
    plt.pie([len(dscf_3),len(scl_3),len(als_3)],
            # labels=['DSCF','Scaling','MCR-ALS'],
            colors=['#7B468C','#2F6B5A','#A8D8CA'],
            labeldistance=1.2,
            pctdistance=0.6)

    # for t in l_text:
    #     t.set_size(30)

    for t in p_text:
        t.set_size(25)
        t.set_color('white')
        t.set_fontfamily('Arial')
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.09, bottom=0.09, right=0.99, top=0.99)
    plt.savefig('fig-3d.svg', dpi=300)
    plt.show()

def draw_pie_bar(dscf,scl,als):
    dscf = np.array(dscf)
    scl = np.array(scl)
    als = np.array(als)

    dscf_3 = dscf[dscf==3]
    dscf_2 = dscf[dscf==2]
    dscf_1 = dscf[dscf==1]
    scl_3 = scl[scl==3]
    scl_2 = scl[scl==2]
    scl_1 = scl[scl==1]
    als_3 = als[als==3]
    als_2 = als[als==2]
    als_1 = als[als==1]

    plt.figure(figsize=(5, 5),dpi=300)

    patches, l_text, p_text = \
    plt.bar([1,2,3],[len(dscf_3),len(scl_3),len(als_3)],
            # labels=['DSCF','Scaling','MCR-ALS'],
        color=['#7B468C','#2F6B5A','#A8D8CA'],
            # width=0.5
            )



    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.09, bottom=0.09, right=0.99, top=0.99)
    plt.savefig('fig-3d-bar.svg', dpi=300)
    plt.show()


def draw_histogram():
    dscf_mean = np.mean(dscf)
    scl_mean = np.mean(scl)
    als_mean = np.mean(als)

    dscf_std = np.std(dscf)
    scl_std = np.std(scl)
    als_std = np.std(als)
    print(dscf_mean,scl_mean,als_mean)
    print(dscf_std,scl_std,als_std)
    plt.figure(figsize=(5, 5),dpi=500)
    ax = plt.subplot(111)
    # plt.bar([1,2,3],[dscf_mean,scl_mean,als_mean],yerr=[dscf_std,scl_std,als_std],capsize=5,color=['#7B468C','#2F6B5A','#A8D8CA'])
    plt.bar([1, 2, 3], [dscf_mean, scl_mean, als_mean], capsize=5,
            color=['#7B468C', '#2F6B5A', '#A8D8CA'])

    # plt.hist(dscf_mean,scl_mean,als_mean,bins=3,color=['#7B468C','#2F6B5A','#A8D8CA'],alpha=1)
    plt.xticks([1,2,3],['DSCF','Scaling','MCR-ALS'],fontsize=20,family='Arial')
    plt.yticks([1,2,3],['1','2','3'],rotation=90,fontsize=20,family='Arial')
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.09, bottom=0.09, right=0.99, top=0.99)
    plt.ylim(1,3.05)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    print(dscf_mean,scl_mean,als_mean)
    plt.savefig('fig-3c.svg',dpi=300)
    plt.show()

# draw_histogram()
# draw_pie(dscf,scl,als)
draw_pie_bar(dscf,scl,als)