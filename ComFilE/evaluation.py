import torch
import matplotlib.pyplot as plt
class evaluate():

    def __init__(self,cls_num=3):
        self.cls_num = cls_num
        # the confusion matrix: target/predict
        self.metrics = torch.zeros((cls_num,cls_num))
        self.sum = 0

    def calculation(self,targets,predicts):
        preds = torch.argmax(predicts, dim=1)
        if len(targets.shape) !=1:
            tars = torch.argmax(targets, dim=1)
        else:
            tars = targets
        for tar,pred in zip(tars,preds):
            self.metrics[tar,pred] += 1
            self.sum += 1
    def eval(self):
        acc = self.metrics/self.sum
        # print(self.metrics/self.sum)
        # print('sum',self.sum)
        acc_sum = 0
        for i in range(self.cls_num):
            acc_sum += acc[i,i]
        print('acc:%.3f' % (acc_sum.item()))
        return acc_sum.item()

    def show(self,thresh=50):
        C = self.metrics.numpy()
        S = C.sum(1)

        M = torch.zeros((self.cls_num,self.cls_num))
        for i in range(len(C)):
            for j in range(len(C)):
                M[j,i] = C[j, i]/S[j]*100
                print(C[j, i])
        plt.matshow(M, cmap='Greens')
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate('%.1f'%M[j,i]+'%', xy=(i, j), horizontalalignment='center', verticalalignment='center',color="white" if M[j,i] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.yticks([0, 1, 2], ['PCa', 'BPH', 'Control'])
        plt.xticks([0,1,2], ['PCa','BPH','Control'])
        plt.rcParams['figure.figsize']=(6,6)
        plt.tight_layout(pad=0.1, h_pad=None, w_pad=None, rect=None)
        plt.show()
