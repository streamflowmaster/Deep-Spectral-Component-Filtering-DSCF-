import torch


def HQI(targets,preds):
    # B,D,L * B,D,L
    B,D,L = targets.shape
    targets = targets.reshape(B*D,L)
    preds = preds.reshape(B*D,L)
    pcc_sum = 0
    # tp =  torch.diag(torch.mm(target,pred.T))
    # tt =  torch.diag(torch.mm(target,target.T))
    # pp = torch.diag(torch.mm(pred,pred.T))
    # pcc = (torch.square(tp)/tt/pp).mean()
    # print(tp.shape,tt.shape,pp.shape)
    for i in range(B*D):
        target = targets[i]
        pred = preds[i]
        pcc_sum += torch.abs(torch.cosine_similarity(target,pred,dim=-1))
    # print(pcc_sum/B/D)
    return pcc_sum/(B*D)

def MSE(target,pred):
    return (target-pred).square().mean().mean()


def idf_acc(targets, preds, dict):
    B, D, L = targets.shape
    dict_len, L_d = dict.shape
    # assert L==L_d, str(L)+'!='+str(L_d)+' wrong dictionary shape!'
    if L<L_d:
        dict = dict[:,:L]
    else:
        targets = targets[:,:,:L_d]
        preds = preds[:,:,:L_d]
        L = L_d
    targets = targets.float()
    preds = preds.float()
    dict = dict.float()
    targets = targets.reshape(B*D,L)
    preds = preds.reshape(B*D,L)
    acc = 0
    idx_tar = torch.argmax(torch.mm(targets, dict.T), dim = 1)
    idx_pred = torch.argmax(torch.mm(preds, dict.T), dim = 1)
    residuals = idx_pred-idx_tar
    for i in range(B*D):
        if residuals[i]==0:
            acc+=1
    print(acc/(B*D))

    return acc/(B*D)

if __name__ == '__main__':

    x1 = torch.rand(3,1,708)
    x2 = torch.rand(3,1,708)*2

    print(HQI(x1,x2))
    print(MSE(x1,x2))