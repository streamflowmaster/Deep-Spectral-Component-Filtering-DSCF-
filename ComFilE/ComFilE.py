import os
import torch
import pickle
from train_filtered_cls_models import train as train_cls
from eva_cls_models import eva as eva_cls
from finetune_filter_models import train as train_filter
from DSCF_Submit.customized_task.DSCF_models_pe_ import Hierarchical_1d_model as filter,DSCFConfig
from cls_models import Hierarchical_1d_cls_model as cls
from cls_ml_models import fetch_ml_model
from dataset_ml import load_data
from eval_cls_ml_models import eva as eva_ml_cls

def robust_fold(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)



def ComFilE(workspace = '../ComFilE30/',
            dict_size = 15,
            device = 'cuda:0',
            logic = 'AND',
            filter_encoder_name='SiT',
            filter_decoder_name='PPS',
            cls_encoder_name = 'ResUnet',
            scale = 'tiny',
            sig_len = 512,
            pretrain_pth='../pretrain/model_512/',
            ):

    logicspace = workspace+logic
    robust_fold(workspace)
    robust_fold(logicspace)
    if scale == 'tiny':
        layers = [3, 6, 8, 3]
        d_layers = [3, 6, 8, 3]
    elif scale == 'large':
        layers = [5, 10, 16, 5]
        d_layers = [3, 6, 8, 3]
    elif scale == 'huge':
        layers = [10, 20, 32, 10]
        d_layers = [5, 10, 16, 5]
    elif scale == 'ultra':
        layers = [40, 80, 128, 40]
        d_layers = [5, 10, 16, 5]
    elif scale == 'ultra_pro':
        layers = [80, 160, 256, 80]
        d_layers = [10, 20, 30, 10]

    model_args = dict(
        inplanes=1,
        outplanes=dict_size,
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        layers=layers,
        d_layers=d_layers,
        device=device,
        mask=0.01,
        patch_size=4,
        embed_dim=64,
        sig_len=sig_len,
        snr = 15,
        val_steps = 10
    )
    config = DSCFConfig(**model_args)
    model_filter = filter(config)

    filter_model_name = filter_encoder_name + '_' + filter_decoder_name + '_' + str(dict_size)+'_' + scale + '.pt'
    filter_model_pth = os.path.join(workspace,filter_model_name)
    if os.path.exists(filter_model_pth):
        model_filter.load_state_dict(torch.load(filter_model_pth))
        train_filter(model=model_filter,
                     device=device,
                     snr=35,
                     save_path=filter_model_pth,
                     dict_size=dict_size,
                     batch_size=100,
                     head_path='../ConstructedData/Training_'+str(dict_size),
                     lr=1e-5 )
    else:
        pretrain_name = encoder_name + '_' + decoder_name + '_' + str(scale) + '.pt'
        print('pretrain_name',pretrain_name)
        model_filter.load_encoder_from_pretrain(os.path.join(pretrain_pth,pretrain_name))
        print('Training Component Filter Model')
        train_filter(model=model_filter,
                     device=device,
                     snr=35,
                     save_path=filter_model_pth,
                     dict_size=dict_size,
                     batch_size=200,
                     head_path='../ConstructedData/Training_'+str(dict_size),
                     lr=1e-5 )

    if cls_encoder_name in ['ResUnet','SiT']:
        model_cls = cls(sig_len=709,layers=[2,2,2,1],encoder_name=cls_encoder_name)
        cls_model_name = logic +'_DictSize_'+str(dict_size)+'_Encoder_'+cls_encoder_name+'.pt'
        cls_model_path = os.path.join(logicspace,cls_model_name)
        if os.path.exists(cls_model_path):
            print(cls_model_path,'Loading CLS Model')
            model_cls.load_state_dict(torch.load(cls_model_path))
        else:
            print('Training CLS Model')
            train_cls(epoches=100,
                      model=model_cls,
                      filter_model=model_filter,
                      lr=1e-4,
                      device=device,
                      save_path=cls_model_path,
                      batch_size=512,
                      snr=35,
                      head_path= '../ConstructedData/Training_'+str(dict_size),
                      dict_size=dict_size,
                      logic_settings=logic+'.yaml')

        filtered_acc = torch.zeros(dict_size)
        unfiltered_acc = eva_cls(component_idx=None,model=model_cls,device=device,filter_model=model_filter,
                snr=35,head_path='../ConstructedData/Val_'+str(dict_size),
                dict_size=dict_size,logic_settings=logic+'.yaml')
        with torch.no_grad():
            for idx in range(dict_size):
                print(idx+1)
                filtered_acc[idx] = eva_cls(component_idx=idx,model=model_cls,device=device,filter_model=model_filter,
                        snr=35,head_path='../ConstructedData/Val_'+str(dict_size),
                        dict_size=dict_size,logic_settings=logic+'.yaml')
        print(unfiltered_acc - filtered_acc)
        torch.save(unfiltered_acc - filtered_acc, os.path.join(logicspace, cls_encoder_name + '_delta_acc.pt'))


    elif cls_encoder_name in ['DecisionTree','RandomForest', 'KNN','GaussianNB','MLP','SVM']:
        model_cls = fetch_ml_model(cls_encoder_name)
        model_save_pth = os.path.join(logicspace,cls_encoder_name+'.pkl')
        Xtest, Ytest = load_data(head_path='../ConstructedData/Testing_' + str(dict_size),
                                 logic_settings=logic + '.yaml')
        if not os.path.exists(model_save_pth):
            print('Training ML Model')
            Xtrain, Ytrain = load_data(head_path='../ConstructedData/Training_'+str(dict_size),logic_settings=logic+'.yaml')
            model_cls.fit(Xtrain, Ytrain)
            with open(model_save_pth, 'wb') as f:
                pickle.dump(model_cls, f)  #
                print('Saving ML Model')
        else:
            print('Loading ML Model')
            with open(model_save_pth, 'rb') as f:
                model_cls = pickle.load(f)
        filtered_acc = torch.zeros(dict_size)
        unfiltered_acc = eva_ml_cls(model=model_cls, test_data=zip(Xtest,Ytest), device=device, filter_model=model_filter,component_idx=None)
        print('unfiltered',unfiltered_acc)
        for idx in range(dict_size):
            filtered_acc[idx] = eva_ml_cls(model=model_cls, test_data=zip(Xtest,Ytest), device=device, filter_model=model_filter,component_idx=idx)
            print(idx+1,filtered_acc[idx])
        print(unfiltered_acc - filtered_acc)
        torch.save(unfiltered_acc - filtered_acc, os.path.join(logicspace, cls_encoder_name+'_delta_acc.pt'))

    print(logic)





if __name__ == '__main__':

    # ComFilE(device='cuda:0', logic='AND',
    #         filter_encoder_name='ResUnet',
    #         filter_decoder_name='PPS',
    #         cls_encoder_name='SiT')
    # ComFilE(device='cuda:0', logic='OR',
    #         filter_encoder_name='ResUnet',
    #         filter_decoder_name='PPS',
    #         cls_encoder_name='SiT')
    dict_size = 50
    encoder_name = 'SiT'
    decoder_name = 'PPS'
    # cls_enc_name = 'RandomForest'
    workspace = '../ComFilE'+str(dict_size)+'/'
    cls_enc_names = ['DecisionTree','RandomForest', 'KNN','GaussianNB','MLP','SVM','ResUnet','SiT']
    logics = ['AND1','OR1','SOFT1','AND','OR','SOFT']
    for cls_enc_name in cls_enc_names:
        for logic in logics:
            for cls_enc_name in cls_enc_names:
                for logic in logics:
                    if os.path.exists(os.path.join(workspace, logic, cls_enc_name + '_delta_acc.pt')):
                        print(logic, cls_enc_name, 'Already Done')
                    else:
                        print(logic, cls_enc_name, 'Start to Run')
                        ComFilE(device='cuda:1', logic=logic,
                                filter_encoder_name=encoder_name,
                                filter_decoder_name=decoder_name, workspace=workspace,
                                cls_encoder_name=cls_enc_name, dict_size=dict_size)