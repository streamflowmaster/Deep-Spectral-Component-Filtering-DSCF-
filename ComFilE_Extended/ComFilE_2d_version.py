import os
import torch
import pickle
from DSCF_Submit.ComFilE.COMFILE import *
from ComFilE_Extended.eval_cls_ml_models import eva as eva_ml_cls

def robust_fold(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

def return_dataset_size(dataset_scale,dict_size):
    if dataset_scale == '':
        return int(dict_size)
    elif dataset_scale == 'small':
        return int(dict_size*0.5)
    elif dataset_scale == 'tiny':
        return int(dict_size*0.1)
    elif dataset_scale == 'micro':
        return int(dict_size*0.01)
    elif dataset_scale == 'nano':
        return int(dict_size*0.001)
    else:
        raise ValueError('dataset_scale must be one of [tiny, small, micro, nano]')


def ComFilE(workspace = '../ComFilE30/',
            dict_size = 15,
            device = 'cuda:0',
            logic = 'AND',
            filter_encoder_name='SiT',
            filter_decoder_name='PPS',
            cls_encoder_name = 'ResUnet',
            dataset_scale = 'tiny'):

    config = DSCFConfig(**model_args)
    model_filter = filter(config)

    filter_model_name = filter_encoder_name + '_' + filter_decoder_name + '_' + str(dict_size)+'_' + scale + '.pt'
    filter_model_pth = os.path.join(workspace,filter_model_name)
    if os.path.exists(filter_model_pth):
        model_filter.load_state_dict(torch.load(filter_model_pth))
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


    if os.path.exists(filter_model_pth):
        model_filter.load_state_dict(torch.load(filter_model_pth))
        # train_filter(model=model_filter, device=device,snr=35,
        #              save_path=filter_model_pth,dict_size=dict_size,batch_size=128,
        #               head_path='../ConstructedData/Training_'+str(dict_size),lr=1e-5 )
    else:
        print('Training Component Filter Model')
        train_filter(model=model_filter, device=device,snr=35,
                     save_path=filter_model_pth,dict_size=dict_size,batch_size=128,
                      head_path='../ConstructedData/Training_'+str(dict_size),lr=1e-5 )

    if cls_encoder_name in ['ResUnet','SiT']:
        model_cls = cls(sig_len=709,layers=[2,2,2,1],encoder_name=cls_encoder_name)
        cls_model_name = logic +'_DictSize_'+str(dict_size)+'_Encoder_'+cls_encoder_name+'.pt'
        cls_model_path = os.path.join(logicspace,cls_model_name)
        if os.path.exists(cls_model_path):
            model_cls.load_state_dict(torch.load(cls_model_path))
        else:
            print('Training CLS Model')
            train_cls(epoches=100,model=model_cls,filter_model=model_filter,lr=1e-4,device=device,
                      save_path=cls_model_path,batch_size=512,snr=35,
                      head_path= '../ConstructedData/Training_'+str(dict_size),
                      dict_size=dict_size,logic_settings=logic+'.yaml')
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


    elif cls_encoder_name in ['DecisionTree','RandomForest', 'KNN','GaussianNB','MLP','SVM',]:
        model_cls = fetch_ml_model(cls_encoder_name)
        model_save_pth = os.path.join(logicspace,cls_encoder_name+'.pkl')
        Xtest, Ytest = load_data(head_path='../ConstructedData/Testing_' + str(dict_size),
                                 logic_settings=logic + '.yaml')
        if not os.path.exists(model_save_pth):
            print('Training ML Model')
            Xtrain, Ytrain = load_data(head_path='../ConstructedData/Training_'+str(dict_size),logic_settings=logic+'.yaml')
            data_size = return_dataset_size(dataset_scale,Ytrain.shape[0])
            model_cls.fit(Xtrain[:data_size], Ytrain[:data_size])
            with open(model_save_pth, 'wb') as f:
                pickle.dump(model_cls, f)  #
                print('Saving ML Model')
        else:
            print('Loading ML Model')
            with open(model_save_pth, 'rb') as f:
                model_cls = pickle.load(f)
        filtered_acc = torch.zeros((dict_size,dict_size))
        unfiltered_acc = eva_ml_cls(model=model_cls, test_data=zip(Xtest,Ytest), device=device, filter_model=model_filter,component_idx=None)
        print('unfiltered',unfiltered_acc)
        for idx in range(dict_size):
            for idx_dual in range(idx,dict_size):
                filtered_acc[idx,idx_dual] = eva_ml_cls(model=model_cls,
                                               test_data=zip(Xtest,Ytest),
                                               device=device,
                                               filter_model=model_filter,
                                               component_idx=[idx,idx_dual])

                filtered_acc[idx_dual,idx] = filtered_acc[idx,idx_dual]
                print(idx,idx_dual,filtered_acc[idx,idx_dual])


        torch.save(filtered_acc,
                   os.path.join(logicspace, cls_encoder_name + dataset_scale + '_filter_acc.pt'))
        print(unfiltered_acc - filtered_acc)
        torch.save(unfiltered_acc - filtered_acc, os.path.join(logicspace, cls_encoder_name+dataset_scale+'_delta_acc.pt'))
    print(logic)





if __name__ == '__main__':


    dict_size = 15
    encoder_name = 'SiT'
    decoder_name = 'PPS'
    cls_enc_name = 'RandomForest'
    data_scale = 'micro'
    workspace = '../ComFilE'+str(dict_size)+'_'+data_scale+'/'

    cls_enc_names = ['DecisionTree','RandomForest', 'KNN','GaussianNB','MLP','SVM']
    logics = ['AND1','OR1','SOFT1','AND','OR','SOFT']
    for cls_enc_name in cls_enc_names:
        for logic in logics:
            if os.path.exists(os.path.join(workspace,logic, cls_enc_name +data_scale+ '_delta_acc.pt')):
                print(logic, cls_enc_name,'Already Done')
            else:
                ComFilE(device='cuda:1', logic=logic,
                    filter_encoder_name=encoder_name,
                    filter_decoder_name=decoder_name,workspace = workspace,
                    cls_encoder_name=cls_enc_name,dict_size=dict_size,
                        dataset_scale=data_scale)