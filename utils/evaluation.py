import matplotlib.pyplot as plt
from ComponentFiltering.Denoising.utils.dataset import train_dataloader

def evaluation(snr,model,device='cpu',batch_size = 16, head_path = '../../ConstructedData/Testing',
                   dict_size = 122,spectra_dict_pth =  '../../SpectraDict/Reference123.txt'):

    train_data = train_dataloader(batch_size=batch_size,spectra_dict_pth=spectra_dict_pth, head_path=head_path, dict_size=dict_size, snr=snr)
    for idx,(inputs,target) in enumerate(train_data):

        inputs = inputs.float().to(device).unsqueeze(1)
        target = target.float().to(device).unsqueeze(1)

        # print(inputs.shape)
        outputs = model(inputs)
        # print('input:', inputs.shape)
        # print('output:',outputs.shape)
        # print('target:', target.shape)

        B,C,Lt = target.shape
        B,C,Lo = outputs.shape
        # print(Lt,Lo)
        if Lt<Lo:
            outputs = outputs[:,:,:Lt]
        else:
            target = target[:,:,:Lo]

        plt.plot(inputs[0, 0, 1:].T.cpu().detach().numpy(), c='g')
        plt.plot(outputs[0,0,1:].T.cpu().detach().numpy()+10,c = 'b')
        plt.plot(target[0,0,1:].T.cpu().detach().numpy()+20,c = 'r')
        plt.show()


def evaluation_one_epoch(snr, data,model,device='cpu',model_name = None):
    if 'SiT' in model_name:
        device = 'cuda:0'
        model = model.to(device)
    inputs, target = data
    inputs = inputs.float().to(device).unsqueeze(1)
    target = target.float().to(device).unsqueeze(1)
    outputs = model(inputs)
    B,C,Lt = target.shape
    B,C,Lo = outputs.shape


    # print(Lt,Lo)
    if Lt<Lo:
        outputs = outputs[:,:,:Lt]
    else:
        target = target[:,:,:Lo]
    return outputs.to('cpu')

