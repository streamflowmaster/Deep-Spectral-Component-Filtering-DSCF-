import torch
import torch.utils.data as tud
import os
from ComponentFiltering.COMFILE.generating_virtual_label import generate_label




def load_data(head_path = '../ConstructedData/Testing', spectra_pth = 'data_with_BG',snr = None,
                 labe_path = 'BG_GT',dict_size = 15, batch = 200,
                 spectra_dict_pth = '../SpectraDict/Reference123.txt',
                 logic_settings = 'AND.yaml',device= 'cuda:0'):
    labe_path = os.path.join(head_path, labe_path)
    spectra_pth = os.path.join(head_path, spectra_pth)
    labels_lib =  os.listdir(labe_path)
    # print(labels_lib)
    spectra_lib = os.listdir(spectra_pth)
    virtual_label_logic = generate_label(logic_file=logic_settings)
    cls_list = torch.zeros((len(labels_lib),batch))
    specs_list = torch.zeros((len(labels_lib),batch,709))

    for i in range(len(labels_lib)):
        labels = torch.load(os.path.join(labe_path, labels_lib[i]))
        spectra = torch.load(os.path.join(spectra_pth, spectra_lib[i]))
        for j in range(batch):
            cls_list[i,j] = virtual_label_logic.generate(abm=labels[j])
            specs_list[i,j] = spectra[j]
    # print(cls_list)
    return specs_list.reshape(len(labels_lib)*batch,-1),\
           cls_list.reshape(len(labels_lib)*batch,1)

