This work, titled “Deep Spectral Component Filtering as a Foundation Model for Spectral Analysis Demonstrated in Metabolic Profiling,” is published in Nature Machine Intelligence. This repository contains code for utilizing a pretrained foundation model tailored for spectral analysis. To enhance accessibility for spectroscopy researchers, the code has been designed with user-friendliness in mind, allowing for a seamless start without requiring complex training frameworks or extensive environment configuration. Additionally, we have provided scripts for fine-tuning, accompanied by clear instructions within them, to assist users in loading their own data and adapting the model to their specific tasks.

# Directory notation
**pretrain**: This folder contains the pretrained model weights and general-purpose tools for utilizing the foundation model.

**customized_task**: This folder includes scripts for applying the pretrained model and finetuning it to suit your specific tasks.

**preprocessing**: This folder provides scripts for preprocessing, along with source files and results for tasks such as infrared paraffin removal and SERS nanoparticle (NPs) removal.

**quantify**: This folder contains scripts for quantification, accompanied by spectral data ready for quantitative analysis.

**ComFilE**: This folder includes scripts and results for the Component Filtering Explanation (ComFilE) method. ComFilE can be used to rank the importance of specific spectral components (e.g., metabolites in serum) and interpret their contributions to distinguishing results (e.g., disease vs. control samples).

**ComFilE_Extended**: This folder contains scripts and results for the k-order Component Filtering Explanation (where k > 1). The k-order ComFilE extends the methodology to analyze the cooperative effects of k spectral components in explaining result distinctions.

# Quick start
The scripts to build DSCF for your personalized work are in the directory "costumized_task". 

To start with the scripts, you should follow the instructions in the 'dataset.py' to load your spectra into the corresponding file folds. 

    Fold 'Component-spec' is for the spectral dictionary of pure substances.
    
    Fold 'Impurity-spec' is for unwanted spectral components to filter out from spectra
    
    Fold 'Pure-spec' is for spectral components to be preserved. 
    
{'dir':'Pure-spec/',
       'tensor_dim':2,
       'spec_tensor_dim':-1,}
       
An attribution dictionary should be innit for each data fold. Tensor_dim is to describe the total dimension of one data file. Spec_tensor_dim is to describe the id of spectral dimension in the data file, ranging from (0,tensor_dim-1).

The output mode can be customized by revising the 'return value' in the __gettitem__ function. 

# Gallery of implicit results behind this work
## Model architecture

DSCF model is a hierarchical local attention encoder-decoder transformer. The detailed components of the model are described in DSCF_model_pe.py.
The following image is the general outline of the general pre-trained model.
The pre-trained weights of the tiny-version model are available and can be downloaded at https://figshare.com/s/2b31ca642313086dcfe6. The weights of larger models will also be open-source after publishing.
![image](https://github.com/user-attachments/assets/56879799-315c-4138-8e49-f273dd2dbd28)


## Preprocessing

Paraffin removal is a general routine in FFPE IR analysis. DSCF model can be tailored for paraffin removal. The following images are results of raw data, paraffin and paraffin-removed data.

![pt_data6-201521653-7hsi_normed](https://github.com/user-attachments/assets/2916b6f5-a878-4fa6-a882-488c586c9812)
![pt_data6-201521653-7removed_normed](https://github.com/user-attachments/assets/83a587d8-ffe6-4161-a3b2-97739ffad1c0)
![pt_data6-201521653-7paraffin_normed](https://github.com/user-attachments/assets/6b2a23ba-3fe7-401c-8976-edb3a0ef8824)


## Explaining for spectral marker

Some of the in-silico explaining results are as follows, where highlighted components are ground truth.
![image](https://github.com/user-attachments/assets/7a398f74-1eed-49bc-80b4-c50d566ada8d)
![image](https://github.com/user-attachments/assets/78093c24-b4c8-4275-b423-b6cea85dacee)

The code for detailed downstream tasks is coming soon after the manuscript is formally published.

2nd-order ComFilE. 
![ComFilE_Extend](https://github.com/user-attachments/assets/549408f2-4294-43ab-8f0c-70fe98e3b76c)

