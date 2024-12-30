Here is the code for pretraining the foundation model for spectral analysis, which is now under peer review so that the partial core source code can be uploaded to the project. 



# Directory notation
utils： The file fold contains general pretraining model weights and usage tools.

Preprocessing: The file fold contains scripts for preprocessing, source files, and results of infrared paraffin removal and SERS NPs removal.

quantify: The file fold contains scripts for quantification and spectral data to be quantified.

ComFilE: The file fold contains scripts and results for the Component Filtering Explanation. 


# Gallery of implicit results behind this work
## Model architecture

DSCF model is a hierarchical local attention encoder-decoder transformer. The detailed components of the model are described in DSCF_model_pe.py.
The following image is the general outline of the general pre-trained model.
The pre-trained weights of the tiny-version model are open source and can be downloaded at https://pan.baidu.com/s/1KemTI4yx2-6SwKzmSYyROQ?pwd=ira8. The weights of larger models will also be open-source after publishing.
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

