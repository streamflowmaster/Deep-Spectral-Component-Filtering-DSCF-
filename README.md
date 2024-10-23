Here is the code for pretraining the foundation model for spectral analysis, which is now under peer review so that the partial core source code can be uploaded to the project. 



# Directory notation
utils： The file fold contains general pretraining model weights and usage tools.

Preprocessing: The file fold contains scripts for preprocessing, source files, and results of infrared paraffin removal and SERS NPs removal.

quantify: The file fold contains scripts for quantification and spectral data to be quantified.

ComFilE: The file fold contains scripts and results for the Component Filtering Explanation. 


# Gallery of charming results during work
## Model architecture

The model is a hierarchical local attention encoder-decoder transformer. The detail components of model are described in DSCF_model_pe.py.
The following image is the general outline of the general pre-trained model.
![image](https://github.com/user-attachments/assets/56879799-315c-4138-8e49-f273dd2dbd28)


## Paraffin remove

![pt_data6-201521653-7hsi_normed](https://github.com/user-attachments/assets/2916b6f5-a878-4fa6-a882-488c586c9812)
![pt_data6-201521653-7removed_normed](https://github.com/user-attachments/assets/83a587d8-ffe6-4161-a3b2-97739ffad1c0)
![pt_data6-201521653-7paraffin_normed](https://github.com/user-attachments/assets/6b2a23ba-3fe7-401c-8976-edb3a0ef8824)




## Explaining for spectral marker
![image](https://github.com/user-attachments/assets/7a398f74-1eed-49bc-80b4-c50d566ada8d)
![image](https://github.com/user-attachments/assets/78093c24-b4c8-4275-b423-b6cea85dacee)

The code for detailed downstream tasks is coming soon after the manuscript is formally published.
