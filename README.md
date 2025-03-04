# Brain-tumor-segmentation
## How to install

1. Download the code: 
   
2. Download the trained model: 


Import module in 3D Slicer:

- Modules —> Developer Tools —> Extension Wizard
- Select Extension —> Choose clone folder —> Open
- Now you can see my Extension in the Module list

Import independent python module

Open Python Console
>>> import pip
>>> pip.main(['install', 'torch', 'monai[nibabel]'])
