# Brain-tumor-segmentation
## How to install

1. Download the code: 

<img width="918" alt="Screenshot 2025-03-04 alle 15 29 57" src="https://github.com/user-attachments/assets/a73ba8b3-4ff8-4a65-a255-f9ba58bf801f" />

2. Unzip in a folder of your choice on your computer

3. Open 3D Slicer
4. Check if General Registration (BRAINS) extension is installed (Modules -> Registration). If not, install it from the extension manager
5. Install "swissskullstripping" extension from the extension manager
6. Restart Slicer
   
7. Import module in 3D Slicer:

   - Slicer -> Edit -> Application Settings -> Modules -> Additional module paths
   - Add path to the "CustomInference" folder
  
<img width="1454" alt="Screenshot 2025-03-05 alle 11 17 39" src="https://github.com/user-attachments/assets/e25676f1-11f8-4eb2-84d0-fdeb5202aa0a" />

   - Save changes
   - Restart Slicer
8. Import independent python modules

   - Open Python Console
     
<img width="409" alt="Screenshot 2025-03-04 alle 15 39 54" src="https://github.com/user-attachments/assets/d512d75c-e94b-48a7-94df-a2b32278bbd9" />

   - Copy and paste the code below:
     
    
    >>> import pip
    >>> pip.main(['install', 'numpy', 'torch', 'monai[nibabel]'])
    
    


6. Download the trained model folder here: [Model](https://drive.google.com/drive/folders/1zyauzbkUiraO2vhI-AeAP7jT8DQyq1ET?usp=sharing)



