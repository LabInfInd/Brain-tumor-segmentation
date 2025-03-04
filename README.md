# Brain-tumor-segmentation
## How to install

1. Download the code: 

<img width="918" alt="Screenshot 2025-03-04 alle 15 29 57" src="https://github.com/user-attachments/assets/a73ba8b3-4ff8-4a65-a255-f9ba58bf801f" />

2. Unzip in a folder of your choice on your computer

3. Open 3D Slicer
   
5. Import module in 3D Slicer:

   - Modules —> Developer Tools —> Extension Wizard 
<img width="528" alt="Screenshot 2025-03-04 alle 15 38 37" src="https://github.com/user-attachments/assets/b15e649e-86c2-4b5a-8203-c8d582d3ab6c" />


   - Select Extension —> Choose code folder —> Open
   - Now you can see the Extension CUSTOM INFERENCE MODULE in the Module list

6. Import independent python modules

   - Open Python Console
     
<img width="409" alt="Screenshot 2025-03-04 alle 15 39 54" src="https://github.com/user-attachments/assets/d512d75c-e94b-48a7-94df-a2b32278bbd9" />

   - Copy the code below:
     
    ```python
    >>> import pip
    >>> pip.main(['install', 'torch', 'monai[nibabel]'])
    ```
    


6. Download the trained model: *LINK*


