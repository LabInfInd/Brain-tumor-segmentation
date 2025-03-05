# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:27:14 2025

@author: Utente
"""



import os
import torch
import numpy as np
from monai.transforms import  EnsureChannelFirstd, Compose, LoadImaged, EnsureTyped, Orientationd, Resized, NormalizeIntensityd, AdjustContrastd, Activations, AsDiscrete
from monai.data import DataLoader, Dataset, decollate_batch
from monai.data.meta_tensor import MetaTensor
from slicer.ScriptedLoadableModule import *
import slicer
import vtk, qt, ctk
import time
from VNetModel.VNetModel import VNetMultiEncoder

class CustomInferenceModules(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Custom Inference Module"
        self.parent.categories = ["MyModules"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name (Your Institution)"]
        self.parent.helpText = "Modulo per inferenza con V-Net in 3D Slicer."
        self.parent.acknowledgementText = "Sviluppato da Your Name, Your Institution."

class CustomInferenceModulesWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.inputVolumes = {}  
        self.logic = CustomInferenceModulesLogic()  

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
    
    
        titleLabel = qt.QLabel("Loading Model")  
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")  
        self.layout.addWidget(titleLabel)
        self.loadModelButton = qt.QPushButton("Select Model")
        self.layout.addWidget(self.loadModelButton)
        self.loadModelButton.connect('clicked(bool)', self.onLoadModelButton)
        
      
        
        titleLabel = qt.QLabel("Select Loaded Volumes")
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")
        self.layout.addWidget(titleLabel)
        
        



        self.volumeSelectors = {}

        for modality in ["T1", "T2", "T1CE", "FLAIR"]:
            rowLayout = qt.QHBoxLayout()
            rowLayout.setSpacing(10)
            label = qt.QLabel(f"{modality}:")
            rowLayout.addWidget(label)

            comboBox = qt.QComboBox()
            comboBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)  


            self.populateVolumeSelector(comboBox)

            rowLayout.addWidget(comboBox)
            self.volumeSelectors[modality] = comboBox

            
            self.layout.addLayout(rowLayout)
            
        self.updateComboBoxes()
        slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.onNodeAdded)
        slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeRemovedEvent, self.onNodeRemoved)


        self.confirmSelectionButton = qt.QPushButton("Confirm Selection")
        self.layout.addWidget(self.confirmSelectionButton)
        self.confirmSelectionButton.connect('clicked(bool)', self.onConfirmSelection)
        
        
        titleLabel = qt.QLabel("Skullstripping") 
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")  
        self.layout.addWidget(titleLabel)
        self.skullstrippingButton = qt.QPushButton("Run Skullstripping")
        self.layout.addWidget(self.skullstrippingButton)
        self.skullstrippingButton.connect('clicked(bool)', self.onSkullStrippingButton)
        
        
        titleLabel = qt.QLabel("Preprocessing")  
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")  
        self.layout.addWidget(titleLabel)
        self.preprocessingButton = qt.QPushButton("Run Preprocessing")
        self.layout.addWidget(self.preprocessingButton)
        self.preprocessingButton.connect('clicked(bool)', self.onPreprocessingButton)
        
        
        
        
        
        
        titleLabel = qt.QLabel("Segmentation")  
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")  
        self.layout.addWidget(titleLabel)
        self.inferenceButton = qt.QPushButton("Run Inference")
        self.layout.addWidget(self.inferenceButton)
        self.inferenceButton.connect('clicked(bool)', self.onInferenceButton)
        
        

        self.modifySegmentationButton = qt.QPushButton("Modify Segmentation")
        self.layout.addWidget(self.modifySegmentationButton)
        self.modifySegmentationButton.connect('clicked(bool)', self.onModifySegmentation)
        


        self.returnToMainButton = qt.QPushButton("Return to Main")

        self.returnToMainButton.connect('clicked(bool)', self.onReturnToMainButton)

       
        
        self.layout.addStretch(1)

    def onLoadModelButton(self):
        modelFolderPath = qt.QFileDialog.getExistingDirectory(self.parent, "Select Model Directory")
        if modelFolderPath:
            self.logic.loadModels(modelFolderPath)
            qt.QMessageBox.information(self.parent, "Model Loaded", "Modelli caricati con successo.")
            
    def updateComboBoxes(self):    
        for comboBox in self.volumeSelectors.values():
            self.populateVolumeSelector(comboBox)
    
    def onNodeAdded(self, caller, event):  
        self.updateComboBoxes()
    
    def onNodeRemoved(self, caller, event):
        self.updateComboBoxes()
        
    def onLoadVolume(self, modality):
        fileDialog = qt.QFileDialog()
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)
        fileDialog.setNameFilter("NIfTI Files (*.nii *.nii.gz)")
        if fileDialog.exec_():
            selectedFile = fileDialog.selectedFiles()[0]
            self.logic.loadVolume(modality, selectedFile)
            qt.QMessageBox.information(self.parent, "Volume Loaded", f"{modality} caricato con successo.")
    
    
    def populateVolumeSelector(self, comboBox):
        
        comboBox.clear() 
        volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")

        for node in volumeNodes:
            storageNode = node.GetStorageNode()
            if storageNode and storageNode.GetFileName():
                filePath = storageNode.GetFileName()
                comboBox.addItem(filePath, node) 

    def onConfirmSelection(self):
       
        for modality, comboBox in self.volumeSelectors.items():
            selectedIndex = comboBox.currentIndex  
            
            if selectedIndex >= 0:  
                selectedNode = comboBox.itemData(selectedIndex)  
                
                if selectedNode:
                    existingNode = slicer.util.getNode(modality) if slicer.util.getNodes(modality) else None
    

                    if existingNode and existingNode == selectedNode:
                        
                        continue
    

                    if selectedNode.GetName() != modality:  
                        selectedNode.SetName(modality) 
    
                    self.inputVolumes[modality] = selectedNode  
                else:
                    print(f"Errore: Nessun nodo selezionato per {modality}") 
                    qt.QMessageBox.warning(self.parent, "Selection Error", f"Please select a volume for {modality}.")
                    return
      
        qt.QMessageBox.information(self.parent, "Volumes Selected", "Volumes successfully selected and renamed!")
                
    def onPreprocessingButton(self):
        modalities = ["T1", "T2", "T1CE", "FLAIR"]
        fixedImage = slicer.util.getNode("T1")  
    
        for modality in modalities:
            movingImage = slicer.util.getNode(modality)
            if movingImage is None:
                print(f"Il volume {modality} non è stato trovato, salto...")
                continue
    

            storageNode = movingImage.GetStorageNode()
            if not storageNode:
                print(f"Errore: Il volume {modality} non ha un nodo di archiviazione, impossibile sovrascrivere.")
                continue
            originalPath = storageNode.GetFullNameFromFileName()
    

            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{modality}_transform")

    

            parameters = {
                "fixedVolume": fixedImage.GetID(),
                "movingVolume": movingImage.GetID(),
                "linearTransform": transformNode.GetID(),
                "interpolationMode": "Linear",
                "initializeTransformMode": "useMomentsAlign",

                "use_histogram_matching": True,
                "outputVolume": movingImage.GetID(),

                "useRigid": True
            }
    

            cliModule = slicer.modules.brainsfit
            cliNode = slicer.cli.runSync(cliModule, None, parameters)
            
    
        print("Volumi registrati e sovrascritti")
        
    
    def onSkullStrippingButton(self):
        
        modalities = ["T1", "T2", "T1CE", "FLAIR"]
        
        for modality in modalities:
            imageNode = slicer.util.getNode(modality)
            if imageNode is None:
                print(f"Il volume {modality} non è stato trovato, salto...")
                continue
    
           
            maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{modality}_brainmask")
    

            parameters = {
                "patientVolume": imageNode.GetID(),
                "patientOutputVolume": imageNode.GetID(),
                "patientMaskLabel": maskVolumeNode.GetID() 
                }

            cliModule = slicer.modules.swissskullstripper
            cliNode = slicer.cli.runSync(cliModule, None, parameters)
    
          
        print("Skull Stripping eseguito")
    


    def onInferenceButton(self):
        
        self.logic.runInference()

    def onReturnToMainButton(self):
        slicer.util.selectModule('CustomInferenceModules')
    


    def onModifySegmentation(self):
        segmentationNode = slicer.util.getNode('Segmentation')
        referenceVolume = slicer.util.getNode("T1CE_Preprocessed") 
        if segmentationNode:
            slicer.modules.segmenteditor.widgetRepresentation().show()
            slicer.util.selectModule('SegmentEditor')
            slicer.util.setSliceViewerLayers(foreground=segmentationNode)
    

            editorWidget = slicer.modules.segmenteditor.widgetRepresentation()
            layout = editorWidget.layout()
            layout.addWidget(self.returnToMainButton)
            self.returnToMainButton.show()
        else:
            qt.QMessageBox.warning(None, "Errore", "Nessuna segmentazione trovata.")
    
            
        




class CustomInferenceModulesLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        self.models = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.transforms = Compose([
    
            EnsureTyped(keys=["T1", "T2", "FLAIR", "T1CE"]),
            Orientationd(keys=["T1", "T2", "FLAIR", "T1CE"], axcodes="RAS"),
            Resized(keys=["T1", "T2", "FLAIR", "T1CE"], spatial_size=(192, 192,150), mode="trilinear", align_corners=True), #!!!!
            NormalizeIntensityd(keys=["T1", "T2", "FLAIR", "T1CE"], nonzero=True, channel_wise=True),
            AdjustContrastd(keys=["T2", "FLAIR"], gamma=1.1)
        ])
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    def loadModels(self, modelFolderPath):
        self.models = []
        for fold in range(5):
            model = VNetMultiEncoder(input_channels=1).to(self.device)
            modelPath = os.path.join(modelFolderPath, f"best_model_fold{fold+1}.pth")
            model.load_state_dict(torch.load(modelPath, map_location=self.device), strict=False)
            model.eval()
            self.models.append(model)

    
          
    def runInference(self):
        if not self.models:
            qt.QMessageBox.warning(None, "Error", "No models loaded. Load models before running inference.")
            return
        
        progressDialog = qt.QProgressDialog("Inferenza in corso...", None, 0, 100)
        progressDialog.setWindowTitle("Inferenza")
        progressDialog.setCancelButton(None)
        progressDialog.show()
        
        slicer.app.processEvents()
        
        sequence_names = ["T1", "T2", "T1CE", "FLAIR"]
        
        # Funzione per ottenere un volume da 3D Slicer come array NumPy
        def get_slicer_volume_as_numpy(volume_name):
            volume_node = slicer.util.getNode(volume_name)  
            image_data = slicer.util.arrayFromVolume(volume_node) 
            return np.moveaxis(image_data, 0, -1), volume_node  # Riordina gli assi (Z, Y, X diventa  X, Y, Z)
        
        # Estrarre i dati da Slicer
        image_dict = {}
        for seq in sequence_names:
            image_numpy, _ = get_slicer_volume_as_numpy(seq)
            image_dict[seq] = np.expand_dims(image_numpy, axis=0)
        
        # Creare una lista di dizionari per MONAI
        data_list = [{"T1": image_dict["T1"], "T2": image_dict["T2"], 
                      "T1CE": image_dict["T1CE"], "FLAIR": image_dict["FLAIR"]}]
        dataset = Dataset(data=data_list, transform=self.transforms)
        dataloader = DataLoader(dataset, batch_size=1)
        
        steps_per_model = 50  
        total_steps = len(self.models) * steps_per_model + 40 
        current_step = 0
        
        def update_progress_bar(step_increment):
            nonlocal current_step
            current_step += step_increment
            progressDialog.setValue(int((current_step / total_steps) * 100))
            slicer.app.processEvents()
        
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                inputs = {key: batch[key].to(self.device) for key in ["T1CE", "T1", "T2", "FLAIR"]}
        
                all_predictions = []
                for j, model in enumerate(self.models):
                    test_output = model(inputs["T1CE"], inputs["T1"], inputs["T2"], inputs["FLAIR"])
                    test_output = [self.post_trans(i) for i in decollate_batch(test_output)]
                    all_predictions.append(torch.stack(test_output))
                    
                    for _ in range(10):
                        update_progress_bar(5)
        
                all_predictions = torch.stack(all_predictions)
                final_prediction = torch.mode(all_predictions, dim=0)[0]
        
                for _ in range(10):
                    update_progress_bar(2)
        
                preprocessed_volumes = {}
                for key in ["T1CE", "T1", "T2", "FLAIR"]:
                    referenceVolume = slicer.util.getNode(key) 
                    preprocessed_array = batch[key].squeeze()  
                    print(f"Shape del volume preprocessato {key} prima della trasposta:", preprocessed_array.shape)
                    
                    preprocessed_array = np.transpose(preprocessed_array, (2, 0, 1))  
                    print(f"Shape del volume preprocessato {key}:", preprocessed_array.shape)
        
                    newVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{key}_Preprocessed")
                    slicer.util.updateVolumeFromArray(newVolume, preprocessed_array)
                    newVolume.SetOrigin(referenceVolume.GetOrigin())  #nuove
                    newVolume.SetSpacing(referenceVolume.GetSpacing())
                    directionMatrix = vtk.vtkMatrix4x4()
                    referenceVolume.GetIJKToRASDirectionMatrix(directionMatrix)
                    newVolume.SetIJKToRASDirectionMatrix(directionMatrix)
 
                    preprocessed_volumes[key] = newVolume
        
                    update_progress_bar(5)
        
                mask_array = final_prediction.cpu().numpy().squeeze()
                self.createSegmentationNode(mask_array, preprocessed_volumes)
                update_progress_bar(10)
        
        progressDialog.setValue(100)
        slicer.app.processEvents()
        
    def createSegmentationNode(self, mask_array, preprocessed_volumes):
            referenceVolume = preprocessed_volumes["T1CE"]
        
            if not isinstance(referenceVolume, slicer.vtkMRMLScalarVolumeNode):
                qt.QMessageBox.warning(None, "Error", "Il volume di riferimento non è valido.")
                return
        
            mask_array = mask_array.astype(np.uint8)
            print("Shape maschera:", mask_array.shape)
            print("Valori unici nella maschera:", np.unique(mask_array))
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolume)
            segmentationNode.CreateDefaultDisplayNodes()
            slicer.app.processEvents()
        
            labels = ["NET", "ED", "ET"]
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        
            for i, label in enumerate(labels):
                segmentId = segmentationNode.GetSegmentation().AddEmptySegment(label)
                mask = mask_array[i].transpose(2,0,1)  
                print("Shape maschera:", mask.shape)
                slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentationNode, segmentId, referenceVolume)
                segmentationNode.GetSegmentation().GetSegment(segmentId).SetColor(colors[i])
        
            
            slicer.app.processEvents()
        
            segmentationDisplayNode = segmentationNode.GetDisplayNode()
            segmentationDisplayNode.SetVisibility2DFill(True)
            segmentationDisplayNode.SetVisibility2DOutline(True)
            segmentationDisplayNode.SetVisibility3D(True)
            
            self.updateSliceViewerLayers(preprocessed_volumes, segmentationNode)
            
    def updateSliceViewerLayers(self, preprocessed_volumes=None, segmentationNode=None):
        if preprocessed_volumes is None:
            try:
                preprocessed_volumes = {key: slicer.util.getNode(f'Preprocessed_{key}') for key in ["T1CE", "T1", "T2", "FLAIR"]}
            except slicer.util.MRMLNodeNotFoundException as e:
                qt.QMessageBox.warning(None, "Error", str(e))
                return
        if segmentationNode is None:
            try:
                segmentationNode = slicer.util.getNode('Segmentation')
            except slicer.util.MRMLNodeNotFoundException as e:
                qt.QMessageBox.warning(None, "Error", str(e))
                return
        
       
        for sliceViewName in ['Red', 'Green', 'Yellow']:
            sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
            sliceLogic = sliceWidget.sliceLogic()
            sliceCompositeNode = sliceLogic.GetSliceCompositeNode()
            sliceCompositeNode.SetBackgroundVolumeID(preprocessed_volumes["T1CE"].GetID())
            sliceCompositeNode.SetForegroundVolumeID(segmentationNode.GetID())
            sliceCompositeNode.SetForegroundOpacity(0.5)

            sliceLogic.FitSliceToAll()
    

        slicer.app.processEvents()
        
    
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
