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
import torch.nn.functional as F

import SimpleITK as sitk
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
        self.sceneObservers = []
        self.logic = CustomInferenceModulesLogic()  
        self.volumeSelectors = {}
        self.volumeNodes = []
        

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
        
        



       

        for modality in ["T1", "T2", "T1CE", "FLAIR"]:
            rowLayout = qt.QHBoxLayout()
            rowLayout.setSpacing(10)
            label = qt.QLabel(f"{modality}:")
            rowLayout.addWidget(label)

            comboBox = qt.QComboBox()
            comboBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)  


            

            rowLayout.addWidget(comboBox)
            self.volumeSelectors[modality] = comboBox

            
            self.layout.addLayout(rowLayout)

        self.updateComboBoxes()
        

        
        self.timer = qt.QTimer()
        self.timer.timeout.connect(self.checkForNewVolumes)
        self.timer.start(1000)  # Controlla ogni secondo
       
    
      
        
    
        

        self.confirmSelectionButton = qt.QPushButton("Confirm Selection")
        self.layout.addWidget(self.confirmSelectionButton)
        self.confirmSelectionButton.connect('clicked(bool)', self.onConfirmSelection)
        
        
        titleLabel = qt.QLabel("Skullstripping") 
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")  
        self.layout.addWidget(titleLabel)
        self.skullstrippingButton = qt.QPushButton("Run Skullstripping")
        self.layout.addWidget(self.skullstrippingButton)
        self.skullstrippingButton.setEnabled(False) 
        self.skullstrippingButton.connect('clicked(bool)', self.onSkullStrippingButton)
        
        
        titleLabel = qt.QLabel("Preprocessing")  
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")  
        self.layout.addWidget(titleLabel)
        self.preprocessingButton = qt.QPushButton("Run Preprocessing")
        self.layout.addWidget(self.preprocessingButton)
        self.preprocessingButton.setEnabled(False)
        self.preprocessingButton.connect('clicked(bool)', self.onPreprocessingButton)
        
        
        
        
        
        
        titleLabel = qt.QLabel("Segmentation")  
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")  
        self.layout.addWidget(titleLabel)
        self.inferenceButton = qt.QPushButton("Run Inference")
        self.layout.addWidget(self.inferenceButton)
        self.inferenceButton.setEnabled(False)
        self.inferenceButton.connect('clicked(bool)', self.onInferenceButton)
        
        

        self.modifySegmentationButton = qt.QPushButton("Modify Segmentation")
        self.layout.addWidget(self.modifySegmentationButton)
        self.modifySegmentationButton.setEnabled(False)
        self.modifySegmentationButton.connect('clicked(bool)', self.onModifySegmentation)
        


        self.returnToMainButton = qt.QPushButton("Return to Main")

        self.returnToMainButton.connect('clicked(bool)', self.onReturnToMainButton)

       
        
        self.layout.addStretch(1)
        
    def updateComboBoxes(self):      
        for comboBox in self.volumeSelectors.values():
            self.populateVolumeSelector(comboBox)
    
    def checkForNewVolumes(self):
        
        currentVolumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        
        if len(currentVolumeNodes) != len(self.volumeNodes):
            self.updateComboBoxes()  

        self.volumeNodes = currentVolumeNodes
    
    def addSceneObservers(self):
        
        self.removeSceneObservers()  
    
        observer1 = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.onSceneUpdated)
        observer2 = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeRemovedEvent, self.onSceneUpdated)
        #observer3 = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.ModifiedEvent, self.onSceneUpdated)
    
        self.sceneObservers.append(observer1)
        self.sceneObservers.append(observer2)

   
        
    def onLoadModelButton(self):
        modelFolderPath = qt.QFileDialog.getExistingDirectory(self.parent, "Select Model Directory")
        if modelFolderPath:
            self.logic.loadModels(modelFolderPath)
            qt.QMessageBox.information(self.parent, "Model Loaded", "Modelli caricati con successo.")

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
            
        slicer.app.processEvents()

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
        self.skullstrippingButton.setEnabled(True)
        #self.preprocessingButton.setEnabled(True)
        #self.inferenceButton.setEnabled(True)
        #self.modifySegmentationButton.setEnabled(True)
                
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
        self.inferenceButton.setEnabled(True)
        
    
    def onSkullStrippingButton(self):
        
        modalities = ["T1", "T2", "T1CE", "FLAIR"]
        
        for modality in modalities:
            imageNode = slicer.util.getNode(modality)
            if imageNode is None:
                print(f"Il volume {modality} non è stato trovato, salto...")
                continue
    
            originalNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{modality}_original")

            
            originalNode.Copy(imageNode)  
         
            newStorageNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            
         
            storageNode = imageNode.GetStorageNode()
            
            if storageNode:
                
                originalNode.SetAndObserveStorageNodeID(newStorageNode.GetID())  
            
               
                filePath = storageNode.GetFileName()
                newStorageNode.SetFileName(filePath)  
            
           
            originalNode.SetName(f"{modality}_original") 

            maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{modality}_brainmask")
    

            parameters = {
                "patientVolume": imageNode.GetID(),
                "patientOutputVolume": imageNode.GetID(),
                "patientMaskLabel": maskVolumeNode.GetID() 
                }

            cliModule = slicer.modules.swissskullstripper
            cliNode = slicer.cli.runSync(cliModule, None, parameters)
    
        self.checkForNewVolumes()
        print("Skull Stripping eseguito")
        self.preprocessingButton.setEnabled(True)
        
    


    def onInferenceButton(self):
        
        self.logic.runInference()
        self.modifySegmentationButton.setEnabled(True)

    def onReturnToMainButton(self):
        slicer.util.selectModule('CustomInferenceModules')
    

    
    def onModifySegmentation(self): 
        # Trova la segmentazione attiva (quella visibile)
        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        activeSegmentation = None
    
        for segNode in segmentationNodes:
            if segNode.GetDisplayNode().GetVisibility():  # Segmentazione visibile
                activeSegmentation = segNode
                break
    
        if not activeSegmentation:
            qt.QMessageBox.warning(None, "Errore", "Nessuna segmentazione visibile trovata.")
            return
    
        # Disabilita gli osservatori dei slice per evitare aggiornamenti indesiderati
        self.logic.disableSliceObservers()
    
        # Seleziona il modulo SegmentEditor
        slicer.util.selectModule('SegmentEditor')
        slicer.app.processEvents()
    
        # Mostra il pannello di modifica della segmentazione
        editorWidget = slicer.modules.segmenteditor.widgetRepresentation()
        layout = editorWidget.layout()
        layout.addWidget(self.returnToMainButton)
        self.returnToMainButton.show()
    
        # Abilita gli osservatori e riallinea le slice
        self.logic.enableSliceObservers()
        self.logic.forceSliceRealignment()
    
        slicer.app.processEvents()




    
            
        




class CustomInferenceModulesLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        self.models = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.updatingSlice = False


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
    
    def disableSliceObservers(self):
        
        for sliceViewName in ['Red', 'Green', 'Yellow']:
            sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
            sliceNode = sliceWidget.sliceLogic().GetSliceNode()
            observerTag = getattr(sliceNode, "_orientationObserver", None)
            if observerTag is not None:
                sliceNode.RemoveObserver(observerTag)
                setattr(sliceNode, "_orientationObserver", None)  
    
        
    def enableSliceObservers(self):
        try:
            # Creiamo una mappa tra volumi e segmentazioni
            volume_mask_mapping = {
                #"T1_original": "Segmentation_T1_original",
                #"T1CE_original": "Segmentation_T1CE_original",
                #"T2_original": "Segmentation_T2_original",
                #"FLAIR_original": "Segmentation_FLAIR_original",
                "T1_original": "Segmentation_original",
                "T2_original": "Segmentation_original",
                "T1CE_original": "Segmentation_original",
                "FLAIR_original": "Segmentation_original",
                #"T1CE_preprocessed": "Segmentation_Preprocessed",
                #"T1_preprocessed": "Segmentation_Preprocessed",
                #"T2_preprocessed": "Segmentation_Preprocessed",
                #"FLAIR_preprocessed": "Segmentation_Preprocessed"
            }
    
            # Trova il volume attivo
            activeVolume = None
            for volumeName in volume_mask_mapping.keys():
                volumeNode = slicer.util.getNode(volumeName)
                if volumeNode and volumeNode.GetDisplayNode().GetVisibility():
                    activeVolume = volumeNode
                    break
    
            if not activeVolume:
                qt.QMessageBox.warning(None, "Errore", "Nessun volume attivo trovato.")
                return
    
            # Trova la segmentazione corrispondente
            segmentationName = volume_mask_mapping.get(activeVolume.GetName(), None)
            if not segmentationName:
                qt.QMessageBox.warning(None, "Errore", f"Nessuna segmentazione trovata per {activeVolume.GetName()}.")
                return
    
            activeSegmentation = slicer.util.getNode(segmentationName)
            if not activeSegmentation:
                qt.QMessageBox.warning(None, "Errore", f"Segmentazione {segmentationName} non trovata.")
                return
    
            # Assicura che solo questa segmentazione sia visibile
            for segNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                segNode.GetDisplayNode().SetVisibility(segNode == activeSegmentation)
    
            # Aggiorna il viewer con il volume e la segmentazione attiva
            #self.updateSliceViewerLayers({activeVolume.GetName(): activeVolume}, activeSegmentation)
            self.updateSliceViewerLayers(activeSegmentation)
            
        except slicer.util.MRMLNodeNotFoundException as e:
            qt.QMessageBox.warning(None, "Errore", str(e))
            return
    
        slicer.app.processEvents()
        self.forceSliceRealignment()

 
    
    
    def forceSliceRealignment(self):
        # Lista di tutti i volumi (originali e preprocessati)
        all_volumes = [
            "T1_original", "T1CE_original", "T2_original", "FLAIR_original",
            #"T1CE_Preprocessed", "T1_Preprocessed", "T2_Preprocessed", "FLAIR_Preprocessed"
        ]
    
        activeVolume = None
    
        # Trova il volume attivo (quello visibile)
        for volumeName in all_volumes:
            volumeNode = slicer.util.getNode(volumeName)
            if volumeNode and volumeNode.GetDisplayNode().GetVisibility():
                activeVolume = volumeNode
                break
    
        if not activeVolume:
            return
    
        view_orientation_map = {
            'Red': 'Axial',
            'Green': 'Coronal',
            'Yellow': 'Sagittal'
        }
    
        for sliceViewName, orientation in view_orientation_map.items():
            sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
            sliceLogic = sliceWidget.sliceLogic()
            sliceNode = sliceLogic.GetSliceNode()
    
            if sliceNode.GetOrientationString() in ["Axial", "Sagittal", "Coronal"]:
                sliceNode.SetOrientation(orientation)
                sliceNode.RotateToVolumePlane(activeVolume)
                sliceLogic.FitSliceToAll()
    
        slicer.app.processEvents()




    
    
    def runInference(self):
        if not self.models:
            qt.QMessageBox.warning(None, "Error", "No models loaded. Load models before running inference.")
            return
    
        progressDialog = qt.QProgressDialog("Please wait ...", None, 0, 0)  # Barra indeterminata
        progressDialog.setWindowTitle("Caricamento in corso")
        progressDialog.setCancelButton(None)
        progressDialog.setMinimumWidth(300) 
        progressDialog.setMinimumHeight(50)  
        progressDialog.show()
        slicer.app.processEvents()
        
        referenceVolume_T1 = slicer.util.getNode("T1_original")
        referenceVolume_T2 = slicer.util.getNode("T2_original")
        referenceVolume_T1CE = slicer.util.getNode("T1CE_original")
        referenceVolume_FLAIR = slicer.util.getNode("FLAIR_original")
        
        
    
        sequence_names = ["T1", "T2", "T1CE", "FLAIR"]
    
        def get_slicer_volume_as_numpy(volume_name):
            volume_node = slicer.util.getNode(volume_name)
            image_data = slicer.util.arrayFromVolume(volume_node)
            return np.moveaxis(image_data, 0, -1), volume_node  # Riordina gli assi
    
        image_dict = {}
        for seq in sequence_names:
            image_numpy, _ = get_slicer_volume_as_numpy(seq)
            image_dict[seq] = np.expand_dims(image_numpy, axis=0)
    
        data_list = [{"T1": image_dict["T1"], "T2": image_dict["T2"],
                      "T1CE": image_dict["T1CE"], "FLAIR": image_dict["FLAIR"]}]
        dataset = Dataset(data=data_list, transform=self.transforms)
        dataloader = DataLoader(dataset, batch_size=1)
    
        steps_per_model = 50
        total_steps = len(self.models) * steps_per_model + 40
        current_step = 0
    
        def update_progress_bar(step_increment):
            nonlocal current_step
            if progressDialog.minimum == 0 and progressDialog.maximum == 0:
                progressDialog.setRange(0, 100)  # Passa alla barra determinata
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
                    update_progress_bar(200 // len(self.models))
    
                all_predictions = torch.stack(all_predictions)
                final_prediction = torch.mode(all_predictions, dim=0)[0]
                update_progress_bar(10)
    
                preprocessed_volumes = {}
                for key in ["T1CE", "T1", "T2", "FLAIR"]:
                    referenceVolume = slicer.util.getNode(key)
                    preprocessed_array = batch[key].squeeze()
                    preprocessed_array = np.transpose(preprocessed_array, (2, 0, 1))
    
                    #newVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{key}_Preprocessed")
                    #slicer.util.updateVolumeFromArray(newVolume, preprocessed_array)
                    #newVolume.SetOrigin(referenceVolume.GetOrigin())
                    #newVolume.SetSpacing(referenceVolume.GetSpacing())
                    #directionMatrix = vtk.vtkMatrix4x4()
                    #referenceVolume.GetIJKToRASDirectionMatrix(directionMatrix)
                    #newVolume.SetIJKToRASDirectionMatrix(directionMatrix)
                    #preprocessed_volumes[key] = newVolume
    
                    update_progress_bar(5)
                
                print(final_prediction.shape)
                
                dims_T1 = referenceVolume_T1.GetImageData().GetDimensions()
                #resized_mask_T1 = F.interpolate(final_prediction, size=(dims_T1), mode='nearest')
                #mask_array_T1 = resized_mask_T1.cpu().numpy().squeeze()
                resized_mask_T1 = F.interpolate(final_prediction, size=(dims_T1), mode='trilinear', align_corners=False)
                resized_mask_T1 = (resized_mask_T1 > 0.5).float()
                mask_array_T1 = resized_mask_T1.cpu().numpy().squeeze()
                
               
        
                
                dims_T1CE = referenceVolume_T1CE.GetImageData().GetDimensions()
                #resized_mask_T1CE = F.interpolate(final_prediction, size=(dims_T1CE), mode='nearest')
                #mask_array_T1CE = resized_mask_T1CE.cpu().numpy().squeeze()
                resized_mask_T1CE = F.interpolate(final_prediction, size=(dims_T1CE), mode='trilinear', align_corners=False)
                resized_mask_T1CE = (resized_mask_T1CE > 0.5).float()
                mask_array_T1CE = resized_mask_T1CE.cpu().numpy().squeeze()
                
                dims_T2 = referenceVolume_T2.GetImageData().GetDimensions()
                #resized_mask_T2 = F.interpolate(final_prediction, size=(dims_T2), mode='nearest')
                #mask_array_T2 = resized_mask_T2.cpu().numpy().squeeze()
                resized_mask_T2 = F.interpolate(final_prediction, size=(dims_T2), mode='trilinear', align_corners=False)
                resized_mask_T2 = (resized_mask_T1CE > 0.5).float()
                mask_array_T2 = resized_mask_T2.cpu().numpy().squeeze()
                
                dims_FLAIR = referenceVolume_FLAIR.GetImageData().GetDimensions()
                #resized_mask_FLAIR = F.interpolate(final_prediction, size=(dims_FLAIR), mode='nearest')
                #mask_array_FLAIR = resized_mask_FLAIR.cpu().numpy().squeeze()
                resized_mask_FLAIR = F.interpolate(final_prediction, size=(dims_FLAIR), mode='trilinear', align_corners=False)
                resized_mask_FLAIR = (resized_mask_FLAIR > 0.5).float()
                mask_array_FLAIR = resized_mask_FLAIR.cpu().numpy().squeeze()
                
                mask_redim = final_prediction.cpu().numpy().squeeze()
                
                
                
                self.createSegmentationNode(mask_array_T1, mask_array_T1CE, mask_array_T2, mask_array_FLAIR,mask_redim) #preprocessed_volumes)
                update_progress_bar(10)
    
        progressDialog.setValue(100)
        slicer.app.processEvents()
        
        
    
    


    
  
    
    
    def createSegmentationNode(self, mask_array_T1, mask_array_T1CE, mask_array_T2, mask_array_FLAIR, mask_redim):#, preprocessed_volumes):
        
        
            
        
            
       
        volume_mask_mapping = {
            #"T1CE_Preprocessed": mask_redim,
            #"T1CE_original": mask_array_T1CE,
            #"Preprocessed": mask_redim,
            "original": mask_array_T1,
            #"T2_original": mask_array_T2,
            #"FLAIR_original": mask_array_FLAIR,
            #"T1_original": mask_array_T1
             
        }
    
        segment_labels = ["NET", "ED", "ET"]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Rosso, Verde, Blu
        active_segmentation_node = None
        
        for volume_name, mask_array in volume_mask_mapping.items():
            #referenceVolume = slicer.util.getNode(volume_name)  
            referenceVolume = slicer.util.getNode(f"T1_{volume_name}")
            
           
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"Segmentation_{volume_name}")
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolume)
            segmentationNode.CreateDefaultDisplayNodes()
            slicer.app.processEvents()
    
            
            mask_array = mask_array.astype(np.uint8)
            print(f" Maschera per {volume_name}")
            print("Shape maschera:", mask_array.shape)
            print("Valori unici nella maschera:", np.unique(mask_array))
    
            for i, label in enumerate(segment_labels):
                segmentId = segmentationNode.GetSegmentation().AddEmptySegment(label)
                mask = mask_array[i].transpose(2, 0, 1)  
                
                print(f"Segmento {label} - Shape: {mask.shape}")
                slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentationNode, segmentId, referenceVolume)
                segmentationNode.GetSegmentation().GetSegment(segmentId).SetColor(colors[i])
    
            
            segmentationDisplayNode = segmentationNode.GetDisplayNode()
            segmentationDisplayNode.SetVisibility2DFill(True)
            segmentationDisplayNode.SetVisibility2DOutline(True)
            segmentationDisplayNode.SetVisibility3D(True)
            
            if active_segmentation_node:
                active_segmentation_node.GetDisplayNode().SetVisibility(False)
            
            active_segmentation_node = segmentationNode  
            
            
            
            #self.updateSliceViewerLayers(preprocessed_volumes, segmentationNode)
            self.updateSliceViewerLayers(segmentationNode)
    
        slicer.app.processEvents()


    
    
    #def updateSliceViewerLayers(self, preprocessed_volumes=None, segmentationNode=None):
    def updateSliceViewerLayers(self, segmentationNode=None):
        #if preprocessed_volumes is None:
            #try:
                #preprocessed_volumes = {key: slicer.util.getNode(f'Preprocessed_{key}') for key in ["T1CE", "T1", "T2", "FLAIR"]}
            #except slicer.util.MRMLNodeNotFoundException as e:
                #qt.QMessageBox.warning(None, "Error", str(e))
                #return
        if segmentationNode is None:
            try:
                segmentationNode = slicer.util.getNode('Mask_Segmentation')
            except slicer.util.MRMLNodeNotFoundException as e:
                qt.QMessageBox.warning(None, "Error", str(e))
                return
    
        
        
        volumeNode = slicer.util.getNode("T1_original")
    
        
        view_orientation_map = {
            'Red': 'Axial',
            'Green': 'Coronal',
            'Yellow': 'Sagittal'
        }
    
        for sliceViewName, defaultOrientation in view_orientation_map.items():
            sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
            sliceLogic = sliceWidget.sliceLogic()
            sliceCompositeNode = sliceLogic.GetSliceCompositeNode()
            sliceNode = sliceLogic.GetSliceNode()
    
           
            sliceCompositeNode.SetBackgroundVolumeID(volumeNode.GetID())
            sliceCompositeNode.SetForegroundVolumeID(segmentationNode.GetID())
            sliceCompositeNode.SetForegroundOpacity(0.5)
    
         
            sliceNode.SetOrientation(defaultOrientation)
            sliceNode.RotateToVolumePlane(volumeNode)
            sliceLogic.FitSliceToAll()
    
    
            def createOrientationCallback(sliceNode, sliceLogic, volumeNode, defaultOrientation):
                def onOrientationChanged(caller, event):
                    currentOrientation = sliceNode.GetOrientationString()
                    if currentOrientation in ["Axial", "Sagittal", "Coronal"]:
                        sliceNode.RotateToVolumePlane(volumeNode)
                        sliceLogic.FitSliceToAll()
                return onOrientationChanged
    
    
            observerTag = getattr(sliceNode, "_orientationObserver", None)
            if observerTag is not None:
                sliceNode.RemoveObserver(observerTag)
    
            observerTag = sliceNode.AddObserver(vtk.vtkCommand.ModifiedEvent, createOrientationCallback(sliceNode, sliceLogic, volumeNode, defaultOrientation))
            setattr(sliceNode, "_orientationObserver", observerTag)
    
        slicer.app.processEvents() 

   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
