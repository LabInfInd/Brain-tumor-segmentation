# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:27:14 2025

@author: Utente
"""

import os
import re
import glob
import time
import platform

import torch
import torch.nn.functional as F
import numpy as np

from monai.transforms import (
    Compose,
    EnsureTyped,
    Orientationd,
    Resized,
    NormalizeIntensityd,
    Activations,
    AsDiscrete,
)
from monai.data import DataLoader, Dataset, decollate_batch

from slicer.ScriptedLoadableModule import *
import slicer
import vtk, qt, ctk

from DICOMLib import DICOMUtils
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

        self.loadDicomFolderButton = qt.QPushButton("Load DICOM folder → convert to NIfTI")
        self.layout.addWidget(self.loadDicomFolderButton)
        self.loadDicomFolderButton.connect('clicked(bool)', self.onLoadDicomFolder)

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
        self.timer.start(1000)

        self.confirmSelectionButton = qt.QPushButton("Confirm Selection")
        self.layout.addWidget(self.confirmSelectionButton)
        self.confirmSelectionButton.connect('clicked(bool)', self.onConfirmSelection)

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

    def onLoadDicomFolder(self):
        dicomDir = qt.QFileDialog.getExistingDirectory(self.parent, "Select DICOM folder")
        if not dicomDir:
            return

        try:
            outDir, volumeNodes = self.logic.loadDicomFolderAndExportAllToNifti(dicomDir)
            self.updateComboBoxes()

            qt.QMessageBox.information(
                self.parent,
                "Done",
                f"Caricati {len(volumeNodes)} volumi e salvati in NIfTI.\n\nOutput:\n{outDir}"
            )
        except Exception as e:
            qt.QMessageBox.warning(self.parent, "DICOM→NIfTI Error", str(e))

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

        self.sceneObservers.append(observer1)
        self.sceneObservers.append(observer2)

    def removeSceneObservers(self):
        for observer in self.sceneObservers:
            slicer.mrmlScene.RemoveObserver(observer)
        self.sceneObservers = []

    def onSceneUpdated(self, caller=None, event=None):
        self.updateComboBoxes()

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
                shortName = os.path.basename(filePath)          
                comboBox.addItem(shortName, node)              
    
        slicer.app.processEvents()

    def onConfirmSelection(self):
        
        for modality, comboBox in self.volumeSelectors.items():
            selectedIndex = comboBox.currentIndex  
            
            if selectedIndex >= 0:  
                selectedNode = comboBox.itemData(selectedIndex)  
                
                if selectedNode:
                    
                    
                    #if self.logic.needsNiftiExport(selectedNode):
                        #niftiPath = self.logic.ensureNiftiOnDisk(selectedNode, modality)
                        #print(f"[→NIfTI] {modality}: {niftiPath}")
                    #else:
                        #print(f"[OK] {modality} già NIfTI")
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
        #self.skullstrippingButton.setEnabled(True)
        #self.preprocessingButton.setEnabled(True)
        self.inferenceButton.setEnabled(True)
        #self.modifySegmentationButton.setEnabled(True)
        self.updateComboBoxes()

    def onInferenceButton(self):
        try:
            self.inferenceButton.setEnabled(False)
            self.confirmSelectionButton.setEnabled(False)
            self.modifySegmentationButton.setEnabled(False)

            self.logic.runFullPipeline()

            self.modifySegmentationButton.setEnabled(True)

        except Exception as e:
            qt.QMessageBox.critical(self.parent, "Pipeline Error", str(e))

        finally:
            self.inferenceButton.setEnabled(True)
            self.confirmSelectionButton.setEnabled(True)

    def onReturnToMainButton(self):
        slicer.util.selectModule('CustomInferenceModules')

    def onModifySegmentation(self):
        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        activeSegmentation = None

        for segNode in segmentationNodes:
            if segNode.GetDisplayNode() and segNode.GetDisplayNode().GetVisibility():
                activeSegmentation = segNode
                break

        if not activeSegmentation:
            qt.QMessageBox.warning(None, "Errore", "Nessuna segmentazione visibile trovata.")
            return

        self.logic.disableSliceObservers()

        slicer.util.selectModule('SegmentEditor')
        slicer.app.processEvents()

        editorWidget = slicer.modules.segmenteditor.widgetRepresentation()
        layout = editorWidget.layout()
        layout.addWidget(self.returnToMainButton)
        self.returnToMainButton.show()

        self.logic.enableSliceObservers()
        self.logic.forceSliceRealignment()

        slicer.app.processEvents()


class CustomInferenceModulesLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        self.models = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.updatingSlice = False
        self.niftiExportDir = os.path.join(slicer.app.temporaryPath, "nifti_from_dicom")

        self.transforms = Compose([
            EnsureTyped(keys=["T1", "T2", "FLAIR", "T1CE"]),
            Orientationd(keys=["T1", "T2", "FLAIR", "T1CE"], axcodes="RAS"),
            Resized(
                keys=["T1", "T2", "FLAIR", "T1CE"],
                spatial_size=(192, 192, 150),
                mode="trilinear",
                align_corners=True
            ),
            NormalizeIntensityd(keys=["T1", "T2", "FLAIR", "T1CE"], nonzero=True, channel_wise=True),
        ])

        self.post_trans = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])

    def isDicomVolume(self, volumeNode):
        uids = volumeNode.GetAttribute("DICOM.instanceUIDs")
        return (uids is not None) and (len(uids) > 0)

    def _safeName(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
        return s[:80] if s else "volume"

    def _uniquePath(self, outDir: str, base: str) -> str:
        p = os.path.join(outDir, f"{base}.nii.gz")
        if not os.path.exists(p):
            return p
        i = 2
        while True:
            p2 = os.path.join(outDir, f"{base}_{i}.nii.gz")
            if not os.path.exists(p2):
                return p2
            i += 1

    def loadDicomFolderAndExportAllToNifti(self, dicomDir, outDir=None):
        if outDir is None:
            folderName = self._safeName(os.path.basename(os.path.normpath(dicomDir)))
            outDir = os.path.join(slicer.app.temporaryPath, "nifti_from_dicom", folderName)

        os.makedirs(outDir, exist_ok=True)

        loadedNodeIDs = []

        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDir, db)
            patientUIDs = db.patients()
            if not patientUIDs:
                raise RuntimeError(f"Nessun DICOM importato da: {dicomDir}")

            for patientUID in patientUIDs:
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

        volumeNodes = []
        for nid in loadedNodeIDs:
            n = slicer.mrmlScene.GetNodeByID(nid)
            if n and n.IsA("vtkMRMLScalarVolumeNode"):
                volumeNodes.append(n)

        if not volumeNodes:
            raise RuntimeError("Nessun vtkMRMLScalarVolumeNode caricato (solo segmentazioni/altro?)")

        for v in volumeNodes:
            base = self._safeName(v.GetName())
            niftiPath = self._uniquePath(outDir, base)

            ok = slicer.util.saveNode(v, niftiPath)
            if not ok:
                raise RuntimeError(f"Impossibile salvare {v.GetName()} in {niftiPath}")

            storage = v.GetStorageNode()
            if storage is None:
                storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
                v.SetAndObserveStorageNodeID(storage.GetID())
            storage.SetFileName(niftiPath)

            print(f"[DICOM→NIfTI] {v.GetName()} -> {niftiPath}")

        return outDir, volumeNodes

    def ensureNiftiOnDisk(self, volumeNode, outBaseName, outDir=None):
        if outDir is None:
            outDir = self.niftiExportDir

        os.makedirs(outDir, exist_ok=True)
        niftiPath = os.path.join(outDir, f"{outBaseName}.nii.gz")

        ok = slicer.util.saveNode(volumeNode, niftiPath)
        if not ok:
            raise RuntimeError(f"Impossibile salvare il volume {volumeNode.GetName()} in NIfTI: {niftiPath}")

        storageNode = volumeNode.GetStorageNode()
        if storageNode is None:
            storageNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volumeNode.SetAndObserveStorageNodeID(storageNode.GetID())

        storageNode.SetFileName(niftiPath)
        return niftiPath

    def needsNiftiExport(self, volumeNode):
        storage = volumeNode.GetStorageNode()
        fn = (storage.GetFileName() if storage else "") or ""
        fn = fn.lower()
        return not (fn.endswith(".nii") or fn.endswith(".nii.gz"))

    def getOrCreateScalarVolumeNode(self, name):
        try:
            return slicer.util.getNode(name)
        except slicer.util.MRMLNodeNotFoundException:
            return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)

    def getOrCreateLinearTransformNode(self, name):
        try:
            return slicer.util.getNode(name)
        except slicer.util.MRMLNodeNotFoundException:
            return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)

    def loadModels(self, modelFolderPath):
        self.models = []

        for fold in range(5):
            model = VNetMultiEncoder(input_channels=1).to(self.device)
            modelPath = os.path.join(modelFolderPath, f"best_model_fold{fold+1}.pth")
            model.load_state_dict(torch.load(modelPath, map_location=self.device), strict=False)
            model.eval()
            self.models.append(model)

    def runFullPipeline(self):
        if not self.models:
            raise RuntimeError("No models loaded. Load models before running inference.")

        required = ["T1", "T2", "T1CE", "FLAIR"]
        for name in required:
            try:
                slicer.util.getNode(name)
            except slicer.util.MRMLNodeNotFoundException:
                raise RuntimeError(f"Missing input volume: {name}")

        progressDialog = qt.QProgressDialog("Running pipeline...", None, 0, 100)
        progressDialog.setWindowTitle("Processing")
        progressDialog.setCancelButton(None)
        progressDialog.setMinimumWidth(350)
        progressDialog.show()
        slicer.app.processEvents()

        try:
            progressDialog.setLabelText("Step 1 - Registration...")
            progressDialog.setValue(5)
            slicer.app.processEvents()
            self.runPreprocessing()

            progressDialog.setLabelText("Step 2 - Skull stripping...")
            progressDialog.setValue(35)
            slicer.app.processEvents()
            self.runSkullStripping()

            progressDialog.setLabelText("Step 3 - Running inference...")
            progressDialog.setValue(60)
            slicer.app.processEvents()
            self.runInference(progressDialog=progressDialog, start_value=60, end_value=100)

            progressDialog.setValue(100)
            slicer.app.processEvents()

        finally:
            progressDialog.close()

    def runPreprocessing(self):
        modalities = ["T1CE", "FLAIR", "T1", "T2"]
        fixedImage = slicer.util.getNode("T1CE")

        for modality in modalities:
            movingImage = slicer.util.getNode(modality)
            if movingImage is None:
                print(f"Il volume {modality} non è stato trovato, salto...")
                continue

            transformNode = self.getOrCreateLinearTransformNode(f"{modality}_transform")
            outputVolumeNode = self.getOrCreateScalarVolumeNode(f"{modality}_original")

            parameters = {
                "fixedVolume": fixedImage.GetID(),
                "movingVolume": movingImage.GetID(),
                "linearTransform": transformNode.GetID(),
                "interpolationMode": "Linear",
                "initializeTransformMode": "useMomentsAlign",
                "outputVolume": outputVolumeNode.GetID(),
                "transformType": "Rigid",
                "numberOfSamples": 0,
                "useRigid": True,
                "useComposite": False,
                "useBSpline": False,
                "useAffine": False,
                "useRigidScale": False,
                "useRigidScaleSkew": False,
                "useSyN": False,
                "useScaleVersor3D": False,
                "useScaleSkewVersor3D": False,
                "splineGridSize": [14, 10, 12],
                "maskProcessingMode": "NOMASK",
                "cropOutput": False,
                "histogramMatch": False,
                "useROIBSpline": False,
                "costMetric": "MMI",
                "medianFilterSize": "0,0,0",
                "removeIntensityOutliers": 0.0,
                "outputVolumePixelType": "float",
                "backgroundFillValue": 0.0,
                "scaleOutputValues": False,
                "numberOfIterations": 1500,
                "maximumStepLength": 0.05,
                "minimumStepLength": 0.001,
                "relaxationFactor": 0.5,
                "translationScale": 1000.0,
                "reproportionScale": 1.0,
                "skewScale": 1.0,
                "maxBSplineDisplacement": 0.0,
                "fixedVolumeTimeIndex": 0,
                "movingVolumeTimeIndex": 0,
                "fixedImageTimeIndex": 0,
                "movingImageTimeIndex": 0,
                "numberOfHistogramBins": 50,
                "numberOfMatchPoints": 10,
                "maskInferiorCutOffFromCenter": 1000.0,
                "ROIAutoDilateSize": 0.0,
                "ROIAutoClosingSize": 9.0,
                "strippedOutputTransform": "",
                "passOutputTransformToBSpline": False,
                "writeOutputTransformsInSinglePrecision": False,
                "failureExitCode": -1,
                "writeTransformOnFailure": False,
                "numberOfThreads": -1,
                "debugLevel": 0,
                "samplingStrategy": "Random",
                "maximumNumberOfCorrections": 25,
                "maximumNumberOfEvaluations": 900,
                "costFunctionConvergenceFactor": 20000000000000.0,
                "projectedGradientTolerance": 0.0,
            }

            cliNode = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)
            status = cliNode.GetStatusString() if cliNode else "Unknown"
            print(f"[Preprocessing] {modality}: {status}")

        print("Volumi registrati / preprocessing completato")

    def runSkullStripping(self):
        modalities = ["T1_original", "T2_original", "T1CE_original", "FLAIR_original"]
        new_modalities = ["T1", "T2", "T1CE", "FLAIR"]

        for modality, modality_new in zip(modalities, new_modalities):
            imageNode = slicer.util.getNode(modality)
            patientOutputVolume = slicer.util.getNode(modality_new)

            if imageNode is None:
                print(f"Il volume {modality} non è stato trovato, salto...")
                continue

            maskVolumeNode = self.getOrCreateScalarVolumeNode(f"{modality}_brainmask")

            parameters = {
                "patientVolume": imageNode.GetID(),
                "patientOutputVolume": patientOutputVolume.GetID(),
                "patientMaskLabel": maskVolumeNode.GetID()
            }

            cliNode = slicer.cli.runSync(slicer.modules.swissskullstripper, None, parameters)
            status = cliNode.GetStatusString() if cliNode else "Unknown"
            print(f"[Skull stripping] {modality}: {status}")

        print("Skull Stripping eseguito")

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
            volume_mask_mapping = {
                "T1_original": "Segmentation_original",
                "T2_original": "Segmentation_original",
                "T1CE_original": "Segmentation_original",
                "FLAIR_original": "Segmentation_original",
            }

            activeVolume = None
            for volumeName in volume_mask_mapping.keys():
                volumeNode = slicer.util.getNode(volumeName)
                if volumeNode and volumeNode.GetDisplayNode() and volumeNode.GetDisplayNode().GetVisibility():
                    activeVolume = volumeNode
                    break

            if not activeVolume:
                qt.QMessageBox.warning(None, "Errore", "Nessun volume attivo trovato.")
                return

            segmentationName = volume_mask_mapping.get(activeVolume.GetName(), None)
            if not segmentationName:
                qt.QMessageBox.warning(None, "Errore", f"Nessuna segmentazione trovata per {activeVolume.GetName()}.")
                return

            activeSegmentation = slicer.util.getNode(segmentationName)
            if not activeSegmentation:
                qt.QMessageBox.warning(None, "Errore", f"Segmentazione {segmentationName} non trovata.")
                return

            for segNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                if segNode.GetDisplayNode():
                    segNode.GetDisplayNode().SetVisibility(segNode == activeSegmentation)

            self.updateSliceViewerLayers(activeSegmentation)

        except slicer.util.MRMLNodeNotFoundException as e:
            qt.QMessageBox.warning(None, "Errore", str(e))
            return

        slicer.app.processEvents()
        self.forceSliceRealignment()

    def forceSliceRealignment(self):
        all_volumes = [
            "T1_original", "T1CE_original", "T2_original", "FLAIR_original",
        ]

        activeVolume = None
        for volumeName in all_volumes:
            volumeNode = slicer.util.getNode(volumeName)
            if volumeNode and volumeNode.GetDisplayNode() and volumeNode.GetDisplayNode().GetVisibility():
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

    def runInference(self, progressDialog=None, start_value=0, end_value=100):
        if not self.models:
            qt.QMessageBox.warning(None, "Error", "No models loaded. Load models before running inference.")
            return

        owns_dialog = False
        if progressDialog is None:
            progressDialog = qt.QProgressDialog("Please wait ...", None, 0, 100)
            progressDialog.setWindowTitle("Inference")
            progressDialog.setCancelButton(None)
            progressDialog.setMinimumWidth(300)
            progressDialog.show()
            owns_dialog = True
            slicer.app.processEvents()

        referenceVolume_T1 = slicer.util.getNode("T1_original")
        referenceVolume_T2 = slicer.util.getNode("T2_original")
        referenceVolume_T1CE = slicer.util.getNode("T1CE_original")
        referenceVolume_FLAIR = slicer.util.getNode("FLAIR_original")

        sequence_names = ["T1", "T2", "T1CE", "FLAIR"]

        def set_progress(pct, label=None):
            value = int(start_value + (end_value - start_value) * pct / 100.0)
            progressDialog.setValue(value)
            if label:
                progressDialog.setLabelText(label)
            slicer.app.processEvents()

        def get_slicer_volume_as_numpy(volume_name):
            volume_node = slicer.util.getNode(volume_name)
            image_data = slicer.util.arrayFromVolume(volume_node)
            return np.moveaxis(image_data, 0, -1), volume_node

        image_dict = {}
        for seq in sequence_names:
            image_numpy, _ = get_slicer_volume_as_numpy(seq)
            image_dict[seq] = np.expand_dims(image_numpy, axis=0)

        data_list = [{
            "T1": image_dict["T1"],
            "T2": image_dict["T2"],
            "T1CE": image_dict["T1CE"],
            "FLAIR": image_dict["FLAIR"]
        }]

        dataset = Dataset(data=data_list, transform=self.transforms)
        dataloader = DataLoader(dataset, batch_size=1)

        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                set_progress(5, "Preparing tensors...")

                inputs = {
                    key: batch[key].to(self.device)
                    for key in ["T1CE", "T1", "T2", "FLAIR"]
                }

                all_predictions = []
                n_models = len(self.models)

                for j, model in enumerate(self.models):
                    test_output = model(inputs["T1CE"], inputs["T1"], inputs["T2"], inputs["FLAIR"])
                    test_output = [self.post_trans(x) for x in decollate_batch(test_output)]
                    all_predictions.append(torch.stack(test_output))

                    #pct = 10 + int((j + 1) / n_models * 50)
                    #set_progress(pct, f"Running model {j+1}/{n_models}...")

                all_predictions = torch.stack(all_predictions)
                final_prediction = torch.mode(all_predictions, dim=0)[0]

                set_progress(70, "Resizing masks...")

                dims_T1 = referenceVolume_T1.GetImageData().GetDimensions()
                resized_mask_T1 = F.interpolate(final_prediction, size=dims_T1, mode='trilinear', align_corners=True)
                resized_mask_T1 = (resized_mask_T1 > 0.5).float()
                mask_array_T1 = resized_mask_T1.cpu().numpy().squeeze()

                dims_T1CE = referenceVolume_T1CE.GetImageData().GetDimensions()
                resized_mask_T1CE = F.interpolate(final_prediction, size=dims_T1CE, mode='trilinear', align_corners=True)
                resized_mask_T1CE = (resized_mask_T1CE > 0.5).float()
                mask_array_T1CE = resized_mask_T1CE.cpu().numpy().squeeze()

                dims_T2 = referenceVolume_T2.GetImageData().GetDimensions()
                resized_mask_T2 = F.interpolate(final_prediction, size=dims_T2, mode='trilinear', align_corners=True)
                resized_mask_T2 = (resized_mask_T2 > 0.5).float()
                mask_array_T2 = resized_mask_T2.cpu().numpy().squeeze()

                dims_FLAIR = referenceVolume_FLAIR.GetImageData().GetDimensions()
                resized_mask_FLAIR = F.interpolate(final_prediction, size=dims_FLAIR, mode='trilinear', align_corners=True)
                resized_mask_FLAIR = (resized_mask_FLAIR > 0.5).float()
                mask_array_FLAIR = resized_mask_FLAIR.cpu().numpy().squeeze()

                mask_redim = final_prediction.cpu().numpy().squeeze()

                set_progress(90, "Creating segmentation nodes...")
                self.createSegmentationNode(
                    mask_array_T1,
                    mask_array_T1CE,
                    mask_array_T2,
                    mask_array_FLAIR,
                    mask_redim
                )

                set_progress(100, "Done")

        if owns_dialog:
            progressDialog.close()

    def createSegmentationNode(self, mask_array_T1, mask_array_T1CE, mask_array_T2, mask_array_FLAIR, mask_redim):
        volume_mask_mapping = {
            "original": mask_array_T1CE,
        }

        segment_labels = ["NECROSI", "FLAIR", "CONTRASTO"]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        active_segmentation_node = None

        for volume_name, mask_array in volume_mask_mapping.items():
            referenceVolume = slicer.util.getNode(f"T1CE_{volume_name}")

            segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode",
                f"Segmentation_{volume_name}"
            )
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolume)
            segmentationNode.CreateDefaultDisplayNodes()
            slicer.app.processEvents()

            mask_array = mask_array.astype(np.uint8)
            print(f"Maschera per {volume_name}")
            print("Shape maschera:", mask_array.shape)
            print("Valori unici nella maschera:", np.unique(mask_array))

            for i, label in enumerate(segment_labels):
                segmentId = segmentationNode.GetSegmentation().AddEmptySegment()
                segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
                segment.SetName(label)

                mask = mask_array[i].transpose(2, 0, 1)
                print(f"Segmento {label} - Shape: {mask.shape}")

                slicer.util.updateSegmentBinaryLabelmapFromArray(
                    mask, segmentationNode, segmentId, referenceVolume
                )
                segmentationNode.GetSegmentation().GetSegment(segmentId).SetColor(colors[i])

            segmentationDisplayNode = segmentationNode.GetDisplayNode()
            segmentationDisplayNode.SetVisibility2DFill(True)
            segmentationDisplayNode.SetVisibility2DOutline(True)
            segmentationDisplayNode.SetVisibility3D(True)

            if active_segmentation_node and active_segmentation_node.GetDisplayNode():
                active_segmentation_node.GetDisplayNode().SetVisibility(False)

            active_segmentation_node = segmentationNode
            self.updateSliceViewerLayers(segmentationNode)

        slicer.app.processEvents()

    def updateSliceViewerLayers(self, segmentationNode=None):
        if segmentationNode is None:
            try:
                segmentationNode = slicer.util.getNode('Mask_Segmentation')
            except slicer.util.MRMLNodeNotFoundException as e:
                qt.QMessageBox.warning(None, "Error", str(e))
                return

        volumeNode = slicer.util.getNode("T1CE_original")

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

            observerTag = sliceNode.AddObserver(
                vtk.vtkCommand.ModifiedEvent,
                createOrientationCallback(sliceNode, sliceLogic, volumeNode, defaultOrientation)
            )
            setattr(sliceNode, "_orientationObserver", observerTag)

        slicer.app.processEvents() 