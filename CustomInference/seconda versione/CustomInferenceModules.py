import os
import re
import glob
import time
import platform
import csv
import traceback
import tempfile
import shutil
from contextlib import contextmanager

try:
    import psutil
except ImportError:
    psutil = None

import torch
import torch.nn.functional as F
import numpy as np

try:
    import nibabel as nib
    from nibabel.processing import resample_from_to, resample_to_output
except ImportError:
    nib = None
    resample_from_to = None
    resample_to_output = None

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    Invertd,
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
        self.radiomicsVolumeSelectors = {}
        self.radiomicsSegmentationSelector = None
        self.volumeNodes = []
        self.segmentationNodes = []

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

        titleLabel = qt.QLabel("Radiomics Feature Extraction")
        titleLabel.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px; margin-bottom: 5px;")
        self.layout.addWidget(titleLabel)

        

        originalsLabel = qt.QLabel(
            "Fixed radiomics inputs: T1_original | T2_original | "
            "T1CE_original | FLAIR_original"
        )
        originalsLabel.setWordWrap(True)
        self.layout.addWidget(originalsLabel)

        rowLayout = qt.QHBoxLayout()
        rowLayout.setSpacing(10)

        label = qt.QLabel("Segmentation:")
        rowLayout.addWidget(label)

        self.radiomicsSegmentationSelector = qt.QComboBox()
        self.radiomicsSegmentationSelector.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        rowLayout.addWidget(self.radiomicsSegmentationSelector)
        self.layout.addLayout(rowLayout)

        self.extractRadiomicsButton = qt.QPushButton("Extract Radiomics Features")
        self.layout.addWidget(self.extractRadiomicsButton)
        self.extractRadiomicsButton.setEnabled(True)
        self.extractRadiomicsButton.connect('clicked(bool)', self.onExtractRadiomicsFeatures)

        self.updateComboBoxes()

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

        for comboBox in self.radiomicsVolumeSelectors.values():
            self.populateVolumeSelector(comboBox)

        if self.radiomicsSegmentationSelector is not None:
            self.populateSegmentationSelector(self.radiomicsSegmentationSelector)

    def checkForNewVolumes(self):
        currentVolumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        currentSegmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")

        if (
            len(currentVolumeNodes) != len(self.volumeNodes)
            or len(currentSegmentationNodes) != len(self.segmentationNodes)
        ):
            self.updateComboBoxes()

        self.volumeNodes = currentVolumeNodes
        self.segmentationNodes = currentSegmentationNodes

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
        currentNode = comboBox.itemData(comboBox.currentIndex) if comboBox.currentIndex >= 0 else None
        comboBox.clear()
        volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        selectedIndex = -1
    
        for idx, node in enumerate(volumeNodes):
            storageNode = node.GetStorageNode()
            if storageNode and storageNode.GetFileName():
                filePath = storageNode.GetFileName()
                shortName = os.path.basename(filePath)
            else:
                shortName = node.GetName()

            comboBox.addItem(shortName, node)

            if currentNode and node.GetID() == currentNode.GetID():
                selectedIndex = idx

        if selectedIndex >= 0:
            comboBox.setCurrentIndex(selectedIndex)
    
        slicer.app.processEvents()

    def populateSegmentationSelector(self, comboBox):
        currentNode = comboBox.itemData(comboBox.currentIndex) if comboBox.currentIndex >= 0 else None
        comboBox.clear()
        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        selectedIndex = -1

        for idx, node in enumerate(segmentationNodes):
            comboBox.addItem(node.GetName(), node)

            if currentNode and node.GetID() == currentNode.GetID():
                selectedIndex = idx

        if selectedIndex >= 0:
            comboBox.setCurrentIndex(selectedIndex)

        slicer.app.processEvents()

    def onConfirmSelection(self):
        selectedNodes = {}

        for modality, comboBox in self.volumeSelectors.items():
            selectedIndex = comboBox.currentIndex
            if selectedIndex < 0:
                qt.QMessageBox.warning(
                    self.parent, "Selection Error",
                    f"Please select a volume for {modality}."
                )
                return

            selectedNode = comboBox.itemData(selectedIndex)
            if selectedNode is None or selectedNode.GetImageData() is None:
                qt.QMessageBox.warning(
                    self.parent, "Selection Error",
                    f"Invalid volume selected for {modality}."
                )
                return

            selectedNodes[modality] = selectedNode

        selectedIDs = [node.GetID() for node in selectedNodes.values()]
        if len(set(selectedIDs)) != 4:
            qt.QMessageBox.warning(
                self.parent, "Selection Error",
                "The same volume was selected for more than one modality."
            )
            return


        self.inputVolumes = selectedNodes
        self.logic.inputVolumeNodeIDs = {
            modality: node.GetID()
            for modality, node in selectedNodes.items()
        }

        summary = "\n".join(
            f"{modality}: {node.GetName()}"
            for modality, node in selectedNodes.items()
        )
        print("[Input selection]\n" + summary)

        qt.QMessageBox.information(
            self.parent,
            "Volumes Selected",
            "Volumes selected using exact MRML node IDs:\n\n" + summary,
        )
        self.inferenceButton.setEnabled(True)

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

    def onExtractRadiomicsFeatures(self):
        try:
            self.extractRadiomicsButton.setEnabled(False)

            segIndex = self.radiomicsSegmentationSelector.currentIndex
            if segIndex < 0:
                qt.QMessageBox.warning(
                    self.parent,
                    "Radiomics Selection Error",
                    "Please select the final/refined segmentation."
                )
                return

            segmentationNode = self.radiomicsSegmentationSelector.itemData(segIndex)
            if not segmentationNode:
                qt.QMessageBox.warning(
                    self.parent,
                    "Radiomics Selection Error",
                    "Invalid segmentation selected."
                )
                return

            # Images are intentionally NOT user-selectable: the standardized
            # radiomics pipeline always uses the registered *_original volumes.
            self.logic.extractRadiomicsFeatures(
                segmentationNode=segmentationNode
            )

        except Exception as e:
            qt.QMessageBox.critical(self.parent, "Radiomics Error", str(e))

        finally:
            self.extractRadiomicsButton.setEnabled(True)

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

        self.inputVolumeNodeIDs = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.updatingSlice = False
        self.niftiExportDir = os.path.join(slicer.app.temporaryPath, "nifti_from_dicom")
        desktopPath = os.path.join(os.path.expanduser("~"), "Desktop")


        if not os.path.isdir(desktopPath):
            scrivaniaPath = os.path.join(os.path.expanduser("~"), "Scrivania")
            if os.path.isdir(scrivaniaPath):
                desktopPath = scrivaniaPath


        os.makedirs(desktopPath, exist_ok=True)


        self.radiomicsOutputDir = os.path.join(desktopPath, "Radiomics_Features")
        os.makedirs(self.radiomicsOutputDir, exist_ok=True)

        # Runtime/QC logging used to document deployability:
        # hardware requirements, CPU/GPU execution, processing time,
        # memory use, pipeline success/failure, and cumulative failure rate.
        self.runtimeQCOutputDir = os.path.join(desktopPath, "BRIDGE_Runtime_QC")
        os.makedirs(self.runtimeQCOutputDir, exist_ok=True)
        self.runtimeQCPath = os.path.join(self.runtimeQCOutputDir, "bridge_runtime_qc.csv")
        self.currentRunID = None
        self.currentRunCSVRow = {}
        self.pipelineTimings = {}
        self.pipelineFailures = []
        self.runtimeMetrics = {}
        self.modelCheckpointPaths = []
        self.modelDiskSizeMB = 0.0


        self.inferenceKeys = ["T1", "T2", "FLAIR", "T1CE"]
        self.brainMaskKey = "BRAIN_MASK"
        self.transformKeys = self.inferenceKeys + [self.brainMaskKey]
        self.modelSpacing = (1.0, 1.0, 1.0)
        self.modelSpatialSize = (192, 192, 150)
        self.cropMargin = (8, 8, 5)

        # Standard radiomics specification. Radiomics is always computed from
        # the registered *_original volumes. The user may select/refine only
        # the segmentation. A COPY of each original volume is skull-stripped
        # with Shared_brainmask BEFORE intensity normalization. Mean/std are
        # estimated only from finite, non-zero voxels inside that brain mask.
        # N4 bias-field correction is first estimated inside Shared_brainmask.
        # The N4-corrected copy is then skull-masked and z-score normalized.
        # The z-score is clipped at +/-3 SD and scaled by 100; PyRadiomics
        # internal normalization is therefore disabled. Texture features are
        # always computed after isotropic 1 mm resampling by PyRadiomics.
        self.radiomicsTargetSpacing = (1.0, 1.0, 1.0)
        self.radiomicsBinWidth = 10.0
        self.radiomicsNormalizeScale = 100.0
        self.radiomicsOutlierSigma = 3.0
        # N4 bias-field correction is applied to a temporary radiomics copy
        # BEFORE skull-masked z-score normalization. Shared_brainmask is used
        # as the N4 mask so extracranial tissues do not drive the estimated field.
        self.radiomicsUseN4BiasCorrection = True
        self.radiomicsN4ShrinkFactor = 4
        self.radiomicsN4Iterations = (50, 50, 30, 20)
        self.radiomicsN4ConvergenceThreshold = 1e-6
        # Refined tumour ROIs should remain inside the conservative shared brain
        # mask. A tiny tolerance is allowed for rasterization/border effects.
        self.radiomicsMaxROIFractionOutsideBrain = 0.001
        # Intermediate NRRD files are required by the current PyRadiomics call,
        # but they are temporary implementation details. Keep them only when
        # explicitly debugging/reproducing a single extraction.
        self.keepRadiomicsIntermediateFiles = False

       
        self.fastRegistration = True
        self.registrationIterations = 300
        self.registrationHistogramBins = 32
        self.registrationSamplingPercentage = {
            "T1": 0.02,
            "T2": 0.02,
            "FLAIR": 0.01,
        }
        # Registration QC thresholds. For a same-exam multimodal brain study,
        # spatial overlap must be high before a candidate can participate in
        # the final selection. Native/same-geometry candidates are held to an
        # even stricter coverage threshold.
        self.registrationMinCoverage = {
            "T1": 0.92,
            "T2": 0.90,
            "FLAIR": 0.90,
        }
        self.nativeGeometryMinCoverage = {
            "T1": 0.95,
            "T2": 0.93,
            "FLAIR": 0.93,
        }
        self.registrationMinNMI = {
            "T1": 0.015,
            "T2": 0.015,
            "FLAIR": 0.015,
        }
        self.nativeGeometryMinNMI = {
            "T1": 0.030,
            "T2": 0.025,
            "FLAIR": 0.025,
        }

        self.post_trans = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])

    def _formatSeconds(self, seconds):
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.2f} s"
        minutes = int(seconds // 60)
        remaining = seconds - 60 * minutes
        return f"{minutes} min {remaining:.1f} s"

    def _runtimeSetMetric(self, key, value):
        try:
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
        except Exception:
            pass
        self.runtimeMetrics[key] = value
        if hasattr(self, "currentRunCSVRow") and self.currentRunCSVRow is not None:
            self.currentRunCSVRow[key] = value

    def _runtimeMemorySnapshot(self):
        snapshot = {
            "cpu_rss_mb": "",
            "ram_used_gb": "",
            "ram_total_gb": "",
            "gpu_allocated_mb": "",
            "gpu_reserved_mb": "",
            "gpu_peak_allocated_mb": "",
            "gpu_peak_reserved_mb": "",
        }

        if psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                vm = psutil.virtual_memory()
                snapshot["cpu_rss_mb"] = process.memory_info().rss / (1024 ** 2)
                snapshot["ram_used_gb"] = vm.used / (1024 ** 3)
                snapshot["ram_total_gb"] = vm.total / (1024 ** 3)
            except Exception as exc:
                snapshot["cpu_memory_note"] = f"psutil error: {exc}"
        else:
            snapshot["cpu_memory_note"] = "psutil not installed"

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                snapshot["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
                snapshot["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
                snapshot["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
                snapshot["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
            except Exception as exc:
                snapshot["gpu_memory_note"] = f"CUDA memory error: {exc}"

        return snapshot

    def _printMemorySnapshot(self, prefix):
        snapshot = self._runtimeMemorySnapshot()
        parts = []
        for key in [
            "cpu_rss_mb",
            "ram_used_gb",
            "ram_total_gb",
            "gpu_allocated_mb",
            "gpu_reserved_mb",
            "gpu_peak_allocated_mb",
            "gpu_peak_reserved_mb",
        ]:
            value = snapshot.get(key, "")
            if value == "":
                continue
            unit = "GB" if key.endswith("_gb") else "MB"
            parts.append(f"{key}={float(value):.2f} {unit}")
        if snapshot.get("cpu_memory_note"):
            parts.append(snapshot["cpu_memory_note"])
        if snapshot.get("gpu_memory_note"):
            parts.append(snapshot["gpu_memory_note"])
        print(f"{prefix} " + ("; ".join(parts) if parts else "memory info unavailable"))

    def _countModelParameters(self, model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return total, trainable

    def _getSlicerVersion(self):
        try:
            return slicer.app.applicationVersion
        except Exception:
            return "unknown"

    def _printRuntimeEnvironment(self):
        print("\n" + "=" * 72)
        print("[Runtime QC] BRIDGE / 3D Slicer execution report")
        print("=" * 72)
        print(f"[Runtime QC] run_id={self.currentRunID}")
        print(f"[Runtime QC] timestamp_start={self.currentRunCSVRow.get('timestamp_start', '')}")
        print(f"[Runtime QC] OS={platform.platform()}")
        print(f"[Runtime QC] machine={platform.machine()}, processor={platform.processor()}")
        print(f"[Runtime QC] CPU logical cores={os.cpu_count()}")
        print(f"[Runtime QC] Python={platform.python_version()}, 3D Slicer={self._getSlicerVersion()}")
        print(f"[Runtime QC] torch={getattr(torch, '__version__', 'unknown')}")
        try:
            import monai
            print(f"[Runtime QC] MONAI={getattr(monai, '__version__', 'unknown')}")
        except Exception:
            print("[Runtime QC] MONAI version unavailable")
        print(f"[Runtime QC] CUDA available={torch.cuda.is_available()}")
        print(f"[Runtime QC] selected device={self.device}")

        self.currentRunCSVRow.update({
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_logical_cores": os.cpu_count(),
            "python_version": platform.python_version(),
            "slicer_version": self._getSlicerVersion(),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "selected_device": self.device,
            "model_count": len(self.models),
            "model_input_size": "x".join(str(v) for v in self.modelSpatialSize),
        })

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                ramTotalGB = vm.total / (1024 ** 3)
                print(f"[Runtime QC] RAM total={ramTotalGB:.2f} GB")
                self.currentRunCSVRow["ram_total_gb"] = f"{ramTotalGB:.2f}"
            except Exception as exc:
                print(f"[Runtime QC] RAM total unavailable: {exc}")
        else:
            print("[Runtime QC] psutil not installed: detailed CPU/RAM memory logging disabled")

        if torch.cuda.is_available():
            try:
                gpuName = torch.cuda.get_device_name(0)
                gpuTotalGB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(f"[Runtime QC] GPU={gpuName}, VRAM total={gpuTotalGB:.2f} GB")
                print(f"[Runtime QC] CUDA runtime={torch.version.cuda}")
                self.currentRunCSVRow["gpu_name"] = gpuName
                self.currentRunCSVRow["gpu_total_vram_gb"] = f"{gpuTotalGB:.2f}"
                self.currentRunCSVRow["cuda_runtime"] = torch.version.cuda
            except Exception as exc:
                print(f"[Runtime QC] GPU details unavailable: {exc}")
        else:
            print("[Runtime QC] GPU not available: running CPU-only inference.")
            print("[Runtime QC] This documents that the module does not require a dedicated GPU.")

        if self.models:
            try:
                totalParams, trainableParams = self._countModelParameters(self.models[0])
                totalEnsembleParams = totalParams * len(self.models)
                print(f"[Runtime QC] parameters per model={totalParams:,} "
                      f"(trainable={trainableParams:,})")
                print(f"[Runtime QC] ensemble parameters={totalEnsembleParams:,}")
                self.currentRunCSVRow["parameters_per_model"] = totalParams
                self.currentRunCSVRow["trainable_parameters_per_model"] = trainableParams
                self.currentRunCSVRow["ensemble_parameters"] = totalEnsembleParams
            except Exception as exc:
                print(f"[Runtime QC] model parameter count unavailable: {exc}")

        if getattr(self, "modelDiskSizeMB", 0.0):
            print(f"[Runtime QC] checkpoint disk size total={self.modelDiskSizeMB:.2f} MB")
            self.currentRunCSVRow["checkpoint_disk_size_total_mb"] = f"{self.modelDiskSizeMB:.2f}"

        self._printMemorySnapshot("[Runtime QC] Memory at start:")

    def _beginRuntimeQC(self, runLabel="pipeline"):
        self.currentRunID = time.strftime("%Y%m%d_%H%M%S")
        self.pipelineTimings = {}
        self.pipelineFailures = []
        self.runtimeMetrics = {}
        self._runtimeWallStart = time.perf_counter()

        self.currentRunCSVRow = {
            "run_id": self.currentRunID,
            "run_label": runLabel,
            "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "RUNNING",
            "error_stage": "",
            "error_message": "",
        }

        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        self._printRuntimeEnvironment()

    @contextmanager
    def _timedStage(self, stageName):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        print(f"\n[Runtime QC] START {stageName}")
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            elapsed = time.perf_counter() - start
            self.pipelineTimings[stageName] = elapsed
            self.pipelineFailures.append({
                "stage": stageName,
                "error": str(exc),
            })
            print(f"[Runtime QC] FAILED {stageName}: {self._formatSeconds(elapsed)}")
            print(f"[Runtime QC] ERROR {stageName}: {exc}")
            self._printMemorySnapshot(f"[Runtime QC] Memory after failed {stageName}:")
            raise
        else:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            elapsed = time.perf_counter() - start
            self.pipelineTimings[stageName] = elapsed
            print(f"[Runtime QC] END {stageName}: {self._formatSeconds(elapsed)}")
            self._printMemorySnapshot(f"[Runtime QC] Memory after {stageName}:")
            self.currentRunCSVRow[f"{stageName}_s"] = f"{elapsed:.3f}"

    def _appendRuntimeQCRow(self, row):
        os.makedirs(self.runtimeQCOutputDir, exist_ok=True)

        existingRows = []
        fieldnames = []
        if os.path.isfile(self.runtimeQCPath):
            try:
                with open(self.runtimeQCPath, "r", newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream, delimiter=";")
                    fieldnames = list(reader.fieldnames or [])
                    existingRows = list(reader)
            except Exception as exc:
                print(f"[Runtime QC] Existing CSV could not be read and will be overwritten: {exc}")

        preferred = [
            "run_id", "run_label", "timestamp_start", "timestamp_end", "status",
            "error_stage", "error_message", "total_pipeline_s",
            "registration_s", "skull_stripping_s", "inference_total_s",
            "nifti_export_s", "prepare_arrays_s", "geometry_round_trip_s",
            "tensor_creation_s", "ensemble_voting_s", "restore_original_geometry_s",
            "create_segmentation_node_s", "selected_device", "cuda_available",
            "gpu_name", "gpu_total_vram_gb", "ram_total_gb", "cpu_logical_cores",
            "parameters_per_model", "ensemble_parameters", "checkpoint_disk_size_total_mb",
            "segmentation_voxels_NCRNET", "segmentation_voxels_ED", "segmentation_voxels_ET",
            "overlap_voxels", "geometry_round_trip_dice",
        ]

        for key in preferred:
            if key not in fieldnames:
                fieldnames.append(key)
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

        with open(self.runtimeQCPath, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
            writer.writeheader()
            for existing in existingRows:
                writer.writerow(existing)
            writer.writerow(row)

        print(f"[Runtime QC] CSV updated: {self.runtimeQCPath}")
        self._printCumulativeFailureRate()

    def _printCumulativeFailureRate(self):
        if not os.path.isfile(self.runtimeQCPath):
            return
        try:
            with open(self.runtimeQCPath, "r", newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
        except Exception as exc:
            print(f"[Runtime QC] Cumulative failure rate unavailable: {exc}")
            return

        completed = [
            row for row in rows
            if row.get("status") in ("SUCCESS", "FAILED")
        ]
        total = len(completed)
        failures = sum(1 for row in completed if row.get("status") == "FAILED")
        successes = sum(1 for row in completed if row.get("status") == "SUCCESS")
        failureRate = (failures / total * 100.0) if total else 0.0
        print(
            f"[Runtime QC] cumulative runs={total}, successes={successes}, "
            f"failures={failures}, failure_rate={failureRate:.1f}%"
        )

    def _finishRuntimeQC(self, success=True, error=None):
        if not hasattr(self, "_runtimeWallStart"):
            return

        totalElapsed = time.perf_counter() - self._runtimeWallStart
        self.currentRunCSVRow["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.currentRunCSVRow["total_pipeline_s"] = f"{totalElapsed:.3f}"
        self.currentRunCSVRow["status"] = "SUCCESS" if success else "FAILED"

        for stage, seconds in self.pipelineTimings.items():
            self.currentRunCSVRow[f"{stage}_s"] = f"{seconds:.3f}"

        if not success:
            if self.pipelineFailures:
                self.currentRunCSVRow["error_stage"] = self.pipelineFailures[-1]["stage"]
                self.currentRunCSVRow["error_message"] = self.pipelineFailures[-1]["error"]
            elif error is not None:
                self.currentRunCSVRow["error_message"] = str(error)
            if error is not None:
                self.currentRunCSVRow["traceback"] = traceback.format_exc()

        finalMemory = self._runtimeMemorySnapshot()
        for key, value in finalMemory.items():
            if value != "":
                if isinstance(value, float):
                    self.currentRunCSVRow[f"final_{key}"] = f"{value:.3f}"
                else:
                    self.currentRunCSVRow[f"final_{key}"] = value

        print("\n" + "-" * 72)
        print("[Runtime QC] Pipeline summary")
        print("-" * 72)
        for stage, seconds in self.pipelineTimings.items():
            print(f"[Runtime QC] {stage}: {self._formatSeconds(seconds)}")
        print(f"[Runtime QC] total_pipeline: {self._formatSeconds(totalElapsed)}")
        print(f"[Runtime QC] status={self.currentRunCSVRow['status']}")
        self._printMemorySnapshot("[Runtime QC] Memory at end:")
        print("-" * 72 + "\n")

        self._appendRuntimeQCRow(self.currentRunCSVRow)

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
        startTotal = time.perf_counter()
        print("\n[Runtime QC] START dicom_import_export")
        try:
            if outDir is None:
                folderName = self._safeName(os.path.basename(os.path.normpath(dicomDir)))
                outDir = os.path.join(slicer.app.temporaryPath, "nifti_from_dicom", folderName)

            os.makedirs(outDir, exist_ok=True)

            loadedNodeIDs = []

            startImport = time.perf_counter()
            with DICOMUtils.TemporaryDICOMDatabase() as db:
                DICOMUtils.importDicom(dicomDir, db)
                patientUIDs = db.patients()
                if not patientUIDs:
                    raise RuntimeError(f"Nessun DICOM importato da: {dicomDir}")

                for patientUID in patientUIDs:
                    loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
            importElapsed = time.perf_counter() - startImport
            print(f"[Runtime QC] dicom_import_load: {self._formatSeconds(importElapsed)}")

            volumeNodes = []
            for nid in loadedNodeIDs:
                n = slicer.mrmlScene.GetNodeByID(nid)
                if n and n.IsA("vtkMRMLScalarVolumeNode"):
                    volumeNodes.append(n)

            if not volumeNodes:
                raise RuntimeError("Nessun vtkMRMLScalarVolumeNode caricato (solo segmentazioni/altro?)")

            startExport = time.perf_counter()
            exportedBytes = 0
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

                try:
                    exportedBytes += os.path.getsize(niftiPath)
                except Exception:
                    pass
                print(f"[DICOM→NIfTI] {v.GetName()} -> {niftiPath}")

            exportElapsed = time.perf_counter() - startExport
            totalElapsed = time.perf_counter() - startTotal
            print(f"[Runtime QC] dicom_nifti_export: {self._formatSeconds(exportElapsed)}")
            print(f"[Runtime QC] imported volumes={len(volumeNodes)}, exported size={exportedBytes / (1024 ** 2):.2f} MB")
            print(f"[Runtime QC] END dicom_import_export: {self._formatSeconds(totalElapsed)}")
            self._printMemorySnapshot("[Runtime QC] Memory after DICOM import/export:")

            return outDir, volumeNodes

        except Exception as exc:
            totalElapsed = time.perf_counter() - startTotal
            print(f"[Runtime QC] FAILED dicom_import_export: {self._formatSeconds(totalElapsed)}")
            print(f"[Runtime QC] ERROR dicom_import_export: {exc}")
            self._printMemorySnapshot("[Runtime QC] Memory after failed DICOM import/export:")
            raise

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

    def _selectedInputNode(self, modality):
        nodeID = self.inputVolumeNodeIDs.get(modality)
        node = slicer.mrmlScene.GetNodeByID(nodeID) if nodeID else None
        if node is None or node.GetImageData() is None:
            raise RuntimeError(
                f"Input {modality} non valido o non più presente nella scena. "
                "Confermare nuovamente la selezione dei volumi."
            )
        return node

    def loadModels(self, modelFolderPath):
        startTotal = time.perf_counter()
        print("\n[Runtime QC] START load_models")
        self.modelCheckpointPaths = []
        self.modelDiskSizeMB = 0.0

        loadedModels = []

        for fold in range(5):
            modelPath = os.path.join(
                modelFolderPath, f"best_model_fold{fold + 1}.pth"
            )
            if not os.path.isfile(modelPath):
                raise RuntimeError(f"Checkpoint non trovato: {modelPath}")

            foldStart = time.perf_counter()
            try:
                checkpointSizeMB = os.path.getsize(modelPath) / (1024 ** 2)
            except Exception:
                checkpointSizeMB = 0.0
            self.modelDiskSizeMB += checkpointSizeMB
            self.modelCheckpointPaths.append(modelPath)

            try:
                checkpoint = torch.load(modelPath, map_location=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Impossibile leggere il checkpoint {modelPath}: {exc}"
                ) from exc

            # Il training corrente salva direttamente model.state_dict(), ma
            # supportiamo anche i contenitori piu' comuni.
            if isinstance(checkpoint, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    candidate = checkpoint.get(key)
                    if isinstance(candidate, dict):
                        checkpoint = candidate
                        break

            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f"Formato checkpoint non supportato: {modelPath} "
                    f"({type(checkpoint).__name__})"
                )

            # Rimuove l'eventuale prefisso aggiunto da torch.nn.DataParallel.
            state = {
                (name[7:] if name.startswith("module.") else name): value
                for name, value in checkpoint.items()
            }

            model = VNetMultiEncoder(input_channels=1).to(self.device)

            try:
                incompatibility = model.load_state_dict(state, strict=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"I tensori del checkpoint non sono compatibili con "
                    f"VNetMultiEncoder per il fold {fold + 1}:\n{exc}"
                ) from exc

            if incompatibility.missing_keys:
                print(
                    f"[Model warning] Fold {fold + 1} - missing keys: "
                    f"{incompatibility.missing_keys}"
                )
            if incompatibility.unexpected_keys:
                print(
                    f"[Model warning] Fold {fold + 1} - unexpected keys: "
                    f"{incompatibility.unexpected_keys}"
                )

            model.eval()
            loadedModels.append(model)
            foldElapsed = time.perf_counter() - foldStart
            print(
                f"[Model] Fold {fold + 1}/5 caricato da: {modelPath} "
                f"({checkpointSizeMB:.2f} MB, {self._formatSeconds(foldElapsed)})"
            )

        self.models = loadedModels

        if self.models:
            try:
                totalParams, trainableParams = self._countModelParameters(self.models[0])
                print(f"[Model] Parametri per modello: total={totalParams:,}, trainable={trainableParams:,}")
                print(f"[Model] Parametri ensemble ({len(self.models)} modelli): {totalParams * len(self.models):,}")
            except Exception as exc:
                print(f"[Model] Impossibile contare i parametri: {exc}")

        totalElapsed = time.perf_counter() - startTotal
        print(f"[Model] Caricati {len(self.models)} checkpoint su {self.device}")
        print(f"[Runtime QC] checkpoint disk size total={self.modelDiskSizeMB:.2f} MB")
        print(f"[Runtime QC] END load_models: {self._formatSeconds(totalElapsed)}")
        self._printMemorySnapshot("[Runtime QC] Memory after model loading:")

    def runFullPipeline(self):
        self._beginRuntimeQC("full_pipeline")
        progressDialog = None

        try:
            if not self.models:
                raise RuntimeError("No models loaded. Load models before running inference.")

            required = ["T1", "T2", "T1CE", "FLAIR"]
            for name in required:
                self._selectedInputNode(name)

            progressDialog = qt.QProgressDialog("Running pipeline...", None, 0, 100)
            progressDialog.setWindowTitle("Processing")
            progressDialog.setCancelButton(None)
            progressDialog.setMinimumWidth(350)
            progressDialog.show()
            slicer.app.processEvents()

            with self._timedStage("registration"):
                progressDialog.setLabelText("Step 1 - Registration...")
                progressDialog.setValue(5)
                slicer.app.processEvents()
                self.runPreprocessing()

            with self._timedStage("skull_stripping"):
                progressDialog.setLabelText("Step 2 - Skull stripping...")
                progressDialog.setValue(35)
                slicer.app.processEvents()
                self.runSkullStripping()

            with self._timedStage("inference_total"):
                progressDialog.setLabelText("Step 3 - Running inference...")
                progressDialog.setValue(60)
                slicer.app.processEvents()
                self.runInference(progressDialog=progressDialog, start_value=60, end_value=100)

            progressDialog.setValue(100)
            slicer.app.processEvents()

        except Exception as exc:
            self._finishRuntimeQC(success=False, error=exc)
            raise

        else:
            self._finishRuntimeQC(success=True)

        finally:
            if progressDialog is not None:
                progressDialog.close()

    def _exactScalarVolumeNode(self, name, create=True):

        matches = [
            node for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if node.GetName() == name
        ]
        node = matches[0] if matches else None
        for extra in matches[1:]:
            slicer.mrmlScene.RemoveNode(extra)

        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", name
            )
        return node

    def _exactLinearTransformNode(self, name, create=True):
        matches = [
            node for node in slicer.util.getNodesByClass("vtkMRMLLinearTransformNode")
            if node.GetName() == name
        ]
        node = matches[0] if matches else None
        for extra in matches[1:]:
            slicer.mrmlScene.RemoveNode(extra)

        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", name
            )
        return node

    def _copyVolumeToNode(self, sourceNode, outputName, arrayKJI=None):

        outputNode = self._exactScalarVolumeNode(outputName)
        if arrayKJI is None:
            arrayKJI = np.array(
                slicer.util.arrayFromVolume(sourceNode), copy=True
            )
        else:
            arrayKJI = np.asarray(arrayKJI)

        slicer.util.updateVolumeFromArray(outputNode, arrayKJI)

        ijkToRas = vtk.vtkMatrix4x4()
        sourceNode.GetIJKToRASMatrix(ijkToRas)
        outputNode.SetIJKToRASMatrix(ijkToRas)
        outputNode.SetName(outputName)
        outputNode.CreateDefaultDisplayNodes()
        return outputNode

    def _runBrainMask(self, imageNode, maskName, previewName):
        """Run SwissSkullStripper on one already-resampled volume.

        The returned mask is intentionally kept raw here. Morphological cleanup
        is applied only after T1CE and FLAIR masks are fused, so a peripheral
        lesion is less likely to be discarded as a small disconnected component.
        """
        if imageNode is None or imageNode.GetImageData() is None:
            raise RuntimeError(f"Volume non valido per skull stripping: {maskName}")

        maskNode = self._exactScalarVolumeNode(maskName)
        previewNode = self._exactScalarVolumeNode(previewName)

        parameters = {
            "patientVolume": imageNode.GetID(),
            "patientOutputVolume": previewNode.GetID(),
            "patientMaskLabel": maskNode.GetID(),
        }
        cliNode = slicer.cli.runSync(
            slicer.modules.swissskullstripper, None, parameters
        )
        status = cliNode.GetStatusString() if cliNode else "Unknown"
        errorText = (
            cliNode.GetErrorText()
            if cliNode and hasattr(cliNode, "GetErrorText")
            else ""
        )
        print(f"[Skull stripping {maskName}] {status}")
        if errorText:
            print(f"[Skull stripping {maskName}] {errorText}")

        if cliNode is None or "Completed" not in status or "error" in status.lower():
            raise RuntimeError(
                f"SwissSkullStripper non ha prodotto {maskName}. "
                f"Stato: {status}. {errorText}"
            )

        mask = slicer.util.arrayFromVolume(maskNode) > 0
        if not np.any(mask):
            raise RuntimeError(f"La brain mask {maskName} e' vuota.")

        occupancy = float(np.count_nonzero(mask)) / float(mask.size)
        self._runtimeSetMetric(f"{maskName}_occupancy_raw", f"{occupancy:.6f}")
        print(f"[Skull stripping {maskName} QC] raw occupancy={occupancy:.4f}")
        return maskNode

    def _createCombinedBrainMask(self, t1ceNode, t1Node, t1ceMaskNode=None):
        """Create a conservative shared mask from T1CE OR registered T1.

        Order of operations: union -> closing -> fill holes -> largest connected
        component -> one-voxel dilation. The T1 is already resampled on the T1CE
        grid before its mask is computed, so the voxel-wise OR is geometrically valid.
        """
        if t1ceMaskNode is None:
            t1ceMaskNode = self._runBrainMask(
                t1ceNode,
                "T1CE_brainmask",
                "T1CE_skullstrip_preview",
            )

        t1MaskNode = self._runBrainMask(
            t1Node,
            "T1_brainmask",
            "T1_skullstrip_preview",
        )

        t1ceMask = slicer.util.arrayFromVolume(t1ceMaskNode) > 0
        t1Mask = slicer.util.arrayFromVolume(t1MaskNode) > 0

        if t1ceMask.shape != t1Mask.shape:
            raise RuntimeError(
                "T1CE e T1 brain mask non hanno la stessa geometria: "
                f"T1CE={t1ceMask.shape}, T1={t1Mask.shape}."
            )

        # Voxel-wise union: a voxel is preserved if either skull-stripper mask
        # considers it intracranial.
        combinedMask = np.logical_or(t1ceMask, t1Mask)

        try:
            from scipy import ndimage

            structure = ndimage.generate_binary_structure(3, 1)

            # Reconnect tiny gaps before connected-component filtering.
            combinedMask = ndimage.binary_closing(
                combinedMask, structure=structure, iterations=1
            )
            combinedMask = ndimage.binary_fill_holes(combinedMask)

            labels, count = ndimage.label(combinedMask, structure=structure)
            if count > 0:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                combinedMask = labels == int(np.argmax(sizes))

            # Small safety margin against skull-strip truncation at the cortex.
            combinedMask = ndimage.binary_dilation(
                combinedMask, structure=structure, iterations=1
            )
        except Exception as exc:
            print(f"[Combined brain mask] Pulizia scipy non applicata: {exc}")

        if not np.any(combinedMask):
            raise RuntimeError("La brain mask combinata T1CE+T1 e' vuota.")

        t1ceCount = int(np.count_nonzero(t1ceMask))
        t1Count = int(np.count_nonzero(t1Mask))
        combinedCount = int(np.count_nonzero(combinedMask))
        occupancy = float(combinedCount) / float(combinedMask.size)
        coords = np.argwhere(combinedMask)
        bbox = tuple(coords.min(axis=0)) + tuple(coords.max(axis=0))

        print(
            "[Combined brain mask QC] "
            f"T1CE={t1ceCount}, T1={t1Count}, UNION+cleanup={combinedCount}, "
            f"occupancy={occupancy:.4f}, bbox KJI={bbox}"
        )
        self._runtimeSetMetric("brainmask_t1ce_voxels", t1ceCount)
        self._runtimeSetMetric("brainmask_t1_voxels", t1Count)
        self._runtimeSetMetric("brainmask_combined_voxels", combinedCount)
        self._runtimeSetMetric("brainmask_occupancy", f"{occupancy:.6f}")
        self._runtimeSetMetric("brainmask_bbox_kji", str(bbox))

        if occupancy < 0.02 or occupancy > 0.70:
            raise RuntimeError(
                "Brain mask combinata non plausibile: occupazione "
                f"{occupancy:.1%} del volume. Controllare T1CE, T1 e SwissSkullStripper."
            )

        sharedMaskNode = self._copyVolumeToNode(
            t1ceNode,
            "Shared_brainmask",
            combinedMask.astype(np.uint8),
        )
        self.sharedBrainMaskNodeID = sharedMaskNode.GetID()
        return sharedMaskNode

    def _runSharedMaskOnReference(self, fixedImage):
        """Create the initial T1CE mask used as QC support for T1 registration."""
        return self._runBrainMask(
            fixedImage,
            "T1CE_brainmask",
            "T1CE_skullstrip_preview",
        )

    @staticmethod
    def _normalizedMutualInformation(valuesA, valuesB, bins=64):

        a = np.asarray(valuesA, dtype=np.float64)
        b = np.asarray(valuesB, dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        a = a[valid]
        b = b[valid]
        if a.size < 1000:
            return 0.0

        def robust01(x):
            low, high = np.percentile(x, [1.0, 99.0])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                return np.zeros_like(x)
            return np.clip((x - low) / (high - low), 0.0, 1.0)

        a = robust01(a)
        b = robust01(b)
        histogram, _, _ = np.histogram2d(
            a, b, bins=bins, range=((0.0, 1.0), (0.0, 1.0))
        )
        total = histogram.sum()
        if total <= 0:
            return 0.0

        pxy = histogram / total
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        nzxy = pxy > 0
        nzx = px > 0
        nzy = py > 0
        hxy = -float(np.sum(pxy[nzxy] * np.log(pxy[nzxy])))
        hx = -float(np.sum(px[nzx] * np.log(px[nzx])))
        hy = -float(np.sum(py[nzy] * np.log(py[nzy])))
        if hx + hy <= 0:
            return 0.0

        mutualInformation = hx + hy - hxy
        return max(0.0, min(1.0, 2.0 * mutualInformation / (hx + hy)))

    def _registrationQuality(self, fixedNode, registeredNode, fixedMaskNode):

        fixed = np.asarray(slicer.util.arrayFromVolume(fixedNode), dtype=np.float32)
        moved = np.asarray(slicer.util.arrayFromVolume(registeredNode), dtype=np.float32)
        mask = slicer.util.arrayFromVolume(fixedMaskNode) > 0

        if fixed.shape != moved.shape or fixed.shape != mask.shape:
            return {
                "score": -1.0,
                "nmi": 0.0,
                "coverage": 0.0,
                "reason": "shape mismatch",
            }

        finite = np.isfinite(fixed) & np.isfinite(moved)
        movedNonzero = np.abs(moved) > 1e-8
        valid = mask & finite & movedNonzero
        maskCount = int(np.count_nonzero(mask))
        validCount = int(np.count_nonzero(valid))
        coverage = validCount / max(maskCount, 1)

        if validCount < 1000:
            return {
                "score": -1.0,
                "nmi": 0.0,
                "coverage": coverage,
                "reason": "too few overlapping voxels",
            }

        indices = np.flatnonzero(valid)
        maximumSamples = 250000
        if indices.size > maximumSamples:
            step = int(np.ceil(indices.size / maximumSamples))
            indices = indices[::step]

        fixedValues = fixed.ravel()[indices]
        movedValues = moved.ravel()[indices]
        nmi = self._normalizedMutualInformation(fixedValues, movedValues)


        score = nmi + 0.15 * min(coverage, 1.0)
        return {
            "score": float(score),
            "nmi": float(nmi),
            "coverage": float(coverage),
            "reason": "ok",
        }

    def _runIdentityResampleCandidate(self, fixedImage, movingImage, modality):
        """Ricampiona usando soltanto la geometria fisica nativa, senza ottimizzazione."""
        outputNode = self._exactScalarVolumeNode(
            f"__REG_{modality}_NativeGeometry"
        )
        transformNode = self._exactLinearTransformNode(
            f"__REG_{modality}_NativeGeometry_transform"
        )
        identity = vtk.vtkMatrix4x4()
        identity.Identity()
        transformNode.SetMatrixTransformToParent(identity)

        parameters = {
            "inputVolume": movingImage.GetID(),
            "referenceVolume": fixedImage.GetID(),
            "outputVolume": outputNode.GetID(),
            "transformationFile": transformNode.GetID(),
            "interpolationType": "linear",
            "defaultPixelValue": 0.0,
        }
        cliNode = slicer.cli.runSync(
            slicer.modules.resamplescalarvectordwivolume,
            None,
            parameters,
        )
        status = cliNode.GetStatusString() if cliNode else "Unknown"
        errorText = cliNode.GetErrorText() if cliNode and hasattr(cliNode, "GetErrorText") else ""
        print(f"[Registration candidate] {modality} / NativeGeometry: {status}")
        if errorText:
            print(f"[Registration candidate] {errorText}")
        success = (
            cliNode is not None
            and "Completed" in status
            and "error" not in status.lower()
            and outputNode.GetImageData() is not None
        )
        return success, outputNode, transformNode, status, errorText

    def _runBrainsFitCandidate(
        self,
        fixedImage,
        movingImage,
        modality,
        initializationMode,
        candidateIndex,
    ):
        suffix = f"{candidateIndex}_{initializationMode}"
        outputNode = self._exactScalarVolumeNode(
            f"__REG_{modality}_{suffix}"
        )
        transformNode = self._exactLinearTransformNode(
            f"__REG_{modality}_{suffix}_transform"
        )

       
        samplingPercentage = self.registrationSamplingPercentage.get(
            modality, 0.01
        )

        parameters = {
            "fixedVolume": fixedImage.GetID(),
            "movingVolume": movingImage.GetID(),
            "outputVolume": outputNode.GetID(),
            "linearTransform": transformNode.GetID(),
            "initializeTransformMode": initializationMode,
            "interpolationMode": "Linear",
            "outputVolumePixelType": "float",
            "backgroundFillValue": 0.0,
            "scaleOutputValues": False,
            "useRigid": True,
            "useScaleVersor3D": False,
            "useScaleSkewVersor3D": False,
            "useAffine": False,
            "useBSpline": False,
            "useSyN": False,
            "useComposite": False,
            "maskProcessingMode": "ROIAUTO",
            "ROIAutoDilateSize": 5.0,
            "ROIAutoClosingSize": 9.0,
            "maskInferiorCutOffFromCenter": 1000.0,
            "histogramMatch": False,
            "medianFilterSize": [0, 0, 0],
            "removeIntensityOutliers": 0.005,
            "costMetric": "MMI",
            "numberOfHistogramBins": self.registrationHistogramBins,
            "samplingPercentage": samplingPercentage,
            "numberOfSamples": 0,
            "numberOfIterations": [self.registrationIterations],
            "maximumStepLength": 0.10,
            "minimumStepLength": [0.0005],
            "relaxationFactor": 0.5,
            "translationScale": 1000.0,
            "failureExitCode": -1,
            "writeTransformOnFailure": False,
            "numberOfThreads": -1,
            "debugLevel": 0,
        }

        cliNode = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)
        status = cliNode.GetStatusString() if cliNode else "Unknown"
        errorText = cliNode.GetErrorText() if cliNode and hasattr(cliNode, "GetErrorText") else ""
        print(
            f"[Registration candidate] {modality} / {initializationMode}: "
            f"{status}"
        )
        if errorText:
            print(f"[Registration candidate] {errorText}")

        success = (
            cliNode is not None
            and "Completed" in status
            and "error" not in status.lower()
            and outputNode.GetImageData() is not None
        )
        return success, outputNode, transformNode, status, errorText

    def _copyLinearTransform(self, sourceNode, outputName):
        outputNode = self._exactLinearTransformNode(outputName)
        matrix = vtk.vtkMatrix4x4()
        sourceNode.GetMatrixTransformToParent(matrix)
        outputNode.SetMatrixTransformToParent(matrix)
        outputNode.SetName(outputName)
        return outputNode

    def _cleanupRegistrationCandidates(self, keepVolumeID=None, keepTransformID=None):
        for node in list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")):
            if node.GetName().startswith("__REG_") and node.GetID() != keepVolumeID:
                slicer.mrmlScene.RemoveNode(node)
        for node in list(slicer.util.getNodesByClass("vtkMRMLLinearTransformNode")):
            if node.GetName().startswith("__REG_") and node.GetID() != keepTransformID:
                slicer.mrmlScene.RemoveNode(node)

    def _samePhysicalGeometry(self, firstNode, secondNode, tolerance=1e-4):
        #True se dimensioni e matrice IJK->RAS coincidono.
        if firstNode is None or secondNode is None:
            return False
        if firstNode.GetImageData() is None or secondNode.GetImageData() is None:
            return False

        if tuple(firstNode.GetImageData().GetDimensions()) != tuple(
            secondNode.GetImageData().GetDimensions()
        ):
            return False

        firstMatrix = vtk.vtkMatrix4x4()
        secondMatrix = vtk.vtkMatrix4x4()
        firstNode.GetIJKToRASMatrix(firstMatrix)
        secondNode.GetIJKToRASMatrix(secondMatrix)

        maximumDifference = max(
            abs(firstMatrix.GetElement(row, column) -
                secondMatrix.GetElement(row, column))
            for row in range(4)
            for column in range(4)
        )
        return maximumDifference <= tolerance
    
    def _registrationQualityIsAcceptable(
        self,
        quality,
        native=False,
        modality=None,
    ):
        if quality is None:
            return False

        if native:
            minimumCoverage = self.nativeGeometryMinCoverage.get(
                modality, 0.93
            )
            minimumNMI = self.nativeGeometryMinNMI.get(
                modality, 0.030
            )
        else:
            minimumCoverage = self.registrationMinCoverage.get(
                modality, 0.90
            )
            minimumNMI = self.registrationMinNMI.get(
                modality, 0.015
            )

        return (
            quality.get("coverage", 0.0) >= minimumCoverage
            and quality.get("nmi", 0.0) >= minimumNMI
        )

    def _registrationCandidateIsNative(self, candidate):
        return candidate.get("initialization") in {
            "SameGeometry",
            "NativeGeometry",
        }

    def _acceptableRegistrationCandidates(self, candidates, modality):
        return [
            candidate
            for candidate in candidates
            if self._registrationQualityIsAcceptable(
                candidate,
                native=self._registrationCandidateIsNative(candidate),
                modality=modality,
            )
        ]

    def _finalizeRegistrationCandidate(self, best, modality):
        finalVolume = self._copyVolumeToNode(
            best["output"], f"{modality}_original"
        )
        finalTransform = self._copyLinearTransform(
            best["transform"], f"{modality}_transform"
        )

        self.registrationQC.append({
            "modality": modality,
            "initialization": best["initialization"],
            "nmi": best["nmi"],
            "coverage": best["coverage"],
            "score": best["score"],
        })
        self._runtimeSetMetric(f"registration_{modality}_initialization", best["initialization"])
        self._runtimeSetMetric(f"registration_{modality}_nmi", f"{best['nmi']:.6f}")
        self._runtimeSetMetric(f"registration_{modality}_coverage", f"{best['coverage']:.6f}")
        self._runtimeSetMetric(f"registration_{modality}_score", f"{best['score']:.6f}")

        print(
            f"[Registration selected] {modality}: "
            f"{best['initialization']}, NMI={best['nmi']:.4f}, "
            f"coverage={best['coverage']:.1%}"
        )
        self._cleanupRegistrationCandidates()
        return finalVolume, finalTransform

    def _registerOneModality(self, fixedImage, movingImage, modality, fixedMaskNode):

        evaluatedCandidates = 0

        def record_candidate(initialization, outputNode, transformNode, quality):
            nonlocal evaluatedCandidates
            evaluatedCandidates += 1
            candidate = {
                "initialization": initialization,
                "output": outputNode,
                "transform": transformNode,
                **quality,
            }
            print(
                f"[Registration QC] {modality} / {initialization}: "
                f"NMI={quality['nmi']:.4f}, "
                f"coverage={quality['coverage']:.3f}, "
                f"score={quality['score']:.4f}"
            )
            return candidate

        def accept_and_finalize(candidate, native):
            if not self._registrationQualityIsAcceptable(
                candidate, native=native, modality=modality
            ):
                return None

            print(
                f"[Registration fast path] {modality}: "
                f"{candidate['initialization']} accettata; stop ricerca candidati."
            )
            self._runtimeSetMetric(
                f"registration_{modality}_acceptable_candidates", 1
            )
            self._runtimeSetMetric(
                f"registration_{modality}_evaluated_candidates",
                evaluatedCandidates,
            )
            return self._finalizeRegistrationCandidate(candidate, modality)

        failedCandidates = []

        # 1) Fast path: se le geometrie coincidono fisicamente, non eseguire BRAINSFit.
        if self.fastRegistration and self._samePhysicalGeometry(
            fixedImage, movingImage
        ):
            quality = self._registrationQuality(
                fixedImage, movingImage, fixedMaskNode
            )
            outputNode = self._copyVolumeToNode(
                movingImage, f"__REG_{modality}_SameGeometry"
            )
            transformNode = self._exactLinearTransformNode(
                f"__REG_{modality}_SameGeometry_transform"
            )
            identity = vtk.vtkMatrix4x4()
            identity.Identity()
            transformNode.SetMatrixTransformToParent(identity)

            candidate = record_candidate(
                "SameGeometry", outputNode, transformNode, quality
            )
            result = accept_and_finalize(candidate, native=True)
            if result is not None:
                return result
            failedCandidates.append(candidate)

        # 2) Ricampionamento nativo sulla griglia T1CE. Se passa il QC, basta questo.
        success, outputNode, transformNode, status, errorText = (
            self._runIdentityResampleCandidate(
                fixedImage, movingImage, modality
            )
        )
        if success:
            quality = self._registrationQuality(
                fixedImage, outputNode, fixedMaskNode
            )
            candidate = record_candidate(
                "NativeGeometry", outputNode, transformNode, quality
            )
            result = accept_and_finalize(candidate, native=True)
            if result is not None:
                return result
            failedCandidates.append(candidate)

        # 3) BRAINSFit viene eseguito soltanto se i fast path non superano il QC.
        # Ogni inizializzazione viene valutata in sequenza e ci si ferma appena
        # compare un candidato accettabile.
        initializationModes = ["Off", "useGeometryAlign", "useMomentsAlign"]
        for candidateIndex, initializationMode in enumerate(initializationModes):
            print(
                f"[Registration] {modality}: BRAINSFit / "
                f"{initializationMode} ({candidateIndex + 1}/{len(initializationModes)})"
            )
            success, outputNode, transformNode, status, errorText = (
                self._runBrainsFitCandidate(
                    fixedImage,
                    movingImage,
                    modality,
                    initializationMode,
                    candidateIndex,
                )
            )
            if not success:
                continue

            quality = self._registrationQuality(
                fixedImage, outputNode, fixedMaskNode
            )
            candidate = record_candidate(
                initializationMode, outputNode, transformNode, quality
            )
            result = accept_and_finalize(candidate, native=False)
            if result is not None:
                return result
            failedCandidates.append(candidate)

        self._runtimeSetMetric(
            f"registration_{modality}_acceptable_candidates", 0
        )
        self._runtimeSetMetric(
            f"registration_{modality}_evaluated_candidates",
            evaluatedCandidates,
        )

        if failedCandidates:
            details = "\n".join(
                f"  - {candidate['initialization']}: "
                f"NMI={candidate['nmi']:.4f}, "
                f"coverage={candidate['coverage']:.1%}, "
                f"score={candidate['score']:.4f}"
                for candidate in failedCandidates
            )
            self._cleanupRegistrationCandidates()
            raise RuntimeError(
                f"Nessuna registrazione accettabile per {modality}.\n"
                f"Candidati valutati:\n{details}"
            )

        self._cleanupRegistrationCandidates()
        raise RuntimeError(
            f"Nessun tentativo di registrazione e' riuscito per {modality}."
        )

    def _writeRegistrationQC(self):
        if not getattr(self, "registrationQC", None):
            return None
        os.makedirs(self.niftiExportDir, exist_ok=True)
        outputPath = os.path.join(
            self.niftiExportDir, "registration_qc.csv"
        )
        with open(outputPath, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "modality", "initialization", "nmi", "coverage", "score"
                ],
                delimiter=";",
            )
            writer.writeheader()
            writer.writerows(self.registrationQC)
        print(f"[Registration QC] Report: {outputPath}")
        return outputPath

    def runPreprocessing(self):
        fixedImage = self._selectedInputNode("T1CE")
        if fixedImage is None or fixedImage.GetImageData() is None:
            raise RuntimeError("T1CE non valida: impossibile creare il riferimento.")

        # T1CE defines the common physical grid.
        fixedOriginal = self._copyVolumeToNode(
            fixedImage, "T1CE_original"
        )
        self.registrationQC = []

        # 1) Initial T1CE mask: used only as registration QC support for T1.
        t1ceMaskNode = self._runSharedMaskOnReference(fixedOriginal)

        # 2) Register T1 first. Its registered mask is then fused with the T1CE
        #    mask to build the final shared skull-stripping mask.
        t1Image = self._selectedInputNode("T1")
        if t1Image is None or t1Image.GetImageData() is None:
            raise RuntimeError("Volume non valido o mancante: T1")

        self._registerOneModality(
            fixedOriginal,
            t1Image,
            "T1",
            t1ceMaskNode,
        )

        t1Original = slicer.util.getNode("T1_original")

        # 3) Conservative multimodal brain mask: T1CE OR registered T1, then
        #    closing/fill/largest-component and a one-voxel safety dilation.
        sharedMaskNode = self._createCombinedBrainMask(
            fixedOriginal,
            t1Original,
            t1ceMaskNode=t1ceMaskNode,
        )

        # 4) Register T2 and FLAIR using the final shared mask for QC.
        for modality in ["T2", "FLAIR"]:
            movingImage = self._selectedInputNode(modality)
            if movingImage is None or movingImage.GetImageData() is None:
                raise RuntimeError(f"Volume non valido o mancante: {modality}")

            self._registerOneModality(
                fixedOriginal,
                movingImage,
                modality,
                sharedMaskNode,
            )

        registrationQCPath = self._writeRegistrationQC()
        self._runtimeSetMetric("registration_qc_path", registrationQCPath or "")
        self._assertInferenceGeometry(
            ["T1_original", "T2_original", "T1CE_original", "FLAIR_original"]
        )
        print(
            "Registrazione robusta completata sulla griglia T1CE; "
            "Shared_brainmask costruita da T1CE+T1."
        )

    def runSkullStripping(self):

        try:
            maskNode = slicer.mrmlScene.GetNodeByID(self.sharedBrainMaskNodeID)
        except Exception:
            maskNode = None
        if maskNode is None:
            maskNode = self._exactScalarVolumeNode(
                "Shared_brainmask", create=False
            )
        if maskNode is None or maskNode.GetImageData() is None:
            raise RuntimeError(
                "Shared_brainmask non disponibile. Rieseguire la registrazione."
            )

        mask = slicer.util.arrayFromVolume(maskNode) > 0
        mapping = {
            "T1_original": "T1",
            "T2_original": "T2",
            "T1CE_original": "T1CE",
            "FLAIR_original": "FLAIR",
        }

        for sourceName, outputName in mapping.items():
            sourceNode = slicer.util.getNode(sourceName)
            image = np.asarray(
                slicer.util.arrayFromVolume(sourceNode), dtype=np.float32
            )
            if image.shape != mask.shape:
                raise RuntimeError(
                    f"Shape incompatibile per {sourceName}: "
                    f"immagine={image.shape}, mask={mask.shape}"
                )

            stripped = np.where(mask, image, 0.0).astype(np.float32)
            self._copyVolumeToNode(sourceNode, outputName, stripped)
            print(
                f"[Shared skull stripping] applicata a {sourceName} -> {outputName}"
            )

        self._assertInferenceGeometry(["T1", "T2", "T1CE", "FLAIR"])
        print("Skull stripping condiviso completato")

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

    def _ijkToRasArray(self, volumeNode):
        matrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(matrix)
        return np.array(
            [[matrix.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )

    def _assertInferenceGeometry(self, sequenceNames):
        
        referenceName = "T1CE_original"
        referenceNode = slicer.util.getNode(referenceName)
        referenceDims = tuple(referenceNode.GetImageData().GetDimensions())
        referenceMatrix = self._ijkToRasArray(referenceNode)

        errors = []
        for name in sequenceNames:
            node = slicer.util.getNode(name)
            dims = tuple(node.GetImageData().GetDimensions())
            matrix = self._ijkToRasArray(node)

            print(f"[Geometry] {name}: dimensions IJK={dims}")
            print(f"[Geometry] {name}: IJK->RAS=\n{matrix}")

            if dims != referenceDims:
                errors.append(
                    f"{name}: dimensioni {dims}, attese {referenceDims}"
                )

            if not np.allclose(matrix, referenceMatrix, rtol=0.0, atol=1e-4):
                maxDifference = float(np.max(np.abs(matrix - referenceMatrix)))
                errors.append(
                    f"{name}: matrice IJK->RAS diversa dalla T1CE "
                    f"(differenza massima {maxDifference:.6g})"
                )

        if errors:
            raise RuntimeError(
                "I volumi non condividono la stessa geometria dopo registrazione "
                "e skull stripping:\n- " + "\n- ".join(errors)
            )

    def _exportInferenceVolumes(self, sequenceNames):

        outDir = os.path.join(slicer.app.temporaryPath, "custom_inference_oriented")
        os.makedirs(outDir, exist_ok=True)

        nodeMap = {name: name for name in sequenceNames}
        nodeMap[self.brainMaskKey] = "Shared_brainmask"

        paths = {}
        for logicalName, nodeName in nodeMap.items():
            node = slicer.util.getNode(nodeName)
            path = os.path.join(outDir, f"{logicalName}.nii.gz")

            if os.path.exists(path):
                os.remove(path)

            # world=True include eventuali trasformazioni parent nella geometria esportata.
            slicer.util.exportNode(
                node,
                path,
                {"useCompression": 1},
                world=True,
            )

            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Esportazione NIfTI fallita per {logicalName}: {path}"
                )

            paths[logicalName] = path
            print(f"[Inference input] {logicalName} -> {path}")

        return paths

    def _normalizeNonzero(self, array):

        output = np.asarray(array, dtype=np.float32).copy()
        valid = np.isfinite(output) & (output != 0)
        if not np.any(valid):
            return output
        values = output[valid]
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        output[valid] = (values - mean) / std
        output[~np.isfinite(output)] = 0.0
        return output

    def _centerCropPadPlan(self, sourceShape, targetShape):

        sourceSlices = []
        targetSlices = []
        for sourceSize, targetSize in zip(sourceShape, targetShape):
            sourceSize = int(sourceSize)
            targetSize = int(targetSize)
            if sourceSize >= targetSize:
                sourceStart = (sourceSize - targetSize) // 2
                sourceEnd = sourceStart + targetSize
                targetStart = 0
                targetEnd = targetSize
            else:
                sourceStart = 0
                sourceEnd = sourceSize
                targetStart = (targetSize - sourceSize) // 2
                targetEnd = targetStart + sourceSize
            sourceSlices.append(slice(sourceStart, sourceEnd))
            targetSlices.append(slice(targetStart, targetEnd))
        return tuple(sourceSlices), tuple(targetSlices)

    def _prepareInferenceArrays(self, inputPaths):
 
        if nib is None or resample_from_to is None or resample_to_output is None:
            raise RuntimeError(
                "Nibabel non è disponibile nell'ambiente Python di Slicer. "
                "Aprire la Python Console ed eseguire: "
                "slicer.util.pip_install('nibabel')"
            )
        referenceImage = nib.load(inputPaths["T1CE"])
        referenceCanonical = nib.as_closest_canonical(referenceImage)

        # Costruisce una griglia RAS isotropa comune. Questa è l'equivalente
        # esplicita di Orientationd + Spacingd.
        targetReference = resample_to_output(
            referenceCanonical,
            voxel_sizes=self.modelSpacing,
            order=1,
            mode="constant",
            cval=0.0,
        )
        targetShape = tuple(int(v) for v in targetReference.shape[:3])
        targetAffine = np.asarray(targetReference.affine, dtype=np.float64)
        targetSpec = (targetShape, targetAffine)

        resampled = {}
        for key in self.inferenceKeys:
            image = nib.load(inputPaths[key])
            output = resample_from_to(
                image,
                targetSpec,
                order=1,
                mode="constant",
                cval=0.0,
            )
            resampled[key] = np.asarray(
                output.dataobj, dtype=np.float32
            )

        maskImage = nib.load(inputPaths[self.brainMaskKey])
        maskResampledImage = resample_from_to(
            maskImage,
            targetSpec,
            order=0,
            mode="constant",
            cval=0.0,
        )
        maskTarget = np.asarray(maskResampledImage.dataobj) > 0.5
        coordinates = np.argwhere(maskTarget)
        if coordinates.size == 0:
            raise RuntimeError(
                "La brain mask è vuota dopo orientamento RAS e spacing 1 mm."
            )

        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        margin = np.asarray(self.cropMargin, dtype=int)
        cropStart = np.maximum(lower - margin, 0).astype(int)
        cropEnd = np.minimum(
            upper + margin, np.asarray(targetShape, dtype=int)
        ).astype(int)
        cropSlices = tuple(
            slice(int(cropStart[i]), int(cropEnd[i])) for i in range(3)
        )
        cropShape = tuple(int(cropEnd[i] - cropStart[i]) for i in range(3))
        sourceSlices, modelSlices = self._centerCropPadPlan(
            cropShape, self.modelSpatialSize
        )

        modelArrays = {}
        for key in self.inferenceKeys:
            cropped = resampled[key][cropSlices]
            modelArray = np.zeros(self.modelSpatialSize, dtype=np.float32)
            modelArray[modelSlices] = cropped[sourceSlices]
            modelArrays[key] = self._normalizeNonzero(modelArray)

        croppedMask = maskTarget[cropSlices]
        modelMask = np.zeros(self.modelSpatialSize, dtype=np.uint8)
        modelMask[modelSlices] = croppedMask[sourceSlices].astype(np.uint8)

        print(
            f"[Geometry explicit] original T1CE={referenceImage.shape[:3]}, "
            f"RAS/spacing={targetShape}, crop={cropShape}, "
            f"model={self.modelSpatialSize}"
        )
        print(
            f"[Geometry explicit] crop start={cropStart.tolist()}, "
            f"end={cropEnd.tolist()}, source slices={sourceSlices}, "
            f"model slices={modelSlices}"
        )

        context = {
            "referenceImage": referenceImage,
            "targetShape": targetShape,
            "targetAffine": targetAffine,
            "cropStart": cropStart,
            "cropEnd": cropEnd,
            "cropShape": cropShape,
            "cropSlices": cropSlices,
            "sourceSlices": sourceSlices,
            "modelSlices": modelSlices,
            "originalMaskImage": maskImage,
            "modelMask": modelMask,
        }
        return modelArrays, context

    def _restoreModelArrayToOriginalIJK(self, modelArray, context):

        array = np.asarray(modelArray)
        if array.ndim == 3:
            array = array[np.newaxis, ...]
        if array.ndim != 4:
            raise RuntimeError(
                f"Array del modello atteso [C,X,Y,Z], ricevuto {array.shape}."
            )
        if tuple(array.shape[1:]) != tuple(self.modelSpatialSize):
            raise RuntimeError(
                f"Forma del modello {array.shape[1:]} diversa da "
                f"{self.modelSpatialSize}."
            )

        channels = int(array.shape[0])
        cropArray = np.zeros(
            (channels,) + tuple(context["cropShape"]), dtype=np.float32
        )
        cropArray[(slice(None),) + context["sourceSlices"]] = array[
            (slice(None),) + context["modelSlices"]
        ]

        targetArray = np.zeros(
            (channels,) + tuple(context["targetShape"]), dtype=np.float32
        )
        targetArray[(slice(None),) + context["cropSlices"]] = cropArray

        referenceImage = context["referenceImage"]
        restoredChannels = []
        for channelIndex in range(channels):
            targetImage = nib.Nifti1Image(
                targetArray[channelIndex], context["targetAffine"]
            )
            restoredImage = resample_from_to(
                targetImage,
                referenceImage,
                order=0,
                mode="constant",
                cval=0.0,
            )
            restoredChannels.append(
                np.asarray(restoredImage.dataobj, dtype=np.float32)
            )
        return np.stack(restoredChannels, axis=0)

    def _geometryRoundTripQC(self, context):

        restoredIJK = self._restoreModelArrayToOriginalIJK(
            context["modelMask"], context
        )[0] > 0.5
        originalIJK = np.asarray(
            context["originalMaskImage"].dataobj
        ) > 0.5

        if restoredIJK.shape != originalIJK.shape:
            raise RuntimeError(
                "Round-trip geometrico con dimensioni diverse: "
                f"{restoredIJK.shape} vs {originalIJK.shape}."
            )

        intersection = int(np.count_nonzero(restoredIJK & originalIJK))
        denominator = int(
            np.count_nonzero(restoredIJK) + np.count_nonzero(originalIJK)
        )
        dice = (2.0 * intersection / denominator) if denominator else 1.0

        originalCoordinates = np.argwhere(originalIJK)
        restoredCoordinates = np.argwhere(restoredIJK)
        if originalCoordinates.size and restoredCoordinates.size:
            shift = float(np.linalg.norm(
                originalCoordinates.mean(axis=0)
                - restoredCoordinates.mean(axis=0)
            ))
        else:
            shift = float("inf")

        print(
            f"[Geometry round-trip explicit] Dice={dice:.5f}, "
            f"centroid shift={shift:.3f} voxel IJK"
        )
        self._runtimeSetMetric("geometry_round_trip_dice", f"{dice:.6f}")
        self._runtimeSetMetric("geometry_round_trip_centroid_shift_voxel", f"{shift:.6f}")
        if dice < 0.90 or shift > 2.0:
            raise RuntimeError(
                "La ricostruzione geometrica esplicita non è affidabile. "
                f"Dice={dice:.4f}, shift={shift:.2f} voxel."
            )
        return dice

    def runInference(self, progressDialog=None, start_value=0, end_value=100):
        if not self.models:
            qt.QMessageBox.warning(
                None, "Error", "No models loaded. Load models before running inference."
            )
            return

        ownsDialog = False
        if progressDialog is None:
            progressDialog = qt.QProgressDialog("Please wait ...", None, 0, 100)
            progressDialog.setWindowTitle("Inference")
            progressDialog.setCancelButton(None)
            progressDialog.setMinimumWidth(300)
            progressDialog.show()
            ownsDialog = True
            slicer.app.processEvents()

        referenceVolumeT1CE = slicer.util.getNode("T1CE_original")
        sequenceNames = ["T1", "T2", "T1CE", "FLAIR"]

        def setProgress(percent, label=None):
            value = int(start_value + (end_value - start_value) * percent / 100.0)
            progressDialog.setValue(value)
            if label:
                progressDialog.setLabelText(label)
            slicer.app.processEvents()

        try:
            setProgress(2, "Checking registered geometry...")
            with self._timedStage("check_registered_geometry"):
                self._assertInferenceGeometry(sequenceNames)

            setProgress(5, "Exporting geometry-aware NIfTI inputs...")
            with self._timedStage("nifti_export"):
                inputPaths = self._exportInferenceVolumes(sequenceNames)

            setProgress(8, "Orienting, spacing and cropping inputs...")
            with self._timedStage("prepare_arrays"):
                modelArrays, geometryContext = self._prepareInferenceArrays(
                    inputPaths
                )

            with self._timedStage("geometry_round_trip"):
                self._geometryRoundTripQC(geometryContext)

            with self._timedStage("tensor_creation"):
                inputs = {
                    key: torch.from_numpy(
                        modelArrays[key][np.newaxis, np.newaxis, ...]
                    ).to(self.device, dtype=torch.float32)
                    for key in ["T1CE", "T1", "T2", "FLAIR"]
                }
                for key, tensor in inputs.items():
                    print(f"[Model input] {key}: shape={tuple(tensor.shape)}, device={tensor.device}")

            allPredictions = []
            numberOfModels = len(self.models)
            with torch.no_grad():
                for modelIndex, model in enumerate(self.models):
                    stageName = f"model_fold_{modelIndex + 1}_inference"
                    with self._timedStage(stageName):
                        testOutput = model(
                            inputs["T1CE"],
                            inputs["T1"],
                            inputs["T2"],
                            inputs["FLAIR"],
                        )
                        probabilities = torch.sigmoid(testOutput)

                        # Mantiene classi mutuamente esclusive, come richiesto.
                        maximumProbabilities, maximumClasses = torch.max(
                            probabilities, dim=1, keepdim=True
                        )
                        exclusiveProbabilities = torch.zeros_like(probabilities)
                        exclusiveProbabilities.scatter_(
                            1, maximumClasses, maximumProbabilities
                        )
                        exclusivePrediction = (
                            exclusiveProbabilities > 0.5
                        ).float()
                        allPredictions.append(exclusivePrediction)

                        channelMeans = probabilities.mean(
                            dim=(0, 2, 3, 4)
                        ).detach().cpu().numpy()
                        channelMaximum = probabilities.amax(
                            dim=(0, 2, 3, 4)
                        ).detach().cpu().numpy()
                        print(
                            f"[Model fold {modelIndex + 1}] probability mean="
                            f"{channelMeans.tolist()}, max={channelMaximum.tolist()}"
                        )
                        self._runtimeSetMetric(
                            f"fold_{modelIndex + 1}_probability_mean_NCRNET",
                            f"{float(channelMeans[0]):.6f}",
                        )
                        self._runtimeSetMetric(
                            f"fold_{modelIndex + 1}_probability_mean_ED",
                            f"{float(channelMeans[1]):.6f}",
                        )
                        self._runtimeSetMetric(
                            f"fold_{modelIndex + 1}_probability_mean_ET",
                            f"{float(channelMeans[2]):.6f}",
                        )
                        self._runtimeSetMetric(
                            f"fold_{modelIndex + 1}_probability_max_NCRNET",
                            f"{float(channelMaximum[0]):.6f}",
                        )
                        self._runtimeSetMetric(
                            f"fold_{modelIndex + 1}_probability_max_ED",
                            f"{float(channelMaximum[1]):.6f}",
                        )
                        self._runtimeSetMetric(
                            f"fold_{modelIndex + 1}_probability_max_ET",
                            f"{float(channelMaximum[2]):.6f}",
                        )

                    percent = 10 + int(
                        (modelIndex + 1) / max(numberOfModels, 1) * 50
                    )
                    setProgress(
                        percent,
                        f"Running model {modelIndex + 1}/{numberOfModels}...",
                    )

                with self._timedStage("ensemble_voting"):
                    allPredictions = torch.stack(allPredictions)
                    finalPrediction = torch.mode(allPredictions, dim=0)[0]

            voxelCounts = finalPrediction.sum(
                dim=(0, 2, 3, 4)
            ).detach().cpu().numpy().astype(int)
            overlaps = (finalPrediction.sum(dim=1) > 1).sum().item()
            print(
                f"[Ensemble] voxel NCR/NET, ED, ET="
                f"{voxelCounts.tolist()}, overlap voxels={int(overlaps)}"
            )
            self._runtimeSetMetric("segmentation_voxels_NCRNET", int(voxelCounts[0]))
            self._runtimeSetMetric("segmentation_voxels_ED", int(voxelCounts[1]))
            self._runtimeSetMetric("segmentation_voxels_ET", int(voxelCounts[2]))
            self._runtimeSetMetric("overlap_voxels", int(overlaps))

            setProgress(70, "Restoring original T1CE geometry...")
            with self._timedStage("restore_original_geometry"):
                predictionModel = finalPrediction[0].detach().cpu().numpy()
                restoredPrediction = self._restoreModelArrayToOriginalIJK(
                    predictionModel, geometryContext
                )
                maskArrayIJK = (restoredPrediction > 0.5).astype(np.uint8)

            expectedIJK = tuple(
                int(v) for v in referenceVolumeT1CE.GetImageData().GetDimensions()
            )
            actualIJK = tuple(int(v) for v in maskArrayIJK.shape[1:])
            print(
                f"[Prediction explicit] restored shape C-I-J-K="
                f"{maskArrayIJK.shape}; expected I-J-K={expectedIJK}"
            )
            self._runtimeSetMetric("prediction_restored_shape_cijk", str(tuple(maskArrayIJK.shape)))
            self._runtimeSetMetric("expected_t1ce_shape_ijk", str(expectedIJK))

            if actualIJK != expectedIJK:
                raise RuntimeError(
                    "La predizione ricostruita non coincide con la griglia "
                    f"T1CE: ottenuta {actualIJK}, attesa {expectedIJK}."
                )

            setProgress(90, "Creating segmentation node...")
            with self._timedStage("create_segmentation_node"):
                self.createSegmentationNode(maskArrayIJK)
            setProgress(100, "Done")

        finally:
            if ownsDialog:
                progressDialog.close()

    def createSegmentationNode(self, maskArrayIJK):

        referenceVolume = slicer.util.getNode("T1CE_original")
        expectedIJK = tuple(referenceVolume.GetImageData().GetDimensions())

        if maskArrayIJK.ndim != 4:
            raise RuntimeError(
                f"Maschera attesa [C,I,J,K], ricevuta {maskArrayIJK.shape}"
            )

        if tuple(maskArrayIJK.shape[1:]) != expectedIJK:
            raise RuntimeError(
                f"Geometria maschera {maskArrayIJK.shape[1:]} diversa "
                f"dalla T1CE {expectedIJK}"
            )


        try:
            previousSegmentation = slicer.util.getNode("Segmentation_original")
            slicer.mrmlScene.RemoveNode(previousSegmentation)
        except slicer.util.MRMLNodeNotFoundException:
            pass

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            "Segmentation_original",
        )
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(
            referenceVolume
        )
        segmentationNode.CreateDefaultDisplayNodes()

        segmentLabels = ["NECROSI", "FLAIR", "CONTRASTO"]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        if maskArrayIJK.shape[0] != len(segmentLabels):
            raise RuntimeError(
                f"Il modello ha prodotto {maskArrayIJK.shape[0]} canali, "
                f"ma sono attesi {len(segmentLabels)}."
            )

        expectedKJI = tuple(reversed(expectedIJK))

        for channelIndex, label in enumerate(segmentLabels):
            # Da MONAI/NIfTI [I,J,K] a Slicer NumPy [K,J,I].
            maskKJI = np.transpose(
                maskArrayIJK[channelIndex],
                (2, 1, 0),
            ).astype(np.uint8, copy=False)

            if tuple(maskKJI.shape) != expectedKJI:
                raise RuntimeError(
                    f"Segmento {label}: forma KJI {maskKJI.shape}, "
                    f"attesa {expectedKJI}."
                )

            segmentId = segmentationNode.GetSegmentation().AddEmptySegment()
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            segment.SetName(label)
            segment.SetColor(colors[channelIndex])

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                maskKJI,
                segmentationNode,
                segmentId,
                referenceVolume,
            )

            print(
                f"[Segmentation] {label}: IJK="
                f"{maskArrayIJK[channelIndex].shape}, KJI={maskKJI.shape}, "
                f"voxels={int(np.count_nonzero(maskKJI))}"
            )

        displayNode = segmentationNode.GetDisplayNode()
        displayNode.SetVisibility2DFill(True)
        displayNode.SetVisibility2DOutline(True)
        displayNode.SetVisibility3D(True)

        self._runtimeSetMetric("segmentation_node_name", segmentationNode.GetName())
        self._runtimeSetMetric("segmentation_node_id", segmentationNode.GetID())
        self.updateSliceViewerLayers(segmentationNode)
        slicer.app.processEvents()

    def _createRadiomicsExtractor(self):
        try:
            from radiomics import featureextractor
        except ImportError:
            raise RuntimeError(
                "PyRadiomics non è installato nell'ambiente Python di Slicer.\n\n"
                "Installa prima le dipendenze dalla Python console di 3D Slicer con:\n"
                "slicer.util.pip_install('pyradiomics')\n"
                "slicer.util.pip_install('scikit-image')\n"
                "slicer.util.pip_install('trimesh')"
            )

        extractor = featureextractor.RadiomicsFeatureExtractor(
            binWidth=float(self.radiomicsBinWidth),
            # Normalization is performed explicitly and identically in both the
            # Slicer and standalone pipelines, only on finite non-zero voxels.
            normalize=False,
            removeOutliers=None,
            resampledPixelSpacing=list(self.radiomicsTargetSpacing),
            interpolator="sitkBSpline",
            # Masks are exported by Slicer directly on the selected image grid.
            # A geometry mismatch should therefore fail instead of being hidden.
            correctMask=False,
            geometryTolerance=1e-5,
            preCrop=True,
        )

        extractor.enableAllFeatures()
        extractor.disableAllImageTypes()
        extractor.enableImageTypeByName("Original")
        extractor.enableImageTypeByName("Wavelet", customArgs={
            "wavelet": "coif1",
            "level": 1,
        })
        extractor.enableImageTypeByName("LoG", customArgs={
            "sigma": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        extractor.enableImageTypeByName("Square")
        extractor.enableImageTypeByName("SquareRoot")
        extractor.enableImageTypeByName("Logarithm")
        extractor.enableImageTypeByName("Exponential")
        extractor.enableImageTypeByName("Gradient")

        try:
            extractor.enableImageTypeByName("LocalBinaryPattern2D")
        except Exception as e:
            print(f"[Radiomics warning] LocalBinaryPattern2D non abilitato: {e}")

        try:
            extractor.enableImageTypeByName("LocalBinaryPattern3D")
        except Exception as e:
            print(f"[Radiomics warning] LocalBinaryPattern3D non abilitato: {e}")

        return extractor

    def _radiomicsOriginalNodes(self):
        """Return the only image nodes allowed by the standardized pipeline."""
        nodeNames = {
            "T1": "T1_original",
            "T2": "T2_original",
            "T1CE": "T1CE_original",
            "FLAIR": "FLAIR_original",
        }
        imageNodes = {}
        missing = []
        for modality, nodeName in nodeNames.items():
            try:
                node = slicer.util.getNode(nodeName)
            except slicer.util.MRMLNodeNotFoundException:
                node = None
            if node is None or node.GetImageData() is None:
                missing.append(nodeName)
            else:
                imageNodes[modality] = node

        if missing:
            raise RuntimeError(
                "Mancano i volumi radiomici standardizzati: " + ", ".join(missing)
            )

        # All registered originals must share the T1CE physical grid before any
        # radiomics-specific resampling is performed by PyRadiomics.
        self._assertInferenceGeometry(list(nodeNames.values()))
        return imageNodes

    def _radiomicsBrainMaskArray(self, referenceNode):
        """Return Shared_brainmask on the common *_original grid.

        The inference preprocessing constructs Shared_brainmask on the T1CE
        reference grid and all *_original volumes are required to share this
        same physical geometry. Radiomics therefore uses this mask to remove
        extracranial tissues before estimating normalization statistics.
        """
        try:
            maskNode = slicer.util.getNode("Shared_brainmask")
        except slicer.util.MRMLNodeNotFoundException:
            maskNode = None
        if maskNode is None or maskNode.GetImageData() is None:
            raise RuntimeError(
                "Shared_brainmask non disponibile. La radiomica standardizzata "
                "richiede la brain mask generata durante il preprocessing, "
                "per rimuovere skull/tessuti extracranici prima della normalizzazione."
            )

        referenceDims = tuple(referenceNode.GetImageData().GetDimensions())
        maskDims = tuple(maskNode.GetImageData().GetDimensions())
        if maskDims != referenceDims:
            raise RuntimeError(
                f"Shared_brainmask con dimensioni {maskDims}, attese {referenceDims}."
            )

        referenceMatrix = self._ijkToRasArray(referenceNode)
        maskMatrix = self._ijkToRasArray(maskNode)
        maxDifference = float(np.max(np.abs(referenceMatrix - maskMatrix)))
        if maxDifference > 1e-4:
            raise RuntimeError(
                "Shared_brainmask non condivide la geometria fisica degli *_original "
                f"(differenza massima IJK->RAS={maxDifference:.6g})."
            )

        maskKJI = np.asarray(slicer.util.arrayFromVolume(maskNode)) > 0
        brainVoxelCount = int(np.count_nonzero(maskKJI))
        if brainVoxelCount < 2:
            raise RuntimeError("Shared_brainmask vuota o non valida.")

        return maskNode, maskKJI.astype(bool, copy=False)

    def _n4BiasCorrectRadiomicsArray(
        self, imageNode, brainMaskNode, modality, workDir
    ):
        """Return an N4-corrected KJI array without modifying *_original.

        N4 is estimated inside Shared_brainmask. A shrink factor is used only
        for estimating the smooth bias field; the estimated field is then
        reconstructed and applied on the full-resolution original grid.
        Temporary NRRD files live in the radiomics working directory.
        """
        if not self.radiomicsUseN4BiasCorrection:
            arrayKJI = np.asarray(
                slicer.util.arrayFromVolume(imageNode), dtype=np.float32
            )
            return arrayKJI, {
                "N4BiasCorrection": False,
                "N4Mask": "",
                "N4ShrinkFactor": "",
                "N4Iterations": "",
                "N4ConvergenceThreshold": "",
            }

        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise RuntimeError(
                "SimpleITK non disponibile: N4 bias field correction e' "
                "obbligatoria nella pipeline radiomica standardizzata."
            ) from exc

        safeModality = self._safeName(modality)
        inputPath = os.path.join(workDir, f"{safeModality}_n4_input.nrrd")
        maskPath = os.path.join(workDir, f"{safeModality}_n4_brainmask.nrrd")

        if not slicer.util.saveNode(imageNode, inputPath):
            raise RuntimeError(
                f"Impossibile salvare input temporaneo N4 per {modality}: {inputPath}"
            )
        if not slicer.util.saveNode(brainMaskNode, maskPath):
            raise RuntimeError(
                f"Impossibile salvare brain mask temporanea N4 per {modality}: {maskPath}"
            )

        image = sitk.ReadImage(inputPath, sitk.sitkFloat32)
        mask = sitk.ReadImage(maskPath)
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

        shrinkFactor = max(1, int(self.radiomicsN4ShrinkFactor))
        if shrinkFactor > 1:
            factors = [shrinkFactor] * image.GetDimension()
            imageForN4 = sitk.Shrink(image, factors)
            maskForN4 = sitk.Shrink(mask, factors)
        else:
            imageForN4 = image
            maskForN4 = mask

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(
            [int(v) for v in self.radiomicsN4Iterations]
        )
        corrector.SetConvergenceThreshold(
            float(self.radiomicsN4ConvergenceThreshold)
        )

        try:
            _ = corrector.Execute(imageForN4, maskForN4)
        except Exception as exc:
            raise RuntimeError(
                f"N4 bias field correction fallita per {modality}: {exc}"
            ) from exc

        # Reconstruct the estimated log-bias field on the FULL original grid,
        # then correct the full-resolution image. This avoids feeding the
        # downsampled N4 image into subsequent radiomics preprocessing.
        logBiasField = corrector.GetLogBiasFieldAsImage(image)
        corrected = image / sitk.Exp(logBiasField)
        correctedKJI = sitk.GetArrayFromImage(corrected).astype(
            np.float32, copy=False
        )

        originalKJI = np.asarray(
            slicer.util.arrayFromVolume(imageNode), dtype=np.float32
        )
        if correctedKJI.shape != originalKJI.shape:
            raise RuntimeError(
                f"N4 ha modificato la shape di {modality}: "
                f"{correctedKJI.shape} vs {originalKJI.shape}."
            )

        stats = {
            "N4BiasCorrection": True,
            "N4Mask": "Shared_brainmask",
            "N4ShrinkFactor": shrinkFactor,
            "N4Iterations": str(tuple(int(v) for v in self.radiomicsN4Iterations)),
            "N4ConvergenceThreshold": float(
                self.radiomicsN4ConvergenceThreshold
            ),
        }
        return correctedKJI, stats

    def _normalizeRadiomicsArray(self, array, brainMask):
        """Skull-strip a copy, then normalize intensities inside brain only.

        The source *_original volume is never modified. Extracranial voxels are
        set to zero in the temporary radiomics copy. Mean/std are estimated on
        finite, non-zero voxels INSIDE Shared_brainmask. The z-score is clipped
        to +/- radiomicsOutlierSigma and multiplied by radiomicsNormalizeScale.
        """
        source = np.asarray(array, dtype=np.float32)
        brainMask = np.asarray(brainMask, dtype=bool)
        if source.shape != brainMask.shape:
            raise RuntimeError(
                f"Shape immagine/brain mask incompatibile: {source.shape} vs {brainMask.shape}."
            )

        finiteInsideBrain = brainMask & np.isfinite(source)
        valid = finiteInsideBrain & (source != 0)
        count = int(np.count_nonzero(valid))
        brainVoxelCount = int(np.count_nonzero(brainMask))
        if count < 2:
            raise RuntimeError(
                "Volume radiomico senza un numero sufficiente di voxel cerebrali non-zero."
            )

        values = source[valid].astype(np.float64, copy=False)
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(std) or std < 1e-8:
            raise RuntimeError(
                f"Deviazione standard non valida per normalizzazione radiomica: {std}"
            )

        normalizedValues = (values - mean) / std
        sigma = self.radiomicsOutlierSigma
        if sigma is not None:
            normalizedValues = np.clip(
                normalizedValues, -float(sigma), float(sigma)
            )
        normalizedValues *= float(self.radiomicsNormalizeScale)

        # Explicit skull stripping: the temporary radiomics image is zero
        # everywhere outside Shared_brainmask. Original MRML volumes are untouched.
        output = np.zeros(source.shape, dtype=np.float32)
        output[valid] = normalizedValues.astype(np.float32)
        output[~np.isfinite(output)] = 0.0

        stats = {
            "NormalizationMask": "Shared_brainmask",
            "SkullStrippingBeforeNormalization": True,
            "BrainMaskVoxelCount": brainVoxelCount,
            "NormalizationMean": mean,
            "NormalizationStd": std,
            "NormalizationVoxelCount": count,
            "NormalizationVoxelFractionOfBrainMask": float(
                count / max(brainVoxelCount, 1)
            ),
            "NormalizationScale": float(self.radiomicsNormalizeScale),
            "NormalizationOutlierSigma": (
                "" if sigma is None else float(sigma)
            ),
        }
        return output, stats

    def _prepareRadiomicsNormalizedNode(
        self, imageNode, modality, brainMaskNode, brainMaskKJI, workDir
    ):
        correctedKJI, n4Stats = self._n4BiasCorrectRadiomicsArray(
            imageNode, brainMaskNode, modality, workDir
        )
        normalizedKJI, stats = self._normalizeRadiomicsArray(
            correctedKJI, brainMaskKJI
        )
        stats.update(n4Stats)
        nodeName = f"__RADIOMICS_{modality}_N4_BRAIN_NORMALIZED"
        normalizedNode = self._copyVolumeToNode(
            imageNode, nodeName, normalizedKJI
        )
        return normalizedNode, stats

    def extractRadiomicsFeatures(
        self,
        progressDialog=None,
        start_value=0,
        end_value=100,
        segmentationNode=None,
    ):
        """Run the standardized radiomics pipeline.

        Only the segmentation is selectable/refinable. Image inputs are always
        T1_original, T2_original, T1CE_original and FLAIR_original.
        """
        if segmentationNode is None:
            try:
                segmentationNode = slicer.util.getNode("Segmentation_original")
            except slicer.util.MRMLNodeNotFoundException:
                raise RuntimeError(
                    "Segmentazione 'Segmentation_original' non trovata. "
                    "Seleziona o crea prima la segmentazione finale."
                )

        imageNodes = self._radiomicsOriginalNodes()
        return self.extractRadiomicsFeaturesFromNodes(
            image_nodes=imageNodes,
            segmentationNode=segmentationNode,
            progressDialog=progressDialog,
            start_value=start_value,
            end_value=end_value,
        )

    def extractRadiomicsFeaturesFromNodes(
        self,
        image_nodes,
        segmentationNode,
        progressDialog=None,
        start_value=0,
        end_value=100,
    ):
        """Compatibility entry point, but enforce the fixed *_original inputs."""
        expectedNodes = self._radiomicsOriginalNodes()
        if image_nodes is not None:
            for modality, expectedNode in expectedNodes.items():
                suppliedNode = image_nodes.get(modality)
                if suppliedNode is None or suppliedNode.GetID() != expectedNode.GetID():
                    raise RuntimeError(
                        "Pipeline radiomica standardizzata: i volumi non sono "
                        "selezionabili. Usare esclusivamente T1_original, "
                        "T2_original, T1CE_original e FLAIR_original."
                    )
        image_nodes = expectedNodes

        ownsDialog = False
        if progressDialog is None:
            progressDialog = qt.QProgressDialog(
                "Extracting standardized radiomics features...", None, 0, 100
            )
            progressDialog.setWindowTitle("Radiomics")
            progressDialog.setCancelButton(None)
            progressDialog.setMinimumWidth(350)
            progressDialog.show()
            ownsDialog = True
            slicer.app.processEvents()

        def setProgress(percent, label=None):
            value = int(start_value + (end_value - start_value) * percent / 100.0)
            progressDialog.setValue(value)
            if label:
                progressDialog.setLabelText(label)
            slicer.app.processEvents()

        try:
            setProgress(0, "Preparing standardized PyRadiomics extractor...")
            extractor = self._createRadiomicsExtractor()

            if segmentationNode is None:
                raise RuntimeError("Nessuna segmentazione selezionata.")

            segmentation = segmentationNode.GetSegmentation()
            segmentNames = []
            for index in range(segmentation.GetNumberOfSegments()):
                segmentId = segmentation.GetNthSegmentID(index)
                segment = segmentation.GetSegment(segmentId)
                if segment:
                    segmentNames.append(segment.GetName())
            if not segmentNames:
                raise RuntimeError("La segmentazione selezionata non contiene segmenti.")

            # Brain mask is defined once on the common *_original/T1CE grid and
            # reused identically for all four MRI modalities.
            brainMaskNode, brainMaskKJI = self._radiomicsBrainMaskArray(
                image_nodes["T1CE"]
            )

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safeSegName = self._safeName(segmentationNode.GetName())
            outDir = os.path.join(
                self.radiomicsOutputDir,
                f"{timestamp}_{safeSegName}_STANDARDIZED",
            )
            os.makedirs(outDir, exist_ok=True)

            if self.keepRadiomicsIntermediateFiles:
                workDir = os.path.join(outDir, "intermediate_nrrd")
                os.makedirs(workDir, exist_ok=True)
            else:
                workDir = tempfile.mkdtemp(
                    prefix="custom_inference_radiomics_",
                    dir=slicer.app.temporaryPath,
                )

            rows = []
            totalJobs = len(image_nodes) * len(segmentNames)
            completedJobs = 0

            for modality, imageNode in image_nodes.items():
                normalizedNode = None
                try:
                    setProgress(
                        5 + completedJobs / max(totalJobs, 1) * 85,
                        f"Standardizing {modality} intensities...",
                    )
                    normalizedNode, normalizationStats = (
                        self._prepareRadiomicsNormalizedNode(
                            imageNode, modality, brainMaskNode, brainMaskKJI, workDir
                        )
                    )

                    safeModality = self._safeName(modality)
                    imagePath = os.path.join(
                        workDir, f"{safeModality}_standardized.nrrd"
                    )
                    ok = slicer.util.saveNode(normalizedNode, imagePath)
                    if not ok:
                        raise RuntimeError(
                            f"Impossibile salvare il volume standardizzato {modality}: "
                            f"{imagePath}"
                        )

                    spacing = tuple(float(v) for v in imageNode.GetSpacing())
                    voxelPhysicalVolume = float(np.prod(spacing))

                    for segmentName in segmentNames:
                        completedJobs += 1
                        setProgress(
                            5 + completedJobs / max(totalJobs, 1) * 85,
                            f"Radiomics: {modality} - {segmentName}...",
                        )

                        segmentId = segmentation.GetSegmentIdBySegmentName(segmentName)
                        if not segmentId:
                            print(f"[Radiomics warning] Segmento non trovato: {segmentName}")
                            continue

                        safeSegment = self._safeName(segmentName)
                        labelmapNode = slicer.mrmlScene.AddNewNodeByClass(
                            "vtkMRMLLabelMapVolumeNode",
                            f"__RADIOMICS_MASK_{safeSegment}_{safeModality}",
                        )
                        segmentIds = vtk.vtkStringArray()
                        segmentIds.InsertNextValue(segmentId)

                        try:
                            # This also handles a manually refined segmentation:
                            # Slicer exports the chosen segment onto the exact
                            # standardized-image physical grid using labelmap logic.
                            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                                segmentationNode,
                                segmentIds,
                                labelmapNode,
                                normalizedNode,
                            )

                            maskArray = slicer.util.arrayFromVolume(labelmapNode)
                            maskVoxelCount = int(np.count_nonzero(maskArray))
                            uniqueValues = np.unique(maskArray)
                            print(
                                f"[Radiomics] {modality} - {segmentName} "
                                f"mask values={uniqueValues}, voxels={maskVoxelCount}"
                            )
                            if maskVoxelCount == 0:
                                print(
                                    f"[Radiomics warning] ROI vuota: "
                                    f"{modality} - {segmentName}. Skip."
                                )
                                continue

                            roiMask = np.asarray(maskArray) > 0
                            outsideBrainVoxels = int(
                                np.count_nonzero(roiMask & (~brainMaskKJI))
                            )
                            outsideBrainFraction = float(
                                outsideBrainVoxels / max(maskVoxelCount, 1)
                            )
                            if (
                                outsideBrainFraction
                                > float(self.radiomicsMaxROIFractionOutsideBrain)
                            ):
                                raise RuntimeError(
                                    f"ROI {segmentName} non contenuta nella Shared_brainmask: "
                                    f"{outsideBrainVoxels}/{maskVoxelCount} voxel "
                                    f"({outsideBrainFraction:.3%}) fuori brain mask. "
                                    "Controllare/refinire segmentazione o brain mask."
                                )
                            if outsideBrainVoxels:
                                print(
                                    f"[Radiomics warning] {segmentName}: "
                                    f"{outsideBrainVoxels} voxel ({outsideBrainFraction:.3%}) "
                                    "fuori Shared_brainmask, entro la tolleranza QC."
                                )

                            maskPath = os.path.join(
                                workDir,
                                f"mask_{safeSegment}_{safeModality}.nrrd",
                            )
                            ok = slicer.util.saveNode(labelmapNode, maskPath)
                            if not ok:
                                raise RuntimeError(
                                    f"Impossibile salvare la maschera {segmentName} "
                                    f"per {modality}: {maskPath}"
                                )

                            result = extractor.execute(
                                imagePath, maskPath, label=1
                            )

                            row = {
                                "Modality": modality,
                                "VolumeNodeName": imageNode.GetName(),
                                "SegmentationNodeName": segmentationNode.GetName(),
                                "Segment": segmentName,
                                "OriginalSpacing": str(spacing),
                                "RadiomicsTargetSpacing": str(
                                    tuple(self.radiomicsTargetSpacing)
                                ),
                                "MaskVoxelCountOriginalGrid": maskVoxelCount,
                                "MaskVolumeMM3OriginalGrid": (
                                    maskVoxelCount * voxelPhysicalVolume
                                ),
                                "ROIVoxelsOutsideBrainMask": outsideBrainVoxels,
                                "ROIFractionOutsideBrainMask": outsideBrainFraction,
                                "BinWidth": float(self.radiomicsBinWidth),
                                "IntermediateFilesKept": bool(self.keepRadiomicsIntermediateFiles),
                                "ImagePath": imagePath if self.keepRadiomicsIntermediateFiles else "TEMPORARY_DELETED_AFTER_EXTRACTION",
                                "MaskPath": maskPath if self.keepRadiomicsIntermediateFiles else "TEMPORARY_DELETED_AFTER_EXTRACTION",
                                **normalizationStats,
                            }

                            # Keep a compact subset of PyRadiomics diagnostics as QC.
                            diagnosticKeep = {
                                "diagnostics_Versions_PyRadiomics",
                                "diagnostics_Versions_SimpleITK",
                                "diagnostics_Image-original_Spacing",
                                "diagnostics_Image-original_Size",
                                "diagnostics_Mask-original_VoxelNum",
                                "diagnostics_Mask-original_VolumeNum",
                                "diagnostics_Mask-original_BoundingBox",
                            }
                            for key, value in result.items():
                                if key.startswith("diagnostics") and key not in diagnosticKeep:
                                    continue
                                row[key] = value

                            rows.append(row)
                            print(
                                f"[Radiomics] Estratte feature standardizzate "
                                f"per {modality} - {segmentName}"
                            )

                        except Exception as e:
                            print(
                                f"[Radiomics ERROR] {modality} - "
                                f"{segmentName}: {e}"
                            )
                        finally:
                            if labelmapNode:
                                slicer.mrmlScene.RemoveNode(labelmapNode)

                finally:
                    if normalizedNode:
                        slicer.mrmlScene.RemoveNode(normalizedNode)

            if not rows:
                raise RuntimeError(
                    "Nessuna feature radiomica estratta. Controlla ROI e geometria."
                )

            setProgress(95, "Writing XLSX...")
            try:
                from openpyxl import Workbook
            except ImportError:
                raise RuntimeError(
                    "openpyxl non è installato nell'ambiente Python di Slicer. "
                    "Installa con: slicer.util.pip_install('openpyxl')"
                )

            xlsxPath = os.path.join(outDir, "radiomics_features.xlsx")
            fieldnames = sorted(set().union(*(row.keys() for row in rows)))
            preferredColumns = [
                "Modality", "VolumeNodeName", "SegmentationNodeName", "Segment",
                "OriginalSpacing", "RadiomicsTargetSpacing",
                "MaskVoxelCountOriginalGrid", "MaskVolumeMM3OriginalGrid",
                "ROIVoxelsOutsideBrainMask", "ROIFractionOutsideBrainMask",
                "N4BiasCorrection", "N4Mask", "N4ShrinkFactor",
                "N4Iterations", "N4ConvergenceThreshold",
                "NormalizationMask", "SkullStrippingBeforeNormalization",
                "BrainMaskVoxelCount", "NormalizationMean", "NormalizationStd",
                "NormalizationVoxelCount",
                "NormalizationVoxelFractionOfBrainMask", "NormalizationScale",
                "NormalizationOutlierSigma", "BinWidth", "IntermediateFilesKept",
                "ImagePath", "MaskPath",
            ]
            orderedFieldnames = preferredColumns + [
                field for field in fieldnames if field not in preferredColumns
            ]

            def excelSafeValue(value):
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        return value.item()
                    return str(value.tolist())
                if isinstance(value, (list, tuple, dict)):
                    return str(value)
                return value

            workbook = Workbook()
            worksheet = workbook.active
            worksheet.title = "Radiomics"
            worksheet.append(orderedFieldnames)
            for row in rows:
                worksheet.append([
                    excelSafeValue(row.get(column, ""))
                    for column in orderedFieldnames
                ])

            qcSheet = workbook.create_sheet("Pipeline_QC")
            qcSheet.append(["Parameter", "Value"])
            qcSheet.append(["Image inputs", "T1_original,T2_original,T1CE_original,FLAIR_original"])
            qcSheet.append(["Segmentation", segmentationNode.GetName()])
            qcSheet.append(["Radiomics image source", "registered *_original volumes"])
            qcSheet.append(["N4 bias field correction", bool(self.radiomicsUseN4BiasCorrection)])
            qcSheet.append(["N4 mask", "Shared_brainmask"])
            qcSheet.append(["N4 shrink factor", int(self.radiomicsN4ShrinkFactor)])
            qcSheet.append(["N4 iterations", str(tuple(self.radiomicsN4Iterations))])
            qcSheet.append(["N4 convergence threshold", float(self.radiomicsN4ConvergenceThreshold)])
            qcSheet.append(["Skull stripping before normalization", True])
            qcSheet.append(["Normalization mask", "Shared_brainmask"])
            qcSheet.append(["Intensity normalization", "N4-corrected; brain-mask finite non-zero z-score; clipped; scaled"])
            qcSheet.append(["Original volumes modified", False])
            qcSheet.append(["Normalization scale", float(self.radiomicsNormalizeScale)])
            qcSheet.append(["Outlier sigma", float(self.radiomicsOutlierSigma)])
            qcSheet.append(["Target spacing mm", str(tuple(self.radiomicsTargetSpacing))])
            qcSheet.append(["Bin width", float(self.radiomicsBinWidth)])
            qcSheet.append(["PyRadiomics normalize", False])
            qcSheet.append(["PyRadiomics correctMask", False])
            qcSheet.append(["Mask export", "Slicer segment exported to each image grid"])
            qcSheet.append(["Max ROI fraction outside brain mask", float(self.radiomicsMaxROIFractionOutsideBrain)])
            qcSheet.append(["Radiomics resampling", "always enabled via PyRadiomics"])
            qcSheet.append(["Image interpolation", "sitkBSpline"])
            qcSheet.append(["Mask interpolation", "nearest-neighbor by PyRadiomics/labelmap handling"])
            qcSheet.append(["Keep intermediate NRRD files", bool(self.keepRadiomicsIntermediateFiles)])

            workbook.save(xlsxPath)
            setProgress(100, "Radiomics completed")
            print(f"[Radiomics] Feature salvate in: {xlsxPath}")
            qt.QMessageBox.information(
                None,
                "Radiomics completed",
                f"Feature radiomiche standardizzate salvate in:\n{xlsxPath}",
            )
            return xlsxPath

        finally:
            if 'workDir' in locals() and workDir and not self.keepRadiomicsIntermediateFiles:
                try:
                    shutil.rmtree(workDir, ignore_errors=False)
                    print(f"[Radiomics] File intermedi temporanei rimossi: {workDir}")
                except Exception as cleanupError:
                    print(
                        f"[Radiomics warning] Impossibile rimuovere la cartella "
                        f"temporanea {workDir}: {cleanupError}"
                    )
            if ownsDialog:
                progressDialog.close()

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
