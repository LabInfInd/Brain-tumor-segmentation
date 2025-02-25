# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:58:31 2025

@author: Utente
"""
import streamlit as st
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, Resized, NormalizeIntensityd, AdjustContrastd
from monai.data import Dataset, DataLoader, decollate_batch

def create_test_loader(uploaded_files):
    # DataLoader MONAI dai file caricati dall'utente
    sequence_types = ["T1CE", "T1", "T2", "FLAIR"]
    selected_files = {}

    # Mapping dei file ai tipi di sequenza
    for file in uploaded_files:
        selected_type = st.selectbox(f"Seleziona il tipo per {file.name}", sequence_types, key=file.name)
        selected_files[selected_type] = file

   
    if set(selected_files.keys()) != set(sequence_types):
        st.warning("Assegna correttamente tutti i file NIfTI ai loro tipi.")
        return None

    # Definizione delle trasformazioni
    val_transform = Compose([
        LoadImaged(keys=["t1ce", "t1", "t2", "flair"]),
        EnsureChannelFirstd(keys=["t1ce", "t1", "t2", "flair"]),
        EnsureTyped(keys=["t1ce", "t1", "t2", "flair"]),
        Orientationd(keys=["t1ce", "t1", "t2", "flair"], axcodes="RAS"),
        Resized(keys=["t1ce", "t1", "t2", "flair"], spatial_size=(192, 192, 150), mode="trilinear", align_corners=True),
        NormalizeIntensityd(keys=["t1ce", "t1", "t2", "flair"], nonzero=True, channel_wise=True),
        AdjustContrastd(keys=["t2", "flair"], gamma=1.1)
    ])

    # Dizionario con i file caricati 
    datalist = {
        "inference": [
            {
                "t1ce": selected_files["T1CE"],
                "t1": selected_files["T1"],
                "t2": selected_files["T2"],
                "flair": selected_files["FLAIR"]
            }
        ]
    }

    # Dataset e DataLoader
    test_dataset = Dataset(data=datalist["inference"], transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader
