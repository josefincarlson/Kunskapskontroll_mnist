import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Laddar modellen
@st.cache_mnistdatamodel
def load_data():
    return joblib.load("svc_mnist.joblib")

model = load_data

st.title("Skriv en siffra mellan 0–9 och få en prediction")

# Fasta canvas-inställningar
canvas_size = 280
stroke_width = 32