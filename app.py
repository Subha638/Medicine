import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

files_needed = {
    "symptoms_disease": "symptoms_df.csv",
    "medications": "medications.csv",
    "diets": "diets.csv",
    "precautions": "precautions_df.csv",
    "workouts": "workout_df.csv"
}

datasets = {}
for key, filename in files_needed.items():
    filepath = os.path.join(BASE_DIR, filename)
    try:
        datasets[key] = pd.read_csv(filepath)
        st.success(f"✅ Loaded {filename}")
    except Exception as e:
        st.error(f"⚠️ Could not load {filename}: {e}")
        # Optionally: fallback to a small sample or stop execution
        st.stop()

# Now you can use datasets["symptoms_disease"], etc.
