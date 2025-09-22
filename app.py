import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

# === Load datasets ===
BASE_DIR = r"C:\Users\subha\Downloads\medicine"
datasets = {}

files_needed = {
    "symptoms_disease": "symptoms_df.csv",
    "medications": "medications.csv",
    "diets": "diets.csv",
    "precautions": "precautions_df.csv",
    "workouts": "workout_df.csv"
}

for key, filename in files_needed.items():
    try:
        datasets[key] = pd.read_csv(f"{BASE_DIR}\\{filename}")
    except:
        st.warning(f"⚠️ {filename} not found")

# === Preprocessing ===
df = datasets["symptoms_disease"].drop_duplicates().fillna("")
le = LabelEncoder()
df["Disease_encoded"] = le.fit_transform(df["Disease"])

symptom_cols = [col for col in df.columns if "Symptom" in col]
df["symptom_text"] = df[symptom_cols].astype(str).apply(lambda x: " ".join(x), axis=1)

tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(df["symptom_text"]).toarray()
df["char_len"] = df["symptom_text"].apply(len)
df["word_count"] = df["symptom_text"].apply(lambda x: len(x.split()))
stats = df[["char_len","word_count"]].values
X_features = np.hstack([X_tfidf, stats])

pca = PCA(n_components=90)
X_fused = pca.fit_transform(X_features)

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_fused, df["Disease_encoded"])

# Train a simple model
model = LogisticRegression(max_iter=2000)
model.fit(X_selected, df["Disease_encoded"])

# === Functions ===
def get_recommendations(disease):
    meds = datasets["medications"].loc[datasets["medications"]["Disease"] == disease, "Medication"].tolist()
    diets = datasets["diets"].loc[datasets["diets"]["Disease"] == disease, "Diet"].tolist()
    prec = datasets["precautions"].loc[datasets["precautions"]["Disease"] == disease, "Precaution"].tolist()
    workouts = datasets["workouts"].loc[datasets["workouts"]["Disease"] == disease, "Workout"].tolist()
    return meds, diets, prec, workouts

def chatbot_response(user_input):
    X_input = tfidf.transform([user_input]).toarray()
    stats_input = np.array([[len(user_input), len(user_input.split())]])
    X_input_full = np.hstack([X_input, stats_input])
    X_input_pca = pca.transform(X_input_full)
    X_input_sel = selector.transform(X_input_pca)

    pred = model.predict(X_input_sel)[0]
    disease = le.inverse_transform([pred])[0]

    meds, diets, prec, workouts = get_recommendations(disease)
    response = {
        "disease": disease,
        "medications": meds,
        "diets": diets,
        "precautions": prec,
        "workouts": workouts
    }
    return response

# === Streamlit UI ===
st.title("🩺 Medical Chatbot")
st.write("Enter your symptoms and get disease prediction with recommendations.")

user_input = st.text_area("Describe your symptoms here:")

if st.button("Get Recommendation"):
    if user_input.strip():
        result = chatbot_response(user_input)
        st.success(f"**Predicted Disease:** {result['disease']}")
        st.write("💊 **Medications:**", ", ".join(result['medications']) if result['medications'] else "N/A")
        st.write("🥗 **Diets:**", ", ".join(result['diets']) if result['diets'] else "N/A")
        st.write("⚠️ **Precautions:**", ", ".join(result['precautions']) if result['precautions'] else "N/A")
        st.write("🏋️ **Workouts:**", ", ".join(result['workouts']) if result['workouts'] else "N/A")
    else:
        st.warning("Please enter symptoms.")
