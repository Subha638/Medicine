import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============ PAGE CONFIG ============
st.set_page_config(page_title="🩺 Health Chatbot", layout="centered")

# ============ LOAD DATA ============
@st.cache_data
def load_data():
    try:
        symptoms_df = pd.read_csv("symptoms_df.csv")
        diets_df = pd.read_csv("diets_df.csv")
        medications_df = pd.read_csv("medications_df.csv")
        precautions_df = pd.read_csv("precautions_df.csv")
        workout_df = pd.read_csv("workout_df.csv")
        return symptoms_df, diets_df, medications_df, precautions_df, workout_df
    except FileNotFoundError:
        st.error("❌ CSV files not found! Please make sure all dataset files are in the same directory.")
        st.stop()

symptoms_df, diets_df, medications_df, precautions_df, workout_df = load_data()

# ============ PREPROCESS & TRAIN MODEL ============
symptom_cols = [col for col in symptoms_df.columns if "Symptom" in col]
all_symptoms = sorted(list(set(sum([symptoms_df[c].dropna().astype(str).tolist() for c in symptom_cols], []))))

# One-hot encode symptoms
X = pd.DataFrame(0, index=symptoms_df.index, columns=all_symptoms)
for idx, row in symptoms_df.iterrows():
    for col in symptom_cols:
        val = str(row[col]).strip()
        if val in all_symptoms:
            X.at[idx, val] = 1
y = symptoms_df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Model evaluation (for console/logging)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# ============ FUNCTIONS ============
def predict_disease(symptoms):
    """Predict disease and top-3 probable diseases"""
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for s in symptoms:
        s = str(s).strip()
        if s in all_symptoms:
            input_vector[s] = 1
    pred = clf.predict(input_vector)[0]
    probs = clf.predict_proba(input_vector)[0]
    top3 = sorted(zip(clf.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    return pred, top3

def get_recommendations(disease):
    """Return diet, medication, precautions, workout"""
    diet = diets_df[diets_df["Disease"] == disease]["Diet"].tolist() or ["No data available"]
    meds = medications_df[medications_df["Disease"] == disease]["Medication"].tolist() or ["No data available"]
    pre = precautions_df[precautions_df["Disease"] == disease].dropna(axis=1).values.flatten().tolist() or ["No data available"]
    work = workout_df[workout_df["disease"] == disease]["workout"].tolist() or ["No data available"]
    return {"Diet": diet, "Medications": meds, "Precautions": pre[:4], "Workouts": work[:4]}

# ============ STREAMLIT UI ============
st.title("🩺 AI Health Chatbot")
st.markdown("Describe your symptoms or select them below to get predictions and personalized health advice.")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("Enter your symptoms (comma separated, e.g. fever, cough, headache):")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]
    pred, top3 = predict_disease(symptoms)
    rec = get_recommendations(pred)

    msg = f"🎯 **Predicted Disease:** {pred}\n\n"
    msg += "**Top Predictions:**\n" + "\n".join([f"- {d}: {round(p,2)}" for d,p in top3])
    msg += "\n\n💡 **Diet:** " + ", ".join(rec["Diet"][:3])
    msg += "\n💊 **Medications:** " + ", ".join(rec["Medications"][:3])
    msg += "\n⚠️ **Precautions:** " + ", ".join(rec["Precautions"][:3])
    msg += "\n🏋️ **Workouts:** " + ", ".join(rec["Workouts"][:3])
    st.session_state.messages.append({"role": "assistant", "content": msg})

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ============ DROPDOWN ALTERNATIVE ============
with st.expander("🔽 Or select symptoms manually"):
    s1 = st.selectbox("Symptom 1", ["None"] + all_symptoms)
    s2 = st.selectbox("Symptom 2", ["None"] + all_symptoms)
    s3 = st.selectbox("Symptom 3", ["None"] + all_symptoms)
    s4 = st.selectbox("Symptom 4", ["None"] + all_symptoms)

    if st.button("Predict Disease"):
        selected = [s for s in [s1, s2, s3, s4] if s != "None"]
        if selected:
            pred, top3 = predict_disease(selected)
            st.success(f"🎯 Predicted Disease: {pred}")
            st.bar_chart(pd.DataFrame(top3, columns=["Disease", "Probability"]).set_index("Disease"))
            rec = get_recommendations(pred)
            st.subheader("💡 Recommendations")
            st.write("**Diet:**", rec["Diet"])
            st.write("**Medications:**", rec["Medications"])
            st.write("**Precautions:**", rec["Precautions"])
            st.write("**Workouts:**", rec["Workouts"])
        else:
            st.warning("Please select at least one symptom.")
