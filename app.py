import streamlit as st
import pandas as pd

# -----------------------------
# Load data from CSVs
# -----------------------------
@st.cache_data
def load_data():
    symptoms_df = pd.read_csv("https://raw.githubusercontent.com/Subha638/Medicine/main/symptoms_df.csv")
    medications_df = pd.read_csv("https://raw.githubusercontent.com/Subha638/Medicine/main/medications.csv")
    diets_df = pd.read_csv("https://raw.githubusercontent.com/Subha638/Medicine/main/diets.csv")
    precautions_df = pd.read_csv("https://raw.githubusercontent.com/Subha638/Medicine/main/precautions_df.csv")
    workout_df = pd.read_csv("https://raw.githubusercontent.com/Subha638/Medicine/main/workout_df.csv")
    return symptoms_df, medications_df, diets_df, precautions_df, workout_df

symptoms_df, medications_df, diets_df, precautions_df, workout_df = load_data()

# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Chatbot response logic
# -----------------------------
def get_bot_response(user_input):
    user_input_lower = user_input.lower()
    
    # Check symptoms
    symptom_matches = symptoms_df[symptoms_df['Symptom'].str.lower().str.contains(user_input_lower)]
    if not symptom_matches.empty:
        return f"Symptoms found: {', '.join(symptom_matches['Symptom'].tolist())}"
    
    # Check medications
    med_matches = medications_df[medications_df['Name'].str.lower().str.contains(user_input_lower)]
    if not med_matches.empty:
        return f"Medication info: {med_matches.iloc[0]['Name']} - {med_matches.iloc[0]['Usage']}"
    
    # Check diets
    diet_matches = diets_df[diets_df['Diet'].str.lower().str.contains(user_input_lower)]
    if not diet_matches.empty:
        return f"Diet info: {diet_matches.iloc[0]['Diet']} - {diet_matches.iloc[0]['Benefits']}"
    
    # Check precautions
    precaution_matches = precautions_df[precautions_df['Precaution'].str.lower().str.contains(user_input_lower)]
    if not precaution_matches.empty:
        return f"Precaution: {precaution_matches.iloc[0]['Precaution']}"
    
    # Check workouts
    workout_matches = workout_df[workout_df['Workout'].str.lower().str.contains(user_input_lower)]
    if not workout_matches.empty:
        return f"Workout: {workout_matches.iloc[0]['Workout']} - {workout_matches.iloc[0]['Instructions']}"
    
    return "Sorry, I couldn't find any information on that. Please try something else."

# -----------------------------
# Streamlit app UI
# -----------------------------
st.set_page_config(page_title="Health Chatbot", page_icon="💊")

st.title("💊 Health Chatbot")
st.write("Ask about symptoms, medications, diets, precautions, or workouts.")

# Input box
user_input = st.text_input("You:", "")

if user_input:
    response = get_bot_response(user_input)
    st.session_state.messages.append({"user": user_input, "bot": response})

# Display chat history
for msg in st.session_state.messages:
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**Bot:** {msg['bot']}")
    st.markdown("---")
