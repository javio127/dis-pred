import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.title("🌎 Catastrophe Prediction & Risk Insights")

# Upload processed dataset
st.sidebar.header("📂 Upload Files")
data_file = st.sidebar.file_uploader("📂 Upload Processed Disaster Data (CSV)", type=["csv"])
rf_severity_file = st.sidebar.file_uploader("📂 Upload Severity Prediction Model (PKL)", type=["pkl"])

# Load dataset if uploaded
if data_file:
    df = pd.read_csv(data_file)
    st.sidebar.success("✅ Dataset uploaded successfully!")
else:
    df = None

# Load trained ML model if uploaded
if rf_severity_file:
    rf_severity = joblib.load(BytesIO(rf_severity_file.read()))
    st.sidebar.success("✅ ML Model Uploaded Successfully!")

    # Extract feature names from the trained model
    FEATURE_SET = list(rf_severity.feature_names_in_)
else:
    rf_severity = None
    FEATURE_SET = []

# Load LLM for novel risk insights
@st.cache_resource
def load_llm():
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_llm()

# Function to get most frequent disaster type for a location & month
def get_most_frequent_disaster(location, month):
    if df is not None:
        subset = df[(df["Location"] == location) & (df["Month"] == month)]
        if not subset.empty:
            return subset["Disaster_Type"].mode()[0]  # Most frequent disaster type
    return "Unknown"

# Function to compute disaster probability
def compute_probability(location, month):
    if df is not None:
        subset = df[(df["Location"] == location) & (df["Month"] == month)]
        return len(subset) / len(df)  # Normalize between 0-1
    return 0  # If no data is available, default to 0

# User selects date & location
selected_date = st.date_input("📅 Select a Date")
if df is not None:
    selected_location = st.selectbox("🌍 Select a Location", df["Location"].unique())

    # Extract month and day
    month, day = selected_date.month, selected_date.day

    # Predict disaster type and probability
    predicted_disaster = get_most_frequent_disaster(selected_location, month)
    probability = compute_probability(selected_location, month)

    # Prepare input for ML model (if uploaded)
    if rf_severity is not None:
        X_input = pd.DataFrame([[month, day, probability, selected_location, predicted_disaster]],
                               columns=["Month", "Day", "Probability", "Location", "Disaster_Type"])

        # One-hot encode categorical variables
        X_input = pd.get_dummies(X_input, columns=["Location", "Disaster_Type"])

        # Ensure all expected features exist in X_input
        for col in FEATURE_SET:
            if col not in X_input.columns:
                X_input[col] = 0  # Add missing features with default value 0

        # Align X_input columns with model expectations
        X_input = X_input[FEATURE_SET]

        # Predict severity (fatalities)
        severity = rf_severity.predict(X_input)[0]
    else:
        severity = "⚠️ Upload ML model to see severity."

    # Display results
    st.subheader("🌪️ Predicted Future Catastrophe")
    st.write(f"📆 **Date:** {selected_date}")
    st.write(f"📍 **Location:** {selected_location}")
    st.write(f"🌪️ **Disaster Type:** {predicted_disaster}")
    st.write(f"📊 **Probability:** {probability:.2f}")
    st.write(f"🔥 **Severity:** {int(severity):,} deaths" if rf_severity is not None else severity)

    # Generate novel risk insight using LLM
    if st.button("🧠 Generate Novel Risk Insight"):
        with st.spinner("🧠 Thinking of novel risks..."):
            prompt = (f"A {predicted_disaster} is predicted in {selected_location} with probability {probability:.2f}. "
                      f"It is estimated to cause {int(severity):,} deaths. "
                      "What are some underappreciated risks or consequences of this disaster?")
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, max_length=50)
            insight = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("🧠 Novel Risk Insight")
            st.write(insight)


