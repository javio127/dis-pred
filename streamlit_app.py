import streamlit as st
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.title("🌎 Catastrophe Prediction & Risk Insights")

# Upload trained model
st.sidebar.header("🔄 Upload ML Model")
rf_severity_file = st.sidebar.file_uploader("📂 Upload Severity Model", type=["pkl"])
if rf_severity_file:
    st.sidebar.success("✅ Model Uploaded Successfully!")
    rf_severity = joblib.load(BytesIO(rf_severity_file.read()))

# Load processed disaster data
@st.cache_data
def load_data():
    return pd.read_csv("processed_disaster_data.csv")

df = load_data()

# Disaster mapping
disaster_mapping = df.groupby(["Location", "Month"])["Disaster_Type"].agg(lambda x: x.mode()[0]).reset_index()

# User selects date & location
selected_date = st.date_input("📅 Select a Date")
selected_location = st.selectbox("🌍 Select a Location", df["Location"].unique())

# Extract date features
month, day = selected_date.month, selected_date.day

# Find most common disaster for this location & month
predicted_disaster = disaster_mapping[
    (disaster_mapping["Location"] == selected_location) & (disaster_mapping["Month"] == month)
]["Disaster_Type"].values[0]

# Compute probability based on past frequency
probability = df[
    (df["Location"] == selected_location) & (df["Month"] == month)
]["Probability"].mean()

# Prepare input for ML model
X_input = pd.DataFrame([[month, day, selected_location, predicted_disaster, probability]],
                       columns=["Month", "Day", "Location", "Disaster_Type", "Probability"])

# Convert categorical variables
X_input = pd.get_dummies(X_input, columns=["Location", "Disaster_Type"], drop_first=True)

# Ensure all feature columns exist
for col in df.columns:
    if col not in X_input.columns and col not in ["Fatalities", "Economic_Loss($)"]:
        X_input[col] = 0  # Add missing columns with default 0

# Predict severity (fatalities)
severity = rf_severity.predict(X_input)[0] if rf_severity_file else "Model not uploaded"

# Display results
st.subheader("🌪️ Predicted Future Catastrophe")
st.write(f"📆 **Date:** {selected_date}")
st.write(f"📍 **Location:** {selected_location}")
st.write(f"🌪️ **Disaster Type:** {predicted_disaster}")
st.write(f"📊 **Probability:** {probability:.2f}")
st.write(f"🔥 **Severity:** {int(severity):,} deaths" if rf_severity_file else "⚠️ Upload the model to see severity.")

# Load LLM for risk insights
@st.cache_resource
def load_llm():
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_llm()

# Generate novel risk insight
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


