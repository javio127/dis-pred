import streamlit as st
import joblib
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load trained models
xgb_model = joblib.load("xgb_disaster_model.pkl")
rf_fatalities = joblib.load("rf_fatalities_model.pkl")
rf_economic = joblib.load("rf_economic_model.pkl")

# Load LLM for novel risk generation
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Disaster mapping
disaster_mapping = {0: "Earthquake", 1: "Flood", 2: "Hurricane", 3: "Wildfire", 4: "Tornado"}

# Streamlit UI
st.title("🌎 Catastrophe Prediction & Risk Insights")

# User selects a date
selected_date = st.date_input("📅 Select a Date")

# Convert date to features
year, month, day = selected_date.year, selected_date.month, selected_date.day
magnitude = st.slider("🌋 Disaster Magnitude (1-10)", min_value=1.0, max_value=10.0, step=0.1)

# User clicks predict
if st.button("🔮 Predict Catastrophe"):
    # Convert user inputs into features
    X_input = np.array([[year, month, day, magnitude, 0]])  # Dummy location ID

    # Make predictions
    disaster_probs = xgb_model.predict_proba(X_input)
    max_prob_index = np.argmax(disaster_probs)
    predicted_disaster = disaster_mapping[max_prob_index]
    probability = disaster_probs[0][max_prob_index]

    fatalities = rf_fatalities.predict(X_input)[0]
    economic_loss = rf_economic.predict(X_input)[0]

    # Display Predictions
    st.subheader("🌪️ Predicted Future Catastrophe")
    st.write(f"📆 **Date:** {selected_date}")
    st.write(f"🌪️ **Disaster Type:** {predicted_disaster}")
    st.write(f"📊 **Probability:** {probability:.2f}")
    st.write(f"💀 **Fatalities:** {int(fatalities):,} deaths")
    st.write(f"💰 **Economic Loss:** ${economic_loss:.2f} billion")

    # Generate novel risk insight using LLM
    with st.spinner("🧠 Thinking of novel risks..."):
        prompt = (f"A {predicted_disaster} is predicted with a probability of {probability:.2f}. "
                  f"It is estimated to cause {int(fatalities):,} deaths and ${economic_loss:.2f} billion in economic loss. "
                  "What are some underappreciated risks or consequences of this disaster?")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_length=50)
        insight = tokenizer.decode(output[0], skip_special_tokens=True)

        st.subheader("🧠 Novel Risk Insight")
        st.write(insight)
