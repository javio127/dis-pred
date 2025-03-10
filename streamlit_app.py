import os
import streamlit as st
import joblib
import numpy as np
from io import BytesIO
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 🚀 Ensure necessary dependencies are installed
os.system("pip install torch sentencepiece protobuf --quiet")

st.title("🌎 Catastrophe Prediction & Risk Insights")

# Upload model files
st.sidebar.header("🔄 Upload Model Files")
xgb_file = st.sidebar.file_uploader("📂 Upload Disaster Model", type=["pkl"])
rf_fatalities_file = st.sidebar.file_uploader("📂 Upload Fatalities Model", type=["pkl"])
rf_economic_file = st.sidebar.file_uploader("📂 Upload Economic Model", type=["pkl"])

if xgb_file and rf_fatalities_file and rf_economic_file:
    st.sidebar.success("✅ Models Uploaded Successfully!")

    # Load models from uploaded files
    xgb_model = joblib.load(BytesIO(xgb_file.read()))
    rf_fatalities = joblib.load(BytesIO(rf_fatalities_file.read()))
    rf_economic = joblib.load(BytesIO(rf_economic_file.read()))

    # Load LLM model for risk insights
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Disaster mapping
    disaster_mapping = {0: "Earthquake", 1: "Flood", 2: "Hurricane", 3: "Wildfire", 4: "Tornado"}

    # User selects a date
    selected_date = st.date_input("📅 Select a Date")
    year, month, day = selected_date.year, selected_date.month, selected_date.day
    magnitude = st.slider("🌋 Disaster Magnitude (1-10)", min_value=1.0, max_value=10.0, step=0.1)

    # Predict button
    if st.button("🔮 Predict Catastrophe"):
        X_input = np.array([[year, month, day, magnitude, 0]])  # Dummy location ID
        disaster_probs = xgb_model.predict_proba(X_input)
        max_prob_index = np.argmax(disaster_probs)
        predicted_disaster = disaster_mapping[max_prob_index]
        probability = disaster_probs[0][max_prob_index]
        fatalities = rf_fatalities.predict(X_input)[0]
        economic_loss = rf_economic.predict(X_input)[0]

        # Format output values for readability
        formatted_fatalities = f"{int(fatalities):,}" if fatalities > 0 else "Minimal impact"
        formatted_economic_loss = f"${economic_loss:.2f} billion"

        st.subheader("🌪️ Predicted Future Catastrophe")
        st.write(f"📆 **Date:** {selected_date}")
        st.write(f"🌪️ **Disaster Type:** {predicted_disaster}")
        st.write(f"📊 **Probability:** {probability:.2f}")
        st.write(f"💀 **Fatalities:** {formatted_fatalities}")
        st.write(f"💰 **Economic Loss:** {formatted_economic_loss}")

        # Generate LLM insight
        with st.spinner("🧠 Thinking of novel risks..."):
            prompt = (f"A {predicted_disaster} is predicted with a probability of {probability:.2f}. "
                      f"It is estimated to cause {formatted_fatalities} fatalities and {formatted_economic_loss} in economic loss. "
                      "What are some underappreciated risks or consequences of this disaster?")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, max_length=50)
            insight = tokenizer.decode(output[0], skip_special_tokens=True)

            # Fix Formatting
            formatted_insight = insight.replace(".", ". ").replace("  ", " ")

            st.subheader("🧠 Novel Risk Insight")
            st.write(formatted_insight)
