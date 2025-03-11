import streamlit as st
import joblib
import numpy as np
from io import BytesIO
from transformers import T5Tokenizer, T5ForConditionalGeneration

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

    # Hardcoded locations
    location_mapping = {
        0: "New York, USA", 1: "California, USA", 2: "Tokyo, Japan",
        3: "Manila, Philippines", 4: "Sydney, Australia", 5: "London, UK"
    }

    # User selects a date & location
    selected_date = st.date_input("📅 Select a Date")
    year, month, day = selected_date.year, selected_date.month, selected_date.day
    selected_location = st.selectbox("📍 Select a Location", options=list(location_mapping.values()))
    encoded_location = [k for k, v in location_mapping.items() if v == selected_location][0]

    # Predict button
    if st.button("🔮 Predict Catastrophe"):
        X_input = np.array([[year, month, day, encoded_location]])  # Date & Location Only
        disaster_probs = xgb_model.predict_proba(X_input)
        max_prob_index = np.argmax(disaster_probs)
        probability = disaster_probs[0][max_prob_index]
        fatalities = rf_fatalities.predict(X_input)[0]
        economic_loss = rf_economic.predict(X_input)[0]

        # Format output values
        formatted_fatalities = f"{int(fatalities):,}" if fatalities > 0 else "Minimal impact"
        formatted_economic_loss = f"${economic_loss:.2f} billion"

        st.subheader("🌪️ Predicted Future Catastrophe")
        st.write(f"📆 **Date:** {selected_date}")
        st.write(f"🌍 **Location:** {selected_location}")
        st.write(f"📊 **Probability:** {probability:.2f}")
        st.write(f"💀 **Fatalities:** {formatted_fatalities}")
        st.write(f"💰 **Economic Loss:** {formatted_economic_loss}")

        # Generate LLM Insight
        with st.spinner("🧠 Thinking of novel risks..."):
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            model = T5ForConditionalGeneration.from_pretrained("t5-base")
            prompt = f"A disaster is predicted with probability {probability:.2f}. Estimated deaths: {formatted_fatalities}. Economic loss: {formatted_economic_loss}. What are some underappreciated risks?"
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, max_length=50)
            insight = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("🧠 Novel Risk Insight")
            st.write(insight)

