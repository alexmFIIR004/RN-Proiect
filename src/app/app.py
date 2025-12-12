import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import time


MODEL_PATH = "models/rn_floor_classifier_v0_skeleton.h5"
CLASS_NAMES = ['asphalt', 'carpet', 'concrete', 'grass', 'tile']


st.set_page_config(
    page_title="Surface Classifier",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Loads the NN model (cached to avoid reloading)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def main():
    st.title("Surface Classification - IMU Sensor")
    st.markdown("### Stage 4")
    
    model = load_model()
    
    if model is None:
        st.error(f"Model not found at {MODEL_PATH}. Please run src/neural_network/model.py first.")
        return

    st.sidebar.header("Input Data")
    st.sidebar.info("Input Source: Random Noise Generator (Default)")
    
    if st.button("Generate Sample & Predict"):
        with st.spinner("Acquiring Data & Processing..."):

            time.sleep(0.5)
            
  
            input_data = np.random.randn(1, 99, 10)
            
            st.subheader("1. Input Data Visualization (IMU Signals)")
  
            chart_data = pd.DataFrame(input_data[0], columns=[f"Feature_{i}" for i in range(10)])
            st.line_chart(chart_data)
            
           
            prediction = model.predict(input_data)
       

            predicted_class_idx = np.argmax(prediction)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx] * 100
            
        
            st.subheader("2. Classification Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Surface", value=predicted_class)
            with col2:
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
   
            st.subheader("3. Probability Distribution")
            prob_df = pd.DataFrame(prediction.T, index=CLASS_NAMES, columns=["Probability"])
            st.bar_chart(prob_df)
            

if __name__ == "__main__":
    main()
