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
            
            # Generare date dummy pentru ambele intrări
            input_imu = np.random.randn(1, 99, 10)
            input_img = np.random.rand(1, 224, 224, 1) # Imagine dummy
            
            st.subheader("1. Input Data Visualization")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("**IMU Signals**")
                chart_data = pd.DataFrame(input_imu[0], columns=[f"Feature_{i}" for i in range(10)])
                st.line_chart(chart_data)
                
            with col_viz2:
                st.markdown("**Camera View (Simulated)**")
                st.image(input_img[0, :, :, 0], caption="Simulated Surface Image", clamp=True)
            
            # Predicție cu modelul multi-modal (listă de input-uri)
            try:
                prediction = model.predict([input_imu, input_img])
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()

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
