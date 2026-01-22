import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import time
import glob
import pickle
import sys

# Add project root to path for config access if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

MODEL_PATH = "models/optimized_model.h5"  # Loaded optimized model from Stage 6
SCALER_PATH = "config/preprocessing_params.pkl"
DATA_DIR = os.path.join(project_root, "data", "test") # Use test data for demo
CLASS_NAMES = ['asphalt', 'carpet', 'concrete', 'grass', 'tile']

st.set_page_config(
    page_title="Surface Classifier",
    layout="wide"
)

@st.cache_resource
def load_resources():
    """Loads the NN model and Scaler."""
    resources = {}
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            resources['model'] = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model not found at {MODEL_PATH}")
        return None

    # Load Scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            resources['scaler'] = pickle.load(f)
    else:
        st.warning(f"Scaler not found at {SCALER_PATH}. Using raw IMU data.")
        resources['scaler'] = None
        
    return resources

def get_random_test_sample():
    """Picks a random sample from the test set."""
    if not os.path.exists(DATA_DIR):
        return None, None, "Data directory not found"
        
    # Get all IMU files
    all_imu_files = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir):
            files = glob.glob(os.path.join(cls_dir, "*_imu.npy"))
            all_imu_files.extend([(f, cls) for f in files])
            
    if not all_imu_files:
        return None, None, "No files found in test set"
        
    # Pick random
    imu_path, true_label = all_imu_files[np.random.randint(len(all_imu_files))]
    base_name = imu_path.replace("_imu.npy", "")
    img_path = f"{base_name}_img.jpg"
    
    if not os.path.exists(img_path):
        # Fallback to just loading whatever
        return get_random_test_sample() # Try again (risky recursion, but low prob of failure)

    return imu_path, img_path, true_label

def preprocess_sample(imu_path, img_path, scaler):
    # Load IMU
    imu_data = np.load(imu_path)
    # Ensure shape (99, 10)
    if imu_data.shape != (99, 10):
        # Allow basic reshape if possible
        if imu_data.size == 990:
             imu_data = imu_data.reshape(99, 10)
    
    # Scale IMU
    if scaler:
        imu_data = scaler.transform(imu_data)
        
    # Add batch dim -> (1, 99, 10)
    imu_input = np.expand_dims(imu_data, axis=0)
    
    # Load Image
    img = tf.keras.utils.load_img(img_path, color_mode='grayscale', target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    return imu_input, img_input

def main():
    st.title("Surface Classification - Inference")
    st.markdown("Trained Model Inference")
    
    resources = load_resources()
    if not resources:
        st.stop()
        
    model = resources['model']
    scaler = resources['scaler']

    st.sidebar.header("Control Panel")
    if st.sidebar.button("Load Random Test Sample"):
        with st.spinner("Loading and Processing..."):
            
            imu_path, img_path, true_label = get_random_test_sample()
            
            if imu_path:
                st.success(f"Loaded sample: {os.path.basename(imu_path)}")
                st.info(f"True Label: **{true_label}**")
                
                # Preprocess
                input_imu, input_img = preprocess_sample(imu_path, img_path, scaler)
                
                # Visualization
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    st.markdown("**IMU Signal (Scaled)**")
                    st.line_chart(pd.DataFrame(input_imu[0], columns=[f"Ax{i}" for i in range(10)]))
                
                with col_viz2:
                    st.markdown("**Surface Image**")
                    st.image(img_path, caption=f"Ground Truth: {true_label}")
                
                # Inference
                start_time = time.time()
                prediction = model.predict([input_imu, input_img])
                inf_time = (time.time() - start_time) * 1000
                
                predicted_idx = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence = prediction[0][predicted_idx] * 100
                
                # Results
                st.markdown("---")
                st.subheader("Classification Results")
                
                # --- STATE MACHINE LOGIC: THRESHOLD CHECK ---
                CONFIDENCE_THRESHOLD = 70.0  # Defined in State Machine logic
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted", predicted_class, delta=None)
                
                # Display Confidence with visual indicator
                if confidence < CONFIDENCE_THRESHOLD:
                    c2.metric("Confidence", f"{confidence:.1f}%", delta="-Low Confidence", delta_color="inverse")
                else:
                    c2.metric("Confidence", f"{confidence:.1f}%", delta="High Confidence")
                    
                c3.metric("Inference Time", f"{inf_time:.1f} ms")
                
                # --- STATE MACHINE LOGIC: ACT vs LOG ---
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f" **UNCERTAIN PREDICTION (<{CONFIDENCE_THRESHOLD}%)**.")
                else:
                    if predicted_class == true_label:
                        st.success(f" **CONFIRMED ({confidence:.1f}%)**.`INFERENCE` → `ACT`")
                    else:
                        st.error(f" Prediction Incorrect. Expected {true_label}")
                # ---------------------------------------------
                
                # Probabilities
                st.bar_chart(pd.DataFrame(prediction.T, index=CLASS_NAMES, columns=["Probability"]))
                
            else:
                st.warning("Could not load a valid sample from data/test. Check dataset structure.")
    
    else:
        st.write("Click the button in the sidebar to load and classify a random sample from the test set.")

if __name__ == "__main__":
    main()
