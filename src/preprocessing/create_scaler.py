import os
import glob
import numpy as np
import pickle
import sys
from sklearn.preprocessing import StandardScaler

# Ensure we can find the project root or src if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import src.neural_network.config as config

def create_and_save_scaler(train_dir, output_path):
    print(f"Scanning {train_dir} for IMU files to fit scaler...")
    
    # Collect all IMU files from ALL classes in training set
    imu_files = []
    for cls in config.CLASS_NAMES:
        cls_path = os.path.join(train_dir, cls)
        if os.path.exists(cls_path):
            files = glob.glob(os.path.join(cls_path, "*_imu.npy"))
            imu_files.extend(files)
    
    print(f"Found {len(imu_files)} training IMU files.")
    
    if len(imu_files) == 0:
        print("Error: No IMU files found. Cannot fit scaler.")
        return

    
    # Data shape: (N_samples, 99, 10) -> to (N_samples * 99, 10)
    
    all_data = []
    for f in imu_files:
        try:
            data = np.load(f)
            # Ensure shape is correct
            if data.shape == config.INPUT_SHAPE_IMU:
                all_data.append(data)
        except Exception as e:
            print(f"Skipping corrupt file {f}: {e}")

    if not all_data:
        print("No valid data loaded.")
        return

    all_data_np = np.array(all_data) # (N, 99, 10)
    print(f"Data shape for scaling: {all_data_np.shape}")
    
    # Flatten time dimension
    N, T, F = all_data_np.shape
    flattened_data = all_data_np.reshape(N * T, F)
    
    scaler = StandardScaler()
    scaler.fit(flattened_data)
    
    print("Scaler fitted.")
    print(f"Mean: {scaler.mean_}")
    print(f"Scale: {scaler.scale_}")
    
    # Save scaler
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaler saved to {output_path}")

if __name__ == "__main__":
    train_dir = os.path.join(project_root, 'data', 'train')
    # Use config dir for scaler
    output_path = os.path.join(project_root, 'config', 'preprocessing_params.pkl')
    
    create_and_save_scaler(train_dir, output_path)
