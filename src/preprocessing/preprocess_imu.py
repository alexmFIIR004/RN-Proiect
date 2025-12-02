import numpy as np

def preprocess_imu(imu_path):
    try:
        data = np.load(imu_path)
    except Exception:
        return None
    if data.shape == (90,):
        data = data.reshape(9, 10)
    return data

def save_processed_imu(imu_data, output_path):
    np.save(output_path, imu_data)
