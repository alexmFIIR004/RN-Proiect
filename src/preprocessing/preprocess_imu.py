import numpy as np
from scipy.ndimage import zoom

def preprocess_imu(imu_path, target_length=99):
    """
    Preprocesează datele IMU: încărcare, redimensionare și interpolare la lungimea țintă.
    Asigură că toate eșantioanele au forma (target_length, 10).
    """
    try:
        data = np.load(imu_path)
    except Exception:
        return None
        
    # Gestionare cazuri vechi (flattened)
    if data.ndim == 1:
        if data.shape[0] == 90:
            data = data.reshape(9, 10)
        elif data.shape[0] == 990:
            data = data.reshape(99, 10)
        else:
            # Încercare de a ghici forma dacă e multiplu de 10
            if data.shape[0] % 10 == 0:
                data = data.reshape(-1, 10)
            else:
                return None

    # Verificare și redimensionare (Interpolare)
    current_length = data.shape[0]
    
    if current_length != target_length:
        # Calculare factori de zoom pentru fiecare axă
        # Axa 0 (timp): target / current
        # Axa 1 (features): 1.0
        zoom_factor = [target_length / current_length, 1.0]
        
        # Aplicare interpolare
        data = zoom(data, zoom_factor, order=1)
        
    return data

def save_processed_imu(imu_data, output_path):
    """
    Salvează datele IMU preprocesate.
    """
    np.save(output_path, imu_data)
