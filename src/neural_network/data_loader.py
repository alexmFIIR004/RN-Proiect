import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import glob
import pickle
import config

class MultiModalDataLoader:
    """
    Helper tool pentru a încărca și converti perechile de date (Imagine + IMU)
    în format numeric (Tensors) pentru modelul AI.
    """
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.class_names = config.CLASS_NAMES
        self.class_indices = {name: i for i, name in enumerate(self.class_names)}
        self.scaler = self._load_scaler()

    def _load_scaler(self):
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'preprocessing_params.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                print(f"Loading scaler from {scaler_path}")
                return pickle.load(f)
        else:
            print(f"Warning: Scaler not found at {scaler_path}. IMU data will not be normalized.")
            return None
        
    def load_data(self, split='train'):
        """
        Încarcă datele pentru un anumit split (train/validation/test).
        Returnează:
            X_imu: array numpy (N, 99, 10)
            X_img: array numpy (N, 224, 224, 1)
            y: array numpy (N,) - etichete
        """
        split_dir = os.path.join(self.base_dir, split)
        
        X_imu_list = []
        X_img_list = []
        y_list = []
        
        print(f"Încărcare date multi-modale din {split_dir}...")
        
        for cls in self.class_names:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
                
            # Găsim fișierele IMU (folosim IMU ca ancoră pentru perechi)
            imu_files = glob.glob(os.path.join(cls_dir, "*_imu.npy"))
            
            for imu_path in imu_files:
                # 1. Încărcare IMU (deja numeric)
                try:
                    imu_data = np.load(imu_path)
                    
                    # Verificare consistență "shape"
                    if imu_data.shape != config.INPUT_SHAPE_IMU:
                        
                        continue

                  
                    if self.scaler:
                       
                        imu_data = self.scaler.transform(imu_data)
                        
                    # 2. Găsire și Conversie Imagine Pereche
                    # Numele este de tipul: asphalt_001_imu.npy -> asphalt_001_img.jpg
                    base_name = imu_path.replace("_imu.npy", "")
                    img_path = f"{base_name}_img.jpg"
                    
                    if not os.path.exists(img_path):
                  
                        continue
                        
                    # Conversie Imagine -> Numere (Tensor)
                    # load_img face resize
                    img = load_img(img_path, color_mode='grayscale', target_size=(config.INPUT_SHAPE_IMG[0], config.INPUT_SHAPE_IMG[1]))
                    img_array = img_to_array(img) # Devine (224, 224, 1) cu valori float32
                    
                    # Normalizare [0, 1]
                    img_array = img_array / 255.0
                    
                    # Adăugare în liste
                    X_imu_list.append(imu_data)
                    X_img_list.append(img_array)
                    y_list.append(self.class_indices[cls])
                    
                except Exception as e:
                    print(f"Eroare la procesarea {imu_path}: {e}")
                    continue

        # Conversie finală la numpy arrays
        X_imu = np.array(X_imu_list)
        X_img = np.array(X_img_list)
        y = np.array(y_list)
        
        print(f"  -> Încărcat {len(y)} perechi (IMU + Imagine) pentru '{split}'.")
        return [X_imu, X_img], y

if __name__ == "__main__":
    # Testare
    loader = MultiModalDataLoader(base_dir="data")
    (X_imu, X_img), y = loader.load_data('train')
    print("Shape IMU:", X_imu.shape)
    print("Shape IMG:", X_img.shape)
    print("Shape Labels:", y.shape)
