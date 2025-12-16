import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
try:
    from . import config
except ImportError:
    import config

def build_floor_classifier_model(input_shape_imu=(99, 10), input_shape_img=(224, 224, 1), num_classes=5):
    """
    Construiește un model MULTI-MODAL (IMU + Imagine) pentru clasificarea suprafețelor.
    
    Args:
        input_shape_imu: Shape date senzori (99, 10)
        input_shape_img: Shape imagini (224, 224, 1)
        num_classes: Numărul de clase de ieșire (5)
        
    Returns:
        Model Keras compilat cu 2 intrări
    """
    # --- RAMURA 1: Procesare IMU (Serii de timp) ---
    input_imu = keras.Input(shape=input_shape_imu, name="imu_input")
    
    x1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(input_imu)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.Dropout(0.2)(x1)
    
    x1 = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.GlobalAveragePooling1D()(x1) # Vector caracteristici IMU
    
    # --- RAMURA 2: Procesare Imagine (Vizual) ---
    input_img = keras.Input(shape=input_shape_img, name="image_input")
    
    # CNN simplu pentru imagini (feature extractor)
    x2 = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.GlobalAveragePooling2D()(x2) # Vector caracteristici Imagine
    
    # --- FUZIUNE: Concatenare caracteristici ---
    combined = layers.concatenate([x1, x2])
    
    # Straturi Dense finale
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    outputs = layers.Dense(num_classes, activation='softmax')(z)
    
    # Definire model cu 2 intrări
    model = keras.Model(inputs=[input_imu, input_img], outputs=outputs, name="floor_classifier_multimodal")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Construire și afișare sumar model
    # Notă: Când rulăm direct scriptul, config-ul importat relativ poate da eroare, 
    # așa că folosim valori default sau un try-except pentru import.
    try:
        model = build_floor_classifier_model(
            input_shape_imu=config.INPUT_SHAPE_IMU,
            input_shape_img=config.INPUT_SHAPE_IMG
        )
    except:
        # Fallback dacă config nu e accesibil direct
        model = build_floor_classifier_model()

    model.summary()
    
    # Salvare schelet model
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/rn_floor_classifier_v0_skeleton.h5")
    print("Scheletul modelului MULTI-MODAL a fost salvat în models/rn_floor_classifier_v0_skeleton.h5")
