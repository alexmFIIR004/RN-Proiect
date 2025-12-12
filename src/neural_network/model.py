import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_floor_classifier_model(input_shape=(99, 10), num_classes=5):
    """
    Construiește un model CNN 1D pentru clasificarea suprafețelor de podea.
    
    Args:
        input_shape: Tuplu (pași_timp, caracteristici) ex., (99, 10)
        num_classes: Numărul de clase de ieșire (5)
        
    Returns:
        Model Keras compilat
    """
    inputs = keras.Input(shape=input_shape)
    
    # Straturi Convoluționale 1D pentru extragerea caracteristicilor din serii temporale
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Global Average Pooling pentru aplatizare
    x = layers.GlobalAveragePooling1D()(x)
    
    # Straturi Dense pentru clasificare
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="floor_classifier_v1")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Construire și afișare sumar model
    model = build_floor_classifier_model()
    model.summary()
    
    # Salvare schelet model
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/rn_floor_classifier_v0_skeleton.h5")
    print("Scheletul modelului a fost salvat în models/rn_floor_classifier_v0_skeleton.h5")
