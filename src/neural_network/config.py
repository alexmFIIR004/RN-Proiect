# Configurare pentru Modulul Rețelei Neurale

# Parametri date
INPUT_SHAPE_IMU = (99, 10)
INPUT_SHAPE_IMG = (224, 224, 1) # Grayscale
NUM_CLASSES = 5
CLASS_NAMES = ['asphalt', 'carpet', 'concrete', 'grass', 'tile']

# Parametri antrenare
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Căi model
MODEL_SAVE_PATH = "models/rn_floor_classifier_v1.h5"
