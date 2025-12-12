# Configurare pentru Modulul Rețelei Neurale

# Parametri date
INPUT_SHAPE = (99, 10)
NUM_CLASSES = 5
CLASS_NAMES = ['asphalt', 'carpet', 'concrete', 'grass', 'tile']

# Parametri antrenare
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Căi model
MODEL_SAVE_PATH = "models/rn_floor_classifier_v1.h5"
