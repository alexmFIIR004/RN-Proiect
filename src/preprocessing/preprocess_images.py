import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224), dark_min=0, light_max=255):
    """
    Preprocesează o imagine: citire, redimensionare, normalizare.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = np.clip(img, dark_min, light_max)
    img = (img - dark_min) / (light_max - dark_min)
    return np.clip(img, 0.0, 1.0)

def save_processed_image(img_array, output_path):
    """
    Salvează o imagine preprocesată (array numpy) ca fișier imagine.
    """
    img_uint8 = (img_array * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), img_uint8)
