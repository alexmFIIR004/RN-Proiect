import augment_images
import generate_imu

def main():
    print("=== Pornire Pipeline Generare Date (Etapa 4) ===")
    
    # Pasul 1: Augmentare Imagini
    print("\n--- Pasul 1: Augmentare Imagini ---")
    augment_images.augment_images()
    
    # Pasul 2: Generare IMU Sintetic
    print("\n--- Pasul 2: Generare Date IMU Sintetice ---")
    generate_imu.generate_synthetic_imu()
    
    print("\n=== Generare Date Completă ===")
    print("Vă rugăm să verificați data/generated/ pentru rezultate.")

if __name__ == "__main__":
    main()
