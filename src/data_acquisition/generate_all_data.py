import augment_images
import generate_imu
import export_to_csv

def main():
    print("=== Pornire Pipeline Generare Date (Etapa 4) ===")
    
    # Pasul 1: Augmentare Imagini
    print("\n--- Pasul 1: Augmentare Imagini ---")
    augment_images.augment_images()
    
    # Pasul 2: Generare IMU Sintetic
    print("\n--- Pasul 2: Generare Date IMU Sintetice ---")
    generate_imu.generate_synthetic_imu()

    # Pasul 3: Export CSV (Data Logging)
    print("\n--- Pasul 3: Export CSV (Data Logging) ---")
    export_to_csv.export_generated_to_csv()
    
    print("\n=== Generare Date Completă ===")
    print("Vă rugăm să verificați data/generated/ pentru rezultate.")

if __name__ == "__main__":
    main()
