import os
import numpy as np
import glob

def calculate_stats(processed_dir):
    """
    Calculează media și deviația standard pentru fiecare clasă din data/processed.
    Returnează un dicționar: {nume_clasă: {'mean': np.array, 'std': np.array}}
    """
    stats = {}
    classes = ["asphalt", "concrete", "grass", "tile", "carpet"] # Include carpet pentru statistici dacă este disponibil
    
    print("Calculare statistici din datele existente...")
    
    for cls in classes:
        cls_dir = os.path.join(processed_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        files = glob.glob(os.path.join(cls_dir, "*_imu.npy"))
        if not files:
            continue
            
        all_data = []
        # Citește un subset pentru a economisi memorie/timp dacă este necesar.
        for f in files:
            try:
                data = np.load(f)
                all_data.append(data)
            except:
                pass
        
        if all_data:
            all_data = np.stack(all_data) # Forma: (N, 99, 10)
            # Calculează media și deviația standard peste eșantioane (axa 0)
            # Luăm media/std pe pas de timp
            mean_signal = np.mean(all_data, axis=0) # (99, 10)
            std_signal = np.std(all_data, axis=0)   # (99, 10)
            
            stats[cls] = {'mean': mean_signal, 'std': std_signal}
            print(f"Statistici calculate pentru {cls}: {len(all_data)} eșantioane.")
            
    return stats

def generate_synthetic_imu():
    processed_dir = os.path.join("data", "processed")
    generated_dir = os.path.join("data", "generated")
    
    stats = calculate_stats(processed_dir)
    
    classes = ["asphalt", "concrete", "grass", "tile"] # Clase țintă
    
    print("Generare date IMU sintetice...")
    
    for cls in classes:
        if cls not in stats:
            print(f"Avertisment: Nu există statistici pentru {cls}. Se sare peste generarea IMU.")
            continue
            
        cls_dir = os.path.join(generated_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        # Găsește toate imaginile (originale + modificate)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith("generated_")]
        
        for img_name in images:
            # Verifică dacă IMU există deja
            base_name = os.path.splitext(img_name)[0]
            imu_name = f"{base_name}_imu.npy"
            imu_path = os.path.join(cls_dir, imu_name)
            
            if os.path.exists(imu_path):
                continue
                
            # Generare date sintetice
            # Eșantionare din N(mean, std)
            mean = stats[cls]['mean']
            std = stats[cls]['std']
            
            # Generare zgomot aleator
            noise = np.random.normal(0, 1, mean.shape)
            synthetic_data = mean + (noise * std)
            
            np.save(imu_path, synthetic_data)
            
    print("Generare IMU sintetic completă.")

if __name__ == "__main__":
    generate_synthetic_imu()
