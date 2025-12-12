import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def generate_evidence():
    """
    Generează:
    1. docs/generated_vs_real.png: Grafic comparativ Semnal Real vs Generat.
    2. docs/data_statistics.csv: Tabel statistic comparativ.
    """
    
    # Căi fișiere
    real_dir = os.path.join("data", "processed")
    generated_dir = os.path.join("data", "generated")
    docs_dir = "docs"
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
    # 1. Generare Grafic Comparativ (Real vs Generat)
    print("Se generează graficul comparativ Real vs Generat...")
    
    # Selectăm o clasă pentru comparație
    target_class = "concrete"
    
    # Găsim o mostră reală
    real_files = glob.glob(os.path.join(real_dir, target_class, "*_imu.npy"))
    # Găsim o mostră generată
    gen_files = glob.glob(os.path.join(generated_dir, target_class, "*_imu.npy"))
    
    if real_files and gen_files:
        real_data = np.load(real_files[0])
        gen_data = np.load(gen_files[0])
        
        # Asigurăm aceeași lungime pentru plotare
        min_len = min(len(real_data), len(gen_data))
        
        plt.figure(figsize=(12, 6))
        
        # Plot Real (Acc Z - de obicei index 2 sau similar, plotăm index 0 pentru comparație generică)
        plt.subplot(1, 2, 1)
        plt.plot(real_data[:min_len, 0], label='Real', color='blue')
        plt.title(f"Date Reale ({target_class})")
        plt.xlabel("Timp")
        plt.ylabel("Amplitudine")
        plt.grid(True)
        
        # Plot Generat
        plt.subplot(1, 2, 2)
        plt.plot(gen_data[:min_len, 0], label='Generat', color='orange')
        plt.title(f"Date Generate ({target_class})")
        plt.xlabel("Timp")
        plt.grid(True)
        
        plt.suptitle(f"Comparație: Semnal Real vs Generat ({target_class})")
        plt.tight_layout()
        
        plot_path = os.path.join(docs_dir, "generated_vs_real.png")
        plt.savefig(plot_path)
        print(f"Graficul a fost salvat în {plot_path}")
    else:
        print("Nu s-au găsit mostre pentru comparație.")

    # 2. Generare Statistici CSV
    print("Se generează CSV-ul cu statistici...")
    stats_data = []
    
    classes = ["asphalt", "concrete", "grass", "tile"]
    
    for cls in classes:
        # Statistici Reale
        r_files = glob.glob(os.path.join(real_dir, cls, "*_imu.npy"))
        if r_files:
            # Încărcăm câteva fișiere pentru a obține statistici medii
            r_data_list = [np.load(f) for f in r_files[:20]] 
            r_concat = np.concatenate(r_data_list, axis=0)
            r_mean = np.mean(r_concat)
            r_std = np.std(r_concat)
        else:
            r_mean, r_std = 0, 0
            
        # Statistici Generate
        g_files = glob.glob(os.path.join(generated_dir, cls, "*_imu.npy"))
        if g_files:
            g_data_list = [np.load(f) for f in g_files[:20]]
            g_concat = np.concatenate(g_data_list, axis=0)
            g_mean = np.mean(g_concat)
            g_std = np.std(g_concat)
        else:
            g_mean, g_std = 0, 0
            
        stats_data.append({
            "Class": cls,
            "Real_Mean": r_mean,
            "Gen_Mean": g_mean,
            "Real_Std": r_std,
            "Gen_Std": g_std
        })
        
    df = pd.DataFrame(stats_data)
    csv_path = os.path.join(docs_dir, "data_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Statisticile au fost salvate în {csv_path}")

if __name__ == "__main__":
    generate_evidence()
