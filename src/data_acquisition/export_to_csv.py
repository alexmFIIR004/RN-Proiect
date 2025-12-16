import os
import numpy as np
import pandas as pd
import glob
import datetime

def export_generated_to_csv():
    """
    Exportă un subset din datele GENERATE (originale) într-un fișier CSV.
    """
    # Sursa: datele generate (originale)
    generated_dir = os.path.join("data", "generated")
    

    output_dir = os.path.join("src", "data_acquisition")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_csv = os.path.join(output_dir, "date_csv.csv")
    
    print(f"Exportare in {output_csv}...")
    
    all_data = []
    classes = ["asphalt", "concrete", "grass", "tile"] 
    
    # Luăm toate mostrele generate sau un subset
    samples_per_class = 10 
    
    for cls in classes:
        cls_dir = os.path.join(generated_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory {cls_dir} not found.")
            continue
            
        files = glob.glob(os.path.join(cls_dir, "*_imu.npy"))
        selected_files = files[:samples_per_class]
        
        for f in selected_files:
            try:
             
                data = np.load(f)
                
              
                mean_vals = np.mean(data, axis=0)
                std_vals = np.std(data, axis=0)
                
              
                row = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "label": cls, 
                    "filename": os.path.basename(f),
                    "source": "synthetic_generation"
                }
                
            
                for i, val in enumerate(mean_vals):
                    row[f"mean_feat_{i}"] = val
                    
                all_data.append(row)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"Am exportat cu succes {len(df)} in {output_csv}")
    else:
        print("No generated data found to export. Make sure you ran generate_all_data.py first.")

if __name__ == "__main__":
    export_generated_to_csv()
