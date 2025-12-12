import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def restructure_dataset():
    base_dir = os.path.abspath("data")
    processed_dir = os.path.join(base_dir, "processed")
    raw_dir = os.path.join(base_dir, "raw")
    generated_dir = os.path.join(base_dir, "generated")
    
    split_dirs = {
        "train": os.path.join(base_dir, "train"),
        "validation": os.path.join(base_dir, "validation"),
        "test": os.path.join(base_dir, "test")
    }
    
    classes = ["asphalt", "carpet", "concrete", "grass", "tile"]
    
    print("=== Pornire Restructurare Set de Date ===")
    
    # 1. Curățare foldere train/validation/test
    print("\n[1] Curățare foldere train/validation/test...")
    for name, path in split_dirs.items():
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for cls in classes:
            os.makedirs(os.path.join(path, cls))
            
    # 2. Reducere Set de Date Public ( 120 eșantioane/clasă)
    print("\n[2] Reducere Set de Date Public ( 120 eșantioane/clasă)...")
    for cls in classes:
        cls_processed_dir = os.path.join(processed_dir, cls)
        cls_raw_dir = os.path.join(raw_dir, cls)
        
        if not os.path.exists(cls_processed_dir):
            continue
            
        # Obține toate eșantioanele publice ( formatul [clasă]_[număr]_img.jpg)
        # Filtrăm "generated_"
        all_files = sorted(os.listdir(cls_processed_dir))
        public_images = [f for f in all_files if f.endswith("_img.jpg") and not f.startswith("generated_")]
        
        # Sortare după număr
        # asphalt_0000001_img.jpg
        public_images.sort()
        
        # Vrem să păstrăm 120.
        samples_to_keep = 120
        
        if len(public_images) <= samples_to_keep:
            print(f"  {cls}: Găsit {len(public_images)} eșantioane publice. Nu este necesară reducerea.")
        else:
            # Logica de ștergere comentată pentru a preveni pierderea accidentală a datelor la rulări repetate
        
            print(f"  {cls}: Găsit {len(public_images)} eșantioane. Logica de reducere dezactivată pentru siguranță.")
            """
            to_delete = public_images[samples_to_keep:]
            print(f"  {cls}: Se șterg {len(to_delete)} eșantioane (se păstrează primele {samples_to_keep}).")
            
            for img_file in to_delete:
                # Construire nume fișiere
                base_name = img_file.replace("_img.jpg", "")
                imu_file = f"{base_name}_imu.npy"
                
                # Ștergere din processed
                try:
                    os.remove(os.path.join(cls_processed_dir, img_file))
                    os.remove(os.path.join(cls_processed_dir, imu_file))
                except OSError as e:
                    print(f"    Eroare la ștergerea din processed: {e}")
                    
                # Ștergere din raw (dacă există)
                # Raw ar putea avea denumiri diferite sau doar imaginea? 
                # Presupunem că raw are aceeași structură sau încercăm doar să ștergem imaginea potrivită
                if os.path.exists(cls_raw_dir):
                    raw_img_path = os.path.join(cls_raw_dir, img_file)
                    if os.path.exists(raw_img_path):
                        os.remove(raw_img_path)
            """
                    

    # 3. Îmbinare Date Generate în Processed
    print("\n[3] Îmbinare Date Generate în Processed...")
    # Clasele generate sunt doar 4 (fără carpet)
    gen_classes = ["asphalt", "concrete", "grass", "tile"]
    
    for cls in gen_classes:
        cls_gen_dir = os.path.join(generated_dir, cls)
        cls_processed_dir = os.path.join(processed_dir, cls)
        
        if not os.path.exists(cls_gen_dir):
            continue
            
        # Obține fișiere generate
        gen_files = [f for f in os.listdir(cls_gen_dir) if f.startswith("generated_")]
        
        print(f"  {cls}: Copiere {len(gen_files)} fișiere din generated în processed.")
        
        for f in gen_files:
            src = os.path.join(cls_gen_dir, f)
            dst = os.path.join(cls_processed_dir, f)
            shutil.copy2(src, dst)

    # 4. Re-împărțire Set de Date (Train/Val/Test)
    print("\n[4] Re-împărțire Set de Date (Train/Val/Test)...")
    # Vrem să împărțim TOATE datele din processed acum.
    # Total așteptat: 1000 eșantioane.
    
    total_samples = 0
    
    for cls in classes:
        cls_processed_dir = os.path.join(processed_dir, cls)
        
        # Obține toate perechile (img + imu)
        all_images = [f for f in os.listdir(cls_processed_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

        valid_samples = []
        for img in all_images:
            base = os.path.splitext(img)[0]
            # Gestionare convenții de denumire
            # Public: asphalt_0000001_img.jpg -> asphalt_0000001_imu.npy
            # Generat: generated_asphalt_001_original.jpg -> generated_asphalt_001_original_imu.npy
            
            if "_img" in base:
                imu_name = base.replace("_img", "_imu") + ".npy"
            else:
                imu_name = base + "_imu.npy"
                
            if os.path.exists(os.path.join(cls_processed_dir, imu_name)):
                valid_samples.append(img)
        
        print(f"  {cls}: Găsit {len(valid_samples)} eșantioane totale.")
        total_samples += len(valid_samples)
        
        # Împărțire
        # 70% Train, 15% Val, 15% Test
        train_imgs, test_val_imgs = train_test_split(valid_samples, test_size=0.3, random_state=42, shuffle=True)
        val_imgs, test_imgs = train_test_split(test_val_imgs, test_size=0.5, random_state=42, shuffle=True)
        
        # Funcție ajutătoare pentru copiere
        def copy_to_split(file_list, split_name):
            dst_dir = os.path.join(split_dirs[split_name], cls)
            for img in file_list:
                # Copiere imagine
                shutil.copy2(os.path.join(cls_processed_dir, img), os.path.join(dst_dir, img))
                
                # Copiere IMU
                base = os.path.splitext(img)[0]
                if "_img" in base:
                    imu_name = base.replace("_img", "_imu") + ".npy"
                else:
                    imu_name = base + "_imu.npy"
                shutil.copy2(os.path.join(cls_processed_dir, imu_name), os.path.join(dst_dir, imu_name))

        copy_to_split(train_imgs, "train")
        copy_to_split(val_imgs, "validation")
        copy_to_split(test_imgs, "test")
        
    print(f"\nTotal eșantioane în setul de date: {total_samples}")
    if total_samples == 1000:
        print("SUCCES: Dimensiunea setului de date corespunde țintei (1000 eșantioane).")
    else:
        print(f"AVERTISMENT: Dimensiunea setului de date {total_samples} != 1000.")

if __name__ == "__main__":
    restructure_dataset()
