import os
import glob
from PIL import Image
import numpy as np
import shutil

def clean_previous_augmentations(base_dir):
    """
    Șterge fișierele generate anterior (care încep cu 'generated_') pentru a asigura o rulare curată.
    Păstrează imaginile originale 'WhatsApp...' sau alte surse.
    """
    print("Curățare date generate anterior...")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("generated_") or "_aug_" in file:
                try:
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    print(f"Eroare la ștergerea {file}: {e}")

def resize_and_crop_center(img, target_size=(224, 224), base_dim=400):
    """
    Redimensionează imaginea astfel încât latura cea mai scurtă să fie base_dim, apoi decupează centrul la base_dim x base_dim.
    Aceasta oferă o pânză suficient de mare pentru rotație fără margini negre.
    În final, decupează centrul la target_size.
    """
    # 1. Redimensionare astfel încât latura cea mai scurtă să fie base_dim
    w, h = img.size
    scale = base_dim / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    
    # 2. Decupare centru la base_dim x base_dim (Pânză pătrată)
    left = (new_w - base_dim) // 2
    top = (new_h - base_dim) // 2
    right = left + base_dim
    bottom = top + base_dim
    img_square = img.crop((left, top, right, bottom))
    
    return img_square

def crop_final(img, target_size=(224, 224)):
    """
    Decupează centrul target_size din imagine.
    """
    w, h = img.size
    left = (w - target_size[0]) // 2
    top = (h - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    return img.crop((left, top, right, bottom))

def augment_images():
    """
    1. Redimensionare original la pătrat mare (400x400).
    2. Aplicare rotații/răsturnări.
    3. Decupare centru 224x224.
    Aceasta evită marginile negre din rotație deoarece decupăm centrul valid.
    """
    base_dir = os.path.join("data", "generated")
    classes = ["asphalt", "concrete", "grass", "tile"]
    
    # Curățare fișiere generate vechi
    clean_previous_augmentations(base_dir)
    
    augmentations = [
        ("rot90", lambda img: img.rotate(90)),
        ("rot180", lambda img: img.rotate(180)),
        ("rot270", lambda img: img.rotate(270)),
        ("flipLR", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("flipTB", lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
        ("rot10", lambda img: img.rotate(10, resample=Image.Resampling.BICUBIC)),
        ("rot_10", lambda img: img.rotate(-10, resample=Image.Resampling.BICUBIC)),
        ("rot20", lambda img: img.rotate(20, resample=Image.Resampling.BICUBIC)),
        ("rot_20", lambda img: img.rotate(-20, resample=Image.Resampling.BICUBIC)),
    ]

    print(f"Începere augmentare standardizată imagini în {base_dir}...")

    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        # Găsire imagini originale (exclude cele generate dacă au scăpat)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("generated_")]
        
        print(f"Găsit {len(images)} imagini originale în {cls}")

        for idx, img_name in enumerate(images):
            img_path = os.path.join(cls_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    # Conversie la Grayscale ('L')
                    img = img.convert('L')
                    
                    # Pas 1: Pregătire Imagine Bază (400x400)
                    # Folosim 400px pentru a asigura că, chiar și cu rotație de ~20-30 grade, centrul de 224px este valid.
                    base_img = resize_and_crop_center(img, base_dim=400)
                    
                    # Salvare versiune originală standardizată (decupată la 224x224)
                    final_original = crop_final(base_img, target_size=(224, 224))
                    original_out_name = f"generated_{cls}_{idx+1:03d}_original.jpg"
                    final_original.save(os.path.join(cls_dir, original_out_name))
                    
             
                    for aug_name, aug_func in augmentations:
                        # Aplicare transformare pe imaginea de bază mare
                        aug_img_large = aug_func(base_img)
                        
                        # Decupare centru 224x224
                        final_aug = crop_final(aug_img_large, target_size=(224, 224))
                        
                        # Salvare
                        aug_out_name = f"generated_{cls}_{idx+1:03d}_aug_{aug_name}.jpg"
                        final_aug.save(os.path.join(cls_dir, aug_out_name))
                        
            except Exception as e:
                print(f"Eroare procesare {img_name}: {e}")

    print("Augmentare imagini completă. Toate imaginile standardizate la 224x224 fără artefacte.")

if __name__ == "__main__":
    augment_images()
