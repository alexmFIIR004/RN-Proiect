# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Marinescu Alexandru]  
**Data:** [25.11.2025]  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```


##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

**Dataset:** 1000 observații (200 per categorie × 5 categorii: asphalt, carpet, concrete, grass, tile)  
**Format IMU:** shape `(99, 10)`, dtype `float64` per fișier `.npy`

**Statistici descriptive per caracteristică :**

| Caracteristică | Mean (min–max) | Std Dev | Median | Q25 – Q75 |
|---------------|----------------|---------|--------|-----------|
| orientation_x | -0.001 (±0.01) | 0.006–0.12 | -0.002 | -0.02 – 0.01 |
| orientation_z | -0.44 (±0.73) | 0.07–0.53 | -0.75 | -0.88 – -0.66 |
| linear_accel_z | 9.74 (7.5–11.3) | 0.07–0.37 | 9.74 | 9.65 – 9.82 |

**Distribuții identificate:**
* **Quaternion (orientation):** Concentrare în jurul axelor preferențiale; asphalt/carpet au distribuții distincte
* **Angular velocity:** Simetrie în jurul zero; variabilitate mare în grass (std aprox. 0.35)
* **Linear acceleration Z:** Distribuție centrată în jurul gravitației (~9.8 m/s²); outlierii sunt în grass/concrete

**Identificarea outlierilor (metoda IQR):**
* **Total outlieri detectați:** 45,032 (4.5% din toate valorile)
* **Categorii cu cei mai mulți outlieri:** asphalt (23,716), grass (10,972), tile (5,680)
* **Caracteristici cu outlieri frecvenți:** orientation_z (7,920), angular_velocity_x/y (6,000), linear_accel

### 3.2 Analiza calității datelor

**Valori lipsă:** 0% – niciun NaN
**Valori infinite:** 0 – toate valorile sunt finite 
**Consistență shape:** 100% – toate fișierele au shape uniform `(99, 10)`

**Corelații între caracteristici:**
* **Orientare (quaternion):** Componentele sunt parțial corelate (normalizare unitară)
* **Accelerație vs. categorie:** linear_accel_z variază semnificativ între suprafețe (concrete: 9.73±0.13, grass: 9.74±0.37)
* **Angular velocity:** Independență relativă între axe; carpet are variabilitate redusă (std ~0.01)

### 3.3 Probleme identificate


**Outlieri numerosi în asphalt:** 23,716 outlieri (4.5% din toate valorile) detectați prin IQR. Probabil cauzati de natura suprafetei in sine.

**Variabilitate mare în grass:** std ridicat pentru angular velocity (0.35) și linear_accel (0.55) – teren neuniform cu iarbă.

**Echilibru clase:** Clasele sunt balansate.

**Calitate date:** Zero valori lipsă/infinite. Analiza datelor nu arată erori sau inconsistență la măsurare.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* Eliminare duplicatelor: nu au fost identificate duplicate în perechile `*_img.jpg` + `*_imu.npy`.
* Valori lipsă/infinite: 0% conform EDA.
* Outlieri: păstrați, deoarece reflectă suprafețele reale; 

### 4.2 Transformarea caracteristicilor

* Imagini (`*_img.jpg`):
  * Redimensionare la `224×224` pixeli
  * Conversie în alb-negru
  * Calibrare luminozitate cu praguri: `dark_min=0`, `light_max=255`
  * Normalizare în intervalul `[0,1]`
* IMU (`*_imu.npy`):
* Clase: echilibrate

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [DA] Structură repository configurată
- [DA] Dataset analizat (EDA realizată)
- [DA] Date preprocesate
- [DA] Seturi train/val/test generate
- [DA] Documentație actualizată în README + `data/README.md`

---
