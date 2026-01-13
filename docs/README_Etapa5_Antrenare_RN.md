# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Marinescu Alexandru  
**Link Repository GitHub:** https://github.com/alexmFIIR004/RN-Proiect  
**Data predării:** 13.01.2026

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [DA] **State Machine** definit și documentat în `docs/state_machine.*`
- [DA] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [DA] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [DA] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`)
- [DA] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [DA] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

**TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:**

Exemplu:
```bash
# 1. Combinare date vechi (Etapa 3) + noi (Etapa 4)
python src/preprocessing/combine_datasets.py

# 2. Refacere preprocesare COMPLETĂ
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42

# Verificare finală:
# data/train/ → trebuie să conțină date vechi + noi
# data/validation/ → trebuie să conțină date vechi + noi
# data/test/ → trebuie să conțină date vechi + noi
```

** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**
- Același `scaler` salvat în `config/preprocessing_params.pkl`
- Aceiași proporții split: 70% train / 15% validation / 15% test
- Același `random_state=42` pentru reproducibilitate

**Verificare rapidă:**
```python
import pandas as pd
train = pd.read_csv('data/train/X_train.csv')
print(f"Train samples: {len(train)}")  # Trebuie să includă date noi
```

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Completați **TOATE** punctele următoare:

1. **Antrenare model** definit în Etapa 4 pe setul final de date (≥40% originale)
2. **Minimum 10 epoci**, batch size 8–32
3. **Împărțire stratificată** train/validation/test: 70% / 15% / 15%
4. **Tabel justificare hiperparametri** (vezi secțiunea de mai jos - OBLIGATORIU)
5. **Metrici calculate pe test set:**
   - **Acuratețe ≥ 65%**
   - **F1-score (macro) ≥ 0.60**
6. **Salvare model antrenat** în `models/trained_model.h5` (Keras/TensorFlow) sau `.pt` (PyTorch) sau `.lvmodel` (LabVIEW)
7. **Integrare în UI din Etapa 4:**
   - UI trebuie să încarce modelul ANTRENAT (nu dummy)
   - Inferență REALĂ demonstrată
   - Screenshot în `docs/screenshots/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 (initial) | Valoare standard pentru Adam, cu ReduceLROnPlateau pentru ajustare fină.
| Batch size | 32 | Optim pentru viteza de antrenare și stabilitate.
| Number of epochs | 50 | Permite convergența completă; Early Stopping este activat - fara overfitting. 
| Optimizer | Adam | Convergență rapidă și eficientă pentru arhitecturi CNN + MLP. 
| Loss function | Sparse Categorical Crossentropy | Clasificare multi-class.
| Activation functions | ReLU (hidden), Softmax (output) | ReLU previne gradient vanishing; Softmax pentru distribuție de probabilitate. |

**Justificare detaliată batch size (exemplu):**
```
Am ales batch_size=32 pentru datasetul nostru de dimensiune medie. 
Acesta permite actualizări frecvente ale greutăților și se încadrează lejer în memoria GPU/CPU disponibilă.
```

**Resurse învățare rapidă:**
- Împărțire date: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html (video 3 min: https://youtu.be/1NjLMWSGosI?si=KL8Qv2SJ1d_mFZfr)  
- Antrenare simplă Keras: https://keras.io/examples/vision/mnist_convnet/ (secțiunea „Training”)  
- Antrenare simplă PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-an-image-classifier (video 2 min: https://youtu.be/ORMx45xqWkA?si=FXyQEhh0DU8VnuVJ)  
- F1-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html (video 4 min: https://youtu.be/ZQlEcyNV6wc?si=VMCl8aGfhCfp5Egi)


---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping** - oprirea antrenării dacă `val_loss` nu scade în 5 epoci consecutive
2. **Learning Rate Scheduler** - `ReduceLROnPlateau` sau `StepLR`
3. **Augmentări relevante domeniu:**
   - Vibrații motor: zgomot gaussian calibrat, jitter temporal
   - Imagini industriale: slight perspective, lighting variation (nu rotații simple!)
   - Serii temporale: time warping, magnitude warping
4. **Grafic loss și val_loss** în funcție de epoci salvat în `docs/loss_curve.png`
5. **Analiză erori context industrial** (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2)

**Indicatori țintă Nivel 2:**
- **Acuratețe ≥ 75%**
- **F1-score (macro) ≥ 0.70**

**Resurse învățare (aplicații industriale):**
- Albumentations: https://albumentations.ai/docs/examples/   
- Early Stopping + ReduceLROnPlateau în Keras: https://keras.io/api/callbacks/   
- Scheduler în PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate 

---

### Nivel 3 – Bonus (până la 100%)

**Punctaj bonus per activitate:**

| **Activitate** |  **Livrabil** |
|----------------|--------------|
| Comparare 2+ arhitecturi diferite | Tabel comparativ + justificare alegere finală în README |
| Export ONNX/TFLite + benchmark latență | Fișier `models/final_model.onnx` + demonstrație <50ms |
| Confusion Matrix + analiză 5 exemple greșite | `docs/confusion_matrix.png` + analiză în README |

**Resurse bonus:**
- Export ONNX din PyTorch: [PyTorch ONNX Tutorial](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- TensorFlow Lite converter: [TFLite Conversion Guide](https://www.tensorflow.org/lite/convert)
- Confusion Matrix analiză: [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

**Implementare specifică proiectului (Surface Classification):**

| **Stare din State Machine** | **Implementare în Etapa 5 (Cod)** |
|-----------------------------|-----------------------------------|
| `ACQUIRE_DATA` | `get_random_test_sample()` - simulează citirea senzorilor |
| `PREPROCESS` | `preprocess_sample()` - aplică `StandardScaler` (IMU) și Resize (Img) |
| `INFERENCE` | `model.predict()` - folosește `models/trained_model.h5` |
| `CALCULATE_CONFIDENCE` | `confidence = prediction.max() * 100` |
| `CHECK_THRESHOLD` | Verificare `if confidence < 70%` în `src/app/app.py` |
| `ACT` (High Conf) | UI afișează "Confirmed" + Setări Robot |
| `LOG` (Low Conf) | UI afișează Warning "Uncertain Prediction" |

**În `src/app/app.py`:**

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Rezultat:**
```
Pe setul de test curent, modelul a obținut o acuratețe de 100%, fără confuzii.
Totuși, într-un scenariu mai complex, se pot face confuzii între 'asphalt' și 'concrete' 
din cauza texturii vizuale.
```

### 2. Ce caracteristici ale datelor cauzează erori?

**Analiză:**
```
Performanța perfectă sugerează că datele sintetice sunt foarte distincte. 
În practică, zgomotul senzorului IMU (vibrații externe) și iluminarea variabilă 
pentru cameră ar putea introduce erori, reducând distincțiile dintre clase.
```

### 3. Ce implicații are pentru aplicația industrială?

**Analiză:**
```
O rată fals pozitivă (confundarea unei suprafețe alunecoase 'tile' cu una aderentă 'asphalt') 
ar fi problematică pentru un robot mobil. Prioritatea este evitarea clasificării 
eronate a suprafețelor cu coeficient de frecare scăzut.
```

### 4. Ce măsuri corective propuneți?

**Măsuri:**
```
1. Colectarea de date reale mai ample pentru a introduce variabilitate naturală.
2. Augmentare mai agresivă a zgomotului pe semnalul IMU.
3. Implementarea unui "confidence threshold" - dacă încrederea maximă < 70%, 
   robotul să încetinească preventiv.
```

---

## Structura Repository-ului la Finalul Etapei 5

**Clarificare organizare:** Vom folosi **README-uri separate** pentru fiecare etapă în folderul `docs/`:

```
proiect-rn-[prenume-nume]/
├── README.md                           # Overview general proiect (actualizat)
├── etapa3_analiza_date.md         # Din Etapa 3
├── etapa4_arhitectura_sia.md      # Din Etapa 4
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              # Din Etapa 4
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── screenshots/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png                # Din Etapa 4
│
├── data/                               # Din Etapa 3-4 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/                     # Contribuția voastră 40%
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/              # Din Etapa 4
│   ├── preprocessing/                 # Din Etapa 3
│   │   └── combine_datasets.py        # NOU (dacă ați adăugat date în Etapa 4)
│   ├── neural_network/
│   │   ├── model.py                   # Din Etapa 4
│   │   ├── train.py                   # NOU - Script antrenare
│   │   └── evaluate.py                # NOU - Script evaluare
│   └── app/
│       └── app.py                     # ACTUALIZAT - încarcă model antrenat
│
├── models/
│   ├── untrained_model.h5             # Din Etapa 4
│   ├── trained_model.h5               # NOU - OBLIGATORIU
│   └── final_model.onnx               # (opțional - Nivel 3 bonus)
│
├── results/                            # NOU - Folder rezultate antrenare
│   ├── training_history.csv           # OBLIGATORIU - toate epoch-urile
│   ├── test_metrics.json              # Metrici finale pe test set
│   └── hyperparameters.yaml           # Hiperparametri folosiți
│
├── config/
│   └── preprocessing_params.pkl       # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt                    # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 4:**
- Adăugat `docs/etapa5_antrenare_model.md` (acest fișier)
- Adăugat `docs/loss_curve.png` (Nivel 2)
- Adăugat `models/trained_model.h5` - OBLIGATORIU
- Adăugat `results/` cu history și metrici
- Adăugat `src/neural_network/train.py` și `evaluate.py`
- Actualizat `src/app/main.py` să încarce model antrenat

---

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

### 1. Setup mediu (dacă nu ați făcut deja)

```bash
pip install -r requirements.txt
```

### 2. Pregătire date (DACĂ ați adăugat date noi în Etapa 4)

```bash
# Combinare + reprocesare dataset complet
python src/preprocessing/combine_datasets.py
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42
```

### 3. Antrenare model

```bash
python src/neural_network/train.py --epochs 50 --batch_size 32 --early_stopping

# Output așteptat:
# Epoch 1/50 - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.7891 - val_accuracy: 0.6823
# ...
# Epoch 23/50 - loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7956
# Early stopping triggered at epoch 23
# Model saved to models/trained_model.h5
```

### 4. Evaluare pe test set

```bash
python src/neural_network/evaluate.py --model models/trained_model.h5

# Output așteptat:
# Test Accuracy: 0.7823
# Test F1-score (macro): 0.7456

```

### 5. Lansare UI cu model antrenat

```bash
streamlit run src/app/app.py

# SAU pentru LabVIEW:
# Deschideți WebVI și rulați main.vi
```

**Testare în UI:**
1. Introduceți date de test (manual sau upload fișier)
2. Verificați că predicția este DIFERITĂ de Etapa 4 (când era random)
3. Verificați că confidence scores au sens (ex: 85% pentru clasa corectă)
4. Faceți screenshot → salvați în `docs/screenshots/inference_real.png`

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [DA] State Machine există și e documentat în `docs/state_machine.*`
- [DA] Contribuție ≥40% date originale verificabilă în `data/generated/`
- [DA] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [DA] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [DA] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [DA] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [DA] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [DA] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [DA] Tabel hiperparametri + justificări completat în acest README
- [DA] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [DA] Model salvat în `models/trained_model.h5` (sau .pt, .lvmodel)
- [DA] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [DA] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
- [DA] UI face inferență REALĂ cu predicții corecte
- [DA] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [DA] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [DA] Early stopping implementat și documentat în cod
- [DA] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [DA] Augmentări relevante domeniu aplicate (NU rotații simple!)
- [DA] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [DA] Analiză erori în context industrial completată (4 întrebări răspunse)
- [DA] Metrici Nivel 2: **Accuracy 99%**, **F1 ≥0.99**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [NU] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [NU] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [NU] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [DA] `requirements.txt` actualizat cu toate bibliotecile noi
- [DA] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
- [DA] Cod nou comentat în limba română sau engleză (minimum 15%)
- [DA] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [DA] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)
- [DA] Fluxul de inferență respectă stările din State Machine
- [DA] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [DA] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [DA] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [DA] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [DA] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
- [DA] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [DA] Push: `git push origin main --tags`
- [DA] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. **`results/training_history.csv`** - toate epoch-urile salvate

4. **`results/test_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "test_accuracy": 0.7823,
  "test_f1_macro": 0.7456,
  "test_precision_macro": 0.7612,
  "test_recall_macro": 0.7321
}
```

5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

7. **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**