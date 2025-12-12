# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Marinescu Alexandru  
**Repository GitHub:** https://github.com/alexmFIIR004/RN-Proiect  
**Data:** 09.12.2025  

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape.

**Livrabil:** Un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). Modelul RN este definit și compilat (fără antrenare avansată).

### IMPORTANT - Ce înseamnă "schelet funcțional":

#### CE TREBUIE SĂ FUNCȚIONEZE:
- Toate modulele pornesc fără erori.
- Pipeline-ul complet rulează end-to-end (de la generare date → până la output UI).
- Modelul RN este definit și compilat (arhitectura există).
- Web Service/UI primește input și returnează output.

#### CE NU E NECESAR ÎN ETAPA 4:
- Model RN antrenat cu performanță bună.
- Hiperparametri optimizați.
- Acuratețe mare pe test set.
- Web Service/UI cu funcționalități avansate.

---

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

Legarea nevoii identificate din Etapa 1-2 cu modulele software construite:

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul nostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| **Clasificare automată a tipului suprafeței pe care se deplasează robotul mobil** | RN primește date IMU (accelerație, viteză unghiulară) și clasifică în timp real suprafața (asphalt, carpet, concrete, grass, tile); latență < 500ms. | **Module 2 (RN)** + **Module 3 (UI)** |
| **Adaptarea strategiei de deplasare a robotului în funcție de suprafață** | Pe baza predicției RN, controlul robotului poate ajusta viteza și aderența (ex: pe iarbă reduce viteza cu 30%). Rezultat: îmbunătățire eficiență energetică cu 15%. | **Module 2 (RN)**  |
| **Generarea de date sintetice pentru antrenare robustă** | Module 1 generează date IMU sintetice bazate pe statistici reale și imagini augmentate pentru a acoperi scenarii rare. | **Module 1 (Data Acquisition)** |

---

## 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

### 2.1 Calculul Contribuției Originale

**Situația actuală (Etapa 3):**
- Total observații din Kaggle (VAST dataset): **1000 samples** (200 per clasă × 5 clase).
- Observații generate în Etapa 3: **0 samples**.

**Realizare Etapa 4:**
- Am redus datasetul public la **600 samples** (120 per clasă).
- Am generat **400 samples** originale (100 per clasă pentru 4 clase: asphalt, concrete, grass, tile).
- **Total Final:** 1000 samples.
- **Procent Original:** 400 / 1000 = **40%**.

### 2.2 Planul de Generare Date Originale

#### **Opțiunea Aleasă: Achiziție Imagini Reale + Augmentare + Generare Sintetică IMU**

**Tipul contribuției:**
- Date achiziționate cu senzori proprii (imagini reale noi).
- Modificare date (rotații imagini).
- Date sintetice prin proceduri statistice (IMU generat din distribuții reale).

#### **Detalii Implementare:**

1.  **Achiziție Imagini Reale (40 imagini):**
    - S-au capturat 10 imagini noi pentru 4 tipuri de suprafață (asphalt, concrete, grass, tile).
    - Sursă: Cameră telefon mobil (simulare perspectivă robot).

2.  **Augmentare Imagini prin Rotație (360 imagini):**
    - Pentru fiecare imagine originală, s-au generat 9 variații prin rotație (±10°...±90°).
    - S-a folosit Resize-Rotate-Crop pentru a evita marginile negre.

3.  **Generare IMU Data Sintetică (400 samples):**
    - Pentru fiecare imagine (originală sau modificata), s-a generat un semnal IMU sintetic.
    - Metoda: Analiză statistică (mean, std) pe datele IMU din Etapa 3 per clasă și generarea de zgomot gaussian colorat care respectă aceste statistici.

**Locația codului:**
- `src/data_acquisition/augment_images.py`: Script augmentare imagini.
- `src/data_acquisition/generate_imu.py`: Script generare IMU sintetic.
- `src/data_acquisition/generate_all_data.py`: Orchestrator.
- `src/data_acquisition/restructure_dataset.py`: Script de unificare și pentruu split(Train/Val/Test).

**Locația datelor:**
- `data/generated/`: Datele originale brute.
- `data/processed/`: Datasetul final unificat (1000 samples).

**Dovezi:**
- Grafic comparativ: `docs/generated_vs_real.png`
- Setup experimental: `docs/acquisition_setup.jpg` (dacă aplicabil)
- Tabel statistici: `docs/data_statistics.csv`

---

## 3. Diagrama State Machine a Întregului Sistem

### 3.1 State Machine Complet

Diagrama se găsește în `docs/state_machine.mermaid`.

**Fluxul Principal:**
`IDLE` → `ACQUIRE_DATA` → `PREPROCESS` → `INFERENCE` → `ACT` → `LOG`

### 3.2 Justificarea State Machine-ului Ales

**Tip arhitectură:** Monitorizare continuă în timp real cu feedback control.

**De ce această arhitectură:**
Robotul mobil trebuie să ia decizii rapid în timp ce se deplasează. Nu putem aștepta procesarea batch.
1.  **ACQUIRE_DATA:** Colectează date de la senzorii IMU (buffer de 1 secundă).
2.  **PREPROCESS:** Normalizează datele și extrage (sau le formatează pentru CNN).
3.  **INFERENCE:** Rețeaua neuronală prezice tipul suprafeței.
4.  **ACT:** Robotul ajustează parametrii motoarelor.
5.  **LOG:** Datele sunt salvate pentru analiză ulterioară.

Starea **ERROR_HANDLER** este critică pentru a asigura că robotul nu se oprește brusc în cazul unei citiri eronate a senzorului.

---

## 4. Scheletul Complet al celor 3 Module

### 4.1 Modul 1: Data Acquisition & Logging (`src/data_acquisition/`)

**Scop:** Generare date originale și pregătirea datasetului final.

**Fișiere:**
- `src/data_acquisition/generate_all_data.py`: Rulează pipeline-ul de generare.
- `src/data_acquisition/restructure_dataset.py`: Construiește structura de foldere (Train/Val/Test) și asigură balansarea claselor.
- `src/data_acquisition/export_to_csv.py`: Exportă datele generate în `src/data_acquisition/date_csv.csv`
**Status:** Funcțional.

### 4.2 Modul 2: Neural Network (`src/neural_network/`)

**Scop:** Definire și compilare model RN.

**Fișiere:**
- `src/neural_network/model.py`: Definește arhitectura CNN 1D.
- `src/neural_network/config.py`: Parametrii de configurare.

**Status:** Funcțional. Modelul se compilează și se salvează în `models/rn_floor_classifier_v0_skeleton.h5`.

**Arhitectură:**
- Input: (99, 10) - Serii de timp IMU.
- Layers: Conv1D -> MaxPool -> Dropout -> Dense.
- Output: 5 clase (Softmax).

### 4.3 Modul 3: Web Service / UI (`src/app/`)

**Scop:** Interfață pentru demonstrarea clasificării.

**Fișiere:**
- `src/app/app.py`: Aplicație Streamlit.

**Status:** Funcțional.
- Încarcă modelul schelet.
- Permite generarea de date random pentru testarea fluxului.
- Afișează clasa prezisă și distribuția probabilităților.

**Rulare:**
```bash
streamlit run src/app/app.py
```

---

## 5. Structura Repository-ului la Finalul Etapei 4

```
RN-Proiect/
├── data/
│   ├── raw/                          # Date brute (Kaggle)
│   ├── generated/                    # Date originale (40%)
│   ├── processed/                    # Dataset final (1000 samples)
│   ├── train/                        # Set antrenare
│   ├── validation/                   # Set validare
│   ├── test/                         # Set testare
│   └── README.md
│
├── src/
│   ├── data_acquisition/             # ← MODUL 1
│   │   ├── generate_all_data.py      # Orchestrator generare
│   │   ├── augment_images.py         # Modificare imagini
│   │   ├── generate_imu.py           # Generare IMU sintetic
│   │   ├── restructure_dataset.py    # Split & Merge
│   │   └── README.md
│   │
│   ├── neural_network/               # ← MODUL 2
│   │   ├── model.py                  # Definitie RN
│   │   └── config.py
│   │
│   └── app/                          # ← MODUL 3
│       ├── app.py                    # Streamlit UI
│       └── requirements_app.txt
│
├── models/                           # ← NOU
│   └── rn_floor_classifier_v0_skeleton.h5
│
├── docs/
│   ├── state_machine.mermaid         # Diagrama State Machine
│   └── screenshots/
│       └──Ui_demo.png
├── README.md                         # Etapa 3
├── README_Etapa4_Arhitectura_SIA.md  # Acest fișier
└── requirements.txt
```

---

## 6. Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [DA] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [DA] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [DA] Cod generare/achiziție date funcțional și documentat
- [DA] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [DA] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [DA] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [DA] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [DA] Cod rulează fără erori (`python src/data_acquisition/...` sau echivalent LabVIEW)
- [DA] Produce minimum 40% date originale din dataset-ul final
- [DA] CSV generat în `src/data_acquisition/date_csv.csv`
- [DA] Documentație în `src/data_acquisition/README.md` cu:
  - [DA] Metodă de generare/achiziție explicată
  - [DA] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [DA] Justificare relevanță date pentru problema voastră
- [DA] Fișiere în `data/generated/` conform structurii

### Modul 2: Neural Network
- [DA] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [DA] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [DA] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [DA] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [DA] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---
