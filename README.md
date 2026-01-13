# Proiect RN - Clasificarea Suprafețelor (Surface Classification)

Acest proiect implementează un sistem de Inteligență Artificială pentru clasificarea tipurilor de suprafețe (Asphalt, Carpet, Concrete, Grass, Tile) utilizând date multimodale: senzori inerțiali (IMU) și imagini.

##  Structură Proiect
* **src/data_acquisition**: Scripturi pentru generarea și colectarea datelor.
* **src/preprocessing**: Scripturi pentru curățarea, normalizarea datelor și generarea Scaler-ului.
* **src/neural_network**: Definirea modelului, antrenarea și evaluarea acestuia.
* **src/app**: Interfața grafică (Web Dashboard) realizată cu Streamlit.
* **models**: Modelele salvate (.h5).
* **config**: Fișiere de configurare și parametri preprocesare (`.pkl`).

---

## Ghid de Instalare și Rulare

Urmați acești pași pentru a rula proiectul pe un calculator nou.

### 1. Configurare Mediu
Asigurați-vă că aveți **Python 3.9+** instalat și un terminal (PowerShell sau CMD).

```bash
# 1. Navigați în folderul proiectului
cd RN-Proiect

# 2. Creați un mediu virtual (recomandat)
python -m venv .venv

# 3. Activați mediul virtual
# Windows (PowerShell):
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Instalați dependențele
pip install -r requirements.txt
```

### 2. Verificare / Generare Resurse
Înainte de a rula, asigurați-vă că scaler-ul (necesar pentru normalizarea datelor) există.

```bash
# Generează config/preprocessing_params.pkl bazat pe datele de antrenare
python src/preprocessing/create_scaler.py
```

### 3. Antrenare Model (Optional)
Dacă doriți să re-antrenați modelul de la zero:

```bash
python src/neural_network/train.py
# Va salva modelul în models/trained_model.h5
```

### 4. Evaluare Model
Pentru a verifica performanța (Acuratețe, Matrice de confuzie) pe setul de test:

```bash
python src/neural_network/evaluate.py
```

### 5. Rulare Aplicație
Lansați interfața grafică pentru a testa inferența în timp real.

```bash
# Comanda standard
streamlit run src/app/app.py

# eroare de path în PowerShell:
python -m streamlit run src/app/app.py
```