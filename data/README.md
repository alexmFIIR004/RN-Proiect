# README – Setul de Date

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor
- Origine: dataset public pe Kaggle – https://www.kaggle.com/code/cpatrickalves/floor-surface-classification
- Titlu: „Classifying the type of flooring surface using data collected by Inertial Measurement Units sensors”
- Autor: Patrick Alves (cpatrickalves@gmail.com)
- Data creării: 23.09.2019
- Sursa primară a datelor: Tampere University – Signal Processing Department (Finlanda)
- Context/competiție: https://www.kaggle.com/c/competicao-dsa-machine-learning-sep-2019/
- Modul de achiziție: fișier extern (.csv) provenit din măsurători ale senzorilor IMU (accelerometru, giroscop, orientare)

### 2.2 Caracteristicile dataset-ului
- Număr de caracteristici per punct de măsurare: 10 (orientare 4 canale, viteză unghiulară 3 canale, accelerație 3 canale)
- Tipuri de date: numerice; organizate în serii temporale
- Format fișiere: CSV
- Structura seriilor: 128 măsurători (puncte) per serie
- Coloane de identificare: `row_id` (ID rând), `series_id` (ID serie; cheie către labeluri/submit), `measurement_number` (index măsurare în cadrul seriei)

Voi genera un nou dataset folosind datele din cel public, deoarece acesta este deja prelucrat si analizat complet pe pagina de kaggle.