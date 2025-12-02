# README – Setul de Date

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor
- Origine: dataset open source IROS 2022 – https://github.com/RIVeR-Lab/vast_data
- Modul de achiziție: Senzori reali
- Titlu: „VAST: Visual and Spectral Terrain Classification in Unstructured Multi-Class Environments”
- Tipuri de suprafețe: Asphalt, Brick, Carpet, Concrete, Grass, Gravel, Ice, Mulch, Sand, Tile, Turf. Se vor utiliza doar Asphalt, Carpet, Concrete, Grass, Tile.
- Modul de achiziție: fișier extern (.jpg) si măsurători ale senzorilor IMU (accelerometru, giroscop, orientare)

### 2.2 Caracteristicile dataset-ului
- **Număr total de observații:** 1000 (200 per categorie × 5 categorii)
- **Număr de caracteristici (features):** 10 (orientare 4 canale, viteză unghiulară 3 canale, accelerație 3 canale)
- **Tipuri de date:** ☑ Numerice / ☑ Temporale / ☑ Imagini
- **Format fișiere:** ☑ JPG (imagini) / ☑ NPY (date IMU)
- **Structura seriilor:** fiecare fișier `.npy` conține un ndarray cu shape `(99, 10)`, dtype `float64`
  - Statistici IMU: min: `-0.7176`, max: `9.802`, mean: `0.8098`, std: `~3.0`

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------||
| orientation_x | numeric | – | Componenta X a orientării (quaternion) | −1 to 1 |
| orientation_y | numeric | – | Componenta Y a orientării (quaternion) | −1 to 1 |
| orientation_z | numeric | – | Componenta Z a orientării (quaternion) | −1 to 1 |
| orientation_w | numeric | – | Componenta W a orientării (quaternion) | −1 to 1 |
| angular_velocity_x | numeric | rad/s | Viteza unghiulară pe axa X (giroscop) | −10 to 10 |
| angular_velocity_y | numeric | rad/s | Viteza unghiulară pe axa Y (giroscop) | −10 to 10 |
| angular_velocity_z | numeric | rad/s | Viteza unghiulară pe axa Z (giroscop) | −10 to 10 |
| linear_accel_x | numeric | m/s² | Accelerație liniară pe axa X | −10 to 10 |
| linear_accel_y | numeric | m/s² | Accelerație liniară pe axa Y | −10 to 10 |
| linear_accel_z | numeric | m/s² | Accelerație liniară pe axa Z (conține gravitate ~9.8) | −1 to 10 |

### 2.4 Ce conține dataset-ul?
- **Imagini:** de la camere orientate în jos (`*_img.jpg`)
- **Date IMU:** snippets de citiri senzori (`*_imu.npy`), shape `(99, 10)`, dtype `float64`
- **Asociere:** fiecare index numeric înseamnă că toate intrările sunt colectate simultan. De exemplu, `0000001_img.jpg` și `0000001_imu.npy` sunt măsurători sincronizate.
