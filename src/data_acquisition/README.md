Modul de Achiziție a Datelor

Acest modul se ocupă de generarea setului de date de „Contribuție Originală” pentru Etapa 4.

    Prezentare generală

Obiectivul este generarea a 400 de eșantioane originale (40% din setul final de date), prin:

    - Achiziție de imagini reale: 40 de imagini capturate manual (10 per clasă: asfalt, beton, iarbă, gresie).

    - Augmentare de imagini: 360 de imagini generate prin rotație (9 variații pentru fiecare imagine originală).

    - Generare sintetică IMU: 400 de semnale IMU generate statistic pe baza distribuției setului de date Kaggle (Etapa 3).

Fișiere:

    -augment_images.py: Aplică rotații imaginilor din data/generated/.

    -generate_imu.py: Calculează statisticile din data/processed/ și generează fișiere IMU sintetice .npy pentru fiecare imagine din data/generated/.

    -generate_all_data.py: Script orchestrator care rulează întregul flux.

    -restructure_dataset.py: Combină datele generate cu datele publice, le reduce la dimensiunea țintă și creează împărțiri train/val/test.

    -export_to_csv.py: Exportă un eșantion din datele generate în src/data_acquisition/generated_data_log.csv.

    -generate_comparison_plots.py: Generează grafice comparative pentru semnalele IMU între clase.

Utilizare:
1. Generarea datelor originale
python src/data_acquisition/generate_all_data.py

2. Construirea setului final de date (Combinare & Împărțire)
python src/data_acquisition/restructure_dataset.py

Ieșire:

Datele generate sunt salvate în data/generated/[class]/.

Fiecare eșantion este compus din:

[nume].jpg: Imaginea (originală sau augmentată).

[nume]_imu.npy: Datele IMU sintetice corespunzătoare.
