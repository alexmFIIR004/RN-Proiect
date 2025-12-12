UI

Acest modul oferă o interfață web bazată pe Streamlit.

Rularea aplicației:

1. Instalează dependențele:

pip install -r src/app/requirements_app.txt


2. Rulează aplicația:

streamlit run src/app/app.py


Funcționalități (Etapa 4)

    -Încarcă modelul schelet al Rețelei Neuronale.

    -Generează date de intrare aleatorii (simulând senzorul IMU).

    -Afișează rezultatele predicției și scorurile de încredere.

    -Vizualizează datele de intrare și distribuția probabilităților.

Limitări

    -Modelul este momentan neantrenat (ponderi aleatorii), astfel încât predicțiile sunt aleatorii.