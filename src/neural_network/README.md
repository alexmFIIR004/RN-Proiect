Modul Rețea Neuronală

Acest modul conține definiția și configurarea Rețelei Neuronale utilizate pentru Clasificarea Suprafeței Podelei.

Arhitectură:

    Modelul este o Rețea Neuronală Convoluțională 1D (CNN), concepută pentru clasificarea seriilor temporale provenite din date IMU.

    Straturi:

    - Intrare (99, 10)

    - Conv1D (64 filtre) + MaxPool + Dropout

    - Conv1D (128 filtre) + MaxPool + Dropout

    - GlobalAveragePooling1D

    - Dense (64) + Dropout

    - Ieșire (5 clase, Softmax)

Utilizare:

    Pentru a construi și salva modelul schelet:

    - python src/neural_network/model.py

Fișiere:

    - model.py: Definiția modelului Keras.

    - config.py: Hiperparametri și constante.
