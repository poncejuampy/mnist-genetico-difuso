# -*- coding: utf-8 -*-
"""Red neuronal totalmente conectada (perceptron multicapa) para reconocer digitos, con Keras."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from tensorflow import keras                            # noqa: E402
from tensorflow.keras import layers                     # noqa: E402


def construir_red(capas_ocultas, tasa_aprendizaje, dropout=0.0):
    """Arma la red 784 -> capas ocultas (ReLU) -> 10 (softmax).

    capas_ocultas: lista con la cantidad de neuronas de cada capa oculta (un 0 significa "esa capa no existe").
    tasa_aprendizaje: paso del optimizador Adam.
    dropout: fraccion de neuronas que se apagan al azar durante el entrenamiento (0 = ninguna)."""
    red = keras.Sequential(name="red_digitos")
    red.add(keras.Input(shape=(784,)))
    for n in capas_ocultas:
        if n > 0:
            red.add(layers.Dense(n, activation="relu"))
            if dropout > 0:
                red.add(layers.Dropout(dropout))
    red.add(layers.Dense(10, activation="softmax"))                  # 10 salidas: una por digito, suman 1
    red.compile(optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return red


def entrenar(red, datos_ent, datos_val, epocas, lote=128, verbose=0):
    """Entrena con retropropagacion y devuelve el historial (perdida y precision por epoca)."""
    x, y = datos_ent
    historial = red.fit(x, y, validation_data=datos_val, epochs=epocas, batch_size=lote, verbose=verbose)
    return historial.history


def precision(red, datos):
    """Precision (fraccion de aciertos) de la red sobre un conjunto (x, y)."""
    x, y = datos
    return float(red.evaluate(x, y, verbose=0)[1])
