# -*- coding: utf-8 -*-
"""Carga de MNIST (viene con Keras) y particion en entrenamiento / validacion / prueba.

MNIST: 70.000 imagenes de digitos manuscritos de 28x28 pixeles en escala de grises (60.000 de entrenamiento + 10.000 de prueba).
Aca se separan 10.000 de las de entrenamiento para VALIDACION (las usa el algoritmo genetico para puntuar redes); el conjunto de PRUEBA
(10.000) no se toca hasta el final, asi la precision que se informa no esta "contaminada" por la busqueda.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # menos mensajes de TensorFlow en la terminal
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np                                      # noqa: E402
from tensorflow import keras                            # noqa: E402

SEMILLA = 42


def cargar_mnist():
    """Devuelve ((x_ent, y_ent), (x_val, y_val), (x_prueba, y_prueba)) con 50.000 / 10.000 / 10.000 imagenes.

    Cada imagen queda aplanada a 784 numeros entre 0 (negro) y 1 (blanco)."""
    (x, y), (x_prueba, y_prueba) = keras.datasets.mnist.load_data()      # la primera vez lo descarga (unos 11 MB)
    x = x.reshape(-1, 784).astype("float32") / 255.0
    x_prueba = x_prueba.reshape(-1, 784).astype("float32") / 255.0
    orden = np.random.default_rng(SEMILLA).permutation(len(x))           # mezcla fija: mismo reparto en todas las corridas
    val, ent = orden[:10000], orden[10000:]
    return (x[ent], y[ent]), (x[val], y[val]), (x_prueba, y_prueba)
