# -*- coding: utf-8 -*-
"""Reconoce digitos con la red entrenada (modelo_final.keras) y el modulo de logica difusa.

Uso:   python reconocer_digito.py                  -> prueba con las imagenes 0 a 4 del conjunto de prueba de MNIST
       python reconocer_digito.py 25 300 4021      -> las imagenes numero 25, 300 y 4021 del conjunto de prueba
       python reconocer_digito.py mi_digito.png    -> una imagen propia (se pasa a 28 x 28 en grises; se espera trazo claro sobre
                                                      fondo oscuro como MNIST, y si el fondo es claro la imagen se invierte sola)
Requiere haber corrido antes entrenar_y_evaluar.py (genera modelo_final.keras).
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np                                      # noqa: E402
from tensorflow import keras                            # noqa: E402

from datos import cargar_mnist                          # noqa: E402
from logica_difusa import ModuloDifuso, describir       # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))


def cargar_imagen(ruta):
    """Lee una imagen cualquiera y la deja como los datos de MNIST: 784 numeros entre 0 y 1, trazo claro sobre fondo oscuro."""
    from PIL import Image
    gris = np.asarray(Image.open(ruta).convert("L").resize((28, 28)), dtype="float32") / 255.0
    if gris.mean() > 0.5:                                # fondo claro (hoja blanca): se invierte
        gris = 1.0 - gris
    return gris.reshape(784)


def main():
    pedidos = sys.argv[1:] or ["0", "1", "2", "3", "4"]
    modelo_ruta = os.path.join(AQUI, "modelo_final.keras")
    if not os.path.exists(modelo_ruta):
        sys.exit("Falta modelo_final.keras: primero ejecutar  python entrenar_y_evaluar.py")
    modelo = keras.models.load_model(modelo_ruta)
    modulo = ModuloDifuso()
    _, _, (x_prueba, y_prueba) = cargar_mnist()
    for pedido in pedidos:
        if pedido.isdigit():
            i = int(pedido)
            if i >= len(x_prueba):
                print(f"El conjunto de prueba tiene {len(x_prueba)} imagenes (0 a {len(x_prueba) - 1}); no existe la {i}")
                continue
            x, nombre = x_prueba[i], f"Imagen #{i} (digito real {int(y_prueba[i])})"
        else:
            x, nombre = cargar_imagen(pedido), f"Archivo {pedido}"
        probs = modelo.predict(x.reshape(1, 784), verbose=0)[0]
        r = modulo.evaluar(probs)
        print(f"{nombre}: {describir(r)}")
        otros = [d for d in np.argsort(probs)[::-1] if d != r["digito"]][:2]
        print("      otros candidatos: " + ", ".join(f"{d} ({100 * probs[d]:.1f} %)" for d in otros))


if __name__ == "__main__":
    main()
