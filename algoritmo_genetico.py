# -*- coding: utf-8 -*-
"""Algoritmo genetico (escrito a mano, sin librerias) que busca la mejor TOPOLOGIA e hiperparametros de la red.

Cromosoma = 4 genes. Cada gen guarda la POSICION de un valor dentro de su lista de valores posibles (ver GENES):
    neuronas_1 | neuronas_2 (0 = sin segunda capa) | tasa_aprendizaje | dropout
Aptitud (fitness) = precision de validacion de la red que describe el cromosoma, entrenada poco tiempo (rapido de calcular).
Operadores: seleccion por torneo, cruce uniforme, mutacion gen a gen y elitismo.
Los PESOS de la red no los evoluciona el genetico: los aprende la retropropagacion (Adam) dentro de cada evaluacion.
"""
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from tensorflow import keras                            # noqa: E402

from datos import SEMILLA                               # noqa: E402
from red_neuronal import construir_red                  # noqa: E402

GENES = {
    "neuronas_1": [16, 32, 64, 128, 256, 512],
    "neuronas_2": [0, 16, 32, 64, 128, 256],            # 0 = la red tiene una sola capa oculta
    "tasa_aprendizaje": [0.0005, 0.001, 0.003, 0.01, 0.03],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
}
NOMBRES = list(GENES)


def decodificar(individuo):
    """[3, 1, 2, 0] -> {'neuronas_1': 128, 'neuronas_2': 16, 'tasa_aprendizaje': 0.003, 'dropout': 0.0}"""
    return {nombre: GENES[nombre][i] for nombre, i in zip(NOMBRES, individuo)}


def aptitud(individuo, datos_ent, datos_val, n_ent, epocas, cache):
    """Entrena la red del individuo (n_ent imagenes, pocas epocas) y devuelve su precision en validacion.

    Usa siempre la misma semilla, asi el mismo individuo da siempre la misma aptitud, y se guarda en 'cache' para no repetir el entrenamiento."""
    clave = tuple(individuo)
    if clave not in cache:
        g = decodificar(individuo)
        keras.utils.set_random_seed(SEMILLA)
        red = construir_red([g["neuronas_1"], g["neuronas_2"]], g["tasa_aprendizaje"], g["dropout"])
        x, y = datos_ent
        red.fit(x[:n_ent], y[:n_ent], epochs=epocas, batch_size=128, verbose=0)
        cache[clave] = float(red.evaluate(datos_val[0], datos_val[1], verbose=0)[1])
        keras.backend.clear_session()                   # libera memoria entre evaluaciones
    return cache[clave]


def _torneo(poblacion, aptitudes, k, rng):
    """Elige k individuos al azar y devuelve el mejor de ellos."""
    participantes = rng.sample(range(len(poblacion)), k)
    return poblacion[max(participantes, key=lambda i: aptitudes[i])]


def _cruce(padre1, padre2, p_cruce, rng):
    """Cruce uniforme: con probabilidad p_cruce, cada gen de los hijos viene de uno u otro padre (50 y 50)."""
    h1, h2 = list(padre1), list(padre2)
    if rng.random() < p_cruce:
        for i in range(len(h1)):
            if rng.random() < 0.5:
                h1[i], h2[i] = h2[i], h1[i]
    return h1, h2


def _mutar(individuo, p_mutacion, rng):
    """Con probabilidad p_mutacion por gen, lo cambia por otro valor posible elegido al azar."""
    for i, nombre in enumerate(NOMBRES):
        if rng.random() < p_mutacion:
            individuo[i] = rng.choice([j for j in range(len(GENES[nombre])) if j != individuo[i]])
    return individuo


def evolucionar(datos_ent, datos_val, tam_poblacion=10, generaciones=6, p_cruce=0.8, p_mutacion=0.25, elite=2, k_torneo=3,
                n_ent=20000, epocas=3, semilla=SEMILLA, log=print):
    """Corre el algoritmo genetico. Devuelve (mejor_individuo, mejor_aptitud, historia, cache).

    historia: una fila por generacion con la mejor, la media y la peor aptitud y el mejor individuo."""
    rng = random.Random(semilla)
    cache = {}
    poblacion = [[rng.randrange(len(GENES[n])) for n in NOMBRES] for _ in range(tam_poblacion)]   # generacion 0: al azar
    historia = []
    for gen in range(generaciones):
        aptitudes = [aptitud(ind, datos_ent, datos_val, n_ent, epocas, cache) for ind in poblacion]
        orden = sorted(range(tam_poblacion), key=lambda i: -aptitudes[i])
        mejor = poblacion[orden[0]]
        fila = {"generacion": gen, "mejor": aptitudes[orden[0]], "media": sum(aptitudes) / len(aptitudes), "peor": aptitudes[orden[-1]],
                "mejor_individuo": decodificar(mejor), "redes_entrenadas_hasta_ahora": len(cache)}
        historia.append(fila)
        g = fila["mejor_individuo"]
        log(f"Generacion {gen}: mejor={fila['mejor']:.4f}  media={fila['media']:.4f}  peor={fila['peor']:.4f}  "
            f"(redes entrenadas: {len(cache)})  mejor red: {g['neuronas_1']}-{g['neuronas_2']} neuronas, lr={g['tasa_aprendizaje']}, dropout={g['dropout']}")
        if gen == generaciones - 1:
            break
        nueva = [list(poblacion[i]) for i in orden[:elite]]            # elitismo: los mejores pasan sin cambios
        while len(nueva) < tam_poblacion:
            p1 = _torneo(poblacion, aptitudes, k_torneo, rng)
            p2 = _torneo(poblacion, aptitudes, k_torneo, rng)
            for hijo in _cruce(p1, p2, p_cruce, rng):
                nueva.append(_mutar(hijo, p_mutacion, rng))
        poblacion = nueva[:tam_poblacion]
    mejor_ind = max(cache, key=cache.get)
    return list(mejor_ind), cache[mejor_ind], historia, cache
