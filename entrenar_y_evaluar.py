# -*- coding: utf-8 -*-
"""Programa principal: reconocimiento de digitos manuscritos (MNIST) con red neuronal + algoritmo genetico + logica difusa.

Pasos:
  1) Carga MNIST (50.000 imagenes para entrenar, 10.000 para validar, 10.000 para probar).
  2) Entrena una red "a mano" (linea de base).
  3) El algoritmo genetico busca la mejor topologia e hiperparametros (usa solo entrenamiento y validacion).
  4) Entrena la red que eligio el genetico con el mismo presupuesto que la base y compara las dos en PRUEBA.
  5) La logica difusa evalua la certeza de cada respuesta y se mide que tanto mejora la precision de las respuestas "seguras".
  6) Guarda graficos (carpeta graficos), numeros (carpeta salidas) y el modelo (modelo_final.keras).

Uso:   python entrenar_y_evaluar.py            (completo: 6 minutos en el entorno de prueba, solo con CPU; en una PC comun puede tardar entre 5 y 15)
       python entrenar_y_evaluar.py --rapido   (prueba chica de 2 o 3 minutos; guarda en carpetas con sufijo _rapido)
"""
import argparse
import json
import os
import time

import numpy as np

import graficos
from algoritmo_genetico import GENES, decodificar, evolucionar
from datos import SEMILLA, cargar_mnist
from logica_difusa import UMBRAL_RECHAZO, UMBRAL_SEGURO, ModuloDifuso, describir
from red_neuronal import construir_red, entrenar, precision
from tensorflow import keras

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="prueba chica (algoritmo genetico reducido y pocas epocas)")
    args = ap.parse_args()
    if args.rapido:
        cfg = dict(pob=4, gens=2, n_ent=5000, ep_fit=1, ep_final=2, n_difuso=2000, sufijo="_rapido")
    else:
        cfg = dict(pob=10, gens=6, n_ent=20000, ep_fit=3, ep_final=10, n_difuso=10000, sufijo="")
    sfx = cfg["sufijo"]
    carp_graf = os.path.join(AQUI, "graficos" + sfx)
    carp_sal = os.path.join(AQUI, "salidas" + sfx)
    os.makedirs(carp_graf, exist_ok=True)
    os.makedirs(carp_sal, exist_ok=True)

    lineas = []

    def log(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    def seccion(titulo):
        log()
        log("=" * 78)
        log(titulo)
        log("=" * 78)

    t_total = time.perf_counter()

    # ------------------------------------------------------------------ 1. datos
    seccion("1) DATOS: MNIST (digitos manuscritos de 28 x 28 pixeles)")
    datos_ent, datos_val, datos_prueba = cargar_mnist()
    log(f"Entrenamiento: {len(datos_ent[0])} imagenes | Validacion: {len(datos_val[0])} | Prueba: {len(datos_prueba[0])} | Pixeles por imagen: 784")
    log("Cantidad por digito en prueba: " + ", ".join(f"{d}:{int((datos_prueba[1] == d).sum())}" for d in range(10)))

    # ------------------------------------------------------------------ 2. linea de base
    seccion(f"2) LINEA DE BASE: red hecha a mano, sin algoritmo genetico ({cfg['ep_final']} epocas)")
    keras.utils.set_random_seed(SEMILLA)
    base = construir_red([128, 0], 0.001, 0.0)
    t0 = time.perf_counter()
    h_base = entrenar(base, datos_ent, datos_val, cfg["ep_final"])
    t_base = time.perf_counter() - t0
    acc_base = precision(base, datos_prueba)
    log(f"Red 784 -> 128 -> 10 | Adam, tasa de aprendizaje 0.001 | sin dropout | parametros: {base.count_params()}")
    log(f"Precision en validacion: {100 * h_base['val_accuracy'][-1]:.2f} %   Precision en PRUEBA: {100 * acc_base:.2f} %   (entrenamiento: {t_base:.0f} s)")

    # ------------------------------------------------------------------ 3. algoritmo genetico
    espacio = int(np.prod([len(v) for v in GENES.values()]))
    seccion("3) ALGORITMO GENETICO: busqueda de la topologia y los hiperparametros")
    log(f"Espacio de busqueda: {espacio} combinaciones posibles | poblacion {cfg['pob']} | {cfg['gens']} generaciones | seleccion por torneo (3) | cruce uniforme 0.8 | "
        f"mutacion 0.25 por gen | elitismo 2")
    log(f"Aptitud = precision en VALIDACION tras entrenar {cfg['ep_fit']} epoca(s) con {cfg['n_ent']} imagenes")
    t0 = time.perf_counter()
    mejor, apt, historia, cache = evolucionar(datos_ent, datos_val, tam_poblacion=cfg["pob"], generaciones=cfg["gens"], n_ent=cfg["n_ent"], epocas=cfg["ep_fit"], log=log)
    t_ga = time.perf_counter() - t0
    g = decodificar(mejor)
    log(f"Redes distintas entrenadas por el genetico: {len(cache)} de {espacio} posibles ({100 * len(cache) / espacio:.1f} % del espacio) en {t_ga:.0f} s")
    log(f"MEJOR INDIVIDUO: {g}  (aptitud {100 * apt:.2f} %)")
    top = sorted(cache.items(), key=lambda kv: -kv[1])[:5]
    log("Las 5 mejores redes probadas:")
    for ind, a in top:
        d = decodificar(ind)
        log(f"   {100 * a:.2f} %   {d['neuronas_1']:>3} - {d['neuronas_2']:>3} neuronas   lr={d['tasa_aprendizaje']:<6} dropout={d['dropout']}")

    # ------------------------------------------------------------------ 4. red final
    seccion(f"4) RED ELEGIDA POR EL GENETICO, entrenada {cfg['ep_final']} epocas con las {len(datos_ent[0])} imagenes (igual presupuesto que la base)")
    keras.utils.set_random_seed(SEMILLA)
    final = construir_red([g["neuronas_1"], g["neuronas_2"]], g["tasa_aprendizaje"], g["dropout"])
    t0 = time.perf_counter()
    h_final = entrenar(final, datos_ent, datos_val, cfg["ep_final"])
    t_final = time.perf_counter() - t0
    acc_final = precision(final, datos_prueba)
    log(f"Parametros: {final.count_params()} | Precision en validacion: {100 * h_final['val_accuracy'][-1]:.2f} %   Precision en PRUEBA: {100 * acc_final:.2f} %   (entrenamiento: {t_final:.0f} s)")
    log()
    log(f"{'':<34}{'Red base':>12}{'Red del genetico':>20}")
    log(f"{'Parametros':<34}{base.count_params():>12}{final.count_params():>20}")
    log(f"{'Precision en validacion (%)':<34}{100 * h_base['val_accuracy'][-1]:>12.2f}{100 * h_final['val_accuracy'][-1]:>20.2f}")
    log(f"{'Precision en PRUEBA (%)':<34}{100 * acc_base:>12.2f}{100 * acc_final:>20.2f}")
    final.save(os.path.join(AQUI, "modelo_final" + sfx + ".keras"))

    x_te, y_te = datos_prueba
    probs = final.predict(x_te, batch_size=1000, verbose=0)
    pred = probs.argmax(axis=1)
    correcto = pred == y_te
    log(f"Errores de la red final en prueba: {int((~correcto).sum())} de {len(y_te)}")
    cm = graficos.matriz_confusion(y_te, pred, os.path.join(carp_graf, "matriz_confusion.png"))
    pares = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j]
    pares.sort(reverse=True)
    log("Confusiones mas frecuentes (real -> la red dijo): " + ", ".join(f"{i}->{j} ({n})" for n, i, j in pares[:5]))

    # ------------------------------------------------------------------ 5. logica difusa
    n_d = cfg["n_difuso"]
    seccion(f"5) LOGICA DIFUSA: certeza de cada respuesta ({n_d} imagenes de prueba)")
    modulo = ModuloDifuso()
    t0 = time.perf_counter()
    res = [modulo.evaluar(p) for p in probs[:n_d]]
    t_difuso = time.perf_counter() - t0
    cert = np.array([r["certeza"] for r in res])
    decision = np.array([r["decision"] for r in res])
    ok = correcto[:n_d]
    log(f"Umbrales fijados de antemano: certeza >= {UMBRAL_SEGURO} 'seguro' | entre {UMBRAL_RECHAZO} y {UMBRAL_SEGURO} 'dudoso' | menos de {UMBRAL_RECHAZO} 'no reconocido'  ({t_difuso:.0f} s)")
    log()
    log(f"{'Decision':<16}{'Imagenes':>10}{'% del total':>13}{'Aciertos':>10}{'Errores':>9}{'Precision (%)':>15}")
    filas = {}
    for d in ("seguro", "dudoso", "no reconocido"):
        m = decision == d
        n = int(m.sum())
        aciertos = int(ok[m].sum())
        prec = 100 * aciertos / n if n else float("nan")
        filas[d] = {"imagenes": n, "porcentaje": 100 * n / n_d, "aciertos": aciertos, "errores": n - aciertos, "precision": prec}
        log(f"{d:<16}{n:>10}{100 * n / n_d:>13.2f}{aciertos:>10}{n - aciertos:>9}{prec:>15.2f}")
    log(f"{'TODAS':<16}{n_d:>10}{100.0:>13.2f}{int(ok.sum()):>10}{int((~ok).sum()):>9}{100 * ok.mean():>15.2f}")
    err_total = int((~ok).sum())
    err_fuera = int(((~ok) & (decision != "seguro")).sum())
    log()
    log(f"De los {err_total} errores de la red, {err_fuera} quedaron marcados como 'dudoso' o 'no reconocido' ({100 * err_fuera / max(err_total, 1):.1f} %): "
        f"el modulo difuso 'avisa' antes de que el usuario se lleve una respuesta mala.")
    log(f"Si solo se aceptan las respuestas 'seguras': precision {filas['seguro']['precision']:.2f} % sobre {filas['seguro']['porcentaje']:.2f} % de las imagenes "
        f"(contra {100 * ok.mean():.2f} % sobre el 100 % sin la capa difusa).")

    # ------------------------------------------------------------------ 6. ejemplos y graficos
    seccion("6) EJEMPLOS DE EJECUCION (primeras imagenes del conjunto de prueba)")
    for i in range(8):
        r = res[i]
        log(f"Imagen #{i}: digito real {int(y_te[i])} -> {describir(r)}")
    graficos.curvas_entrenamiento(h_base, h_final, os.path.join(carp_graf, "curvas_entrenamiento.png"))
    graficos.evolucion_ga(historia, os.path.join(carp_graf, "evolucion_genetico.png"))
    graficos.membresia(modulo, os.path.join(carp_graf, "conjuntos_difusos.png"))
    zmin, zmax = graficos.superficie(modulo, os.path.join(carp_graf, "superficie_difusa.png"))
    graficos.precision_vs_cobertura(cert, ok, UMBRAL_SEGURO, UMBRAL_RECHAZO, os.path.join(carp_graf, "precision_vs_cobertura.png"))
    graficos.arquitectura(g, os.path.join(carp_graf, "arquitectura.png"))
    seguros_ok = [i for i in range(n_d) if decision[i] == "seguro" and ok[i]][:4]
    dudosos = [i for i in range(n_d) if decision[i] != "seguro"][:4]
    errores = [i for i in range(n_d) if not ok[i]][:4]
    graficos.ejemplos(x_te, y_te, res, [seguros_ok, dudosos, errores], ["Seguros y correctos", "Dudosos / no reconocidos", "Errores de la red"],
                      os.path.join(carp_graf, "ejemplos.png"))
    log(f"Graficos guardados en: {os.path.relpath(carp_graf, AQUI)}")
    log(f"Superficie difusa: certeza minima {zmin:.2f}, maxima {zmax:.2f}")
    log(f"Tiempo total: {time.perf_counter() - t_total:.0f} s")

    # ------------------------------------------------------------------ guardar numeros
    resultados = {
        "config": cfg, "mejor_individuo": g, "aptitud_mejor": apt, "redes_entrenadas": len(cache), "espacio": espacio, "historia_ga": historia,
        "base": {"parametros": base.count_params(), "val": h_base["val_accuracy"][-1], "prueba": acc_base, "segundos": t_base},
        "final": {"parametros": final.count_params(), "val": h_final["val_accuracy"][-1], "prueba": acc_final, "segundos": t_final},
        "segundos_ga": t_ga, "segundos_difuso": t_difuso, "difuso": filas, "errores": err_total, "errores_marcados": err_fuera,
        "superficie_certeza": [zmin, zmax], "confusiones_top": [(i, j, n) for n, i, j in pares[:5]],
        "top5": [(decodificar(ind), a) for ind, a in top],
    }
    with open(os.path.join(carp_sal, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(carp_sal, "predicciones_prueba.npz"), real=y_te, pred=pred, probs=probs.astype("float32"), certeza=cert)
    with open(os.path.join(carp_sal, "salida_entrenar_y_evaluar.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


if __name__ == "__main__":
    main()
