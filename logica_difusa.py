# -*- coding: utf-8 -*-
"""Modulo de logica difusa (Mamdani, scikit-fuzzy) que decide cuanta CERTEZA tiene la respuesta de la red neuronal.

La red devuelve 10 numeros (uno por digito) que suman 1. Se toman dos:
    confianza = lo que se "parece" la imagen al digito ganador (la mayor salida, 0 a 1)
    margen    = ventaja del ganador sobre el segundo (mayor - segunda mayor, 0 a 1)
y las reglas difusas las combinan en una CERTEZA de 0 a 100:
    SI confianza ALTA  Y margen GRANDE                  ENTONCES certeza ALTA
    SI confianza ALTA  Y margen MEDIO o CHICO           ENTONCES certeza MEDIA
    SI confianza MEDIA Y margen MEDIO o GRANDE          ENTONCES certeza MEDIA
    SI confianza MEDIA Y margen CHICO                   ENTONCES certeza BAJA
    SI confianza BAJA                                   ENTONCES certeza BAJA
Con la certeza se decide: >= 70 "seguro", entre 30 y 70 "dudoso", < 30 "no reconocido" (umbrales fijados de antemano).
Los conjuntos de la salida (baja/media/alta) son los mismos triangulos del ejemplo del frenado ABS visto en clase.
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

UMBRAL_SEGURO = 70
UMBRAL_RECHAZO = 30

U_PROB = np.arange(0, 1.0001, 0.01)
U_CERTEZA = np.arange(0, 101, 1)


def crear_sistema():
    """Arma las variables, los conjuntos difusos y las 5 reglas. Devuelve (confianza, margen, certeza, simulacion)."""
    confianza = ctrl.Antecedent(U_PROB, 'confianza')
    margen = ctrl.Antecedent(U_PROB, 'margen')
    certeza = ctrl.Consequent(U_CERTEZA, 'certeza')

    confianza['baja'] = fuzz.trapmf(U_PROB, [0, 0, 0.4, 0.65])
    confianza['media'] = fuzz.trimf(U_PROB, [0.4, 0.65, 0.9])
    confianza['alta'] = fuzz.trapmf(U_PROB, [0.7, 0.9, 1, 1])

    margen['chico'] = fuzz.trapmf(U_PROB, [0, 0, 0.2, 0.5])
    margen['medio'] = fuzz.trimf(U_PROB, [0.2, 0.5, 0.8])
    margen['grande'] = fuzz.trapmf(U_PROB, [0.5, 0.8, 1, 1])

    certeza['baja'] = fuzz.trimf(U_CERTEZA, [0, 0, 40])
    certeza['media'] = fuzz.trimf(U_CERTEZA, [30, 50, 70])
    certeza['alta'] = fuzz.trimf(U_CERTEZA, [60, 100, 100])

    reglas = [
        ctrl.Rule(confianza['alta'] & margen['grande'], certeza['alta']),
        ctrl.Rule(confianza['alta'] & (margen['medio'] | margen['chico']), certeza['media']),
        ctrl.Rule(confianza['media'] & (margen['medio'] | margen['grande']), certeza['media']),
        ctrl.Rule(confianza['media'] & margen['chico'], certeza['baja']),
        ctrl.Rule(confianza['baja'], certeza['baja']),
    ]
    return confianza, margen, certeza, ctrl.ControlSystemSimulation(ctrl.ControlSystem(reglas))


class ModuloDifuso:
    """Envuelve el sistema difuso: recibe las 10 salidas de la red y devuelve la certeza y la decision."""

    def __init__(self):
        self.confianza, self.margen, self.certeza, self.sim = crear_sistema()

    def certeza_de(self, confianza, margen):
        """Certeza (0 a 100) para un par (confianza, margen). Defuzificacion por centroide."""
        self.sim.input['confianza'] = float(np.clip(confianza, 0, 1))
        self.sim.input['margen'] = float(np.clip(margen, 0, 1))
        self.sim.compute()
        return float(self.sim.output['certeza'])

    def evaluar(self, probabilidades):
        """probabilidades: las 10 salidas de la red. Devuelve un diccionario con el digito, el parecido, la certeza y la decision."""
        p = np.asarray(probabilidades, dtype=float)
        orden = np.argsort(p)[::-1]
        d1, d2 = int(orden[0]), int(orden[1])
        confianza, margen = float(p[d1]), float(p[d1] - p[d2])
        c = self.certeza_de(confianza, margen)
        decision = "seguro" if c >= UMBRAL_SEGURO else ("dudoso" if c >= UMBRAL_RECHAZO else "no reconocido")
        return {"digito": d1, "segundo": d2, "parecido": confianza, "parecido_segundo": float(p[d2]),
                "margen": margen, "certeza": c, "decision": decision}


def describir(r):
    """Frase para mostrarle al usuario, con el estilo de la consigna ("se parece en un 80 % al numero 3")."""
    if r["decision"] == "seguro":
        return f"Se parece en un {100 * r['parecido']:.1f} % al {r['digito']} -> SEGURO (certeza difusa {r['certeza']:.1f})"
    if r["decision"] == "dudoso":
        return (f"DUDOSO: se parece en un {100 * r['parecido']:.1f} % al {r['digito']} y en un {100 * r['parecido_segundo']:.1f} % al {r['segundo']} "
                f"(certeza difusa {r['certeza']:.1f})")
    return f"NO RECONOCIDO: el mejor candidato es el {r['digito']} con solo {100 * r['parecido']:.1f} % (certeza difusa {r['certeza']:.1f})"
