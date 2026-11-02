# Reconocimiento de dígitos manuscritos (MNIST) con red neuronal, algoritmo genético y lógica difusa

Proyecto integrador de Inteligencia Artificial — Tecnicatura Superior en Desarrollo de Software, Instituto Superior General San Martín.

Integrantes: Ponce Juan Pablo, [COMPLETAR: resto del grupo]

## Qué hace

El programa reconoce dígitos escritos a mano (imágenes de 28 x 28 píxeles del conjunto MNIST) combinando tres temas de la materia:

1. **Red neuronal** (Keras): un perceptrón multicapa que recibe los 784 píxeles y devuelve 10 números, uno por dígito. Se interpretan como "se parece en un X % al dígito N".
2. **Algoritmo genético** (escrito a mano, sin librerías): busca la mejor forma de la red (neuronas de la capa 1, neuronas de la capa 2, tasa de aprendizaje y dropout). Los pesos los aprende la retropropagación; el genético solo elige la topología y los hiperparámetros.
3. **Lógica difusa** (scikit-fuzzy, tipo Mamdani): toma la confianza de la red (la salida más alta) y el margen (diferencia con la segunda) y calcula una **certeza** de 0 a 100. Con eso cada respuesta queda como `seguro`, `dudoso` o `no reconocido`.

## Cómo ejecutarlo

Se necesita Python 3 y conexión a Internet la primera vez (Keras descarga MNIST, unos 11 MB).

```
pip install -r requirements.txt
python entrenar_y_evaluar.py            # corrida completa (6 minutos en el entorno de prueba, solo CPU)
python entrenar_y_evaluar.py --rapido   # prueba chica (menos de 1 minuto), guarda en carpetas con sufijo _rapido
python reconocer_digito.py 0 92 158 104 232     # demo con imágenes del conjunto de prueba
python reconocer_digito.py mi_digito.png        # demo con una imagen propia
```

`reconocer_digito.py` usa el modelo ya entrenado (`modelo_final.keras`), así que se puede probar sin volver a entrenar.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `datos.py` | Carga MNIST y lo reparte en 50.000 (entrenar), 10.000 (validar) y 10.000 (probar). |
| `red_neuronal.py` | Arma, entrena y evalúa la red (Keras). |
| `algoritmo_genetico.py` | Algoritmo genético: cromosoma de 4 genes, torneo, cruce uniforme, mutación y elitismo. |
| `logica_difusa.py` | Sistema difuso: 2 entradas, 1 salida, 5 reglas, defuzificación por centroide. |
| `entrenar_y_evaluar.py` | Programa principal: corre todo y guarda gráficos y resultados. |
| `reconocer_digito.py` | Demo: reconoce dígitos con el modelo entrenado. |
| `graficos.py` | Funciones que dibujan los gráficos (matplotlib). |
| `graficos/` | Gráficos generados por el programa. |
| `salidas/` | Resultados en texto y JSON de la corrida completa. |
| `modelo_final.keras` | Red entrenada elegida por el genético. |

## Resultados de la corrida completa (semilla 42)

El conjunto de prueba se usó una sola vez, al final.

| | Red base (784-128-10) | Red elegida por el genético (784-256-256-10) |
|---|---|---|
| Parámetros | 101.770 | 269.322 |
| Precisión en validación | 97,31 % | 97,52 % |
| Precisión en prueba | 97,73 % | 97,80 % |

Con la lógica difusa (umbrales fijados antes de mirar los resultados: certeza mayor o igual a 70 = seguro, entre 30 y 70 = dudoso, menos de 30 = no reconocido), sobre las 10.000 imágenes de prueba:

| Decisión | Imágenes | Precisión |
|---|---|---|
| seguro | 9.791 (97,91 %) | 98,65 % |
| dudoso | 128 (1,28 %) | 63,28 % |
| no reconocido | 81 (0,81 %) | 49,38 % |

De los 220 errores de la red, 88 (40 %) quedaron marcados como `dudoso` o `no reconocido`.

## Limitaciones

- La diferencia entre la red base y la elegida por el genético es de 0,07 puntos (7 imágenes de 10.000) con una sola semilla: no alcanza para decir que una sea mejor que la otra.
- El genético evalúa cada red con un presupuesto chico (3 épocas, 20.000 imágenes) para poder probar muchas; la red final se entrena con 10 épocas y 50.000 imágenes. Lo que gana en la búsqueda no siempre se mantiene en el entrenamiento completo. Probó 43 de las 900 combinaciones posibles (4,8 %).
- La salida softmax de la red no es una verdadera pertenencia difusa: se usa como "grado de parecido" y la lógica difusa trabaja sobre ese número.
- La mejora de la capa difusa es chica (97,80 % a 98,65 % aceptando solo las respuestas seguras) y se paga rechazando o marcando como dudoso el 2,09 % de las imágenes.

## Versiones con las que se probó

Python 3.13.16, tensorflow-cpu 2.21.0 (Keras 3.15.1), scikit-fuzzy 0.5.0, NumPy 2.5.4, SciPy 1.18.1, NetworkX 3.7, Matplotlib 3.11.2, Pillow 13.0.0.

scikit-fuzzy no instala solo SciPy ni NetworkX, por eso están en `requirements.txt`.

## Enlace al repositorio

https://github.com/poncejuampy/mnist-genetico-difuso
