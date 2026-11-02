# -*- coding: utf-8 -*-
"""Todos los graficos del proyecto (matplotlib). Cada funcion guarda un PNG; ninguna abre ventanas."""
import matplotlib

matplotlib.use("Agg")                                   # sin ventana: solo guarda los archivos PNG
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

ROJO = "#A31F34"
AZUL = "#1f77b4"
NARANJA = "#ff7f0e"
VERDE = "#2ca02c"
GRIS = "#7f7f7f"


def _guardar(fig, ruta):
    fig.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(fig)


def curvas_entrenamiento(h_base, h_final, ruta):
    """Precision y perdida de validacion por epoca: red hecha a mano contra la elegida por el genetico."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for h, nombre, color in ((h_base, "Red base (a mano)", AZUL), (h_final, "Red elegida por el genetico", ROJO)):
        ep = np.arange(1, len(h["accuracy"]) + 1)
        a1.plot(ep, 100 * np.array(h["val_accuracy"]), marker="o", color=color, label=nombre)
        a2.plot(ep, h["loss"], linestyle="--", color=color, label=f"{nombre} - entrenamiento")
        a2.plot(ep, h["val_loss"], color=color, label=f"{nombre} - validacion")
    a1.set_title("Precision en validacion por epoca")
    a1.set_xlabel("Epoca")
    a1.set_ylabel("Precision (%)")
    a2.set_title("Perdida (entropia cruzada) por epoca")
    a2.set_xlabel("Epoca")
    a2.set_ylabel("Perdida")
    for a in (a1, a2):
        a.grid(True, linestyle="--", alpha=0.5)
        a.legend(fontsize=8)
    fig.tight_layout()
    _guardar(fig, ruta)


def evolucion_ga(historia, ruta):
    """Mejor, media y peor aptitud de cada generacion del algoritmo genetico."""
    gen = [f["generacion"] for f in historia]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(gen, [100 * f["mejor"] for f in historia], marker="o", color=ROJO, linewidth=2, label="Mejor")
    ax.plot(gen, [100 * f["media"] for f in historia], marker="s", color=AZUL, label="Media")
    ax.plot(gen, [100 * f["peor"] for f in historia], marker="v", color=GRIS, label="Peor")
    ax.set_xlabel("Generacion (0 = poblacion al azar)")
    ax.set_ylabel("Aptitud = precision en validacion (%)")
    ax.set_title("Evolucion del algoritmo genetico")
    ax.set_xticks(gen)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _guardar(fig, ruta)


def matriz_confusion(y_real, y_pred, ruta):
    """Matriz de confusion 10x10: el color indica la fraccion de cada fila y el numero, la cantidad de imagenes."""
    cm = np.zeros((10, 10), dtype=int)
    for r, p in zip(y_real, y_pred):
        cm[r, p] += 1
    frac = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Digito que dijo la red")
    ax.set_ylabel("Digito real")
    ax.set_title("Matriz de confusion (10.000 imagenes de prueba)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Fraccion de la fila")
    fig.tight_layout()
    _guardar(fig, ruta)
    return cm


def membresia(modulo, ruta):
    """Conjuntos difusos de las dos entradas y de la salida del modulo de logica difusa."""
    fig, ejes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, var, titulo, xlab in ((ejes[0], modulo.confianza, "Entrada: confianza (mayor salida de la red)", "Parecido al digito ganador"),
                                  (ejes[1], modulo.margen, "Entrada: margen (ganador menos segundo)", "Ventaja sobre el segundo"),
                                  (ejes[2], modulo.certeza, "Salida: certeza", "Certeza (0 a 100)")):
        for nombre, term in var.terms.items():
            ax.plot(var.universe, term.mf, linewidth=2, label=nombre)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Membresia")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _guardar(fig, ruta)


def superficie(modulo, ruta, n=21):
    """Superficie de control: certeza para cada par (confianza, margen)."""
    conf = np.linspace(0, 1, n)
    mar = np.linspace(0, 1, n)
    z = np.zeros((n, n))
    for i, m in enumerate(mar):
        for j, c in enumerate(conf):
            z[i, j] = modulo.certeza_de(c, m)
    fig = plt.figure(figsize=(7.5, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(conf, mar)
    ax.plot_surface(X, Y, z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Confianza")
    ax.set_ylabel("Margen")
    ax.set_zlabel("Certeza")
    ax.set_title("Superficie de control del modulo difuso")
    ax.view_init(elev=28, azim=-125)
    _guardar(fig, ruta)
    return float(z.min()), float(z.max())


def ejemplos(x, y_real, resultados, indices_por_fila, titulos_filas, ruta):
    """Cuadricula de imagenes de prueba con lo que dijo la red y la logica difusa."""
    filas = len(indices_por_fila)
    cols = max(len(i) for i in indices_por_fila)
    fig, ejes = plt.subplots(filas, cols, figsize=(2.55 * cols, 3.15 * filas), squeeze=False)
    for f, idxs in enumerate(indices_por_fila):
        for c in range(cols):
            ax = ejes[f][c]
            ax.axis("off")
            if c >= len(idxs):
                continue
            i = idxs[c]
            r = resultados[i]
            ax.imshow(x[i].reshape(28, 28), cmap="gray_r")
            ok = r["digito"] == int(y_real[i])
            ax.set_title(f"#{i}  real {int(y_real[i])} | red {r['digito']}\n{100 * r['parecido']:.1f} %  {r['decision'].upper()}\ncerteza {r['certeza']:.0f}",
                         fontsize=8.5, color=("black" if ok else ROJO))
        ejes[f][0].text(-0.1, 0.5, titulos_filas[f], transform=ejes[f][0].transAxes, rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, ruta)


def precision_vs_cobertura(certeza, correcto, umbral_seguro, umbral_rechazo, ruta):
    """Si solo se aceptan las respuestas con certeza >= t: que fraccion de imagenes se responde y con que precision."""
    ts = np.arange(0, 88, 1)
    cobertura = [100 * np.mean(certeza >= t) for t in ts]
    prec = [100 * np.mean(correcto[certeza >= t]) if np.any(certeza >= t) else np.nan for t in ts]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(ts, prec, color=ROJO, linewidth=2, label="Precision de las respuestas aceptadas")
    ax.set_xlabel("Umbral de certeza t (se acepta si certeza >= t)")
    ax.set_ylabel("Precision (%)", color=ROJO)
    p_validas = [v for v in prec if not np.isnan(v)]
    ax.set_ylim(min(p_validas) - 0.4, min(100.2, max(p_validas) + 0.4))
    ax2 = ax.twinx()
    ax2.plot(ts, cobertura, color=AZUL, linewidth=2, linestyle="--", label="Imagenes respondidas (cobertura)")
    ax2.set_ylabel("Cobertura (% de las 10.000 imagenes)", color=AZUL)
    ax2.set_ylim(min(cobertura) - 2, 100.5)
    for t, nombre in ((umbral_rechazo, "rechazo"), (umbral_seguro, "seguro")):
        ax.axvline(t, color=GRIS, linestyle=":")
        ax.text(t + 0.7, 0.03, f"umbral {nombre} ({t})", rotation=90, fontsize=8, color=GRIS, transform=ax.get_xaxis_transform())
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    fig.legend(l1 + l2, n1 + n2, loc="lower center", ncol=2, fontsize=8, frameon=False)
    ax.set_title("Lo que gana la logica difusa: menos cobertura, mas precision")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _guardar(fig, ruta)


def arquitectura(g, ruta):
    """Diagrama de bloques del sistema. g: datos del mejor individuo (para escribir la topologia)."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    def caja(x, y, w, h, texto, color, tam=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=color, ec="#333333", lw=1.3))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=tam)

    def flecha(x1, y1, x2, y2, texto=None, dy=0.18):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333333"))
        if texto:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, texto, ha="center", fontsize=8, color="#333333")

    capas = f"784 -> {g['neuronas_1']}" + (f" -> {g['neuronas_2']}" if g["neuronas_2"] else "") + " -> 10"
    caja(0.1, 1.4, 1.8, 1.5, "Imagen\n28 x 28 pixeles\n(784 numeros)", "#f3f3f3")
    caja(2.6, 1.2, 2.9, 1.9, f"RED NEURONAL\n(perceptron multicapa)\n{capas}\nReLU + softmax", "#cfe3f7")
    caja(6.2, 1.4, 1.9, 1.5, "10 salidas\n\"se parece X %\"\na cada digito", "#f3f3f3")
    caja(8.8, 1.0, 2.1, 2.3, "LOGICA DIFUSA\nconfianza + margen\n5 reglas Mamdani\ncentroide -> certeza", "#d9f0d3")
    caja(11.6, 1.4, 1.3, 1.5, "Decision:\nseguro /\ndudoso /\nno reconoc.", "#fbe3c8", tam=8.5)
    caja(2.2, 3.7, 3.7, 0.95, "ALGORITMO GENETICO\nelige capas, neuronas, tasa y dropout", "#f6d3d8")
    flecha(1.9, 2.15, 2.6, 2.15)
    flecha(5.5, 2.15, 6.2, 2.15)
    flecha(8.1, 2.15, 8.8, 2.15)
    flecha(10.9, 2.15, 11.6, 2.15)
    flecha(4.05, 3.7, 4.05, 3.1)
    ax.text(4.25, 3.4, "topologia e hiperparametros", fontsize=8, color="#333333", va="center")
    ax.set_title("Arquitectura general del sistema", fontsize=12)
    _guardar(fig, ruta)
