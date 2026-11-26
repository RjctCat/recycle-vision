"""Prueba de funcionamiento con cámara: control visual de ingreso en vivo.

Uso (raíz del proyecto, venv activo):
    python src/camara.py --categorias plasticos                       # contenedor de plásticos
    python src/camara.py --categorias papel carton --contenedor C2    # contenedor con 2 categorías
    python src/camara.py --categorias vidrio --continuo               # clasifica solo, cada 1 s
Teclas:  ESPACIO = clasificar el objeto frente a la cámara | q o ESC = salir
Cada decisión se guarda en capturas/<aceptado|rechazado|sin_verificar>/ y en capturas/registro.csv
"""
import argparse
import os
import time

import cv2

from control_ingreso import (RESIDUOS, ACEPTADO, RECHAZADO, SIN_VERIFICAR, VACIO, Clasificador,
                             decidir, leer_umbral, registrar)

COLOR = {ACEPTADO: (79, 158, 46), RECHAZADO: (59, 59, 214), SIN_VERIFICAR: (13, 194, 244), VACIO: (163, 151, 138)}   # BGR


def texto(img, s, y, color=(255, 255, 255), escala=0.7, grosor=2):
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categorias", nargs="+", required=True, choices=RESIDUOS)
    ap.add_argument("--contenedor", default="C1")
    ap.add_argument("--camara", type=int, default=0, help="índice de la cámara (0, 1, ...)")
    ap.add_argument("--umbral", type=float, default=None)
    ap.add_argument("--continuo", action="store_true", help="clasificar automáticamente cada --cada s")
    ap.add_argument("--cada", type=float, default=1.0)
    a = ap.parse_args()

    umbral = a.umbral if a.umbral is not None else leer_umbral()
    print("Cargando modelo...")
    clf = Clasificador()
    cap = cv2.VideoCapture(a.camara, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.camara)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara. Pruebe --camara 1 o cierre otras apps que la usen.")

    ultimo = None            # (clase, conf, decisión)
    t_ultimo = 0.0
    print(f"Contenedor {a.contenedor} acepta: {', '.join(a.categorias)} | umbral {umbral:.3f}")
    print("ESPACIO = clasificar | q = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer de la cámara."); break
        clasificar = False
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == 32:
            clasificar = True
        if a.continuo and time.time() - t_ultimo >= a.cada:
            clasificar = True
        if clasificar:
            clase, conf, _ = clf.predecir_bgr(frame)
            dec = decidir(clase, conf, a.categorias, umbral)
            if dec != VACIO:                       # contenedor vacío: no hay nada que decidir ni registrar
                registrar(frame, a.contenedor, a.categorias, clase, conf, dec, origen="camara")
            ultimo, t_ultimo = (clase, conf, dec), time.time()
            print(f"{clase:>10} {conf:6.1%}  ->  {dec}")

        vista = frame.copy()
        texto(vista, f"Contenedor {a.contenedor}: {', '.join(a.categorias)}  |  umbral {umbral:.2f}", 28)
        if ultimo:
            clase, conf, dec = ultimo
            cv2.rectangle(vista, (0, 0), (vista.shape[1] - 1, vista.shape[0] - 1), COLOR[dec], 8)
            texto(vista, "CONTENEDOR VACIO" if dec == VACIO else dec.replace("_", " "), vista.shape[0] - 52, COLOR[dec], 1.1, 3)
            texto(vista, f"Prediccion: {clase} ({conf:.0%})", vista.shape[0] - 16)
        else:
            texto(vista, "ESPACIO = clasificar | q = salir", vista.shape[0] - 16)
        cv2.imshow("ReCycle - control visual de ingreso", vista)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
