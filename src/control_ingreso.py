"""Control visual de ingreso: clasifica un residuo y decide ACEPTADO / RECHAZADO / SIN VERIFICAR.

Regla (umbral leído de reports/umbral_elegido.json):
    clase "vacio" con confianza >= umbral (no hay objeto)                -> VACIO (no se decide ni se registra)
    categoría predicha en las categorías del contenedor y confianza >= umbral -> ACEPTADO
    categoría predicha fuera de las del contenedor   y confianza >= umbral -> RECHAZADO
    confianza < umbral                                                   -> SIN_VERIFICAR
Los casos SIN_VERIFICAR se guardan en capturas/sin_verificar/ para revisión remota / reentrenamiento.

Uso como script (probar con una imagen, sin cámara):
    python src/control_ingreso.py --imagen ruta.jpg --categorias plasticos
    python src/control_ingreso.py --imagen ruta.jpg --categorias papel carton --contenedor C2
"""
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

CLASES = ["plasticos", "carton", "papel", "vidrio", "metales",
          "organicos", "textiles", "baterias", "otros", "vacio"]
RESIDUOS = CLASES[:-1]        # las 9 categorías que puede tener asignadas un contenedor
RUTA_MODELO = Path("models/recycle_mobilenet.keras")
RUTA_UMBRAL = Path("reports/umbral_elegido.json")
CAPTURAS = Path("capturas")
ACEPTADO, RECHAZADO, SIN_VERIFICAR, VACIO = "ACEPTADO", "RECHAZADO", "SIN_VERIFICAR", "VACIO"


def leer_umbral(defecto=0.825):
    try:
        return float(json.loads(RUTA_UMBRAL.read_text(encoding="utf-8"))["umbral"])
    except Exception:
        return defecto


def decidir(clase_predicha, confianza, categorias_contenedor, umbral):
    """Regla de control visual de ingreso (Tabla 3 de la propuesta)."""
    if confianza < umbral:
        return SIN_VERIFICAR
    if clase_predicha == "vacio":
        return VACIO
    return ACEPTADO if clase_predicha in categorias_contenedor else RECHAZADO


def cargar_modelo(img=224):
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(RUTA_MODELO, compile=False, safe_mode=False)
    except Exception:
        from entrenar import construir          # misma arquitectura del entrenamiento
        modelo, _ = construir(img)
        modelo.load_weights(RUTA_MODELO)
        return modelo


class Clasificador:
    def __init__(self, img=224):
        self.img = img
        self.modelo = cargar_modelo(img)

    def predecir_rgb(self, rgb):
        """rgb: arreglo HxWx3 uint8 (RGB). Devuelve (clase, confianza, probabilidades)."""
        import cv2
        import numpy as np
        x = cv2.resize(rgb, (self.img, self.img), interpolation=cv2.INTER_LINEAR)
        x = np.expand_dims(x.astype("float32"), 0)      # el modelo ya incluye el preprocesamiento
        p = self.modelo.predict(x, verbose=0)[0]
        i = int(p.argmax())
        return CLASES[i], float(p[i]), p

    def predecir_bgr(self, bgr):
        import cv2
        return self.predecir_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def registrar(imagen_bgr, contenedor, categorias, clase, conf, decision, origen=""):
    """Guarda la captura en capturas/<decision>/ y agrega una fila a capturas/registro.csv."""
    import cv2
    carpeta = CAPTURAS / decision.lower()
    carpeta.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    nombre = f"{marca}_{contenedor}_{clase}_{conf:.2f}.jpg"
    cv2.imwrite(str(carpeta / nombre), imagen_bgr)
    nuevo = not (CAPTURAS / "registro.csv").exists()
    with open(CAPTURAS / "registro.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha_hora", "contenedor", "categorias_asignadas", "prediccion",
                        "confianza", "decision", "archivo", "origen"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), contenedor, "|".join(categorias),
                    clase, f"{conf:.4f}", decision, str(Path(decision.lower()) / nombre), origen])
    return carpeta / nombre


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagen", required=True)
    ap.add_argument("--categorias", nargs="+", required=True, choices=RESIDUOS)
    ap.add_argument("--contenedor", default="C1")
    ap.add_argument("--umbral", type=float, default=None)
    a = ap.parse_args()

    umbral = a.umbral if a.umbral is not None else leer_umbral()
    img = cv2.imread(a.imagen)
    if img is None:
        raise SystemExit(f"No se pudo leer la imagen: {a.imagen}")
    clf = Clasificador()
    clase, conf, p = clf.predecir_bgr(img)
    dec = decidir(clase, conf, a.categorias, umbral)
    print(f"Predicción: {clase} (confianza {conf:.1%}) | umbral {umbral:.3f}")
    print(f"Contenedor {a.contenedor} acepta: {', '.join(a.categorias)}")
    print(f"DECISIÓN: {dec}")
    top = sorted(zip(CLASES, p), key=lambda t: -t[1])[:3]
    print("Top 3:", " | ".join(f"{c} {v:.1%}" for c, v in top))
    if dec != VACIO:
        destino = registrar(img, a.contenedor, a.categorias, clase, conf, dec, origen="archivo")
        print(f"Registrado en capturas/{dec.lower()}/ ({destino.name})")


if __name__ == "__main__":
    main()
