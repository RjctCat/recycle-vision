"""ReCycle Vision - interfaz web (prototipo + monitoreo y análisis).

Guardar como src/web_app.py y ejecutar desde la raíz del proyecto:
    python src/web_app.py
Abrir:  http://localhost:5000
Requiere:  pip install flask pyserial

Variables de entorno opcionales
  RECYCLE_HOST=0.0.0.0         permitir acceso desde otros equipos de la red
  RECYCLE_DEMO=1               usa un modelo falso, solo para probar la interfaz sin TensorFlow
  RECYCLE_ARDUINO_PORT=COM3    indicar explícitamente el puerto COM del Arduino
"""
import atexit
import base64
import csv
import io
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, Response, jsonify, request, send_from_directory

from compuerta_arduino import CompuertaArduino
from control_ingreso import VACIO, decidir          # misma regla que camara.py

BASE = Path(__file__).resolve().parent.parent
CAP = BASE / "capturas_web"
REENT = BASE / "data" / "reentrenamiento"          # casos revisados, listos para reentrenar
REG = CAP / "registro_web.json"
DEMO = os.environ.get("RECYCLE_DEMO") == "1"
HOST = os.environ.get("RECYCLE_HOST", "127.0.0.1")

CLASES_DEF = ["plasticos", "carton", "papel", "vidrio", "metales",
              "organicos", "textiles", "baterias", "otros", "vacio"]


def leer_clases():
    """Orden de clases con el que se entrenó el modelo (models_colab/clases.json)."""
    try:
        return json.load(open(BASE / "models_colab" / "clases.json", encoding="utf-8"))
    except Exception:
        return CLASES_DEF


def leer_umbral():
    """Umbral elegido en Colab desde models_colab/umbral_elegido_colab.json."""
    try:
        d = json.load(open(BASE / "models_colab" / "umbral_elegido_colab.json", encoding="utf-8"))
        return float(d.get("umbral", 0.86)), d.get("prueba")
    except Exception:
        return 0.86, None


CLASES = leer_clases()
RESIDUOS = [c for c in CLASES if c != "vacio"]
UMBRAL, REF = leer_umbral()


class ModeloFalso:
    """Solo para probar la interfaz sin TensorFlow (RECYCLE_DEMO=1)."""
    def predict(self, x, verbose=0):
        rng = np.random.default_rng(int(abs(float(x.sum()))) % (2 ** 32))
        z = rng.normal(0, 1, len(CLASES))
        z[rng.integers(len(CLASES))] += rng.uniform(0.5, 5)
        p = np.exp(z - z.max())
        return (p / p.sum())[None]


def cargar_modelo():
    if DEMO:
        print("MODO DEMO: modelo falso, las predicciones son aleatorias.")
        return ModeloFalso()
    import tensorflow as tf
    ruta = BASE / "models_colab" / "recycle_mobilenet_colab.keras"
    try:
        m = tf.keras.models.load_model(
            ruta, compile=False, safe_mode=False,
            custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input})
    except Exception as e:
        print("Carga directa falló (", e, ") -> construir() + load_weights()")
        from entrenar import construir
        m, _ = construir(224)
        m.load_weights(ruta)
    n = m.output_shape[-1]
    if n != len(CLASES):
        raise SystemExit(f"El modelo tiene {n} salidas pero models_colab/clases.json lista {len(CLASES)} clases. "
                         "¿Se reentrenó el modelo? Ejecute de nuevo entrenar.py.")
    return m


print("Cargando modelo...")
MODELO = cargar_modelo()
MODELO.predict(np.zeros((1, 224, 224, 3), "float32"), verbose=0)       # calentar
print(f"Modelo listo. Clases: {len(CLASES)} | umbral = {UMBRAL}")

COMPUERTA = CompuertaArduino(cooldown_seg=2.5)
COMPUERTA.conectar()

CAP.mkdir(exist_ok=True)
LOCK = threading.Lock()        # protege REGISTRO
MLOCK = threading.Lock()       # una predicción a la vez


def cargar_registro():
    try:
        r = json.load(open(REG, encoding="utf-8"))
    except Exception:
        return []
    for e in r:                                  # compatibilidad con registros anteriores
        e.setdefault("t", 0)
        e.setdefault("etiqueta", None)
        if "decision" not in e and "dec" in e:
            e["decision"] = e["dec"]
    return r


REGISTRO = cargar_registro()
_pend = 0


def guardar(forzar=False):
    """Escribe el registro en disco cada 5 lecturas (o de inmediato si forzar). Llamar con LOCK."""
    global _pend
    _pend += 1
    if not forzar and _pend < 5:
        return
    _pend = 0
    tmp = REG.with_suffix(".tmp")
    tmp.write_text(json.dumps(REGISTRO, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REG)


def _limpiar():
    guardar(True)
    COMPUERTA.cerrar()


atexit.register(_limpiar)
app = Flask(__name__)


@app.post("/api/clasificar")
def clasificar():
    d = request.get_json(silent=True) or {}
    cats = [c for c in d.get("categorias", []) if c in RESIDUOS]
    try:
        raw = base64.b64decode(d["imagen"].split(",", 1)[1])
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    except Exception:
        return jsonify(error="imagen inválida"), 400
    if not cats:
        return jsonify(error="sin categorías asignadas"), 400

    x = np.asarray(img, "float32")[None]
    t0 = time.perf_counter()
    with MLOCK:
        p = MODELO.predict(x, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000
    orden = np.argsort(p)[::-1]
    i, j = int(orden[0]), int(orden[1])
    clase, conf = CLASES[i], float(p[i])
    dec = decidir(clase, conf, cats, UMBRAL)

    # Enviar orden a la compuerta Arduino únicamente cuando la decisión es ACEPTADO
    COMPUERTA.notificar_decision(dec)

    archivo = None
    if dec != VACIO:                                     # el contenedor vacío no se guarda
        nombre = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{clase}_{conf:.2f}.jpg"
        (CAP / dec).mkdir(parents=True, exist_ok=True)
        img.save(CAP / dec / nombre, quality=90)
        archivo = f"{dec}/{nombre}"
    e = {"t": round(time.time(), 3), "modo": d.get("modo", ""), "cats": cats, "pred": clase,
         "conf": round(conf, 4), "decision": dec, "p2": CLASES[j], "c2": round(float(p[j]), 4),
         "ms": round(ms, 1), "archivo": archivo, "etiqueta": None}
    with LOCK:
        REGISTRO.append(e)
        guardar()
    top = [[CLASES[k], float(p[k])] for k in orden[:3]]
    return jsonify(clase=clase, conf=conf, decision=dec, top=top, ms=ms)


@app.get("/api/config")
def config():
    return jsonify(umbral=UMBRAL, clases=CLASES, referencia=REF)


@app.get("/api/registro")
def registro():
    with LOCK:
        r = [dict(e) for e in REGISTRO[-20000:]]
    reent = {c: (len(list((REENT / c).glob("*.jpg"))) if (REENT / c).exists() else 0) for c in CLASES}
    return jsonify(ahora=time.time(), umbral=UMBRAL, referencia=REF, clases=CLASES, reent=reent, registro=r)


@app.post("/api/corregir")
def corregir():
    d = request.get_json(silent=True) or {}
    et = d.get("etiqueta") or None
    if et is not None and et not in CLASES:
        return jsonify(error="etiqueta desconocida"), 400
    with LOCK:
        for e in REGISTRO:
            if e.get("archivo") and e["archivo"] == d.get("archivo"):
                nombre = Path(e["archivo"]).name
                if e.get("etiqueta"):                    # quitar la copia anterior
                    (REENT / e["etiqueta"] / nombre).unlink(missing_ok=True)
                e["etiqueta"] = et
                if et:
                    dest = REENT / et
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy(CAP / e["archivo"], dest / nombre)
                break
        else:
            return jsonify(error="registro no encontrado"), 404
        guardar(True)
    return jsonify(ok=True)


@app.get("/api/exportar.csv")
def exportar():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["fecha_hora", "contenedor", "categorias_asignadas", "prediccion", "confianza", "decision",
                "segunda_opcion", "confianza_segunda", "inferencia_ms", "etiqueta_real", "archivo"])
    with LOCK:
        filas = list(REGISTRO)
    for e in filas:
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.get("t", 0))), e.get("modo", ""),
                    "|".join(e.get("cats", [])), e.get("pred", ""), e.get("conf", ""), e.get("decision", ""),
                    e.get("p2", ""), e.get("c2", ""), e.get("ms", ""), e.get("etiqueta") or "", e.get("archivo") or ""])
    return Response("\ufeff" + buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=registro_recycle.csv"})


@app.get("/img/<path:p>")
def img(p):
    return send_from_directory(CAP, p)


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>ReCycle Vision</title><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --mc:#f5c518;--bg:#090d16;--card:#111827;--card-elev:#162033;--line:#1f2937;--line-soft:rgba(255,255,255,0.06);
  --tx:#f1f5f9;--mut:#94a3b8;--deep:#0b1120;--hbg:rgba(9,13,22,0.82);--okc:rgba(16,185,129,0.22);--nok:rgba(239,68,68,0.22);
  --dot:rgba(255,255,255,0.045);
  --shadow:0 12px 32px -8px rgba(0,0,0,0.6), 0 4px 12px -2px rgba(0,0,0,0.4);--shadow-sm:0 4px 12px rgba(0,0,0,0.35);
  --glow-strength:14%;--chip-mix:78%;--chip-bg:20%;
}
[data-theme=light]{
  --bg:#f4f6f9;--card:#ffffff;--card-elev:#f8fafc;--line:#e2e8f0;--line-soft:rgba(15,23,42,0.05);
  --tx:#0f172a;--mut:#64748b;--deep:#f1f5f9;--hbg:rgba(255,255,255,0.86);--okc:#dcfce7;--nok:#fee2e2;--dot:rgba(15,23,42,0.045);
  --shadow:0 10px 30px -5px rgba(15,23,42,0.06), 0 4px 10px -2px rgba(15,23,42,0.03);--shadow-sm:0 2px 8px rgba(15,23,42,0.04);
  --glow-strength:10%;--chip-mix:48%;--chip-bg:16%;
}
*{box-sizing:border-box}
body{margin:0;color:var(--tx);font-family:"Plus Jakarta Sans","Segoe UI",system-ui,sans-serif;min-height:100vh;
  background:radial-gradient(ellipse 80% 45% at 50% -10%, color-mix(in srgb,var(--mc) var(--glow-strength),transparent), transparent 70%),
    radial-gradient(circle, var(--dot) 1px, transparent 1px) 0 0/24px 24px, var(--bg);
  transition:background .35s ease, color .25s ease;-webkit-font-smoothing:antialiased}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 32px;border-bottom:1px solid var(--line);
  position:sticky;top:0;z-index:20;background:var(--hbg);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:var(--shadow-sm)}
header::after{content:"";position:absolute;left:0;bottom:-1px;height:3px;width:100%;
  background:linear-gradient(90deg,#f5c518,#b07a3f,#4f8ef7,#2fa863,#9aa5b1,#8a5a2b,#a855f7,#ef6a3a);opacity:.9}
.logo{display:flex;align-items:center;gap:14px}
.logo-icon{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;
  background:color-mix(in srgb,var(--mc) 16%,var(--card));border:1px solid color-mix(in srgb,var(--mc) 35%,var(--line));
  color:color-mix(in srgb,var(--mc) 82%,var(--tx));box-shadow:0 4px 12px color-mix(in srgb,var(--mc) 20%,transparent);transition:.35s}
h1{margin:0;font-size:21px;font-weight:800;letter-spacing:-.3px}
h1 span{color:color-mix(in srgb,var(--mc) 82%,var(--tx));transition:.35s}
.sub{font-size:12.5px;color:var(--mut);font-weight:500;margin-top:1px}
nav{display:flex;align-items:center;gap:8px;background:var(--deep);padding:5px;border-radius:13px;border:1px solid var(--line)}
nav button{display:inline-flex;align-items:center;gap:7px;background:transparent;border:0;color:var(--mut);padding:8px 16px;border-radius:9px;
  cursor:pointer;font-size:14px;font-weight:600;font-family:inherit;transition:all .2s ease}
nav button:hover{color:var(--tx);background:var(--line-soft)}
nav button.on{background:var(--mc);color:#0f172a;font-weight:700;box-shadow:0 2px 10px color-mix(in srgb,var(--mc) 40%,transparent)}
#tt{border-left:1px solid var(--line);border-radius:0 9px 9px 0;margin-left:2px;padding-left:14px}
main{max-width:1280px;margin:auto;padding:24px 28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--shadow);transition:background .3s, border-color .3s}
h3{margin:0 0 16px;font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--mut);display:flex;align-items:center;justify-content:space-between;gap:10px}
h3 small{font-size:11px;font-weight:500;letter-spacing:0;text-transform:none}
.note{color:var(--mut);font-size:12.5px;line-height:1.5;margin:10px 0 0}
.modos{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px;margin-bottom:22px}
.modo{text-align:left;padding:14px 16px;border-radius:15px;border:1.5px solid var(--line);background:var(--card);color:var(--tx);cursor:pointer;
  transition:all .22s cubic-bezier(.2,.8,.2,1);display:flex;gap:13px;align-items:center;font-family:inherit;box-shadow:var(--shadow-sm)}
.modo .mi{width:46px;height:46px;border-radius:12px;flex:none;display:grid;place-items:center;background:color-mix(in srgb,var(--c) 16%,var(--deep));
  color:color-mix(in srgb,var(--c) 80%,var(--tx));border:1px solid color-mix(in srgb,var(--c) 28%,transparent);transition:.22s}
.modo b{font-size:15.5px;font-weight:700;display:block;margin-bottom:5px;letter-spacing:-.2px}
.modo:hover{transform:translateY(-2px);border-color:color-mix(in srgb,var(--c) 55%,var(--line));background:var(--card-elev)}
.modo.on{border-color:var(--c);background:color-mix(in srgb,var(--c) 9%,var(--card));
  box-shadow:0 8px 24px -4px color-mix(in srgb,var(--c) 30%,transparent), inset 0 0 0 1px color-mix(in srgb,var(--c) 30%,transparent)}
.modo.on .mi{background:var(--c);color:#0f172a;border-color:var(--c)}
.chip{display:inline-flex;align-items:center;font-size:11.5px;font-weight:700;padding:2.5px 9px;border-radius:999px;margin-right:4px;
  background:color-mix(in srgb,var(--c) var(--chip-bg),var(--deep));color:color-mix(in srgb,var(--c) var(--chip-mix),var(--tx));
  border:1px solid color-mix(in srgb,var(--c) 40%,transparent)}
.grid{display:grid;grid-template-columns:1.22fr 1fr;gap:20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.mt{margin-top:20px}
@media(max-width:900px){.grid,.grid2{grid-template-columns:1fr}header{padding:12px 16px;flex-wrap:wrap;gap:10px}}
.vid{position:relative;background:#05070a;border-radius:14px;overflow:hidden;aspect-ratio:4/3;border:2px solid color-mix(in srgb,var(--mc) 50%,var(--line));
  box-shadow:inset 0 0 24px rgba(0,0,0,.6);transition:.35s}
video{width:100%;height:100%;object-fit:cover;display:block;transform:scaleX(-1)}
#guia{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);height:70%;aspect-ratio:1;pointer-events:none;box-shadow:0 0 0 999px rgba(0,0,0,0.48);border-radius:12px}
#guia i{position:absolute;width:28px;height:28px;border:3.5px solid var(--mc);transition:.35s}
#guia i:nth-child(1){top:-1px;left:-1px;border-right:0;border-bottom:0;border-top-left-radius:12px}
#guia i:nth-child(2){top:-1px;right:-1px;border-left:0;border-bottom:0;border-top-right-radius:12px}
#guia i:nth-child(3){bottom:-1px;left:-1px;border-right:0;border-top:0;border-bottom-left-radius:12px}
#guia i:nth-child(4){bottom:-1px;right:-1px;border-left:0;border-top:0;border-bottom-right-radius:12px}
.vid.scan::after{content:"";position:absolute;left:0;right:0;height:3px;background:var(--mc);box-shadow:0 0 20px 5px var(--mc);animation:sc .95s ease-in-out infinite}
@keyframes sc{0%{top:0}50%{top:100%}100%{top:0}}
.live{position:absolute;top:12px;left:12px;background:rgba(9,13,22,0.78);backdrop-filter:blur(6px);color:#f8fafc;font-weight:700;letter-spacing:.6px;
  padding:5px 11px;border-radius:999px;font-size:11.5px;display:flex;align-items:center;gap:7px;border:1px solid rgba(255,255,255,0.12)}
.live i{width:8px;height:8px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px #ef4444;animation:pl 1.3s infinite}
@keyframes pl{50%{opacity:.25}}
.tag{position:absolute;bottom:12px;left:12px;background:var(--mc);color:#0f172a;font-weight:800;padding:5px 13px;border-radius:9px;font-size:12.5px;box-shadow:0 4px 12px rgba(0,0,0,0.3);transition:.35s}
.ctl{display:flex;gap:16px;align-items:center;margin-top:16px;flex-wrap:wrap;color:var(--mut);font-size:13.5px;font-weight:600;background:var(--deep);padding:12px 14px;border-radius:13px;border:1px solid var(--line)}
.ctl label{display:inline-flex;align-items:center;gap:7px;cursor:pointer}
select{background:var(--card);color:var(--tx);border:1px solid var(--line);padding:6px 10px;border-radius:8px;font-family:inherit;font-size:13px;cursor:pointer}
.btn{background:var(--mc);color:#0f172a;border:0;padding:11px 22px;border-radius:11px;font-weight:800;font-size:14.5px;font-family:inherit;cursor:pointer;
  box-shadow:0 4px 14px color-mix(in srgb,var(--mc) 38%,transparent);transition:all .2s ease;display:inline-flex;align-items:center;gap:8px;text-decoration:none}
.btn:hover{transform:translateY(-1px);filter:brightness(1.06)}
.btn.sec{background:var(--deep);color:var(--tx);border:1px solid var(--line);box-shadow:none;padding:8px 14px;font-size:13px}
input[type=range],input[type=checkbox]{accent-color:var(--mc);cursor:pointer}
.tip-box{display:flex;align-items:center;gap:9px;color:var(--mut);margin:12px 0 0;font-size:12.5px;font-weight:500}
.res{display:flex;gap:22px;align-items:center;min-height:185px;border:2px solid var(--dc,var(--line));background:color-mix(in srgb,var(--dc,transparent) 10%,var(--card));transition:all .3s ease}
.res .big{font-size:26px;font-weight:800;letter-spacing:-.4px;color:color-mix(in srgb,var(--dc,var(--tx)) 72%,var(--tx));margin:4px 0 8px}
.res .ico{color:color-mix(in srgb,var(--dc,var(--mut)) 82%,var(--tx));display:inline-flex;padding:8px;border-radius:12px;background:color-mix(in srgb,var(--dc,var(--mut)) 14%,var(--deep))}
.ring{position:relative;width:118px;height:118px;flex:none}
.ring svg{transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:9.5}
.ring .bg{stroke:var(--line)}
.ring .fg{stroke:var(--dc,var(--mut));stroke-linecap:round;transition:stroke-dashoffset .55s cubic-bezier(.2,.8,.2,1)}
.ring b{position:absolute;inset:0;display:grid;place-items:center;font-size:25px;font-weight:800;letter-spacing:-.5px}
.top-row{display:flex;justify-content:space-between;font-size:13.5px;font-weight:600;margin-bottom:4px}
.bar{height:9px;background:var(--deep);border-radius:999px;margin:0 0 12px;overflow:hidden;border:1px solid var(--line-soft)}
.bar i{display:block;height:100%;border-radius:999px;background:var(--c,var(--mc));transition:width .5s cubic-bezier(.2,.8,.2,1)}
.hist{display:flex;flex-direction:column;gap:8px;font-size:13px}
.hist div{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:10px;background:var(--deep);border:1px solid var(--line);border-left:4px solid var(--dc);transition:transform .15s}
.hist div:hover{transform:translateX(2px)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px;margin-bottom:20px}
.kpi{border-left:5px solid var(--c);position:relative;overflow:hidden;padding:16px 18px}
.kpi::after{content:"";position:absolute;right:-15px;top:-15px;width:70px;height:70px;border-radius:50%;background:color-mix(in srgb,var(--c) 12%,transparent);pointer-events:none}
.kpi b{font-size:24px;font-weight:800;display:block;letter-spacing:-.6px;margin-bottom:2px;white-space:nowrap}
.kpi span{color:var(--mut);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.9px}
.donut{display:flex;gap:26px;align-items:center;flex-wrap:wrap}
.dn{width:152px;height:152px;border-radius:50%;display:grid;place-items:center;flex:none;box-shadow:var(--shadow-sm)}
.dn div{width:100px;height:100px;border-radius:50%;background:var(--card);display:grid;place-items:center;font-size:23px;font-weight:800;border:1px solid var(--line)}
.leg div{margin:8px 0;display:flex;align-items:center;gap:9px;font-size:13.5px}
.leg i{width:12px;height:12px;border-radius:4px;background:var(--c);flex:none}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:14px}
.gal>div{background:var(--deep);border:1px solid var(--line);border-top:4px solid var(--c);border-radius:14px;padding:10px;font-size:13px;box-shadow:var(--shadow-sm);transition:transform .2s}
.gal>div:hover{transform:translateY(-2px)}
.gal img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:9px;margin-bottom:8px;border:1px solid var(--line)}
.gal select{width:100%;margin-top:8px;padding:7px 9px;border-radius:9px;font-size:12.5px;font-weight:600}
table{border-collapse:collapse;font-size:13px;width:100%}
td,th{border:1px solid var(--line);padding:7px 9px;text-align:center}
th{background:var(--deep);font-weight:700;color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.5px}
tr:hover td{background:var(--line-soft)}
.dtag{padding:3px 10px;border-radius:999px;font-size:11.5px;font-weight:700;background:color-mix(in srgb,var(--dc) 18%,var(--deep));
  color:color-mix(in srgb,var(--dc) var(--chip-mix),var(--tx));border:1px solid color-mix(in srgb,var(--dc) 38%,transparent);white-space:nowrap}
.hid{display:none}
.tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:18px;justify-content:space-between}
.seg{display:inline-flex;background:var(--deep);border:1px solid var(--line);border-radius:11px;padding:4px;gap:2px}
.seg button{background:transparent;border:0;color:var(--mut);padding:7px 14px;border-radius:8px;font-family:inherit;font-size:13px;font-weight:700;cursor:pointer}
.seg button.on{background:var(--mc);color:#0f172a}
.chart{width:100%;height:auto;display:block}
.chart text{font-family:inherit}
.empty{color:var(--mut);font-size:13.5px;padding:18px 0;text-align:center}
.sbar{display:flex;height:16px;border-radius:999px;overflow:hidden;background:var(--deep);border:1px solid var(--line-soft)}
.sbar i{display:block;height:100%}
.row{margin-bottom:14px;font-size:13.5px}
.row .top-row{margin-bottom:6px}
.ins{display:flex;flex-direction:column;gap:10px}
.ins div{display:flex;gap:11px;align-items:flex-start;font-size:13.5px;line-height:1.45;padding:10px 12px;border-radius:11px;background:var(--deep);border:1px solid var(--line);border-left:4px solid var(--c)}
.simk{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:14px 0}
.simk div{background:var(--deep);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.simk b{display:block;font-size:20px;font-weight:800;letter-spacing:-.4px}
.simk span{font-size:11px;color:var(--mut);font-weight:700;text-transform:uppercase;letter-spacing:.7px}
.sl{display:flex;align-items:center;gap:14px;font-weight:700;font-size:14px}
.sl input{flex:1}
.lg{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin-top:6px}
.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
footer{text-align:center;color:var(--mut);font-size:12.5px;font-weight:500;padding:24px;border-top:1px solid var(--line);margin-top:28px}
</style></head><body>
<header>
  <div class="logo">
    <div class="logo-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 1-15.5 6.2M3 12a9 9 0 0 1 15.5-6.2M18 2v4h-4M6 22v-4h4"/></svg></div>
    <div><h1>Re<span>Cycle</span> Vision</h1><div class="sub">Control visual inteligente de ingreso de residuos</div></div>
  </div>
  <nav>
    <button id="t1" class="on" onclick="tab(1)">Prototipo</button>
    <button id="t2" onclick="tab(2)">Monitoreo y análisis</button>
    <button id="tt" onclick="tema()"></button>
  </nav>
</header>
<main>
<section id="v1">
 <div class="modos" id="modos"></div>
 <div class="grid">
  <div class="card">
   <div class="vid" id="vid">
    <video id="v" autoplay playsinline muted></video>
    <div id="guia"><i></i><i></i><i></i><i></i></div>
    <div class="live"><i></i>EN VIVO</div>
    <div class="tag" id="tag"></div>
   </div>
   <div class="ctl">
    <button class="btn" onclick="clasificar()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
      Clasificar
    </button>
    <label>Cámara: <select id="camSelect" onchange="cambiarCamara()"></select></label>
    <label><input type="checkbox" id="cont"> Continuo</label>
    <label>Zona <input type="range" id="zona" min="40" max="100" value="70"></label>
   </div>
   <p class="tip-box">Fondo uniforme, buena iluminación y el objeto centrado dentro del recuadro. Tecla ESPACIO = clasificar.</p>
  </div>
  <div>
   <div id="res" class="card res"></div>
   <div class="card mt"><h3>Top 3 predicciones del modelo</h3><div id="top"><span style="color:var(--mut);font-size:13.5px">Sin análisis realizados aún</span></div><div style="color:var(--mut);font-size:12.5px;margin-top:4px" id="info"></div></div>
   <div class="card mt"><h3>Confianza de las últimas lecturas <small id="umb"></small></h3><div id="spark"><div class="empty">Aún no hay lecturas</div></div></div>
   <div class="card mt"><h3>Últimos ingresos</h3><div class="hist" id="hist"><span style="color:var(--mut);font-size:13.5px">Sin ingresos registrados aún</span></div></div>
  </div>
 </div>
</section>

<section id="v2" class="hid">
 <div class="tools">
  <div class="seg" id="rango"></div>
  <div style="display:flex;gap:10px;align-items:center"><span style="color:var(--mut);font-size:12.5px" id="act"></span><a class="btn sec" href="/api/exportar.csv">Exportar CSV</a></div>
 </div>
 <div class="kpis" id="kpis"></div>
 <div class="grid">
  <div class="card"><h3>Decisiones del sistema</h3><div class="donut" id="donut"></div></div>
  <div class="card"><h3>Lectura automática del periodo</h3><div class="ins" id="ins"></div></div>
 </div>
 <div class="card mt"><h3>Actividad en el tiempo <small>lecturas por intervalo</small></h3><div id="tl"></div><div class="lg" id="tlg"></div></div>
 <div class="grid2 mt">
  <div class="card"><h3>Distribución de la confianza <small>¿qué tan seguro está el modelo?</small></h3><div id="hc"></div><div class="lg" id="hcg"></div></div>
  <div class="card"><h3>Confianza de las últimas 60 lecturas</h3><div id="sc"></div></div>
 </div>
 <div class="grid2 mt">
  <div class="card"><h3>Resultado por contenedor</h3><div id="cont2"></div><div class="lg" id="cog"></div></div>
  <div class="card"><h3>Materiales detectados <small>lecturas · confianza media</small></h3><div id="clases"></div></div>
 </div>
 <div class="card mt"><h3>Simulador de umbral <small>¿qué pasaría con otro umbral?</small></h3>
  <div class="sl"><span>Umbral</span><input type="range" id="simU" min="30" max="99" value="82"><b id="simV"></b></div>
  <div class="simk" id="simk"></div><div id="simc"></div><div class="lg" id="simg"></div><p class="note" id="simn"></p></div>
 <div class="grid2 mt">
  <div class="card"><h3>Dudas frecuentes <small>casos sin verificar: 1ª opción → 2ª opción</small></h3><div id="pares"></div>
   <p class="note">Si una pareja domina, el modelo confunde esos dos materiales: conviene reunir más ejemplos de ellos o ubicarlos en contenedores distintos.</p></div>
  <div class="card"><h3>Reentrenamiento <small>casos revisados listos para usar</small></h3><div id="reent"></div>
   <p class="note">Las imágenes etiquetadas en la bandeja se copian a <b>data/reentrenamiento/&lt;clase&gt;/</b>. Súmelas al dataset y vuelva a entrenar para adaptar el modelo al entorno real.</p></div>
 </div>
 <div class="card mt"><h3>Matriz de confusión <small id="mn"></small></h3><div style="overflow:auto" id="mat"></div><div style="overflow:auto;margin-top:16px" id="prf"></div>
  <p class="note">Se calcula solo con los casos que usted etiquetó. Como se revisan sobre todo los dudosos, no representa la exactitud global del sistema: úsela para ver qué materiales se confunden. Para auditar también los aceptados y rechazados, cambie el filtro de la bandeja.</p></div>
 <div class="card mt"><h3><span>Bandeja de revisión</span>
   <span style="display:flex;gap:10px;align-items:center"><select id="gfil" onchange="pintar()"><option value="pend">Sin verificar pendientes</option><option value="ACEPTADO">Aceptados</option><option value="RECHAZADO">Rechazados</option><option value="todos">Todos</option></select><span id="np" style="color:#d97706"></span></span></h3>
  <div class="gal" id="gal"></div></div>
 <div class="card mt"><h3>Últimos registros</h3><div style="overflow:auto" id="ult"></div></div>
</section>
</main>
<footer>MobileNetV2 con transferencia de aprendizaje · 9 categorías de residuos + contenedor vacío · Redes Neuronales, UDABOL</footer>
<script>
const $=id=>document.getElementById(id);
const svg=(p,s=24)=>`<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
const IC={plasticos:'<path d="M10 2h4v3l2 3v12a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2V8l2-3z"/><path d="M8 12h8"/>',
 papel:'<path d="M6 3h9l4 4v14H6z"/><path d="M15 3v4h4M9 12h7M9 16h7"/>',
 vidrio:'<path d="M6 3h12l-1.5 17a2 2 0 0 1-2 1.8h-5a2 2 0 0 1-2-1.8z"/><path d="M7.5 9h9"/>',
 organicos:'<path d="M5 19c0-9 5-14 15-14 0 10-5 15-14 15M5 19l8-8"/>',
 baterias:'<rect x="3" y="8" width="16" height="9" rx="2"/><path d="M21 11v3M7 11v3M10 11v3"/>',
 textiles:'<path d="M8 3l-5 4 3 3 2-1v12h8V9l2 1 3-3-5-4a4 4 0 0 1-8 0z"/>',
 otros:'<path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14"/>'};
const CC={plasticos:"#eab308",carton:"#b07a3f",papel:"#3b82f6",vidrio:"#10b981",metales:"#64748b",organicos:"#8a5a2b",textiles:"#a855f7",baterias:"#f97316",otros:"#6b7280",vacio:"#94a3b8"};
const NOM={plasticos:"Plásticos",carton:"Cartón",papel:"Papel",vidrio:"Vidrio",metales:"Metales",organicos:"Orgánicos",textiles:"Textiles",baterias:"Baterías",otros:"Otros",vacio:"Vacío"};
const SH={plasticos:"Plást.",carton:"Cart.",papel:"Papel",vidrio:"Vidr.",metales:"Met.",organicos:"Org.",textiles:"Text.",baterias:"Bat.",otros:"Otros",vacio:"Vacío"};
const MODOS={"Plásticos":{c:["plasticos"],col:"#eab308",ic:"plasticos"},"Papel y Cartón":{c:["papel","carton"],col:"#3b82f6",ic:"papel"},
 "Vidrio y Metales":{c:["vidrio","metales"],col:"#10b981",ic:"vidrio"},"Orgánicos":{c:["organicos"],col:"#8a5a2b",ic:"organicos"},
 "Baterías":{c:["baterias"],col:"#f97316",ic:"baterias"},"Textiles":{c:["textiles"],col:"#a855f7",ic:"textiles"},"Otros":{c:["otros"],col:"#6b7280",ic:"otros"}};
const DEC={ACEPTADO:{t:"Ingreso aceptado",s:"Aceptados",col:"#10b981",p:'<path d="M5 13l4 4L19 7"/>'},
 RECHAZADO:{t:"Ingreso rechazado",s:"Rechazados",col:"#ef4444",p:'<path d="M6 6l12 12M18 6L6 18"/>'},
 SIN_VERIFICAR:{t:"Sin verificar",s:"Sin verificar",col:"#f59e0b",p:'<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.01"/>'},
 VACIO:{t:"Contenedor vacío",s:"Vacío",col:"#94a3b8",p:'<rect x="4" y="7" width="16" height="12" rx="2" stroke-dasharray="3 3"/><path d="M4 11h16"/>'}};
const ORD=["ACEPTADO","RECHAZADO","SIN_VERIFICAR","VACIO"];
const chip=x=>`<span class="chip" style="--c:${CC[x]||'#888'}">${NOM[x]||x}</span>`;
const fp=(x,d=1)=>(x*100).toFixed(d).replace(".",",")+" %";
let modo="Plásticos",ocupado=false,timer=null,hist=[],lect=[],streamActual=null,CFG={umbral:0.825,clases:[],referencia:null};
let REG=null,rango=0;

function pintarModos(){$("modos").innerHTML=Object.entries(MODOS).map(([k,m])=>`<button class="modo ${k==modo?'on':''}" style="--c:${m.col}" onclick="setModo('${k}')"><div class="mi">${svg(IC[m.ic],24)}</div><div><b>${k}</b><div>${m.c.map(chip).join("")}</div></div></button>`).join("")}
function inicio(){const m=MODOS[modo];$("res").style.removeProperty("--dc");$("res").innerHTML=`<div class="ico" style="color:color-mix(in srgb,${m.col} 80%,var(--tx));padding:16px">${svg(IC[m.ic],52)}</div><div><div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--mut)">Contenedor activo</div><div class="big" style="color:var(--tx)">${modo}</div><div style="color:var(--mut);font-size:14px">Acerque un residuo al recuadro de captura</div></div>`}
function setModo(k){modo=k;const m=MODOS[k];document.documentElement.style.setProperty("--mc",m.col);$("tag").textContent="Contenedor: "+k;pintarModos();inicio()}
function tab(n){$("v1").classList.toggle("hid",n!=1);$("v2").classList.toggle("hid",n!=2);$("t1").className=n==1?"on":"";$("t2").className=n==2?"on":"";if(n==2)cargar()}

/* ---------- cámara ---------- */
async function listarCamaras(){try{const ds=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput'),sel=$("camSelect"),act=streamActual&&streamActual.getVideoTracks()[0]?.getSettings().deviceId;
 sel.innerHTML="";ds.forEach((d,i)=>{const o=document.createElement('option');o.value=d.deviceId;o.text=d.label||`Cámara ${i+1}`;if(d.deviceId===act)o.selected=true;sel.appendChild(o)})}catch(e){console.log("Error al listar cámaras:",e)}}
async function iniciarCamara(id=null){if(streamActual)streamActual.getTracks().forEach(t=>t.stop());
 try{streamActual=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},deviceId:id?{exact:id}:undefined}});$("v").srcObject=streamActual;await listarCamaras()}
 catch(e){alert("No se pudo iniciar la cámara seleccionada: "+e)}}
function cambiarCamara(){iniciarCamara($("camSelect").value)}
$("zona").oninput=e=>$("guia").style.height=e.target.value+"%";
$("cont").onchange=e=>{clearInterval(timer);if(e.target.checked)timer=setInterval(clasificar,1500)};
document.addEventListener("keydown",e=>{if(e.code==="Space"&&document.body===e.target&&!$("v1").classList.contains("hid")){e.preventDefault();clasificar()}});

/* ---------- gráficos (SVG sin librerías, funcionan sin internet) ---------- */
const vacioMsg=t=>`<div class="empty">${t||"Sin datos en este periodo"}</div>`;
function ejesY(W,H,m,ymax,ticks,fmt){let s="";for(let i=0;i<=ticks;i++){const y=m.t+(H-m.t-m.b)*(1-i/ticks);
 s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" style="stroke:var(--line)"/><text x="${m.l-6}" y="${y+3.5}" text-anchor="end" font-size="10" style="fill:var(--mut)">${fmt(ymax*i/ticks)}</text>`}return s}
function fmtT(ts,span){const d=new Date(ts*1000);return span>172800?d.toLocaleDateString([],{day:"2-digit",month:"2-digit"})+" "+d.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}):d.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})}
function leyenda(keys){return keys.map(k=>`<span><i style="background:${DEC[k].col}"></i>${DEC[k].s}</span>`).join("")}

function timeline(d,ahora,r){
 if(!d.length)return vacioMsg();
 const W=640,H=215,m={l:34,r:8,t:10,b:26},nb=24;
 let ini=r?ahora-r:d.reduce((a,e)=>e.t>0&&e.t<a?e.t:a,Infinity);if(!isFinite(ini))ini=ahora-60;
 let fin=ahora;if(fin-ini<60)ini=fin-60;const w=(fin-ini)/nb,span=fin-ini;
 const B=Array.from({length:nb},()=>({ACEPTADO:0,RECHAZADO:0,SIN_VERIFICAR:0,VACIO:0}));
 d.forEach(e=>{if(e.t<ini)return;const k=Math.min(nb-1,Math.floor((e.t-ini)/w));B[k][e.decision]=(B[k][e.decision]||0)+1});
 const mx=Math.max(1,...B.map(b=>ORD.reduce((a,k)=>a+b[k],0))),ymax=Math.max(4,Math.ceil(mx/4)*4),bw=(W-m.l-m.r)/nb,ph=H-m.t-m.b;
 let s=ejesY(W,H,m,ymax,4,v=>Math.round(v));
 B.forEach((b,k)=>{let y=H-m.b;const x=m.l+k*bw+bw*.12,hh=fmtT(ini+(k+.5)*w,span);
  ORD.forEach(o=>{const n=b[o];if(!n)return;const h=ph*n/ymax;y-=h;s+=`<rect x="${x}" y="${y}" width="${bw*.76}" height="${h}" rx="2" fill="${DEC[o].col}"><title>${hh} · ${DEC[o].s}: ${n}</title></rect>`})});
 [0,6,12,18,23].forEach(k=>{s+=`<text x="${m.l+(k+.5)*bw}" y="${H-8}" text-anchor="middle" font-size="10" style="fill:var(--mut)">${fmtT(ini+(k+.5)*w,span)}</text>`});
 return `<svg class="chart" viewBox="0 0 ${W} ${H}">${s}</svg>`}

function histConf(d,u){
 const nv=d.filter(e=>e.decision!=="VACIO");if(!nv.length)return vacioMsg();
 const W=640,H=215,m={l:34,r:8,t:14,b:26},B=Array.from({length:10},()=>({ACEPTADO:0,RECHAZADO:0,SIN_VERIFICAR:0}));
 nv.forEach(e=>{B[Math.min(9,Math.floor(e.conf*10))][e.decision]++});
 const mx=Math.max(1,...B.map(b=>b.ACEPTADO+b.RECHAZADO+b.SIN_VERIFICAR)),ymax=Math.max(4,Math.ceil(mx/4)*4),bw=(W-m.l-m.r)/10,ph=H-m.t-m.b;
 let s=ejesY(W,H,m,ymax,4,v=>Math.round(v));
 B.forEach((b,k)=>{let y=H-m.b;const x=m.l+k*bw+bw*.08;
  ["ACEPTADO","RECHAZADO","SIN_VERIFICAR"].forEach(o=>{const n=b[o];if(!n)return;const h=ph*n/ymax;y-=h;s+=`<rect x="${x}" y="${y}" width="${bw*.84}" height="${h}" fill="${DEC[o].col}"><title>${k*10}–${k*10+10} %: ${DEC[o].s} ${n}</title></rect>`})});
 for(let k=0;k<=10;k+=2)s+=`<text x="${m.l+k*bw}" y="${H-8}" text-anchor="middle" font-size="10" style="fill:var(--mut)">${k*10}%</text>`;
 const xu=m.l+u*(W-m.l-m.r);s+=`<line x1="${xu}" x2="${xu}" y1="${m.t-4}" y2="${H-m.b}" stroke-dasharray="4 3" style="stroke:var(--tx)" stroke-width="1.5"/><text x="${xu+5}" y="${m.t+6}" font-size="10" font-weight="700" style="fill:var(--tx)">umbral ${fp(u)}</text>`;
 return `<svg class="chart" viewBox="0 0 ${W} ${H}">${s}</svg>`}

function serieConf(arr,u,H=190){
 const a=arr.filter(e=>e.decision!=="VACIO").slice(-60);if(a.length<2)return vacioMsg("Se necesitan al menos 2 lecturas");
 const W=640,m={l:34,r:10,t:10,b:20},n=a.length,xs=i=>m.l+i*(W-m.l-m.r)/(n-1),ys=v=>m.t+(H-m.t-m.b)*(1-v);
 let s=ejesY(W,H,m,1,4,v=>Math.round(v*100)+"%");
 s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${ys(u)}" y2="${ys(u)}" stroke-dasharray="4 3" style="stroke:var(--tx)" stroke-width="1.4"/><text x="${W-m.r}" y="${ys(u)-5}" text-anchor="end" font-size="10" font-weight="700" style="fill:var(--tx)">umbral</text>`;
 s+=`<polyline fill="none" style="stroke:var(--mut)" stroke-width="1.5" points="${a.map((e,i)=>xs(i)+","+ys(e.conf)).join(" ")}"/>`;
 a.forEach((e,i)=>{s+=`<circle cx="${xs(i)}" cy="${ys(e.conf)}" r="4" fill="${DEC[e.decision].col}"><title>${NOM[e.pred]||e.pred} · ${fp(e.conf)} · ${DEC[e.decision].s}</title></circle>`});
 return `<svg class="chart" viewBox="0 0 ${W} ${H}">${s}</svg>`}

/* ---------- simulador de umbral ---------- */
function decSim(e,u){return e.conf<u?"SIN_VERIFICAR":(e.cats||[]).includes(e.pred)?"ACEPTADO":"RECHAZADO"}
function simular(){
 if(!REG)return;const u=$("simU").value/100;$("simV").textContent=fp(u);
 const d=datos().filter(e=>e.decision!=="VACIO"),n=d.length;
 const dec=d.map(e=>decSim(e,u)),ac=dec.filter(x=>x==="ACEPTADO").length,re=dec.filter(x=>x==="RECHAZADO").length,sv=n-ac-re;
 const lab=d.filter(e=>e.etiqueta),dl=lab.filter(e=>e.conf>=u),ok=dl.filter(e=>e.pred===e.etiqueta).length;
 $("simk").innerHTML=[["Decididos",n?fp((ac+re)/n):"—","#10b981"],["Sin verificar",n?fp(sv/n):"—","#f59e0b"],["Aceptados / rechazados",n?`${ac} / ${re}`:"—","#3b82f6"],
  ["Exactitud (revisados)",dl.length>=5?fp(ok/dl.length):"—","#a855f7"]].map(([a,b,c])=>`<div style="border-left:4px solid ${c}"><b>${b}</b><span>${a}</span></div>`).join("");
 if(!n){$("simc").innerHTML=vacioMsg();$("simg").innerHTML="";$("simn").textContent="";return}
 const W=640,H=200,m={l:34,r:10,t:10,b:24},xs=v=>m.l+(v-.3)/.69*(W-m.l-m.r),ys=v=>m.t+(H-m.t-m.b)*(1-v);
 let s=ejesY(W,H,m,1,4,v=>Math.round(v*100)+"%"),p1=[],p2=[];
 for(let k=30;k<=99;k++){const t=k/100,c=d.filter(e=>e.conf>=t).length;p1.push(xs(t)+","+ys(c/n));
  const l=lab.filter(e=>e.conf>=t);if(l.length>=5)p2.push(xs(t)+","+ys(l.filter(e=>e.pred===e.etiqueta).length/l.length))}
 s+=`<polyline fill="none" stroke="#10b981" stroke-width="2.2" points="${p1.join(" ")}"/>`;
 if(p2.length>1)s+=`<polyline fill="none" stroke="#a855f7" stroke-width="2.2" points="${p2.join(" ")}"/>`;
 for(let k=30;k<=100;k+=10)s+=`<text x="${xs(Math.min(k,99)/100)}" y="${H-7}" text-anchor="middle" font-size="10" style="fill:var(--mut)">${k}%</text>`;
 s+=`<line x1="${xs(CFG.umbral)}" x2="${xs(CFG.umbral)}" y1="${m.t}" y2="${H-m.b}" stroke-dasharray="4 3" style="stroke:var(--mut)"/><text x="${xs(CFG.umbral)+4}" y="${m.t+10}" font-size="10" style="fill:var(--mut)">activo</text>`;
 s+=`<line x1="${xs(u)}" x2="${xs(u)}" y1="${m.t}" y2="${H-m.b}" style="stroke:var(--tx)" stroke-width="1.8"/>`;
 $("simc").innerHTML=`<svg class="chart" viewBox="0 0 ${W} ${H}">${s}</svg>`;
 $("simg").innerHTML=`<span><i style="background:#10b981"></i>% decididos automáticamente</span><span><i style="background:#a855f7"></i>exactitud entre decididos (casos revisados)</span>`;
 const R=CFG.referencia;
 $("simn").textContent=(R&&R.cobertura!=null?`Referencia de laboratorio (conjunto de prueba, umbral ${fp(CFG.umbral)}): ${fp(R.cobertura)} decididos y ${fp(R.exactitud_aceptados)} de exactitud entre aceptados. `:"")+
  "Un umbral más alto reduce los errores pero deriva más casos a revisión. La exactitud usa solo los casos etiquetados, que son sobre todo los dudosos."}

/* ---------- monitoreo ---------- */
const RANGOS=[["15 min",900],["1 hora",3600],["24 horas",86400],["Todo",0]];
function pintarRango(){$("rango").innerHTML=RANGOS.map(([t,s])=>`<button class="${s===rango?'on':''}" onclick="rango=${s};pintar()">${t}</button>`).join("")}
function datos(){if(!REG)return[];if(!rango)return REG.registro;const lim=REG.ahora-rango;return REG.registro.filter(e=>e.t>=lim)}
function pct(a,p){if(!a.length)return 0;const s=[...a].sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(p*(s.length-1)+.5))]}

function insights(d,nv,c,ms,pares){
 const L=[],add=(c,t)=>L.push([c,t]),n=nv.length;
 if(!d.length)return[["#94a3b8","Todavía no hay lecturas en este periodo. Use la pestaña Prototipo para generar datos."]];
 if(n<30)add("#94a3b8",`Solo ${n} ingresos analizados: las conclusiones serán más fiables con más de 100.`);
 if(n){const sv=c.SIN_VERIFICAR/n;
  if(sv>.25)add("#f59e0b",`El ${fp(sv,0)} de los ingresos queda sin verificar. Mejore la iluminación, use fondo uniforme o acerque el objeto; si continúa, revise el umbral en el simulador.`);
  else add("#10b981",`El ${fp(1-sv,0)} de los ingresos se decide automáticamente (${fp(c.SIN_VERIFICAR/n,0)} pasa a revisión).`);
  const porM={};nv.forEach(e=>{(porM[e.modo]=porM[e.modo]||{n:0,r:0}).n++;if(e.decision==="RECHAZADO")porM[e.modo].r++});
  Object.entries(porM).filter(([k,v])=>v.n>=10&&v.r/v.n>.3).forEach(([k,v])=>add("#ef4444",`El contenedor «${k}» rechaza el ${fp(v.r/v.n,0)} de sus ingresos: puede haber residuos mal depositados o confusiones del modelo. Revise los rechazados en la bandeja.`))}
 if(pares.length&&pares[0][1]>=3)add("#f59e0b",`Las dudas se concentran en ${NOM[pares[0][0].split("|")[0]]} → ${NOM[pares[0][0].split("|")[1]]} (${pares[0][1]} casos). Reúna más ejemplos de esos materiales.`);
 const nvac=c.VACIO;if(nvac)add("#94a3b8",`El contenedor se vio vacío en ${nvac} lecturas (${fp(nvac/d.length,0)} del periodo).`);
 if(ms.length&&pct(ms,.95)>500)add("#ef4444",`La inferencia es lenta (p95 ${pct(ms,.95).toFixed(0)} ms). Cierre otras aplicaciones o reduzca la frecuencia del modo continuo.`);
 const pend=nv.filter(e=>e.decision==="SIN_VERIFICAR"&&!e.etiqueta&&e.archivo).length;
 if(pend)add("#3b82f6",`Hay ${pend} casos sin verificar pendientes de etiquetar. Etiquetarlos alimenta el reentrenamiento.`);
 const lab=d.filter(e=>e.etiqueta);if(lab.length>=10)add("#a855f7",`De ${lab.length} casos revisados, el modelo acertó ${fp(lab.filter(e=>e.pred===e.etiqueta).length/lab.length,0)} (muestra sesgada hacia los dudosos).`);
 return L}

function pintar(){
 if(!REG)return;pintarRango();const d=datos(),nv=d.filter(e=>e.decision!=="VACIO"),n=nv.length;
 const c={ACEPTADO:0,RECHAZADO:0,SIN_VERIFICAR:0,VACIO:0};d.forEach(e=>{c[e.decision]=(c[e.decision]||0)+1});
 const ms=d.map(e=>e.ms).filter(x=>x>0),rit=REG.registro.filter(e=>e.t>=REG.ahora-300).length/5;
 $("act").textContent="Actualizado "+new Date().toLocaleTimeString();
 $("kpis").innerHTML=[["Ingresos analizados",n,"var(--mc)"],["Aceptados",c.ACEPTADO,"#10b981"],["Rechazados",c.RECHAZADO,"#ef4444"],["Sin verificar",c.SIN_VERIFICAR,"#f59e0b"],
  ["Decisión automática",n?fp((c.ACEPTADO+c.RECHAZADO)/n,0):"—","#14b8a6"],["Lecturas «vacío»",c.VACIO,"#94a3b8"],
  ["Inferencia p50 / p95",ms.length?pct(ms,.5).toFixed(0)+" / "+pct(ms,.95).toFixed(0)+" ms":"—","#3b82f6"],["Lecturas por minuto",rit.toFixed(1).replace(".",","),"#6366f1"],["Umbral activo",fp(REG.umbral),"#a855f7"]]
  .map(([a,b,k])=>`<div class="card kpi" style="--c:${k}"><b>${b}</b><span>${a}</span></div>`).join("");
 const a=n?100*c.ACEPTADO/n:0,b=n?a+100*c.RECHAZADO/n:0;
 $("donut").innerHTML=`<div class="dn" style="background:${n?`conic-gradient(#10b981 0 ${a}%,#ef4444 ${a}% ${b}%,#f59e0b ${b}% 100%)`:'var(--line)'}"><div>${n?a.toFixed(0)+"%":"—"}</div></div>
  <div class="leg">${["ACEPTADO","RECHAZADO","SIN_VERIFICAR"].map(k=>`<div style="--c:${DEC[k].col}"><i></i><span>${DEC[k].t}: <b>${c[k]||0}</b> (${n?(100*(c[k]||0)/n).toFixed(0):0} %)</span></div>`).join("")}<div style="color:var(--mut);font-size:12px;margin-top:10px">El centro indica la tasa de aceptación</div></div>`;
 const P={};nv.filter(e=>e.decision==="SIN_VERIFICAR"&&e.p2).forEach(e=>{const k=e.pred+"|"+e.p2;P[k]=(P[k]||0)+1});
 const pares=Object.entries(P).sort((x,y)=>y[1]-x[1]).slice(0,6);
 $("ins").innerHTML=insights(d,nv,c,ms,pares).map(([k,t])=>`<div style="--c:${k}">${t}</div>`).join("");
 $("tl").innerHTML=timeline(d,REG.ahora,rango);$("tlg").innerHTML=leyenda(ORD);
 $("hc").innerHTML=histConf(d,REG.umbral);$("hcg").innerHTML=leyenda(["ACEPTADO","RECHAZADO","SIN_VERIFICAR"]);
 $("sc").innerHTML=serieConf(d,REG.umbral);
 /* por contenedor */
 const M={};nv.forEach(e=>{(M[e.modo]=M[e.modo]||{ACEPTADO:0,RECHAZADO:0,SIN_VERIFICAR:0})[e.decision]++});
 $("cont2").innerHTML=Object.keys(M).length?Object.entries(M).map(([k,v])=>{const t=v.ACEPTADO+v.RECHAZADO+v.SIN_VERIFICAR;
  return `<div class="row"><div class="top-row"><span>${k}</span><span style="color:var(--mut)">${t} ingresos</span></div><div class="sbar">${["ACEPTADO","RECHAZADO","SIN_VERIFICAR"].map(o=>`<i style="flex:${v[o]};background:${DEC[o].col}" title="${DEC[o].s}: ${v[o]}"></i>`).join("")}</div></div>`}).join(""):vacioMsg();
 $("cog").innerHTML=leyenda(["ACEPTADO","RECHAZADO","SIN_VERIFICAR"]);
 /* materiales */
 const cl=REG.clases,mx=Math.max(1,d.length),K={};cl.forEach(k=>K[k]={n:0,s:0});d.forEach(e=>{if(K[e.pred]){K[e.pred].n++;K[e.pred].s+=e.conf}});
 $("clases").innerHTML=cl.map(k=>`<div class="top-row"><span>${NOM[k]||k}</span><span style="color:var(--mut)">${K[k].n}${K[k].n?" · "+fp(K[k].s/K[k].n,0):""}</span></div><div class="bar" style="--c:${CC[k]}"><i style="width:${100*K[k].n/mx}%"></i></div>`).join("");
 simular();
 $("pares").innerHTML=pares.length?pares.map(([k,v])=>{const [x,y]=k.split("|");return `<div class="row"><div class="top-row"><span>${chip(x)} → ${chip(y)}</span><span style="color:var(--mut)">${v}</span></div><div class="bar" style="--c:${CC[x]}"><i style="width:${100*v/pares[0][1]}%"></i></div></div>`}).join(""):vacioMsg("Sin casos dudosos en este periodo");
 const tr=Object.values(REG.reent).reduce((a,b)=>a+b,0);
 $("reent").innerHTML=`<div class="kpi card" style="--c:#6366f1;margin-bottom:14px;padding:12px 16px"><b>${tr}</b><span>imágenes etiquetadas</span></div>`+(tr?cl.filter(k=>REG.reent[k]).map(k=>`<div class="top-row"><span>${NOM[k]}</span><span style="color:var(--mut)">${REG.reent[k]}</span></div><div class="bar" style="--c:${CC[k]}"><i style="width:${100*REG.reent[k]/Math.max(...Object.values(REG.reent))}%"></i></div>`).join(""):vacioMsg("Aún no hay casos etiquetados"));
 /* matriz de confusión: filas = etiqueta real, columnas = predicción */
 const lab=d.filter(e=>e.etiqueta),mat={};lab.forEach(e=>{(mat[e.etiqueta]=mat[e.etiqueta]||{})[e.pred]=(mat[e.etiqueta][e.pred]||0)+1});
 $("mn").textContent=lab.length?`${lab.length} casos revisados`:"";
 if(!lab.length){$("mat").innerHTML=vacioMsg("Etiquete casos en la bandeja para construir la matriz");$("prf").innerHTML=""}
 else{const mxc=Math.max(1,...cl.flatMap(r=>cl.map(p=>(mat[r]||{})[p]||0)));
  let h=`<table><tr><th>real ↓ / predicho →</th>${cl.map(x=>`<th style="color:color-mix(in srgb,${CC[x]} 75%,var(--tx))">${SH[x]}</th>`).join("")}</tr>`;
  cl.forEach(r=>{h+=`<tr><th style="color:color-mix(in srgb,${CC[r]} 75%,var(--tx))">${SH[r]}</th>`+cl.map(p=>{const v=(mat[r]||{})[p]||0;const al=v?(.15+.6*v/mxc):0;
   return `<td style="${v?`background:${r===p?`rgba(16,185,129,${al})`:`rgba(239,68,68,${al})`};font-weight:700`:''}">${v||""}</td>`}).join("")+"</tr>"});
  $("mat").innerHTML=h+"</table>";
  let q=`<table><tr><th>Clase</th><th>Revisados (real)</th><th>Recall</th><th>Precisión</th></tr>`;
  cl.forEach(k=>{const tp=(mat[k]||{})[k]||0,fil=Object.values(mat[k]||{}).reduce((a,b)=>a+b,0),col=cl.reduce((a,r)=>a+((mat[r]||{})[k]||0),0);
   if(fil||col)q+=`<tr><td>${chip(k)}</td><td>${fil}</td><td>${fil?fp(tp/fil,0):"—"}</td><td>${col?fp(tp/col,0):"—"}</td></tr>`});
  $("prf").innerHTML=q+"</table>"}
 /* últimos registros */
 $("ult").innerHTML="<table><tr><th>Hora</th><th>Contenedor</th><th>Detectado</th><th>Confianza</th><th>2ª opción</th><th>Decisión</th><th>ms</th></tr>"+
  d.slice(-8).reverse().map(e=>`<tr><td style="color:var(--mut)">${e.t?new Date(e.t*1000).toLocaleTimeString():"—"}</td><td><b>${e.modo}</b></td><td>${chip(e.pred)}</td><td>${fp(e.conf,0)}</td><td style="color:var(--mut)">${e.p2?(NOM[e.p2]||e.p2)+" "+fp(e.c2,0):"—"}</td><td><span class="dtag" style="--dc:${DEC[e.decision].col}">${DEC[e.decision].t}</span></td><td style="color:var(--mut)">${(e.ms||0).toFixed(0)}</td></tr>`).join("")+"</table>";
 /* bandeja (no se repinta mientras hay un selector abierto) */
 const f=$("gfil").value,pend=nv.filter(e=>e.decision==="SIN_VERIFICAR"&&!e.etiqueta&&e.archivo);
 $("np").textContent=pend.length+" pendientes";
 if(!$("gal").contains(document.activeElement)){
  const lista=d.filter(e=>e.archivo&&(f==="todos"||(f==="pend"?(e.decision==="SIN_VERIFICAR"&&!e.etiqueta):e.decision===f))).slice(-24).reverse();
  $("gal").innerHTML=lista.map(e=>`<div style="--c:${CC[e.pred]}"><img src="/img/${e.archivo}" loading="lazy"><div style="display:flex;justify-content:space-between;align-items:center">${chip(e.pred)} <b>${fp(e.conf,0)}</b></div><div style="margin-top:6px"><span class="dtag" style="--dc:${DEC[e.decision].col}">${DEC[e.decision].s}</span></div>
   <select onchange="corregir('${e.archivo}',this.value)"><option value="">Asignar etiqueta real...</option>${cl.map(x=>`<option value="${x}" ${e.etiqueta===x?"selected":""}>${NOM[x]}</option>`).join("")}</select></div>`).join("")||"<span style='color:var(--mut);font-size:13.5px'>No hay casos para este filtro</span>"}}
async function cargar(){try{REG=await (await fetch("/api/registro")).json();CFG.umbral=REG.umbral;CFG.referencia=REG.referencia;pintar()}catch(e){console.log(e)}}
async function corregir(a,l){await fetch("/api/corregir",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({archivo:a,etiqueta:l})});$("gal").blur();cargar()}
$("simU").oninput=simular;
setInterval(()=>{if(!$("v2").classList.contains("hid"))cargar()},4000);

/* ---------- prototipo: clasificar ---------- */
async function clasificar(){
 if(ocupado)return;ocupado=true;$("vid").classList.add("scan");
 const v=$("v"),s=Math.min(v.videoWidth,v.videoHeight)*$("zona").value/100,c=document.createElement("canvas");c.width=c.height=224;
 c.getContext("2d").drawImage(v,(v.videoWidth-s)/2,(v.videoHeight-s)/2,s,s,0,0,224,224);
 try{const r=await (await fetch("/api/clasificar",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({imagen:c.toDataURL("image/jpeg",.9),categorias:MODOS[modo].c,modo})})).json();
 if(r.error){console.log(r.error)}else{
 const d=DEC[r.decision];$("res").style.setProperty("--dc",d.col);
 $("res").innerHTML=`<div class="ring"><svg viewBox="0 0 100 100" width="118" height="118"><circle cx="50" cy="50" r="42" class="bg"/><circle cx="50" cy="50" r="42" class="fg" style="stroke-dasharray:264;stroke-dashoffset:${264*(1-r.conf)}"/></svg><b>${(r.conf*100).toFixed(0)}%</b></div>
 <div><div class="ico">${svg(d.p,30)}</div><div class="big">${d.t}</div><div style="font-size:14px;color:var(--mut)">${r.decision==="VACIO"?"No hay residuo en la zona de captura":"Detectado: "+chip(r.clase)}</div></div>`;
 $("top").innerHTML=r.top.map(x=>`<div class="top-row"><span>${NOM[x[0]]||x[0]}</span><span style="color:var(--mut)">${(x[1]*100).toFixed(1)} %</span></div><div class="bar" style="--c:${CC[x[0]]}"><i style="width:${x[1]*100}%"></i></div>`).join("");
 $("info").textContent=`Tiempo de inferencia: ${r.ms.toFixed(0)} ms · Umbral de confianza: ${fp(CFG.umbral)}`;
 hist.unshift({...r,modo});hist=hist.slice(0,7);
 $("hist").innerHTML=hist.map(h=>`<div style="--dc:${DEC[h.decision].col}"><span>${chip(h.clase)} <b>${(h.conf*100).toFixed(0)} %</b></span><span class="dtag" style="--dc:${DEC[h.decision].col}">${DEC[h.decision].t}</span></div>`).join("");
 lect.push({conf:r.conf,decision:r.decision,pred:r.clase});lect=lect.slice(-40);$("spark").innerHTML=serieConf(lect,CFG.umbral,150)}}catch(e){console.log(e)}
 $("vid").classList.remove("scan");ocupado=false}

const SOL='<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5"/>',LUNA='<path d="M21 13A9 9 0 1 1 11 3a7 7 0 0 0 10 10z"/>';
function tema(t){if(typeof t!=="string")t=document.documentElement.dataset.theme==="light"?"dark":"light";
 document.documentElement.dataset.theme=t;try{localStorage.setItem("tema",t)}catch(e){}
 $("tt").innerHTML=t==="light"?svg(LUNA,15)+" Oscuro":svg(SOL,15)+" Claro"}
let t0=null;try{t0=localStorage.getItem("tema")}catch(e){}
tema(t0||(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark"));

(async()=>{try{CFG=await (await fetch("/api/config")).json()}catch(e){}
 $("umb").textContent="umbral "+fp(CFG.umbral);setModo("Plásticos");iniciarCamara()})();
</script></body></html>"""

if __name__ == "__main__":
    print(f"Abra http://localhost:5000  (acceso desde la red: {'sí' if HOST == '0.0.0.0' else 'no, solo este equipo'})")
    app.run(host=HOST, port=5000, debug=False, threaded=True)