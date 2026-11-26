# ReCycle Vision

Control visual inteligente de ingreso de residuos para contenedores inteligentes de la empresa **ReCycle**.

Una red neuronal convolucional (MobileNetV2 con transferencia de aprendizaje) reconoce el tipo de residuo que ingresa a un contenedor y una regla de decisión **acepta, rechaza o deja sin verificar** el ingreso según las categorías asignadas a ese contenedor. Un Arduino con servomotor abre la compuerta solo cuando el ingreso es aceptado.

> Proyecto académico — Redes Neuronales, Semestre 8, Paralelo A — Universidad de Aquino Bolivia (UDABOL), Ingeniería de Sistemas.
> **Autores:** Ronaldo Jose Canaviri Tola y Juan David Figueredo Cruz
> **Docente:** Ing. Roger Rolando Perez Rojas

---

## Cómo funciona

```
Cámara (navegador) → Flask → Modelo MobileNetV2 → Regla de decisión → Arduino (compuerta)
```

1. El navegador captura una imagen de la cámara y la envía a `/api/clasificar`.
2. Flask la redimensiona a 224 × 224 y el modelo devuelve una probabilidad por clase.
3. La regla de decisión compara la clase y la confianza con las categorías del contenedor y el umbral (0,86):

| Situación | Resultado |
|---|---|
| Confianza menor al umbral | `SIN_VERIFICAR` (se guarda para revisión remota) |
| Clase `vacio` con confianza ≥ umbral | `VACIO` (no se decide ni se registra) |
| Clase dentro de las categorías del contenedor | `ACEPTADO` (abre la compuerta) |
| Clase fuera de las categorías del contenedor | `RECHAZADO` |

4. Cada lectura queda registrada para monitoreo, exportación a CSV y reentrenamiento.

## Clases

Diez clases: `plasticos`, `carton`, `papel`, `vidrio`, `metales`, `organicos`, `textiles`, `baterias`, `otros` y `vacio` (contenedor sin objeto).

## Resultados (conjunto de prueba, 1310 imágenes)

| Métrica | Valor |
|---|---|
| Exactitud | 94,12 % |
| F1 macro | 0,943 |
| Tamaño del modelo | 21,87 MB |
| Inferencia en CPU (Colab) | 245 ms por imagen (mediana) |
| Umbral de confianza | 0,86 |
| Exactitud entre los casos aceptados | 97,84 % |
| Decisión automática | 89,36 % |
| Sin verificar | 10,64 % |

Las confusiones principales se dan entre vidrio, plásticos y metales. Con la cámara real el desempeño fue menor que en el conjunto de prueba, porque los datasets públicos tienen fondos y condiciones distintas a las del contenedor.

## Estructura del repositorio

```
recycle_vision/
├── src/
│   ├── web_app.py            # Aplicación web (Flask): prototipo + monitoreo y análisis
│   ├── control_ingreso.py    # Regla de decisión, clasificador y registro
│   ├── camara.py             # Prueba con cámara desde la consola (OpenCV)
│   └── compuerta_arduino.py  # Comunicación serial con el Arduino
├── models_colab/
│   ├── recycle_mobilenet_colab.keras          # Modelo entrenado
│   ├── recycle_mobilenet_colab.weights.h5     # Pesos del modelo
│   ├── clases.json                            # Orden de las clases
│   ├── umbral_elegido_colab.json              # Umbral y su desempeño
│   ├── metricas_test_colab.json               # Métricas en prueba
│   ├── historial.json                         # Historial de entrenamiento
│   └── *.csv                                  # Reportes y barrido de umbrales
├── notebooks/                # Cuaderno de entrenamiento en Google Colab
└── requirements.txt
```

## Instalación

Requisitos: Python 3.10 o superior, una cámara web y (opcional) un Arduino.

```bash
git clone https://github.com/TU_USUARIO/recycle-vision.git
cd recycle-vision

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

## Uso

### Aplicación web

```bash
python src/web_app.py
```

Abrir <http://localhost:5000>. Tiene dos pestañas: **Prototipo** (clasificación en vivo con cámara) y **Monitoreo y análisis** (indicadores, gráficos, corrección de etiquetas y exportación a CSV).

Variables de entorno opcionales:

| Variable | Efecto |
|---|---|
| `RECYCLE_ARDUINO_PORT=COM3` | Fija el puerto serial del Arduino (si no, usa el primero disponible) |
| `RECYCLE_HOST=0.0.0.0` | Permite el acceso desde otros equipos de la red |
| `RECYCLE_DEMO=1` | Usa un modelo falso, solo para probar la interfaz |

### API

| Ruta | Método | Descripción |
|---|---|---|
| `/api/clasificar` | POST | Clasifica una imagen y aplica la regla de decisión |
| `/api/config` | GET | Umbral y clases |
| `/api/registro` | GET | Registro de eventos |
| `/api/corregir` | POST | Asigna la etiqueta real a un caso y lo copia a la carpeta de reentrenamiento |
| `/api/exportar.csv` | GET | Exporta el registro en CSV |

### Conexión con el Arduino

- Comunicación serial a **9600 baudios**.
- Cuando la decisión es `ACEPTADO`, Python envía `ABRIR\n`; el servomotor se mueve de 0° a 70° y regresa a 0°.
- Hay un tiempo de espera de 2,5 s entre aperturas.
- Si no se detecta un Arduino, el sistema funciona en **modo simulación** y solo imprime la acción.

## Entrenamiento

El modelo se entrenó en Google Colab con transferencia de aprendizaje en dos etapas:

1. **Cabeza de clasificación** con la red base congelada (Adam, tasa de aprendizaje 1e-3, 6 épocas).
2. **Ajuste fino** de las últimas 30 capas, sin tocar BatchNorm (Adam, tasa de aprendizaje 1e-5, hasta 15 épocas).

Se usó aumento de datos (giro, rotación, zoom, brillo y contraste), Dropout de 0,3, pesos de clase, parada temprana y lotes de 32 imágenes.

**Datasets:** [TrashNet](https://github.com/garythung/trashnet) y *Garbage classification (12 classes)*, unificados en nueve categorías más la clase `vacio`, con 8.923 imágenes en total. Los datos **no están incluidos** en este repositorio.

## Limitaciones

- El modelo clasifica **un objeto por imagen**; la detección de varios objetos con YOLO no dio resultados aceptables con datasets existentes.
- La clase `vacio` tiene muy pocas imágenes de prueba (13), por lo que su evidencia es limitada.
- El desempeño con la cámara es parcial y depende del fondo, la iluminación y que el objeto esté completo y centrado.
- Recomendación: construir un dataset propio, con imágenes tomadas en las condiciones reales del contenedor, y reentrenar.

## Autores

- Ronaldo Jose Canaviri Tola
- Juan David Figueredo Cruz

Universidad de Aquino Bolivia (UDABOL) — Facultad de Ciencias y Tecnología, Ingeniería de Sistemas.
