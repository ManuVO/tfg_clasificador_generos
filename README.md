# 🎵 Clasificador de géneros musicales (WIP)

Este repositorio forma parte de un **Trabajo Fin de Grado (TFG)** dedicado a la **clasificación automática de géneros musicales** a partir de audio.  
El proyecto implementa un pipeline completo: desde la **preparación de un dataset**, la **generación de espectrogramas log-mel**, el **entrenamiento de redes neuronales convolucionales (CNNs)** y su **evaluación**, hasta sentar las bases de una futura aplicación capaz de predecir el género musical de cualquier archivo de audio.

---

## 🔄 Flujo del pipeline

### Versión gráfica
*(Ubicada en `docs/pipeline.png`, se puede generar con Draw.io, Mermaid u otra herramienta.)*

### Versión textual
```text
🎼 Dataset
       │
       ▼
🎚️ Preprocesamiento + extracción de características
       │
       ▼
🧠 Entrenamiento de la red neuronal
       │
       ▼
📦 Modelo entrenado
       │
       ▼
🎵 Nuevo archivo de audio
       │
       ▼
✅ Clasificación → Género musical
```

---

## 🎯 Objetivos del proyecto
- 🚀 **Meta a corto plazo (MVP):** ofrecer una experiencia reproducible desde terminal: el usuario prepara un dataset, entrena un modelo y obtiene métricas y reportes.  
- 📦 **Estado actual:** scripts para preprocesar datos, entrenar y evaluar modelos, además de utilidades para comprobar la integridad del pipeline.  
- 🌈 **Visión a largo plazo:** evolucionar hacia una interfaz (desktop o móvil) que permita cargar o grabar audio y obtener el género musical en tiempo real.  
- 🔮 **Posibles extensiones:** integrar datasets adicionales como FMA, MSD+Last.fm, MagnaTagATune, MTG-Jamendo, Homburg, Ballroom, ISMIR 2004, GiantSteps, AudioSet o Spotify MPD.  

---

## 🗺️ Mapa del repositorio
```text
📁 tfg_clasificador_generos
├── 📂 configs/             → Plantillas de configuración (base, datasets, overrides)
├── 📂 data/                → Datos brutos (`raw`) y procesados (`processed`)
├── 📂 experiments/         → Resultados de entrenamiento (checkpoints, métricas, logs)
├── 📂 reports/             → Salidas auxiliares (figuras, informes, comparativas)
├── 📂 scripts/             → Utilidades rápidas (ej. `test_pipeline.py`)
├── 📂 src/
│   ├── 📂 data/            → Descarga y preprocesamiento de datasets
│   ├── 📂 features/        → Extracción de espectrogramas y augmentations
│   ├── 📂 models/          → Definición de arquitecturas (ej. `CNNBaseline`)
│   ├── 🧾 train.py         → Entrenamiento end-to-end
│   └── 🧾 eval.py          → Evaluación de checkpoints
├── 🧾 config.yaml          → Configuración maestra del proyecto
├── 🧾 requirements.txt     → Dependencias base
├── 🧾 init_env.ps1         → Script para entorno virtual en PowerShell
└── 🧾 README.md            → Esta guía
```

---

## 🧰 ¿Qué encontrarás aquí?
- 🎚️ **Preprocesamiento reproducible:** descarga, segmentación y normalización de datasets.  
- 📊 **Extracción de características:** espectrogramas log-mel, estadísticas y augmentations.  
- 🧠 **Modelos base:** CNN ligera con early stopping y scheduler.  
- 📈 **Entrenamiento controlado:** configuraciones YAML, semillas fijas y logs en TensorBoard.  
- ✅ **Evaluación automática:** métricas CSV/JSON, clasificación por género y matriz de confusión.  
- 🧪 **Verificación rápida del pipeline:** script sintético para comprobar la correcta conversión audio→espectrograma.  

---

## 🧱 Requisitos previos
- 🐍 Python 3.10 (recomendado) con `pip` y `venv`.  
- 🎧 FFmpeg en el `PATH`.  
- 💻 Sistemas soportados: Windows 10/11, macOS, Linux.  
- ⚡ Opcional: GPU con CUDA para acelerar el entrenamiento.  
- 📥 Opcional: Kaggle CLI configurada para descargar GTZAN automáticamente.  

---

## ⚙️ Configuración inicial
### Windows / PowerShell
```powershell
./init_env.ps1
```
Crea `.venv`, instala dependencias, configura `PYTHONPATH` y activa el entorno.

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎧 Preparar el dataset GTZAN
1. Descarga el dataset GTZAN desde una fuente fiable.  
2. Colócalo dentro de `data/raw/`, quedando así:  
   ```text
   data/raw/gtzan/genres_original/<GENRE>/*.wav
   ```
   *(El dataset ya viene estructurado en carpetas de géneros; basta con ubicarlo en `data/raw/`.)*  
3. Ejecuta el preprocesado:  
   ```bash
   python src/data/preprocess.py --dataset gtzan --project-config config.yaml
   ```
   Esto genera:  
   - `data/processed/gtzan/segments/`  
   - `data/processed/gtzan/metadata.csv`  
   - `data/processed/gtzan/norm_stats.json`  

---

## 🏋️ Entrenar un modelo
```bash
python src/train.py --dataset gtzan --project-config config.yaml
```
Resultados en `experiments/<nombre_experimento>/run_<timestamp>/`:  
- `checkpoints/best_model.pt`  
- `metrics.csv`, `classification_report.txt`  
- Logs para TensorBoard (`tensorboard --logdir experiments`)  

---

## 🧪 Evaluar un checkpoint
```bash
python src/eval.py --run_dir experiments/<experimento>/run_<timestamp>/
```
Genera reportes (`classification_report.txt`, `confusion_matrix.png`) en el directorio del run o en `--out_dir` personalizado.

---

## ⚡ Personalizar la configuración
Configura datasets, parámetros de audio, augmentations, arquitectura y entrenamiento editando los YAML de `configs/`.

Ejemplo:
```bash
python src/train.py --config configs/mi_experimento.yaml
```

### Ajustes recomendados para `gtzan`
- **Augmentations en forma de onda (`augmentation.waveform`)** → se han rebajado las probabilidades a valores entre 0.3 y 0.4 para que cada clip tenga más frecuencia sin distorsiones extremas. Así la red aprende patrones limpios y, al mismo tiempo, mantiene algo de diversidad para generalizar.
- **SpecAugment (`augmentation.spec_augment`)** → ahora aplica menos máscaras (2 de frecuencia, 3 temporales) con un ancho moderado. Esto oculta partes del espectrograma para evitar sobreajuste, pero deja suficiente información para que la validación no se dispare.
- **Épocas y tamaño de lote** → `training.epochs` sube a 40 y `batch_size` se mantiene en 32. Con lotes pequeños el optimizador ve más iteraciones por época y la paciencia extra permite que el aprendizaje estabilice antes de que el early stopping corte.
- **Tasa de aprendizaje y weight decay** → `learning_rate` se fija en `1e-3` para arrancar más fuerte y `weight_decay` se reduce a `1e-4`. Este equilibrio acelera la caída de la pérdida de entrenamiento sin forzar demasiada penalización L2 que frene la capacidad del modelo.
- **Scheduler** → `ReduceLROnPlateau` conserva un factor moderado (`0.5`) y amplía la `patience` a 3 con `threshold=0.005`. Solo baja el *LR* cuando la validación deja de mejorar de manera significativa, evitando reducciones prematuras.
- **Early stopping** → `patience=10` y `min_delta=0.005` filtran mejoras marginales. El entrenamiento solo se detendrá cuando la validación esté estancada varios ciclos consecutivos.
- **Semilla (`seed`)** → sigue en 42 para reproducibilidad. Cambiarla te permite comprobar la robustez del entrenamiento.
- **Barra de progreso (`progress_bar`)** → permanece desactivada en la base y se activa únicamente en `configs/datasets/gtzan/training.yaml`. Puedes omitir esta clave en datasets donde no quieras barra; el código asume `false` si no existe.

---

## 🔍 Comprobar el pipeline
```bash
python scripts/test_pipeline.py
```
Verifica con un audio sintético que la transformación a espectrograma funciona correctamente.  

---

## 📝 Notas adicionales
- Los checkpoints se guardan en `experiments/`; el mejor modelo se guarda como `checkpoints/best_model.pt`.
- Ajusta `training.num_workers` si tu hardware limita hilos.
- Controla la regularización L2 del optimizador con `training.weight_decay` (por defecto `1e-4`; fija `0.0` para desactivarla).
- Cambios en rutas/datasets → ejecutar de nuevo el preprocesado.

---

## 🗓️ Hoja de ruta
1. **CLI de inferencia:** comando `python src/predict.py --audio <ruta>` que devuelva el género estimado.  
2. **Validación externa:** probar con audios reales y documentar resultados.  
3. **Feedback de usuarios:** mejorar usabilidad y mensajes en terminal.  
4. **Extensión a nuevos datasets:** explorar FMA, MSD+Last.fm, MagnaTagATune, MTG-Jamendo, Homburg, Ballroom, ISMIR 2004, GiantSteps, AudioSet, Spotify MPD.  
5. **Interfaz futura:** prototipos visuales (desktop/móvil).  

---

📌 **Estado:** proyecto en construcción (WIP).  
Cada contribución acerca este clasificador a su visión final: una herramienta reproducible, accesible y académicamente sólida para la clasificación de géneros musicales.  
