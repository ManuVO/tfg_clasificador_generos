# 🎵 Clasificador de géneros musicales – Proyecto de Fin de Grado (WIP – Work in Progress) ![WIP](https://img.shields.io/badge/status-WIP-orange)

Este repositorio alberga un **clasificador automático de géneros musicales** basado en redes neuronales. Abarca todo el pipeline: desde la preparación de un dataset y la generación de espectrogramas log-mel, hasta el entrenamiento y evaluación de modelos. El objetivo final es proporcionar una herramienta que permita predecir el género de cualquier archivo de audio, a la vez que sirve como caso de estudio para técnicas de **aprendizaje automático** y buenas prácticas de **ingeniería de software**.

> ℹ️ **Estado**: proyecto en construcción (MVP). Este documento evoluciona junto con el código y la planificación. Cada contribución acerca el proyecto a su visión final: una herramienta reproducible, accesible y académicamente sólida para la clasificación de géneros musicales.

---
## 🎯 Hitos del proyecto

Para ofrecer una visión global del avance, se establecen varios hitos que marcan momentos clave en el desarrollo.  
A diferencia del backlog, estos hitos representan grandes entregables y puntos de control:

1. **Cierre del análisis y captura de requisitos** – recopilación y validación de todos los requisitos funcionales y no funcionales.  
2. **Diseño y arquitectura definidos** – elección de la arquitectura del sistema y especificación de módulos y flujos de datos.  
3. **Preprocesamiento y pipeline de datos operativo** – scripts de descarga, segmentación y normalización listos.  
4. **Modelo CNN entrenado** – red neuronal implementada y entrenamiento base completado con el dataset GTZAN.  
5. **Evaluación exhaustiva y ajuste de hiperparámetros** – generación de métricas, análisis de resultados y refinamiento de configuraciones.  
6. **Implementación de la CLI de inferencia** – desarrollo de un comando que permita predecir el género de audios nuevos.  
7. **Integración de datasets adicionales y pruebas externas** – incorporación de nuevos conjuntos de datos y validación con audios reales.  
8. **Entrega de la memoria y defensa del TFG** – redacción de la documentación final y presentación del proyecto.  

Estos hitos permiten orientar el progreso general sin detallar todas las tareas individuales que ya aparecen en el backlog.

---

## 📑 Metodología y organización

El trabajo se gestiona siguiendo la metodología ágil **SCRUM**, que divide el desarrollo en **sprints** cortos con entregas incrementales.  

Para visualizar el progreso se utiliza un tablero **Kanban**, con columnas «Por hacer», «En progreso», «En revisión» y «Hecho».  

Las tareas se crean a partir de **historias de usuario** que describen requisitos funcionales y técnicos.  

A continuación se ofrece una visión general del **backlog actual**, incluyendo las tareas completadas y las planificadas.

---

## 📋 Backlog y estado de tareas


| Estado | Tarea | Sprint | Notas |
|--------|-------|--------|-------|
| ✅ | Configurar repositorio Git y estructura de carpetas | Sprint 0 | Proyecto inicializado y estructura preparada |
| ✅ | Definir alcance, objetivos y meta del proyecto | Sprint 0 | Documento de objetivos listo |
| ✅ | Investigación del estado del arte y bibliografía | Sprint 0 | Revisión realizada para contextualizar |
| ✅ | Seleccionar y descargar dataset GTZAN | Sprint 0 | Dataset ubicado en `data/raw/gtzan/` |
| ✅ | Crear entorno virtual y configurar dependencias | Sprint 0 | Uso de `.venv` e instalación de `requirements.txt` |
| ✅ | Planificar sprints y hoja de ruta inicial | Sprint 0 | Backlog inicial y roadmap definidos |
| 🔄 | Capturar requisitos funcionales y no funcionales | Sprint 1 | Documentar necesidades del sistema |
| 🔄 | Definir historias de usuario y backlog priorizado | Sprint 1 | Convertir requisitos en user stories |
| 🔄 | Especificar criterios de aceptación | Sprint 1 | Qué debe cumplirse para considerar completada una tarea |
| 🔄 | Modelado conceptual (diagramas de clases/ER) | Sprint 1 | Identificar entidades y relaciones |
| 🔄 | Validar y refinar requisitos con el tutor | Sprint 1 | Revisar con stakeholders |
| ⏳ | Diseñar arquitectura de software y módulos | Sprint 2 | Definir capas y responsabilidades |
| ✅ | Seleccionar framework de deep learning (PyTorch) | Sprint 2 | Framework establecido para el proyecto |
| ⏳ | Definir estructura de datos y formatos | Sprint 2 | Especificar I/O de los módulos |
| ⏳ | Diseñar red neuronal base (CNN) | Sprint 2 | Elegir arquitectura inicial |
| ⏳ | Plan de pruebas unitarias y estructura de CI/CD | Sprint 2 | Integrar pruebas automatizadas |
| ✅ | Desarrollar script de descarga y preprocesamiento | Sprint 3 | Preprocesamiento reproducible implementado |
| ✅ | Segmentar y normalizar audio | Sprint 3 | Generación de clips y normalización incluida |
| ✅ | Implementar extracción de espectrogramas log-mel | Sprint 3 | Extracción de características completada |
| ✅ | Generar metadata y estadísticas del dataset | Sprint 3 | Archivo `metadata.csv` y `norm_stats.json` |
| ✅ | Verificar pipeline con pruebas sintéticas | Sprint 3 | Script `test_pipeline.py` disponible |
| ✅ | Implementar modelo CNN base | Sprint 4 | `models/CNNBaseline` definido |
| ✅ | Implementar loop de entrenamiento y early stopping | Sprint 4 | Script `train.py` con validación y parada temprana |
| ✅ | Implementar métricas y logging (TensorBoard) | Sprint 4 | Métricas y logs generados en `experiments/` |
| ✅ | Entrenar modelo con GTZAN | Sprint 4 | Primer entrenamiento completado |
| 🔄 | Ajustar augmentations e hiperparámetros | Sprint 4 | Ajuste según recomendaciones |
| ✅ | Guardar checkpoints y modelos óptimos | Sprint 4 | Checkpoints almacenados |
| ✅ | Desarrollar script de evaluación de checkpoints | Sprint 5 | Script `eval.py` implementado |
| ✅ | Generar reportes de métricas y confusión | Sprint 5 | `classification_report.txt` y matriz de confusión |
| 🔄 | Analizar resultados y corregir errores | Sprint 5 | Interpretar métricas y ajustar |
| 🔄 | Ajustar hiperparámetros y probar arquitecturas | Sprint 5 | Experimentar con diferentes configuraciones |
| ⏳ | Comparar resultados con benchmarks y documentar | Sprint 5 | Documentar comparaciones con otros enfoques |
| ⏳ | Diseñar e implementar CLI de inferencia (`predict.py`) | Sprint 6 | Permitirá clasificar audios desde terminal |
| ⏳ | Prototipo de interfaz gráfica (desktop/móvil) | Sprint 6 | Investigar frameworks y UX |
| ⏳ | Integrar nuevos datasets (FMA, MSD+Last.fm, etc.) | Sprint 6 | Ampliar la generalización |
| 🧪 | Validar predicciones con audios externos | Sprint 6 | Probar con datos no vistos |
| ⏳ | Recoger feedback de usuarios y mejorar UX | Sprint 6 | Incorporar sugerencias |
| ⏳ | Desarrollar pruebas unitarias y de integración | Sprint 7 | Asegurar la calidad del código |
| ⏳ | Realizar pruebas de rendimiento y estabilidad | Sprint 7 | Evaluar consumo de recursos |
| ⏳ | Asegurar cobertura de código y configurar linters | Sprint 7 | Controlar calidad y estilo |
| ⏳ | Documentar API y código (docstrings, comentarios) | Sprint 7 | Facilitar mantenimiento |
| ⏳ | Redactar memoria del TFG y presentación | Sprint 7 | Parte final de la documentación |
| ⏳ | Revisión final con el tutor | Sprint 8 | Validación de entrega |
| ⏳ | Ajustar backlog y resolver issues pendientes | Sprint 8 | Cierre de tareas |
| ⏳ | Preparar versión final del repositorio | Sprint 8 | Limpieza y tag de release |
| ⏳ | Presentar y defender el TFG | Sprint 8 | Exposición y defensa oficial |

---

### 📌 Leyenda de estados

- ✅ **Completado**: la tarea ha sido terminada y validada.  
- 🔄 **En progreso**: la tarea está actualmente en curso.  
- ⏳ **Pendiente**: planificada pero todavía no iniciada.  
- 🧪 **En pruebas**: fase de verificación o validación.  

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

## 🧱 Requisitos previos
- 🐍 Python 3.10 (recomendado) con `pip` y `venv`.  
- 🎧 FFmpeg en el `PATH`.  
- 💻 Sistemas soportados: Windows 10/11, macOS, Linux.  
- ⚡ Opcional: GPU con CUDA para acelerar el entrenamiento.  
- 📥 Opcional: Kaggle CLI configurada para descargar GTZAN automáticamente.  

---

## ⚙️ Instalación y configuración inicial

### 1. Clonar el repositorio:
```bash
git clone https://github.com/ManuVO/tfg_clasificador_generos.git
cd tfg_clasificador_generos
```

### 2. Crear y activar un entorno virtual:

**En Windows/PowerShell**:
```powershell
./init_env.ps1
```
Esto crea `.venv`, instala las dependencias y configura `PYTHONPATH`.

**En macOS/Linux**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Asegúrate de tener **FFmpeg** instalado y accesible.

### 3. Configurar parámetros:
La configuración maestra se encuentra en `config.yaml`.  
Puedes crear archivos YAML específicos en `configs/` para distintos experimentos.

---

## 📊 Preparar el dataset

El proyecto está pensado para ser independiente del origen del dataset.  
La versión actual se centra en **GTZAN**, que debes descargar manualmente o mediante Kaggle. Sigue estos pasos:

1. Descarga GTZAN desde una fuente fiable.  
2. Colócalo en `data/raw/gtzan/genres_original/*.wav`.  
3. Ejecuta el preprocesado para segmentar y normalizar el audio:

```bash
python src/data/preprocess.py --dataset gtzan --project-config config.yaml
```

Se generarán los segmentos en `data/processed/gtzan/segments/` y el fichero `metadata.csv`.

Si deseas utilizar otros datasets (FMA, MSD+Last.fm, etc.), adapta las rutas y añade configuraciones en `configs/datasets/`.  
El backlog incluye la integración de nuevos datasets.

---

## 🤖 Entrenar un modelo

Con el dataset preprocesado, entrena una red neuronal ejecutando:

```bash
python src/train.py --dataset gtzan --project-config config.yaml
```

Los resultados se almacenan en `experiments/<fecha>/<hora>/` e incluyen:

- `checkpoints/best_model.pt` → el modelo con mejor rendimiento.  
- `metrics.csv` y `classification_report.txt` → métricas en CSV y texto.  
- Carpetas de logs para TensorBoard.  

Si deseas usar un YAML específico, indícalo con `--config <ruta_yaml>`.

---

## ✅ Evaluar un checkpoint

Para evaluar un modelo entrenado ejecuta:

```bash
python src/eval.py --run_dir experiments/<fecha>/<hora>
```

Esto generará un `classification_report.txt` y una matriz de confusión en el directorio de la ejecución.  
Puedes especificar `--out_dir` para almacenar los resultados en otra ubicación.

---

## ⚙️ Personalizar la configuración y entrenamientos

El archivo `config.yaml` define parámetros generales: ubicación de datos, tamaño de segmentos, arquitectura base, optimizador, etc.  
Para ajustes específicos crea YAMLs en `configs/` y pásalos al script de entrenamiento.  

Algunas recomendaciones para **GTZAN** son:

- **Augmentations** en forma de onda: reducir probabilidades a 0.3–0.4 para evitar distorsiones extremas.  
- **SpecAugment**: aplicar 2 máscaras de frecuencia y 3 temporales, de ancho moderado.  
- **Épocas y tamaño de lote**: entrenar 40 épocas con `batch_size=32`.  
- **Tasa de aprendizaje y weight decay**: `learning_rate=1e-3` y `weight_decay=1e-4`.  
- **Scheduler**: usar `ReduceLROnPlateau` con `factor=0.5` y `patience=3`.  
- **Early stopping**: `patience=10` y `min_delta=0.005`.  

---

## 🧪 Verificar el pipeline rápidamente

Si quieres comprobar que la transformación de audio a espectrograma funciona correctamente sin necesidad de entrenar un modelo completo, ejecuta:

```bash
python scripts/test_pipeline.py
```

Genera un audio sintético, aplica el preprocesado y muestra el resultado en forma de espectrograma.

---

## 📝 Notas finales

- Los checkpoints se guardan en `experiments/`; el mejor modelo es `checkpoints/best_model.pt`.  
- Ajusta `training.num_workers` o `training.weight_decay` según tu hardware y preferencias.  
- Si cambias rutas o añades nuevos datasets, vuelve a ejecutar el preprocesado para actualizar los segmentos.  

### 🤝 Agradecimientos

Gracias a todas las personas que se interesan por este proyecto y siguen su evolución.  
Se trata de un trabajo de fin de grado en desarrollo, por lo que, además de mi esfuerzo, está abierto a colaboraciones externas. Tu comprensión y apoyo son muy apreciados.
