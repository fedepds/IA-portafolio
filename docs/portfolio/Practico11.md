
# Fine-Tuning YOLOv8 para Detección de Productos

En esta práctica, se aborda un problema organizacional (automatización en supermercados) traduciéndolo en una solución técnica de IA. Se evalúan las limitaciones de un modelo de detección genérico (YOLOv8), se implementa una estrategia de fine-tuning para adaptarlo a un dominio específico (detección de frutas) y se compara el rendimiento de múltiples arquitecturas. Finalmente, el modelo afinado se despliega en un pipeline de tracking para un caso de uso de conteo en tiempo real.

## Objetivos
- Traducir una necesidad organizacional (ej. conteo de inventario) en una solución de IA funcional.
- Evaluar el rendimiento "zero-shot" de un modelo pre-entrenado (YOLOv8n en COCO) y demostrar sus limitaciones en un dominio específico.
- Implementar un experimento de comparativa de arquitecturas (YOLOv5/v8) para seleccionar el modelo con el mejor balance de precisión, velocidad y tamaño.
- Aplicar una estrategia de fine-tuning (transfer learning) para especializar un modelo YOLO en la detección de un nuevo conjunto de clases (frutas).
- Desplegar el modelo afinado en un pipeline funcional de tracking (con Norfair) para resolver un caso de uso práctico (conteo de productos en una cinta).
- Analizar cuantitativamente el rendimiento y los fallos del sistema para definir las próximas iteraciones de mejora.

## Metodología

### 1. Diagnóstico y Evaluación "Zero-Shot"
- **Problema**: Se partió de la necesidad de un supermercado de automatizar tareas como el conteo de productos.
- **Modelo Base**: Se cargó un modelo `yolov8n` pre-entrenado en el dataset COCO.
- **Prueba Inicial**: Se evaluó el modelo en una imagen del dominio del problema (pasillo de supermercado).
- **Resultado**: La solución genérica ("off-the-shelf") fracasó. El modelo detectó solo 3 objetos, con clases incorrectas (ej. "orange" por "apple") y confianza muy baja (< 0.37), estableciendo la justificación de negocio para un fine-tuning personalizado.

### 2. Comparativa de Arquitecturas (Trabajo 1)
- **Objetivo**: Determinar la arquitectura con el mejor balance (trade-off) para el problema.
- **Dataset**: Se utilizó un dataset de "Fruit Detection", resolviendo un problema de MLOps al reparar el archivo `data.yaml` (rutas incorrectas) para asegurar la reproducibilidad.
- **Experimento**: Se ejecutó un experimento controlado entrenando 5 modelos distintos (YOLOv5n, v5s, YOLOv8n, v8s, v8m).
- **Parámetros**: El entrenamiento fue idéntico para todos: 10 épocas, `imgsz=416` y usando solo el 25% de los datos (`fraction=0.25`) para una comparativa ágil.
- **Métricas**: Se midió mAP@0.5, mAP@0.5:0.95, tiempo de inferencia (ms) y tamaño del modelo (MB).

### 3. Fine-Tuning y Evaluación (Proyecto Principal)
- **Estrategia**: Se aplicó transfer learning sobre el modelo `yolov8n` (según el análisis principal del proyecto).
- **Análisis de Datos**: Se analizó la distribución del dataset, detectando un desbalance de clases (muchas `Orange`, pocas `Pineapple`) para anticipar el rendimiento.
- **Entrenamiento**: Se entrenó el modelo por 15 épocas usando el 100% de los datos de entrenamiento.
- **Métricas**: Se comparó el rendimiento Pre-Fine-Tuning (F1-Score de 0.0) con el Post-Fine-Tuning (F1-Score, Recall, Precisión) usando el set de prueba.

### 4. Despliegue en Pipeline de Tracking
- **Caso de Uso**: Simular el conteo de productos en una cinta transportadora.
- **Herramientas**: Se integró el detector YOLOv8 afinado con el tracker `Norfair`.
- **Estrategia Simbiótica**: Se configuró una estrategia de dos pasos para maximizar el rendimiento del pipeline:
    1.  **Detector (Modelo YOLO)**: Configurado con `conf=0.2` (umbral bajo) para maximizar el **Recall** (encontrarlo todo, aunque genere ruido o Falsos Positivos).
    2.  **Tracker (Norfair)**: Configurado con `initialization_delay=3` (retardo de 3 frames) para maximizar la **Precisión** (filtrar el ruido y las detecciones "fantasma" momentáneas, confiando solo en detecciones estables).

## Resultados Principales

### 1. Comparativa de Arquitecturas

El experimento de comparativa (Trabajo 1) arrojó un claro "punto óptimo" (sweet spot).

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Inferencia (ms) | Tamaño (MB) |
|---|---:|---:|---:|---:|
| YOLOv5n | 0.336 | 0.204 | 1.8 ms | 5.2 MB |
| YOLOv5s | 0.352 | 0.221 | 3.3 ms | 18.5 MB |
| YOLOv8n | 0.357 | 0.217 | 1.8 ms | 6.2 MB |
| **YOLOv8s** | **0.400** | **0.256** | **3.7 ms** | **22.5 MB** |
| YOLOv8m | 0.375 | 0.237 | 9.2 ms | 52.0 MB |

- **Análisis de Trade-off**:
    - **YOLOv8s** fue el claro ganador en precisión (mAP 0.400), superando a las versiones "nano" (n) con un costo de inferencia (3.7 ms) que sigue siendo trivial para el tiempo real (>270 FPS).
    - **YOLOv8m** (Medium) fue contraproducente: fue 2.5 veces más lento y, sorprendentemente, *menos preciso* que `v8s`, indicando *under-fitting* (sub-entrenamiento) debido a su mayor necesidad de datos y épocas.

### 2. Impacto del Fine-Tuning (Proyecto Principal)

El fine-tuning transformó un modelo inútil en una solución competente.

| Métrica | YOLOv8n Pre-entrenado (COCO) | YOLOv8n Fine-tuned (Frutas) | Mejora |
|---|---:|---:|---:|
| **F1-Score** | 0.0 | **0.800** | +Inf |
| **Recall** | ~0.0 | **1.0** | +100% |
| **Precisión**| ~0.0 | **0.667** | +66.7% |

- **Análisis de Rendimiento**:
    - El modelo afinado demostró una "personalidad" específica: un **Recall perfecto (1.0)**, indicando que es excelente para no omitir ningún producto.
    - Su debilidad fue una **Precisión de 0.667**, lo que significa que, para lograr ese Recall, genera algunos Falsos Positivos (detecciones "fantasma"). Esta debilidad fue tratada en la siguiente etapa (Tracking).

### 3. Resultados del Pipeline de Tracking
- **Éxito Funcional**: El sistema funcionó y resolvió el caso de uso. Contó 13 productos que pasaron por la cinta, clasificándolos correctamente (5 bananas, 4 manzanas, 4 naranjas).
- **Estabilidad**: El tracking fue robusto (ej. "Track 2 - Orange" duró 340 de 343 frames), validando la estrategia simbiótica de detector+tracker.
- **Diagnóstico de Fallos**: Se identificó un problema clave: la "fragmentación de IDs" en las bananas (se generaron 5 IDs de track para 2-3 bananas reales). El análisis determinó que la causa raíz no era el tracker, sino el modelo (detector) que "parpadeaba" (fallaba en detectar) en ciertos frames.

## Conclusiones
- El fine-tuning no es opcional, es un paso crítico. Los modelos "off-the-shelf" fallan en dominios específicos. Esta práctica demostró una creación de valor medible, llevando un modelo de **0.0 a 0.800 F1-Score**.
- La selección de la arquitectura es un trade-off. El análisis comparativo demostró que **YOLOv8s** ofrece el "punto óptimo" de precisión y velocidad.
- Los modelos más grandes (como `v8m`) no son inherentemente mejores y pueden rendir peor si no se les entrena con suficientes datos o épocas (under-fitting).
- Un pipeline de IA (ej. Detector + Tracker) es más robusto que sus componentes individuales. El tracker `Norfair` filtró exitosamente los Falsos Positivos (baja Precisión) del detector `YOLO`, mientras que el detector `YOLO` aseguró que no se omitiera ningún objeto (alto Recall).
- El análisis de fallos (ej. "fragmentación de IDs" en bananas) es fundamental, ya que define la siguiente acción estratégica: no se necesita un mejor tracker, se necesitan más datos de entrenamiento de bananas para mejorar el detector.

## Reflexión Personal
Esta práctica ejecutó un proyecto de IA de extremo a extremo, reflejando perfectamente el salto de un problema de negocio a una solución técnica. No nos enfocamos en inventar un nuevo algoritmo, sino en aplicar herramientas open-source (Ultralytics, Norfair) a un problema concreto. Se evaluaron las limitaciones de la herramienta base, se aplicó un fine-tuning para adaptarla (creando valor medible) y se la integró en un pipeline funcional que resolvió la necesidad organizacional. Finalmente, se analizaron las nuevas limitaciones (IDs de bananas) para definir el próximo ciclo de mejora ágil.
