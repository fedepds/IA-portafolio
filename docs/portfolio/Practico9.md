
# CNN vs Transfer Learning (CIFAR-10)

En esta práctica se comparan dos enfoques fundamentales para la clasificación de imágenes en el dataset CIFAR-10:  
(1) una Red Neuronal Convolucional (CNN) simple entrenada desde cero, y  
(2) un modelo de vanguardia (EfficientNetB0) aplicando Transfer Learning.

El objetivo es evaluar y contrastar el rendimiento, la eficiencia del entrenamiento y el impacto de la arquitectura, demostrando la importancia de seleccionar la estrategia de modelo adecuada según el dominio del problema.

---

## Objetivos

- Implementar CNNs usando TensorFlow/Keras para clasificación de imágenes.  
- Aplicar Transfer Learning (Extracción de Características y Fine-Tuning) con modelos pre-entrenados de Keras Applications.  
- Procesar datasets de imágenes (CIFAR-10) utilizando `ImageDataGenerator` para el aumento de datos.  
- Evaluar modelos usando métricas de clasificación (accuracy, F1-score).  
- Comparar las arquitecturas CNN vs Transfer Learning en un problema de imágenes de baja resolución.

---

## Metodología

### Preprocesamiento

- **Dataset:** `keras.datasets.cifar10` (50,000 imágenes de entrenamiento, 10,000 de test).  
- **Clases:** 10 (airplane, automobile, bird, etc.).  
- **Dimensiones:** 32x32x3.  
- **Normalización:** Escalado de píxeles (0–255) → (0–1).  
- **Codificación:** Etiquetas en formato one-hot encoding.  
- **Batch Size:** 128.

---

### Arquitectura 1: CNN Simple (Desde Cero)

Modelo secuencial optimizado para inputs de 32x32:

1. `Conv2D(32, (3, 3), activation='relu')`  
2. `MaxPooling2D((2, 2))`  
3. `Conv2D(64, (3, 3), activation='relu')`  
4. `MaxPooling2D((2, 2))`  
5. `Flatten()`  
6. `Dense(512, activation='relu')`  
7. `Dense(10, activation='softmax')` (Clasificador)  

**Parámetros entrenables:** 2,122,186.

---

### Arquitectura 2: Transfer Learning (EfficientNetB0)

- **Base:** `applications.EfficientNetB0` (pre-entrenado en ImageNet, `include_top=False`, `pooling='avg'`).  
- **Cabezal:** `BatchNormalization()` → `Dense(256, 'relu')` → `Dropout(0.5)` → `Dense(10, 'softmax')`.  
- **Parámetros totales:** 4,385,197.

---

### Pipeline de Augmentation

Se utilizó `ImageDataGenerator` para ambos modelos a fin de mejorar la robustez y asegurar una comparación justa.

### Transformaciones aplicadas:
- rotation_range = 15
- width_shift_range = 0.1
- height_shift_range = 0.1
- horizontal_flip = True
- zoom_range = 0.1


---

### Entrenamiento y Evaluación

- **Optimizador:** `AdamW` (LR = 0.001).  
- **Loss:** `categorical_crossentropy`.  
- **Callbacks:** `EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)`.

**CNN Simple:**  
Entrenamiento estándar (10 épocas) usando `datagen.flow()`.

**Transfer Learning (2 Etapas):**

1. **Etapa 1 – Extracción:**  
   - Base congelada (`base_model.trainable = False`).  
   - Entrenamiento solo del cabezal (10 épocas, detenida en la 5).  
   - Parámetros entrenables: 333,066.  

2. **Etapa 2 – Fine-Tuning:**  
   - Se descongelaron las últimas 10 capas.  
   - LR reducido a `1e-4`.  
   - Entrenamiento por 10 épocas, detenido en la 8.

---

## Resultados Principales

| Modelo | val_accuracy | Observaciones |
|:--|:--:|:--|
| CNN Simple | **71.40%** | Buen rendimiento, convergencia estable, sin overfitting. |
| EfficientNetB0 (TL) | **14.56%** | Fallo total de aprendizaje; accuracy cercana al azar. |

**Análisis del fallo:**
- Etapa 1: `val_accuracy` máxima 13.37%.  
- Etapa 2: Divergencia del modelo (`val_loss` hasta 4.88).  
- F1-scores nulos (0.00) en 5 de 10 clases para TL.  
- CNN simple mostró F1-scores sólidos (ej. 0.82 para "automobile").  

La CNN simple fue **56.84% superior** al enfoque de Transfer Learning.

---

## Conclusiones

- Contrario a lo esperado, Transfer Learning (EfficientNetB0) tuvo un rendimiento drásticamente inferior (14.56%) que la CNN simple (71.40%).  
- El fallo sugiere incompatibilidad entre el modelo pre-entrenado y el dataset.  
- **Hipótesis:** EfficientNetB0 (entrenado en ImageNet, con imágenes grandes y detalladas) no transfiere bien a CIFAR-10 (imágenes pequeñas, 32x32).  
- En este caso, una CNN diseñada específicamente para las dimensiones del problema superó al modelo complejo.

---

## Reflexión Personal

Esta práctica demuestra que el Transfer Learning no es una "bala de plata".  
Es esencial validar que la arquitectura pre-entrenada sea adecuada al dominio y las dimensiones del dataset.  

Un modelo más complejo (como EfficientNetB0) no garantiza mejores resultados si el contexto de los datos es distinto (ImageNet vs. CIFAR-10).

---

## Próximo Paso

- Probar arquitecturas más ligeras (ej. MobileNetV2).  
- Intenté aplicar el modelo a otro tipo de dataset, pero surgieron muchos problemas con los pesos preentrenados: incompatibilidades de tamaño/escala de entrada y comportamiento inestable al cargar los pesos, lo que impidió un reentrenamiento fiable.  
- Analizar el impacto de las capas `BatchNormalization` durante la congelación, que podrían haber agravado la inestabilidad estadística con datos muy distintos a ImageNet.
