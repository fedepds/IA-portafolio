# Data Augmentation y XAI con Transfer Learning

En este proyecto apliqué **Transfer Learning** a problemas complejos de clasificación multi-clase, integrando técnicas avanzadas de explicabilidad:

- **Data Augmentation**: Pipeline robusto con transformaciones múltiples (flip, rotation, zoom, brightness, contrast) para simular variabilidad real.
- **Transfer Learning con Fine-tuning en dos etapas**: EfficientNetB0 pre-entrenado en ImageNet, primero congelado y luego fine-tuning completo.
- **Explicabilidad (XAI)**: Implementación de GradCAM e Integrated Gradients para entender y auditar las decisiones del modelo.
- **Diagnóstico sistemático**: Búsqueda automática de errores en predicciones y análisis visual de por qué el modelo falla.
- **Generalización multi-dominio**: Validación en dos datasets (Oxford Flowers102 y Stanford Dogs) para demostrar transferibilidad de la metodología.

Aprendí que la explicabilidad no es un "extra" sino una herramienta crítica de debugging, auditoría y generación de confianza en modelos de producción.

---

## Objetivos

- Aplicar Transfer Learning (EfficientNetB0) para clasificación multi-clase con datasets complejos.
- Implementar pipeline de Data Augmentation para simular variaciones de captura real.
- Experimentar con estrategia de fine-tuning en dos etapas (congelado + descongelado).
- Implementar callbacks (EarlyStopping) para prevenir overfitting y guardar el mejor modelo.
- Implementar GradCAM e Integrated Gradients para auditar decisiones del modelo.
- Diagnosticar errores sistemáticamente usando XAI para identificar causas de predicciones incorrectas.
- Validar metodología en múltiples dominios (flores y perros) para demostrar generalización.

---

## Metodología

### Datasets Utilizados

#### 1. Oxford Flowers102
- **Descripción**: 102 especies de flores comunes en el Reino Unido
- **Tamaño**: ~8,189 imágenes (train + test)
- **Subset usado**: 5,000 train / 1,000 test (para iteración rápida)
- **Desafío**: Alta similitud inter-clase, variabilidad en fondos y condiciones de iluminación

#### 2. Stanford Dogs  
- **Descripción**: 120 razas de perros
- **Tamaño**: ~20,580 imágenes (train + test)
- **Subset usado**: 8,000 train / 1,000 test
- **Desafío**: Razas con características visuales muy similares, variabilidad en poses y contextos

### Preprocesamiento

- Carga con TensorFlow Datasets (TFDS) usando `as_supervised=True`
- Redimensionado a 224×224 (input estándar de EfficientNet)
- Normalización con `preprocess_input` de EfficientNet (rangos específicos para ImageNet)
- Split estratégico: subsets para experimentación rápida

### Arquitectura (Transfer Learning en 2 Etapas)

**Base**: EfficientNetB0 pre-entrenado en ImageNet

**Etapa 1 - Entrenamiento de cabezal (base congelada)**:
- `base_model.trainable = False`
- Cabezal: GlobalAveragePooling2D → Dropout(0.3) → Dense(num_classes, softmax)
- Optimizador: Adam (lr por defecto)
- Epochs: 7 con EarlyStopping(patience=3)

**Etapa 2 - Fine-tuning completo**:
- `base_model.trainable = True` (descongelar todas las capas)
- Re-compilación con learning rate bajo: Adam(lr=1e-5)
- Epochs: 10 con EarlyStopping(patience=3)
- Callback `restore_best_weights=True` para recuperar el mejor modelo

### Pipeline de Augmentation

Capas de Keras aplicadas durante entrenamiento:
- `RandomFlip("horizontal")` - flip horizontal para simetría
- `RandomRotation(0.15)` - rotación ±15% para invarianza rotacional
- `RandomZoom(0.2)` - zoom ±20% para simular distancias variables
- `RandomBrightness(0.2)` - variación de brillo ±20%
- `RandomContrast(0.2)` - variación de contraste ±20%

### Entrenamiento y Callbacks

- **Optimizador**: Adam (lr=1e-3 etapa 1, lr=1e-5 etapa 2)
- **Loss**: sparse_categorical_crossentropy (labels como enteros)
- **Callbacks**: EarlyStopping con `monitor='val_loss'`, `patience=3`, `restore_best_weights=True`
- **Estrategia**: Entrenamiento en 2 fases evita catastrophic forgetting y optimiza convergencia

### Análisis de Explicabilidad (XAI)

**GradCAM (Gradient-weighted Class Activation Mapping)**:
- Aplicado sobre última capa convolucional (detección automática)
- Generación de heatmaps mostrando regiones de atención del modelo
- Visualización en 3 vistas: original, heatmap, overlay

**Integrated Gradients**:
- Baseline: imagen negra (tensor de ceros)
- 50 pasos de interpolación entre baseline y imagen real
- Atribución pixel-wise más precisa que GradCAM
- Visualización con colormap 'inferno' para destacar píxeles importantes

**Diagnóstico de Errores**:
- Búsqueda automática de predicciones incorrectas en test set
- Aplicación de GradCAM sobre errores para análisis de causa raíz
- Clasificación de tipos de error: correlación espuria (fondo), confusión honesta (similitud visual), baja confianza

---

## Resultados

### Experimento 1: Oxford Flowers102

**Rendimiento**:
- Val accuracy máximo: ~81.5% (época 6)
- Test accuracy final: ~78.8%
- Train accuracy final: ~98.2%

**Análisis de Overfitting**:
- Brecha train-val: ~20 puntos porcentuales
- Val loss mínimo en época 6 (~0.7544), luego aumentó
- EarlyStopping activado para prevenir degradación adicional

**GradCAM - Predicciones correctas**:
- Atención focalizada en pétalos, estambres y centro reproductivo de flores
- Ignorancia de fondo y follaje → validación de que el modelo no usa correlaciones espurias
- Confianza alta en regiones discriminativas correctas

**Integrated Gradients**:
- Atribuciones más finas en estructuras florales específicas
- Mayor precisión pixel-wise que GradCAM
- Útil para análisis detallado de características discriminativas

**Diagnóstico de Errores**:
- Errores típicos: confusión entre flores con colores/formas similares
- GradCAM mostró que en algunos errores el modelo miraba partes correctas pero insuficientes
- Casos de baja confianza correlacionados con fondos complejos

### Experimento 2: Stanford Dogs

**Rendimiento**:
- Mejora notable con fine-tuning en 2 etapas
- EarlyStopping efectivo para prevenir overfitting
- Test accuracy superior a Flowers102 (razas de perros tienen características más distintivas)

**Validación de Generalización**:
- Mismo pipeline funcionó exitosamente en dominio diferente
- GradCAM mostró atención en características caninas relevantes: orejas, hocico, pelaje
- Metodología XAI consistente y reproducible across dominios

---

## Conclusiones

### Técnicas

- **Transfer Learning con EfficientNetB0** es altamente efectivo para clasificación multi-clase compleja (102-120 clases)
- **Fine-tuning en 2 etapas** (congelado → descongelado con lr bajo) evita catastrophic forgetting y optimiza convergencia
- **EarlyStopping con restore_best_weights** es crítico para evitar overfitting y recuperar el mejor modelo automáticamente
- **Data Augmentation** mejora robustez pero no elimina completamente el overfitting cuando hay limitación de datos
- Técnicas avanzadas recomendadas: Mixup/CutMix, regularización L2, mayor volumen de datos

### Explicabilidad (XAI)

- **GradCAM** es herramienta esencial de debugging: identifica si el modelo usa características correctas o correlaciones espurias
- **Integrated Gradients** proporciona atribuciones más precisas, útil para análisis detallado
- **Diagnóstico de errores con XAI** permite clasificar tipos de fallo y priorizar mejoras
- XAI transforma modelos de "caja negra" a sistemas auditables y confiables

### Aplicación Práctica

**Generación de Confianza**:
- Visualizaciones XAI demuestran a usuarios/expertos que el modelo funciona por razones correctas
- Validación de que no hay "trampas" (ej: mirar fondos en lugar de sujeto)
- Crítico para adopción en contextos profesionales (botánica, veterinaria)

**Mitigación de Riesgos**:
- Modelo sin explicabilidad presenta riesgos: confusión peligrosa (flores venenosas, diagnósticos incorrectos)
- XAI permite auditoría pre-deployment y monitoreo continuo
- Facilita debugging sistemático vs trial-and-error

### Generalización Multi-Dominio

- Metodología validada exitosamente en 2 dominios (flores, perros)
- Pipeline transferible a otros problemas de clasificación visual
- Técnicas XAI consistentes regardless del dominio específico

---

## Reflexión Personal

- Aprendí que accuracy es insuficiente: necesitamos entender **cómo** el modelo llega a sus predicciones
- GradCAM reveló que algunos modelos "adivinan bien" mirando cosas incorrectas → éxito superficial pero modelo frágil
- La diferencia entre "modelo que funciona" y "modelo en el que podemos confiar" radica en la explicabilidad
- Fine-tuning en 2 etapas demostró ser superior a fine-tuning directo o congelamiento permanente
- EarlyStopping es herramienta crítica pero requiere configuración cuidadosa (métrica, patience, restore_best_weights)

---

## Exploraciones Futuras

**Mejoras de Rendimiento**:
- Implementar Mixup/CutMix para regularización adicional durante entrenamiento
- Probar arquitecturas más grandes (EfficientNetB3-B7) con mayor capacidad
- Aumentar volumen de datos (usar dataset completo + augmentation offline)

**Técnicas Avanzadas XAI**:
- LIME (Local Interpretable Model-agnostic Explanations) para comparación
- Saliency Maps y Attention Rollout para análisis adicional
- Quantitative evaluation de mapas de atención vs ground truth annotations

**MLOps**:
- Implementar ModelCheckpoint para tracking de múltiples checkpoints
- A/B testing de variantes de modelo en producción
- Pipeline de monitoreo continuo de explicabilidad (drift detection)

---

## 📓 Notebook

**[Abrir en Google Colab](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT3/Practico10.ipynb)**
