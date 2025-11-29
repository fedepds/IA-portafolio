# Data Augmentation y XAI con Transfer Learning (Flowers102)

En este proyecto apliqué **Transfer Learning** a un problema complejo de clasificación (102 especies de flores), integrando técnicas avanzadas:

- **Data Augmentation**: Pipeline robusto con transformaciones múltiples para simular variabilidad real.
- **Transfer Learning**: Fine-tuning de EfficientNetB0 pre-entrenado en ImageNet.
- **Explicabilidad (XAI)**: Implementación de GradCAM e Integrated Gradients para auditar decisiones del modelo.
- **Diagnóstico de overfitting**: Análisis de curvas de entrenamiento y propuestas de mejora.

Demostré que la explicabilidad no es un "extra" sino una herramienta crítica de debugging y confianza en producción.

--- a un dataset complejo (Flowers102), enfocándonos en robustez (Data Augmentation) y confianza (Explicabilidad: GradCAM e Integrated Gradients).

## Objetivos
- Aplicar Transfer Learning (EfficientNetB0) para clasificación de 102 clases.
- Implementar pipeline de Data Augmentation para simular variaciones de captura.
- Diagnosticar sobreajuste analizando curvas de entrenamiento.
- Implementar GradCAM e Integrated Gradients para auditar decisiones del modelo.
- Usar XAI como herramienta de diagnóstico de errores.

## Metodología

### Preprocesamiento
- Carga de oxford_flowers102 con TFDS.
- Subset usado para iteración rápida (~5k train, 1k test).
- Redimensionado a 224×224 y normalización de EfficientNet.

### Arquitectura (Transfer Learning)
- Base: EfficientNetB0 preentrenado en ImageNet.
- Fine-tuning (base_model.trainable = True).
- Cabezal: GlobalAveragePooling2D + Dense(102, activation='softmax').

### Pipeline de Augmentation
- Capas Keras: RandomFlip, RandomRotation (0.125), RandomZoom (0.2),
  RandomTranslation (0.2), RandomBrightness (0.2), RandomContrast (0.2).

### Entrenamiento y Evaluación
- Optimizador: Adam.
- Loss: sparse_categorical_crossentropy.
- Entrenamiento: 10 épocas sobre el subset.
- Monitoreo de train/val loss y accuracy.

### Análisis de Explicabilidad (XAI)
- GradCAM aplicado sobre la capa de convolución superior (top_conv).
- Integrated Gradients para atribuciones más detalladas.
- Evaluación de predicciones correctas e incorrectas para diagnóstico.

## Resultados principales
- Rendimiento: val_accuracy máximo ~81.5% (época 6); test_accuracy final ~78.8%.
- Sobreajuste: train_accuracy ~98.2% vs val_accuracy ~78.8% (brecha ~20 puntos).
- val_loss tocó mínimo en época 6 (≈0.7544) y luego aumentó, indicando sobreajuste.
- GradCAM: en aciertos la atención se centra en la flor.
- Integrated Gradients: atribuciones más finas en estambres y partes relevantes.
- Diagnóstico de errores: adaptación de GradCAM para auditar por qué falla (fondo, confusiones entre especies similares).

## Conclusiones
- Transfer Learning (EfficientNetB0) es efectivo para problemas de visión con muchas clases.
- Data Augmentation básico ayuda, pero no evita el overfitting en entrenamiento limitado.
- Técnicas avanzadas (Mixup/CutMix, regularización) recomendadas para mejorar generalización.
- XAI (GradCAM/IG) es esencial para confiar y depurar modelos en contextos reales.

## Reflexión personal
- Aprendizaje práctico sobre la importancia de logs y visualizaciones más allá del accuracy.
- GradCAM demostró ser una herramienta de debugging crucial.
- Este ejercicio muestra la diferencia entre "modelo que funciona" y "modelo en el que podemos confiar".

## Próximo paso
- Implementar ModelCheckpoint para guardar el mejor modelo (época 6).
- Probar Mixup/CutMix y regularización adicional para reducir overfitting.

---

## 📓 Notebook

**[Abrir en Google Colab](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT3/Practico10.ipynb)**