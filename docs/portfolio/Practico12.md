# Fine-Tuning de SAM para Segmentación de Inundaciones

En este proyecto apliqué **fine-tuning de un modelo de fundación** (Segment Anything Model - SAM) para un problema crítico: detección de áreas inundadas en imágenes satelitales. Demostré:

- **Transfer learning eficiente**: Fine-tuning selectivo del decoder (4.33% de parámetros) logrando mejoras significativas.
- **Evaluación zero-shot**: Análisis del rendimiento base con prompts de punto y caja.
- **Comparación arquitectónica**: Benchmarking contra U-Net especialista para validar la estrategia.
- **Métricas de segmentación**: IoU, Dice, Precision y Recall para evaluación exhaustiva.

Este proyecto muestra cómo adaptar modelos generalistas de gran escala a dominios específicos con eficiencia computacional.

## Habilidades Demostradas
- Aplicar un modelo de fundación (SAM) a un dominio de segmentación novedoso.
- Evaluar el rendimiento "zero-shot" de SAM usando prompts de punto y caja.
- Implementar una estrategia de fine-tuning eficiente, entrenando únicamente el decodificador de máscaras de SAM.
- Comparar cuantitativamente (IoU, Dice) el rendimiento del modelo pre-entrenado vs. el afinado.
- Entrenar un modelo especialista (U-Net con backbone ResNet34) como baseline competitivo.
- Analizar las ventajas y desventajas entre un modelo generalista afinado (SAM) y un especialista entrenado desde cero (U-Net).

## Metodología

### 1. Preparación y Análisis del Dataset
- **Dataset**: Se utilizó el "Flood Area Segmentation" de Kaggle, que contiene imágenes satelitales y sus correspondientes máscaras binarias indicando la presencia de agua.
- **Carga y Exploración**: Se cargaron 100 muestras, analizando la estructura de directorios y visualizando pares de imagen/máscara para entender la tarea.

### 2. Evaluación Zero-Shot de SAM
- **Modelo**: Se cargó el modelo `sam_vit_b_01ec64.pth`.
- **Prompts**: Se evaluó el rendimiento del modelo pre-entrenado sin ningún tipo de fine-tuning, utilizando dos tipos de prompts generados a partir de las máscaras de verdad (`ground truth`):
    - **Point Prompt**: Un único punto en el centro de la región de agua.
    - **Box Prompt**: Un cuadro delimitador (`bounding box`) que encierra toda la región de agua.
- **Métricas**: Se midió el rendimiento inicial usando IoU, Dice, Precision y Recall.

### 3. Fine-tuning de SAM
- **Estrategia**: Se aplicó una técnica de fine-tuning eficiente (Parameter-Efficient Fine-Tuning, PEFT):
    - **Congelado**: Se congelaron los pesos del `image_encoder` y `prompt_encoder`, que son la mayor parte del modelo.
    - **Entrenamiento**: Se entrenó únicamente el `mask_decoder`, que representa menos del 5% de los parámetros totales.
- **Dataset**: Se creó un `torch.utils.data.Dataset` que redimensiona las imágenes a 1024x1024 (tamaño nativo de SAM) y genera `point prompts` aleatorios dentro de las áreas de agua para cada muestra.
- **Loss Function**: Se utilizó una pérdida combinada de `Binary Cross-Entropy (BCE)` y `Dice Loss` para un entrenamiento más estable.
- **Entrenamiento**: Se entrenó el modelo durante 17 epochs con un `batch size` pequeño (2) debido al alto consumo de memoria de SAM.

### 4. Entrenamiento del Modelo U-Net
- **Arquitectura**: Como comparativa, se implementó un modelo `U-Net` con un backbone `ResNet34` pre-entrenado en ImageNet, utilizando la librería `segmentation-models-pytorch`.
- **Entrenamiento**: Se entrenó el modelo completo sobre el mismo dataset, pero con imágenes redimensionadas a 256x256, lo que permitió un `batch size` mucho mayor (16).

## Resultados Principales

| Métrica      | SAM Pre-entrenado (Point) | SAM Fine-tuned (Point) | Mejora (%) | U-Net (ResNet34) |
|--------------|---------------------------|------------------------|------------|------------------|
| **Mean IoU** | 0.7389                    | **0.8911**             | +20.59%    | **0.9084**       |
| **Mean Dice**| 0.8183                    | **0.9348**             | +14.24%    | **0.9501**       |

- **Rendimiento Base**: El modelo SAM pre-entrenado ya mostraba una capacidad de segmentación decente (IoU ~0.74), pero fallaba en casos complejos o con áreas de agua delgadas.
- **Impacto del Fine-tuning**: El fine-tuning del decodificador mejoró drásticamente el rendimiento, aumentando el IoU medio en más de 20 puntos porcentuales. El modelo afinado demostró ser mucho más preciso y robusto para el dominio específico de inundaciones.
- **Comparación con U-Net**: El modelo U-Net, un especialista entrenado de extremo a extremo, alcanzó el mejor rendimiento general (IoU ~0.91), superando ligeramente al SAM afinado.

## Conclusiones
- El fine-tuning es una estrategia extremadamente efectiva para adaptar modelos de fundación como SAM a tareas específicas, logrando mejoras significativas con un coste computacional reducido (solo se entrenó el 4.33% de los parámetros).
- SAM afinado se convierte en un segmentador de alto rendimiento, cerrando en gran medida la brecha con arquitecturas especialistas como U-Net.
- Existe un trade-off:
    - **U-Net**: Ofrece el máximo rendimiento pero requiere un entrenamiento completo y es un modelo "especialista" que solo sirve para esta tarea.
    - **SAM Fine-tuned**: Logra un rendimiento casi a la par, manteniendo la flexibilidad de la arquitectura original y requiriendo un entrenamiento mucho más eficiente.

## Reflexión Personal
Esta práctica demuestra el poder de los modelos de fundación y el paradigma de "pre-entrenar y luego afinar". En lugar de construir un modelo desde cero, podemos tomar un modelo generalista y, con un esfuerzo computacional relativamente bajo, especializarlo para que sobresalga en una tarea concreta. La comparación con U-Net subraya que, si bien los modelos especialistas aún pueden tener una ligera ventaja, los modelos de fundación afinados son una alternativa increíblemente potente y eficiente.

---

## 📓 Notebook

**[Abrir en Google Colab](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT3/Practico12.ipynb)**