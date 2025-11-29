

# Fine-Tuning de Transformers para Sentimiento Financiero

En esta práctica, se aborda un problema organizacional (medir el sentimiento del mercado) traduciéndolo en una solución técnica de IA (clasificación NLP). Se establece y evalúa un *baseline* clásico (TF-IDF), demostrando sus limitaciones semánticas. Luego, se implementa una estrategia de *fine-tuning* para adaptar modelos Transformer preentrenados (genéricos vs. de dominio) al problema. Finalmente, se analizan los artefactos de entrenamiento (overfitting), se compara el espacio latente (UMAP) y se evalúan técnicas avanzadas (balanceo de clases) para seleccionar el modelo óptimo.

## Objetivos
- Traducir una necesidad organizacional (análisis de sentimiento) en una solución técnica de NLP.
- Implementar y evaluar un *baseline* estadístico (TF-IDF + Regresión Logística) para establecer un benchmark de rendimiento.
- Aplicar *fine-Tuning* (transfer learning) para especializar modelos Transformer preentrenados (`bert-base-uncased` y `ProsusAI/finbert`) en un dataset de dominio.
- Comparar arquitecturas (genérica vs. específica de dominio) para seleccionar el modelo con el mejor balance de precisión, velocidad y estabilidad.
- Diagnosticar el *overfitting* analizando las curvas de pérdida (Training vs. Validation).
- Demostrar visualmente (con UMAP) la superioridad del espacio de características (embeddings) de un Transformer frente a TF-IDF.
- Evaluar técnicas avanzadas (balanceo de clases) y medir su impacto real en el rendimiento.

## Metodología

### 1. Análisis Exploratorio (EDA) y Baseline
- **Problema**: Se partió de la necesidad de clasificar el sentimiento de ~12k tweets financieros (Dataset: `zeroshot/twitter-financial-news-sentiment`).
- **Análisis de Datos**: El EDA reveló un **severo desbalance de clases**, con la clase "Neutral" (2) dominando sobre "Bullish" (1) y "Bearish" (0). Esto justificó el uso de **F1-macro** como métrica principal.
- **Baseline Clásico**: Se implementó un pipeline de `TF-IDF` (con n-gramas 1,2) y `LogisticRegression`. Fue crucial usar `class_weight="balanced"` para forzar al modelo a prestar atención a las clases minoritarias.
- **Diagnóstico Baseline**: Una proyección UMAP sobre las *features* de TF-IDF mostró un "blob" caótico, donde las tres clases estaban completamente mezcladas, prediciendo un rendimiento pobre.

### 2. Fine-Tuning de Transformer Genérico
- **Objetivo**: Superar el *baseline* estadístico usando un modelo que entienda el contexto semántico.
- **Modelo**: Se seleccionó `bert-base-uncased`, un modelo genérico preentrenado en texto general.
- **Entrenamiento**: Se aplicó *fine-tuning* usando el `Trainer` de Hugging Face por 6 épocas. Se monitorearon las métricas de validación por época.

### 3. Fine-Tuning de Transformer de Dominio (Extensión)
- **Hipótesis**: Un modelo preentrenado en texto financiero (`ProsusAI/finbert`) debería superar al modelo genérico.
- **Experimento**: Se repitió el *fine-Tuning* con `ProsusAI/finbert` por 3 épocas (esperando una convergencia más rápida).
- **Análisis**: Se comparó el F1-macro, el tiempo de entrenamiento y las curvas de pérdida contra el modelo genérico.

### 4. Análisis de Espacio Latente (Extensión)
- **Objetivo**: Demostrar *visualmente* por qué el Transformer supera al *baseline*.
- **Técnica**: Se extrajeron los *logits* (la capa de salida previa a la clasificación) del modelo `FinBERT` entrenado.
- **Visualización**: Se aplicó UMAP a estos *logits* para proyectarlos en 2D y comparar la separabilidad de las clases contra el "blob" caótico del TF-IDF.

### 5. Evaluación de Balanceo de Clases (Extensión)
- **Hipótesis**: Dado que el F1 del *baseline* mejoró con el balanceo, aplicar esta técnica al Transformer podría mejorar aún más el F1-macro.
- **Técnica**: Se creó una subclase `WeightedTrainer` que sobreescribe `compute_loss`, aplicando pesos (`nn.CrossEntropyLoss(weight=...)`) para penalizar más los errores en las clases minoritarias.
- **Análisis**: Se comparó el F1-macro final de `FinBERT + Balanced` contra `FinBERT` estándar.

## Resultados Principales

### 1. Comparativa de Modelos (Baseline vs. Transformers)

El *fine-tuning* de Transformers demostró un salto cuántico en rendimiento, superando al *baseline* estadístico por ~10 puntos de F1-macro.

| Modelo | F1-Macro | Notas |
| :--- | ---: | :--- |
| Baseline (TF-IDF + LR Balanced) | 0.7321 | Techo de rendimiento estadístico. |
| **Genérico (BERT-base)** | **0.8265** | **Mejor F1.** Logrado en la época 6. |
| Dominio (FinBERT - Sin Balanceo) | 0.8216 | F1 casi idéntico, pero más eficiente. |
| Dominio (FinBERT + Balanced) | 0.8196 | El balanceo *empeoró* el rendimiento. |

### 2. Diagnóstico de Arquitecturas (Genérico vs. Dominio)

El F1-score más alto no contó toda la historia. El análisis de las curvas de entrenamiento reveló un claro ganador organizacional:

- **Genérico (`bert-base-uncased`):** Logró el F1 más alto (0.8265), pero a un costo alto. El entrenamiento mostró un **overfitting severo** después de la época 2 (Training Loss a 0.02, Validation Loss disparado de 0.37 a 0.68). El modelo estaba "memorizando".
- **Dominio (`ProsusAI/finbert`):** Logró un F1 casi idéntico (0.8216) pero fue **más eficiente y estable**. Alcanzó su rendimiento máximo en **3 épocas** (6.5 min) vs 6 épocas (15.5 min) del genérico, y sus curvas de pérdida fueron mucho más saludables (sin overfitting severo).

**Conclusión del Trade-off**: `FinBERT` es la elección superior para producción, ofreciendo el mismo rendimiento con la mitad del costo de entrenamiento y mayor estabilidad.

### 3. Impacto Visual del Fine-Tuning (El "Blob" vs. los "Continentes")

La visualización UMAP validó por qué los Transformers ganaron:
- **TF-IDF (Baseline)**: [Imagen de blob caótico de UMAP] Mostró un "blob" caótico donde las clases 0, 1 y 2 eran indistinguibles.
- **FinBERT (Transformer)**: [Imagen de UMAP con clusters separados] Mostró tres "continentes" de clases claros y semánticamente separados. El Transformer no "encontró" una estructura, **la creó**, fabricando un espacio de características separable que explica el salto de 10 puntos en F1.

### 4. Resultado de Técnicas Avanzadas (El Balanceo Falló)

La hipótesis de la Extensión 5 falló. Aplicar balanceo de clases al Transformer **perjudicó** el rendimiento (F1 0.8216 -> 0.8196).

- **Análisis**: A diferencia del modelo estadístico, el Transformer (con su mecanismo de auto-atención) fue lo suficientemente robusto para manejar el desbalance de clases por sí mismo. La "sobre-corrección" manual (forzar los pesos) desvió al modelo y empeoró su capacidad de generalización.

## Conclusiones
- El *fine-tuning* no es opcional, es un paso crítico. Los modelos Transformer generaron **+10 puntos de F1-macro** sobre el *baseline* estadístico, demostrando el valor de entender el contexto semántico.
- El F1-score más alto no es el "mejor" modelo. El genérico `bert-base` (0.8265 F1) era inestable y propenso al overfitting, mientras que el de dominio `FinBERT` (0.8216 F1) fue la **opción superior para producción** (más rápido, más estable, mismo rendimiento).
- El EDA visual (UMAP) es clave para el diagnóstico. Se demostró visualmente que TF-IDF no podía separar las clases ("blob"), mientras que el Transformer sí lo hizo ("continentes").
- No todas las técnicas "avanzadas" ayudan. El balanceo de clases fue crítico para el *baseline* simple, pero fue **contraproducente** para el Transformer avanzado, que ya manejaba el desbalance. Se debe medir y validar, no asumir.

## Reflexión Personal
Esta práctica ejecutó un proyecto de NLP de extremo a extremo, desde la justificación del problema hasta la selección de un modelo listo para producción. El proceso reflejó perfectamente el ciclo de vida de MLOps:
1.  Establecer un **Baseline** (TF-IDF) medible.
2.  Probar una solución moderna (**Transformers**) y demostrar su superioridad.
3.  **Diagnosticar** el entrenamiento (overfitting en `bert-base`).
4.  **Comparar trade-offs** (eficiencia y estabilidad de `FinBERT` vs. F1 marginal de `bert-base`).
5.  **Validar hipótesis** (el balanceo de clases falló).

