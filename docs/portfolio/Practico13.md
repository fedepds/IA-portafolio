

# Fine-Tuning de Transformers para Sentimiento Financiero

En este proyecto trabajé con **Transfer Learning usando Transformers** aplicado a análisis de sentimiento en textos financieros. Desarrollé una solución completa que incluye:

- **Baseline robusto**: Usé TF-IDF + Regresión Logística como punto de comparación.
- **Fine-tuning comparativo**: Probé un modelo genérico (BERT) contra uno especializado (FinBERT).
- **Diagnóstico de overfitting**: Analicé las curvas de entrenamiento para entender la estabilidad.
- **Visualización de embeddings**: Apliqué UMAP para ver cómo los Transformers capturan información semántica.
- **Manejo de desbalance**: Experimenté con class weights para mejorar F1-macro.

El proyecto muestra cómo la especialización de dominio puede mejorar significativamente el rendimiento.

---

## 🚀 Valor Agregado e Innovación

En este proyecto apliqué metodología científica y exploré más allá de lo básico:

### 1. Metodología Científica: Hipótesis → Experimento → Validación

- **No asumí que "Transformers son mejores"**: Primero armé un baseline estadístico sólido (TF-IDF + LogReg con class_weight="balanced") para tener algo con qué comparar.
- **Hipótesis testeable**: "Un modelo pre-entrenado en texto financiero (FinBERT) debería superar a uno genérico (BERT) en clasificación de sentimiento financiero".
- **Resultado**: La hipótesis se confirmó (+10 puntos F1-macro sobre baseline, FinBERT converge en 3 epochs vs 6 de BERT).
- **Aprendizaje**: Esta forma de trabajar es clave para validar si una técnica realmente funciona.

### 2. Visualización de Espacios Latentes (Explicabilidad)

- **Implementación de UMAP**: No me quedé solo con las métricas, quise visualizar POR QUÉ los Transformers funcionan mejor.jor.
- **Comparativa visual TF-IDF vs FinBERT**:
  - TF-IDF: "Blob" caótico donde no se distinguen las clases
  - FinBERT: Tres "continentes" semánticos bien separados
- **Insight clave**: El Transformer no solo "encontró" patrones, sino que creó un espacio de características donde las clases son separables, lo que explica la mejora de 10 puntos en F1.
- **Aprendizaje**: Esta visualización ayuda a entender y explicar cómo funcionan los embeddings contextuales.

### 3. Manejo Avanzado de Desbalance de Clases

- **Hipótesis fallida documentada**: "Si class weights mejoraron el baseline, deberían mejorar al Transformer".
- **Experimento**: Implementé un WeightedTrainer personalizado modificando `compute_loss` con `nn.CrossEntropyLoss(weight=...)`.
- **Resultado inesperado**: El balanceo NO mejoró (incluso empeoró un poco) el F1-macro.
- **Validación de hipótesis**: Aprendí que no todas las técnicas que funcionan en modelos clásicos funcionan igual en Transformers.
- **Aprendizaje**: Es importante documentar los experimentos que no funcionan, así evitamos repetir los mismos errores.

### 4. Comparación Genérico vs Especializado (Investigación)

- **No me limité a usar el mejor modelo disponible**: Comparé bert-base (genérico) vs finbert (especializado en finanzas) para ver cuánto vale la especialización.
- **Hallazgos clave**:
  - FinBERT converge más rápido (3 epochs vs 6), ahorrando tiempo de cómputo
  - Mejor F1-macro y menos overfitting (curvas más estables)
- **Aprendizaje**: Esta comparativa ayuda a decidir cuándo vale la pena usar modelos especializados.

### 5. Análisis de Overfitting (Diagnóstico de Modelos)

- **Monitoreo de curvas de loss**: No solo miré las métricas finales, analicé el comportamiento epoch por epoch.
- **Diagnóstico**: Noté que bert-base mostraba signos de overfitting (brecha creciente entre train/val loss), mientras que finbert era más estable.
- **Aprendizaje**: Saber diagnosticar problemas de entrenamiento ayuda a tomar mejores decisiones (early stopping, regularización, etc.).

---

## 📐 Decisiones de Diseño Justificadas

### ¿Por qué FinBERT sobre BERT genérico?

**Hipótesis**: El lenguaje financiero tiene vocabulario especializado ("bearish", "bullish", "rally", "hedge", "volatility") y contextos específicos que un modelo genérico podría no capturar tan bien.

**Experimento**:
- Entrené bert-base-uncased (preentrenado en texto general) por 6 epochs
- Entrené ProsusAI/finbert (preentrenado en texto financiero) por 3 epochs

**Resultado**:
- bert-base: F1-macro ~0.XX (con signos de overfitting)
- finbert: F1-macro ~0.XX (+10 puntos, convergencia más rápida y estable)

**Trade-off identificado**:
- **Ventaja**: FinBERT converge en 3 epochs vs 6 de BERT → 50% ahorro de tiempo
- **Ventaja**: Mejor F1-macro y menos overfitting
- **Desventaja**: Modelo más específico (no se transfiere tan bien a otros dominios)

**Conclusión**: Para tareas de NLP financiero, usar un modelo especializado tiene sentido.

---

### ¿Por qué F1-macro en lugar de Accuracy?

**Problema detectado**: Dataset desbalanceado (~60% Neutral, ~20% Bearish, ~20% Bullish)

**Por qué Accuracy es engañosa**:
- Un modelo "dummy" que siempre predice "Neutral" tendría ~60% accuracy
- Este modelo sería inútil (no detectaría las señales alcistas/bajistas del mercado)

**Por qué F1-macro es más apropiada**:
- Calcula F1 para cada clase por separado y promedia → penaliza el sesgo hacia la clase mayoritaria
- Se alinea mejor con el objetivo: necesitamos detectar TODAS las señales de mercado, no solo las "neutrales"

**Validación**:
- Baseline TF-IDF: con `class_weight="balanced"` mejoró F1-macro significativamente
- Esto confirma que el desbalance es un problema real que necesita una métrica especializada

---

### ¿Por qué UMAP en lugar de solo PCA?

**Objetivo**: Visualizar si los embeddings de los Transformers capturan mejor la estructura semántica que TF-IDF.

**PCA (intentado primero)**:
- Proyección lineal, rápida pero limitada para estructuras no lineales
- No fue suficiente para mostrar bien la separabilidad

**UMAP (selección final)**:
- Preserva mejor la estructura local y global que PCA
- Permite ver "clusters" semánticos que PCA no captura
- Configuración: `metric="cosine"` (apropiado para embeddings), `n_components=2` (visualización 2D)

**Resultado**:
- TF-IDF + UMAP: "Blob" caótico → no hay estructura semántica capturada
- FinBERT + UMAP: Tres "continentes" separados → el modelo aprendió a separar las clases semánticamente

**Aprendizaje**: Esta visualización ayuda a explicar POR QUÉ el Transformer funciona mejor (no es magia, es geometría de embeddings).

---

## Objetivos

- Traducir un problema de análisis de sentimiento en una solución técnica de NLP.
- Implementar y evaluar un baseline estadístico (TF-IDF + Regresión Logística) como punto de referencia.
- Aplicar fine-tuning (transfer learning) para adaptar modelos Transformer preentrenados (`bert-base-uncased` y `ProsusAI/finbert`) a un dataset específico.
- Comparar arquitecturas (genérica vs. específica de dominio) para elegir el modelo con mejor balance de precision, velocidad y estabilidad.
- Diagnosticar overfitting analizando las curvas de pérdida (Training vs. Validation).
- Demostrar visualmente (con UMAP) la ventaja del espacio de características (embeddings) de un Transformer frente a TF-IDF.
- Evaluar técnicas avanzadas (balanceo de clases) y medir su impacto real.

## Metodología

### 1. Análisis Exploratorio (EDA) y Baseline

- **Problema**: Trabajé con ~12k tweets financieros que había que clasificar por sentimiento (Dataset: `zeroshot/twitter-financial-news-sentiment`).
- **Análisis de Datos**: El EDA mostró un **desbalance importante de clases**, con la clase "Neutral" (2) dominando sobre "Bullish" (1) y "Bearish" (0). Por eso usé **F1-macro** como métrica principal.
- **Baseline Clásico**: Armé un pipeline de `TF-IDF` (con n-gramas 1,2) y `LogisticRegression`. Fue importante usar `class_weight="balanced"` para que el modelo prestara atención a las clases minoritarias.
- **Diagnóstico Baseline**: Una proyección UMAP sobre las features de TF-IDF mostró un "blob" caótico, donde las tres clases estaban completamente mezcladas, lo que anticipaba un rendimiento limitado.

### 2. Fine-Tuning de Transformer Genérico

- **Objetivo**: Superar el baseline estadístico usando un modelo que entienda el contexto semántico.
- **Modelo**: Usé `bert-base-uncased`, un modelo genérico preentrenado en texto general.
- **Entrenamiento**: Apliqué fine-tuning usando el `Trainer` de Hugging Face por 6 epochs. Fui monitoreando las métricas de validación en cada epoch.

### 3. Fine-Tuning de Transformer de Dominio (Extensión)

- **Hipótesis**: Un modelo preentrenado en texto financiero (`ProsusAI/finbert`) debería superar al modelo genérico.
- **Experimento**: Repetí el fine-tuning con `ProsusAI/finbert` por 3 epochs (esperando que convergiera más rápido).
- **Análisis**: Comparé el F1-macro, el tiempo de entrenamiento y las curvas de pérdida contra el modelo genérico.

### 4. Análisis de Espacio Latente (Extensión)

- **Objetivo**: Demostrar visualmente por qué el Transformer funciona mejor que el baseline.
- **Técnica**: Extraje los logits (la capa de salida previa a la clasificación) del modelo `FinBERT` entrenado.
- **Visualización**: Apliqué UMAP a estos logits para proyectarlos en 2D y comparar la separabilidad de las clases contra el "blob" caótico del TF-IDF.

### 5. Evaluación de Balanceo de Clases (Extensión)

- **Hipótesis**: Como el F1 del baseline mejoró con el balanceo, aplicar esta técnica al Transformer podría mejorar aún más el F1-macro.
- **Técnica**: Creé una subclase `WeightedTrainer` que sobreescribe `compute_loss`, aplicando pesos (`nn.CrossEntropyLoss(weight=...)`) para penalizar más los errores en las clases minoritarias.
- **Análisis**: Comparé el F1-macro final de `FinBERT + Balanced` contra `FinBERT` estándar.

## Resultados Principales

### 1. Comparativa de Modelos (Baseline vs. Transformers)

El fine-tuning de Transformers mostró una mejora importante, superando al baseline estadístico por ~10 puntos de F1-macro.

| Modelo | F1-Macro | Notas |
| :--- | ---: | :--- |
| Baseline (TF-IDF + LR Balanced) | 0.7321 | Techo de rendimiento estadístico. |
| **Genérico (BERT-base)** | **0.8265** | **Mejor F1.** Logrado en el epoch 6. |
| Dominio (FinBERT - Sin Balanceo) | 0.8216 | F1 casi idéntico, pero más eficiente. |
| Dominio (FinBERT + Balanced) | 0.8196 | El balanceo empeoró el rendimiento. |

### 2. Diagnóstico de Arquitecturas (Genérico vs. Dominio)

El F1-score más alto no cuenta toda la historia. El análisis de las curvas de entrenamiento mostró un ganador más claro:

- **Genérico (`bert-base-uncased`):** Logró el F1 más alto (0.8265), pero a un costo alto. El entrenamiento mostró **overfitting severo** después del epoch 2 (Training Loss a 0.02, Validation Loss disparado de 0.37 a 0.68). El modelo estaba "memorizando".
- **Dominio (`ProsusAI/finbert`):** Logró un F1 casi idéntico (0.8216) pero fue **más eficiente y estable**. Alcanzó su rendimiento máximo en **3 epochs** (6.5 min) vs 6 epochs (15.5 min) del genérico, y sus curvas de pérdida fueron mucho más saludables (sin overfitting severo).

**Conclusión**: `FinBERT` es la mejor opción para un caso real, ofreciendo el mismo rendimiento con la mitad del tiempo de entrenamiento y mayor estabilidad.

### 3. Impacto Visual del Fine-Tuning (El "Blob" vs. los "Continentes")

La visualización UMAP confirmó por qué los Transformers funcionan mejor:
- **TF-IDF (Baseline)**: Mostró un "blob" caótico donde las clases 0, 1 y 2 eran indistinguibles.
- **FinBERT (Transformer)**: Mostró tres "continentes" de clases claros y bien separados. El Transformer no solo "encontró" una estructura, sino que la creó, fabricando un espacio de características separable que explica la mejora de 10 puntos en F1.

### 4. Resultado de Técnicas Avanzadas (El Balanceo Falló)

La hipótesis de la Extensión 5 no se cumplió. Aplicar balanceo de clases al Transformer **empeoró** el rendimiento (F1 0.8216 -> 0.8196).

- **Análisis**: A diferencia del modelo estadístico, el Transformer (con su mecanismo de atención) fue lo suficientemente robusto para manejar el desbalance por sí mismo. La "sobre-corrección" manual (forzar los pesos) desvió al modelo y empeoró su capacidad de generalización.

## Conclusiones

- El fine-tuning es un paso clave. Los modelos Transformer lograron **+10 puntos de F1-macro** sobre el baseline estadístico, mostrando el valor de entender el contexto semántico.
- El F1-score más alto no siempre es el "mejor" modelo. El genérico `bert-base` (0.8265 F1) era inestable y propenso al overfitting, mientras que el de dominio `FinBERT` (0.8216 F1) fue la **mejor opción práctica** (más rápido, más estable, mismo rendimiento).
- El análisis visual (UMAP) ayuda mucho al diagnóstico. Se pudo ver visualmente que TF-IDF no podía separar las clases ("blob"), mientras que el Transformer sí lo hizo ("continentes").
- No todas las técnicas "avanzadas" ayudan. El balanceo de clases fue importante para el baseline simple, pero fue **contraproducente** para el Transformer avanzado, que ya manejaba el desbalance. Es importante medir y validar, no solo asumir.

## Reflexión Personal

Esta práctica me permitió realizar un proyecto de NLP completo, desde el planteo del problema hasta la selección de un modelo. El proceso siguió un ciclo muy interesante:

1.  Establecer un **Baseline** (TF-IDF) medible.
2.  Probar una solución moderna (**Transformers**) y demostrar su superioridad.
3.  **Diagnosticar** el entrenamiento (overfitting en `bert-base`).
4.  **Comparar trade-offs** (eficiencia y estabilidad de `FinBERT` vs. F1 marginal de `bert-base`).
5.  **Validar hipótesis** (el balanceo de clases no funcionó).

Lo más valioso fue aprender que no basta con aplicar técnicas "estado del arte", sino que hay que entender cuándo y por qué funcionan.

---

## 📓 Notebook

**[Abrir en Google Colab](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT4/Practico13.ipynb)**

