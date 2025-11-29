# Validación Cruzada y Selección de Modelos

## 🎯 Objetivo
El propósito de este práctico fue aprender a:
1. Prevenir **data leakage** utilizando pipelines.
2. Implementar **validación cruzada** (cross-validation) robusta.
3. Comparar múltiples modelos de forma sistemática.
4. Analizar métricas de estabilidad y seleccionar el modelo más adecuado.

---

## 📊 Contexto y Dataset
- **Problema**: predecir **abandono estudiantil** y **éxito académico** en educación superior.  
- **Objetivo aplicado**: identificar estudiantes en riesgo para implementar estrategias de apoyo.  
- **Variables**: 36 características (demográficas, académicas, socioeconómicas).  
- **Valor**: mejorar la **retención estudiantil** y reducir la tasa de abandono.  

---

## 🔧 Proceso

### 1. Preparación de datos
- Se construyeron **pipelines** para estandarizar el preprocesamiento y evitar fugas de datos.  
- Se aplicó división estratificada de los datos para mantener balance entre clases.

### 2. Validación cruzada
- Se utilizó **K-Fold Cross Validation** para obtener estimaciones más confiables del rendimiento de los modelos.  
- Esta técnica permitió observar no solo la media de la métrica, sino también la **variabilidad entre folds**.

### 3. Modelos evaluados
- **DummyClassifier**  
  - Modelo base que predice siempre la clase más frecuente.  
  - Sirve como referencia mínima: cualquier modelo útil debe superarlo.  

- **Regresión Logística**  
  - Modelo lineal para clasificación.  
  - Estima la probabilidad de pertenecer a una clase mediante la función logística.  
  - Sencillo, interpretable y suele ser muy robusto.  

- **Random Forest**  
  - Conjunto de múltiples árboles de decisión entrenados en subconjuntos de datos.  
  - Reduce la varianza de un árbol individual y mejora la generalización.  

- **Gradient Boosting (ej. XGBoost/LightGBM)**  
  - Algoritmo de ensamble que construye árboles de forma secuencial, cada uno corrigiendo los errores del anterior.  
  - Tiende a ser muy preciso, aunque más sensible a parámetros y con mayor riesgo de sobreajuste.  

---

## 📈 Resultados

- El **DummyClassifier** sirvió como referencia mínima.  
- La **Regresión Logística** mostró un rendimiento sólido y estable.  
- **Random Forest** y **Gradient Boosting** presentaron buena performance, pero con mayor varianza en algunos folds.  
- Según los boxplots de validación cruzada, la **Logistic Regression fue la más estable**, mientras que las diferencias en bigotes de otros modelos pudieron deberse a outliers o folds atípicos.

---

## 🔍 Análisis
- La **estabilidad del modelo** es tan importante como la métrica promedio: un modelo con menor varianza ofrece predicciones más confiables.  
- En este caso, la Regresión Logística resultó competitiva y consistente, lo que la hace un buen candidato para producción.  

---

## 🚀 Reflexión y mejoras
- Se recomienda probar **nuevas features derivadas** (interacciones, combinaciones de variables socioeconómicas y académicas).  
- Explorar métodos de **regularización** (Ridge, Lasso) en la regresión logística.  
- Implementar técnicas de **balanceo de clases** (SMOTE, undersampling) si la variable objetivo está desbalanceada.  
- Comparar con modelos más complejos como **Redes Neuronales** o **Stacking** para ensambles.  

---
