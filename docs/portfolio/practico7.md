# Fundamentos de Redes Neuronales (MLP)
**Unidad 2 – Fundamentos de Deep Learning**

---

## 📘 Descripción

En este proyecto profundicé en los fundamentos del **Deep Learning**, implementando **redes neuronales artificiales (MLPs)** desde cero. Demostré mi dominio de:

- Arquitecturas neuronales: desde perceptrones simples hasta redes multicapa complejas.
- Técnicas de regularización (Dropout) para prevenir overfitting.
- Optimización de hiperparámetros: funciones de activación, optimizadores y configuraciones de capas.
- Evaluación sistemática del rendimiento con métricas de clasificación.

Implementé experimentos controlados para analizar cómo cada componente impacta en la convergencia, precisión y capacidad de generalización.

---

## 🎯 Habilidades Demostradas

- Comprender la estructura y funcionamiento de un **perceptrón artificial**.  
- Implementar redes neuronales **multicapa (MLP)** utilizando *Keras* y *TensorFlow*.  
- Explorar **diferentes funciones de activación** (`relu`, `tanh`, `logistic`).  
- Comparar **optimizadores** (`adam`, `sgd`, `rmsprop`) y su efecto sobre la convergencia.  
- Aplicar **regularización** mediante `Dropout` para prevenir el overfitting.  
- Evaluar el desempeño de las redes con **métricas de clasificación** (accuracy, precision, recall, F1-score).  
- Analizar la **estabilidad y capacidad de generalización** de cada configuración.

---

## 📊 Dataset

Se emplearon datasets clásicos para tareas de clasificación supervisada, entre ellos:

- **XOR dataset** (problema no lineal, ideal para observar limitaciones del perceptrón).  
- **Datasets de referencia de Scikit-Learn**, como *Breast Cancer* o *Digits*, para probar distintas arquitecturas MLP.  

Las variables de entrada fueron normalizadas para asegurar un entrenamiento estable, siguiendo las recomendaciones vistas en clase.

---

## ⚙️ Desarrollo

El flujo de trabajo siguió el **pipeline estándar de Deep Learning**:

1. **Configuración del entorno TensorFlow / Keras**  
   - Inicialización del entorno con control de GPU y seeds de reproducibilidad.  

2. **Diseño de modelos**  
   - 🔹 *Perceptrón simple:* red de una sola capa lineal, probada sobre el problema XOR.  
   - 🔹 *MLP multicapa:* se experimentó con arquitecturas `(4,)`, `(10,)`, `(4,4)`, `(10,5)` y `(64,32,16)`.

3. **Funciones de activación**  
   - Comparación entre `relu`, `tanh` y `logistic`, observando efectos sobre la convergencia y la capacidad de aprendizaje.

4. **Regularización**  
   - Incorporación de **Dropout** (0.2–0.5) para evitar overfitting y mejorar la robustez del modelo.

5. **Optimización**  
   - Se testearon distintos **optimizadores** (`adam`, `sgd`, `rmsprop`), ajustando el *learning rate* y parámetros de momentum.

6. **Entrenamiento y validación**  
   - División de datos en *train/test*.  
   - Monitoreo de la pérdida y accuracy por época.  
   - Evaluación con métricas de rendimiento.

7. **Visualización y análisis**  
   - Gráficas de pérdida y accuracy.  
   - Reportes de clasificación y comparación de resultados.

---

## 📈 Resultados

- El **perceptrón simple** no logró resolver el problema XOR, confirmando su limitación para separar clases no lineales.  
- Las **redes multicapa (MLP)** sí aprendieron correctamente la frontera de decisión, evidenciando la importancia de la **no linealidad y profundidad**.  
- **Funciones de activación:**  
  - `relu` ofreció la mejor convergencia y desempeño general.  
  - `tanh` mostró estabilidad pero menor velocidad.  
  - `logistic` tuvo problemas de saturación y gradientes pequeños.  
- **Optimizadores:**  
  - `adam` alcanzó los mejores resultados con mínima configuración.  
  - `sgd` fue más estable pero requirió mayor tuning.  
  - `rmsprop` funcionó bien con tasas de aprendizaje pequeñas.  
- **Dropout** (0.3–0.5) ayudó a mejorar la generalización reduciendo el sobreajuste.  

📊 *El modelo final alcanzó una accuracy promedio superior al 95% en los datasets utilizados.*

---

## 🧩 Conclusiones

- Las **redes neuronales multicapa** permiten modelar relaciones complejas que los modelos lineales no pueden capturar.  
- La elección de **arquitectura, activación y optimizador** influye directamente en la capacidad de aprendizaje y estabilidad del modelo.  
- El **Dropout** y otras formas de regularización son esenciales para evitar el sobreajuste y mejorar la robustez del aprendizaje.  
- Este práctico consolida la comprensión de los principios fundamentales de **Deep Learning**, sentando las bases para avanzar hacia arquitecturas más complejas como las **CNNs** y el **Transfer Learning** en la siguiente unidad.

---

## 🔗 Referencias

- Kurucz, J. F. – *Fundamentos del Aprendizaje Automático*, Unidad 2 (Deep Learning).  
- **TensorFlow / Keras API Docs:**  
  [https://www.tensorflow.org/api_docs/python/tf/keras](https://www.tensorflow.org/api_docs/python/tf/keras)  
- **Scikit-learn documentation:**  
  [https://scikit-learn.org/](https://scikit-learn.org/)  
- Goodfellow, I., Bengio, Y., & Courville, A. – *Deep Learning* (MIT Press, 2016)

---

## 📓 Notebook

[Ver Notebook Completo](UT2/practico7.ipynb)