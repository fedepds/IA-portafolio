# Clasificación de Vinos con Machine Learning

## 📌 Práctico: Análisis del Wine Quality Dataset

### 🎯 Objetivo
Aplicar técnicas de **regresión** y **clasificación** sobre el dataset de vinos (rojo y blanco) con el fin de predecir y analizar la calidad de los vinos en función de sus características fisicoquímicas.  

Se exploran modelos de **regresión lineal, Ridge, Lasso** y un modelo de **clasificación binaria con Regresión Logística**.
### 🔹 Ridge Regression (L2 Regularization)
- Agrega una penalización proporcional al **cuadrado de los coeficientes**.  
- No elimina variables, pero reduce el valor de los coeficientes grandes.  
- Útil cuando todas las variables aportan algo de información.  
### 🔹 Lasso Regression (L1 Regularization)
- Agrega una penalización proporcional al **valor absoluto de los coeficientes**.  
- Puede llevar algunos coeficientes a **0**, realizando selección automática de variables.  
- Útil cuando sospechamos que solo unas pocas variables son realmente relevantes.  


---

### 📂 Archivos
- `WineQuality.ipynb`: Notebook principal con el desarrollo de la práctica.  
- `data/`: Contiene los datasets `winequality-red.csv` y `winequality-white.csv`.  
---

### 📊 Proceso
1. **Importación de librerías y carga de datos**  
   - Se integran los datasets de vino tinto y blanco (total: 6497 registros, 13 columnas).  
   - Se crea la columna `type` para diferenciar el tipo de vino.  

2. **Exploración de datos (EDA)**  
   - Distribución de la variable objetivo `quality`.  
   - Matriz de correlación entre variables fisicoquímicas.  
   - La mayoría de los vinos se concentran en calidades 5, 6 y 7.  

3. **Preprocesamiento**  
   - Separación de variables predictoras y target.  
   - Escalamiento de variables con `StandardScaler`.  
   - División en train/test (80/20).  

4. **Modelos de Regresión**  
   - Se entrenaron y evaluaron:
     - **Regresión Lineal**
     - **Ridge Regression**
     - **Lasso Regression**
   - Métricas: MSE, MAE, R².  

5. **Clasificación Binaria**  
   - Se definió `quality_bin`: vinos de calidad **alta (≥7)** vs. **baja (<7)**.  
   - Se entrenó un modelo de **Regresión Logística**.  
   - Métricas: Accuracy, Precision, Recall, F1-score y Matriz de confusión.  

6. **Comparación Red vs White**  
   - Se evaluaron modelos de regresión por separado en vinos tintos y blancos.  
   - Se obtuvo mejor ajuste para vinos tintos que para blancos.  

---

### 📈 Resultados
- **Regresión (todos los vinos):**
  - 📊 Linear Regression → R² = 0.260  
  - 📊 Ridge → R² = 0.260  
  - 📊 Lasso → R² = 0.261  
  > El ajuste es bajo, lo que indica que la calidad no se explica fácilmente solo por variables fisicoquímicas.  

- **Clasificación Binaria (alta vs baja calidad):**
  - Accuracy = 0.82  
  - Precision clase 1 (alta calidad) = 0.61  
  - Recall clase 1 = 0.27  
  > El modelo identifica bien los vinos de baja calidad, pero tiene dificultad para detectar los de alta calidad.  

- **Comparación por tipo:**
  - 🍷 Red Wine → R² = 0.403  
  - 🍷 White Wine → R² = 0.265  
  > Los vinos tintos se predicen mejor que los blancos en este dataset.  

---

### 🚀 Reflexión
Este práctico permitió observar las diferencias entre enfoques de **regresión y clasificación supervisada** aplicados a un mismo dataset:  
- Los modelos lineales no alcanzan alto poder predictivo en la calidad del vino.  
- La clasificación binaria muestra desequilibrio entre clases (pocos vinos de alta calidad).  
- El tipo de vino influye en el rendimiento del modelo, siendo más fácil predecir la calidad en vinos tintos.  

En conclusión, el dataset es ideal para comprender limitaciones de los modelos lineales y la importancia del balance de clases en clasificación.  

---

## 📓 Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT1/Practico4/TA4Wines.ipynb)

[Ver Notebook en GitHub](UT1/Practico4/TA4Wines.ipynb)
