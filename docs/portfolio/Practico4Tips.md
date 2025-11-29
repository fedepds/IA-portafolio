# Regresión con Tips Dataset

## 📌 Práctico: Análisis del Dataset Tips

### 🎯 Objetivo
Aplicar técnicas de **regresión lineal**, **Ridge** y **Lasso** sobre el dataset de **propinas (Tips)**, con el fin de predecir el monto de la propina en función de variables como el total de la cuenta, día, hora y si el cliente es fumador.  

Este práctico permite observar cómo la **regularización** ayuda a mejorar la estabilidad de los modelos y, en el caso de Lasso, a realizar selección de variables.

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
- `TA4Tips.ipynb`: Notebook principal con el desarrollo de la práctica.  
- `data/tips.csv`: Dataset de propinas.  

---

### 📊 Proceso
1. **Importación de librerías y carga de datos**  
   - Se trabajó con el dataset `tips.csv` de `seaborn`.  
   - Incluye 244 registros con 7 variables (numéricas y categóricas).  

2. **Exploración de datos (EDA)**  
   - Estadísticas descriptivas de `total_bill`, `tip` y `size`.  
   - Análisis de variables categóricas: sexo, fumador, día, hora.  
   - Relación entre el total de la cuenta y la propina.  

3. **Preprocesamiento**  
   - Conversión de variables categóricas con One-Hot Encoding.  
   - División en train/test.  
   - Escalamiento de variables numéricas.  

4. **Modelos de Regresión y Clasificación**  
   - Se entrenaron y evaluaron:
     - **Regresión Lineal**
     - **Ridge Regression**
     - **Lasso Regression**
     - **Regresión Logística (clasificación binaria: propina alta vs baja)**  
   - Métricas: R², MAE, RMSE para regresión; Accuracy, Precision, Recall y F1-score para clasificación.  

---

### 📈 Resultados
- **Regresión Lineal**
  - 📊 MAE = 0.67  
  - 📊 RMSE = 0.84  
  - 📊 R² = 0.437  

- **Regularización**
  - 📊 Ridge → R² = 0.423  
  - 📊 Lasso → R² = 0.457  

- **Clasificación Binaria (Propina Alta vs Baja)**
  - Accuracy = 0.735  
  - Precision = 0.609  
  - Recall = 0.778  
  - F1-score = 0.683  
  - Matriz de confusión:
    ```
    [[22  9]
     [ 4 14]]
    ```
  - El modelo identifica bien las propinas bajas (recall 0.71) y logra un buen desempeño para las propinas altas (recall 0.78).  

---

### 🚀 Reflexión
Este práctico permitió comparar distintos enfoques en un dataset pequeño y mixto (variables numéricas y categóricas):  
- **Ridge** estabiliza los coeficientes pero no mejora sustancialmente el R² respecto a la regresión lineal.  
- **Lasso** obtuvo el mejor desempeño al simplificar el modelo seleccionando automáticamente las variables más relevantes.  
- La **clasificación binaria con Regresión Logística** alcanzó un 73% de accuracy, mostrando que el modelo es útil para distinguir entre propinas altas y bajas, aunque el recall de la clase alta aún puede mejorar.  

En conclusión, el dataset **Tips** es ideal para comprender cómo los métodos de regularización (Ridge y Lasso) impactan en la predicción y cómo un enfoque de clasificación puede aportar otra perspectiva en el análisis.  

---

## 📓 Notebook

[Ver Notebook Completo](UT1/Practico4/TA4Tips%20.ipynb)
