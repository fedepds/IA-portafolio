# Regresión Lineal y Logística

## 🎯 Objetivo
Aplicar y comprender dos de los modelos más clásicos de machine learning:
1. **Regresión Lineal** → para predecir valores numéricos continuos.  
2. **Regresión Logística** → para resolver problemas de clasificación binaria.  

---

## 📋 Lo que aprendimos
- Diferencia entre regresión lineal y logística.
- Cómo cargar y explorar datasets reales.
- Cómo dividir datos en entrenamiento y prueba con `train_test_split`.
- Entrenar modelos paso a paso (`fit`) y hacer predicciones (`predict`).
- Evaluar modelos con métricas específicas para cada caso.

---

## 🏠 Parte 1: Regresión Lineal - Precios de Casas (Boston Housing)

### Contexto de negocio
Una inmobiliaria en Boston quiere **predecir el valor medio de casas** en miles de USD, basándose en características del barrio (criminalidad, número de habitaciones, acceso a autopistas, etc.).

### Proceso
1. **Dataset**: Boston Housing (`medv` es la variable objetivo).  
2. **Preparación**:
   - `X` → todas las variables independientes.  
   - `y` → columna `medv` (precio de la vivienda).  
3. **Entrenamiento**:
   - `LinearRegression()` de scikit-learn.  
   - Entrenamiento con `fit(X_train, y_train)`.  
4. **Predicciones**:
   - Se usó `predict(X_test)`.  
5. **Evaluación con métricas**:
   - **MAE (Error Absoluto Medio)** → error promedio en miles de USD.  
   - **RMSE (Raíz del Error Cuadrático Medio)** → penaliza más los errores grandes.  
   - **R² (Coeficiente de Determinación)** → indica qué porcentaje de la variabilidad explica el modelo.  
   - **MAPE (Error Porcentual Absoluto Medio)** → error relativo en %.  

---

## 🏥 Parte 2: Regresión Logística - Diagnóstico Médico (Cáncer de Mama)

### Contexto de negocio
Un hospital necesita un sistema que ayude a **clasificar tumores** como:  
- **0 = Maligno**  
- **1 = Benigno**  

basado en 30 características celulares.

### Proceso
1. **Dataset**: Breast Cancer Wisconsin (incluido en scikit-learn).  
2. **Preparación**:
   - `X_cancer` → variables predictoras.  
   - `y_cancer` → etiqueta (0 o 1).  
3. **Entrenamiento**:
   - `LogisticRegression(max_iter=5000)` para asegurar convergencia.  
4. **Predicciones**:
   - Con `predict(X_test_cancer)`.  
5. **Evaluación con métricas de clasificación**:
   - **Accuracy**: porcentaje total de aciertos.  
   - **Precision**: de los casos predichos como benignos, cuántos eran correctos.  
   - **Recall (Sensibilidad)**: de todos los casos benignos reales, cuántos detectó.  
   - **F1-Score**: balance entre precisión y recall.  
   - **Matriz de Confusión**: muestra aciertos y errores →  
     - Falsos positivos: predecir benigno cuando es maligno.  
     - Falsos negativos: predecir maligno cuando es benigno (el error más grave en medicina).  

---

## ❓ Preguntas de reflexión

- **Diferencia principal**:  
  - Lineal → valores continuos.  
  - Logística → categorías.  

- **¿Por qué dividir datos en train/test?**  
  Para evitar overfitting y comprobar que el modelo generaliza.  

- **¿Qué significa exactitud del 95%?**  
  Que el modelo acierta en 95 de cada 100 predicciones.  

- **Error más peligroso en medicina**:  
  Predecir “benigno” cuando en realidad es maligno (falso negativo).  

---

## 🔍 Comparación de modelos

| Aspecto               | Regresión Lineal                         | Regresión Logística                    |
|------------------------|------------------------------------------|----------------------------------------|
| Qué predice            | Valores continuos (números reales)       | Categorías (etiquetas)                 |
| Ejemplo de uso         | Precios de casas, salario de empleados   | Diagnóstico médico, spam/no spam       |
| Rango de salida        | Cualquier número real                    | Probabilidad entre 0 y 1               |
| Métrica principal      | MAE, RMSE, R²                            | Precision, Recall, F1-Score            |

---

## 🚀 Reflexión Final
- Para **salario de un empleado** → regresión lineal.  
- Para **spam en emails** → regresión logística.  
- Separar train/test es crucial para evaluar si el modelo funciona en la práctica.  

---

