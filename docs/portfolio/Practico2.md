# Feature Engineering y Modelo Base

## 🎯 Descripción
En este proyecto extendí el análisis exploratorio del dataset Titanic, demostrando mis habilidades en:
1. Técnicas avanzadas de imputación de datos faltantes con enfoque inteligente.
2. Ingeniería de características (feature engineering) para extraer información relevante de variables existentes.
3. Entrenamiento y evaluación de modelos predictivos, estableciendo líneas base robustas.

---

## 📊 Dataset
- **Fuente**: [Titanic - Kaggle](https://www.kaggle.com/c/titanic)
- **Variable Importante**: `Survived` (0 = no sobrevivió, 1 = sobrevivió).

---

## 🔧 Metodología Aplicada

### 1. Preprocesamiento Inteligente de Datos
Implementé estrategias específicas de imputación según la naturaleza de cada variable:
- **Embarked**: Imputé valores faltantes con la moda (valor más frecuente), método apropiado para variables categóricas.
- **Fare**: Utilicé la mediana para evitar distorsión por valores extremos (outliers).
- **Age**: Apliqué imputación estratégica calculando la mediana por grupos (`Sex` y `Pclass`), logrando estimaciones más precisas y contextualizadas.

### 2. Ingeniería de Características (Feature Engineering)
Creé nuevas variables derivadas para capturar patrones ocultos en los datos:
- **`FamilySize`**: Combiné `SibSp` + `Parch` + 1 para cuantificar el tamaño del grupo familiar.
- **`IsAlone`**: Variable binaria indicando si el pasajero viajaba solo (1) o acompañado (0).
- **`Title`**: Extraíe títulos sociales de la columna `Name` (Mr., Mrs., Miss., etc.) para capturar estatus social y patrones demográficos, agrupando títulos raros en una categoría unificada.

Estas variables capturaron hipótesis de dominio: las familias podían tener ventajas/desventajas en supervivencia, y el estatus social influyó en el acceso a botes salvavidas.

### 3. Transformación de Variables Categóricas
Apliqué **one-hot encoding** mediante `pd.get_dummies` para convertir variables categóricas (`Sex`, `Embarked`, `Title`) en representaciones numéricas binarias procesables por modelos de machine learning.

### 4. Modelado y Evaluación
Entrene y comparé dos aproximaciones:
- **DummyClassifier (Baseline)**: Establecí una línea base prediciendo siempre la clase mayoritaria. Utilicé `class_weight="balanced"` para manejar desbalance de clases.
- **Regresión Logística**: Implementé un modelo de clasificación supervisada, validando la mejora frente al baseline.
- Utilicé `train_test_split` para división apropiada de datos y evitar sobreajuste.

---

## 📈 Resultados

- **Baseline (DummyClassifier)**: Acc ≈ *0.62*.  
- **Regresión Logística**: Acc ≈ *0.79*.  

📌 El modelo de regresión logística **supera claramente al baseline**, lo que confirma que las features creadas y el preprocesamiento aportan información valiosa.

---

## 🔍 Análisis de la matriz de confusión
- **Falsos positivos**: casos donde el modelo predijo supervivencia pero en realidad no ocurrió.  
- **Falsos negativos**: casos donde el modelo predijo no supervivencia pero la persona sí sobrevivió.  
👉 En este contexto, los **falsos negativos son más graves**, porque implican “no salvar” a alguien que sí podía sobrevivir.  

El modelo tiende a equivocarse más con los **no sobrevivientes**.

---

## 🚀 Reflexión y mejoras
- El **feature engineering** aportó mucho valor: especialmente `Title` y `FamilySize`.  
- A futuro, se podrían crear nuevas variables a partir de:
  - La cabina (`Cabin`) → ubicación en el barco.
  - El billete (`Ticket`) → posibles grupos de viaje.  

Esto abriría la puerta a modelos más complejos como **árboles de decisión o random forest**.

---

## 📓 Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT1/Practico2/Practico2Pizarro.ipynb)

[Ver Notebook en GitHub](UT1/Practico2/Practico2Pizarro.ipynb)
