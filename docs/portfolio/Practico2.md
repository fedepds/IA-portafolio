# Feature Engineering y Modelo Base

## 🎯 Objetivo
Ir más allá del análisis exploratorio inicial del dataset Titanic para:
1. Manejar valores faltantes con imputación inteligente.
2. Crear nuevas variables (feature engineering) que capturen información relevante.
3. Entrenar un **modelo base** con regresión logística y compararlo con un **baseline** (DummyClassifier).

---

## 📊 Dataset
- **Fuente**: [Titanic - Kaggle](https://www.kaggle.com/c/titanic)
- **Variable Importante**: `Survived` (0 = no sobrevivió, 1 = sobrevivió).

---

## 🔧 Proceso

### 1. Preprocesamiento
- **Embarked** → valores faltantes imputados con la moda, por ser el mas común
- **Fare** → valores faltantes imputados con la mediana para evitar que valores extremos la distorsionen
- **Age** → l.                 a mediana de la edad se calculó para cada grupo de Sex y Pclass, lo que resulta en una imputación más precisa.

### 2. Feature Engineering 
- Es crear nuevas variables a partir de las existentes para mejorar el rendimiento del modelo.
- `FamilySize` = `SibSp` + `Parch` + 1  
- `IsAlone` = 1 si el pasajero viajaba solo, 0 en caso contrario 
- FamilySize y IsAlone: Se crearon para capturar el tamaño de la familia de un pasajero, combinando SibSp y Parch. La intuición es que las familias podrían haber tenido una ventaja o desventaja en la supervivencia.
- `Title` = extracción del título desde la columna `Name` (Mr., Mrs., Miss., Rare, etc.), para poder agrupar a las personas quitando el estatus social.  

### 3. Transformación
- **one-hot encoding** (`pd.get_dummies`). Para convertir las variables categóricas (Sex, Embarked, Title) en variables numéricas binarias que el modelo pueda procesar.


### 4. Modelos evaluados
- **DummyClassifier** → baseline (predice siempre la clase más frecuente).Su precisión nos dio el rendimiento mínimo que cualquier modelo útil debería superar.
- **LogisticRegression** → Este modelo resuelve problemas de clasificación.
- **train_test_split:** → Función para dividir los datos de entrenamiento y de prueba

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
