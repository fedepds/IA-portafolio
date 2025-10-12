# Práctico 1 - Análisis Exploratorio del Titanic

## 🎯 Objetivo
Familiarizarse con el dataset **Titanic - Kaggle** y aplicar un **análisis exploratorio de datos (EDA)** para:
1. Entender la estructura del dataset y sus variables.  
2. Detectar valores faltantes y su impacto.  
3. Analizar la distribución de la variable objetivo (`Survived`).  
4. Explorar relaciones entre supervivencia y variables como sexo, clase y edad.  

---

## 📊 Dataset
- **Fuente**: [Titanic - Kaggle](https://www.kaggle.com/c/titanic)  
- **Archivos usados**:  
  - `train.csv`: dataset de entrenamiento con la variable objetivo (`Survived`).  
  - `test.csv`: dataset de test sin la columna de supervivencia.  
- **Variables importantes**:  
  - `Survived` → objetivo (0 = no sobrevivió, 1 = sobrevivió).  
  - `Pclass` → clase del pasajero.  
  - `Sex` → sexo.  
  - `Age` → edad.  
  - `SibSp` → número de hermanos/cónyuges a bordo.  
  - `Parch` → número de padres/hijos a bordo.  
  - `Fare` → tarifa pagada.  
  - `Embarked` → puerto de embarque.  

---

## 🔧 Proceso

### 1. Carga y Exploración
- Lectura de los archivos `train.csv` y `test.csv`.  
- Inspección inicial con:
  - `.shape` → tamaño del dataset.  
  - `.columns` → nombres de las columnas.  
  - `.head()` → primeras filas.  
  - `.info()` y `.describe()` → resumen de tipos y estadísticas.  

### 2. Valores Faltantes
- Análisis con `.isna().sum()` para identificar columnas incompletas.  
- Las más afectadas fueron `Age`, `Cabin` y `Embarked`.  

### 3. Variable Objetivo
- Distribución de `Survived`:  
  - Aproximadamente el **38% sobrevivió** y el **62% no lo hizo**.  

### 4. Visualizaciones
- **Supervivencia por sexo** → las mujeres tuvieron mucha mayor probabilidad de sobrevivir.  
- **Supervivencia por clase (Pclass)** → los pasajeros de primera clase tuvieron más chances.  
- **Distribución de edad vs supervivencia** → se analizaron patrones etarios.  
- **Mapa de calor de correlaciones** → permitió observar relaciones entre variables numéricas (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`).  

---

## 📈 Resultados
- El dataset presenta **valores faltantes relevantes** en `Age` y `Cabin`.  
- Variables con mayor relación con la supervivencia:  
  - **Sexo** (ventaja para mujeres).  
  - **Clase** (mejor supervivencia en primera clase).  
- La variable `Fare` muestra diferencias entre clases, lo que también influye en la supervivencia.  

---

## 🔍 Reflexión
- El análisis exploratorio permite identificar las variables clave que influenciaron la supervivencia.  
- Este paso es fundamental antes de aplicar modelos de machine learning.  
- Futuras mejoras:  
  - Imputar inteligentemente `Age` y `Embarked`.  
  - Extraer nuevas variables (`Title` desde `Name`, tamaño de familia).  
  - Evaluar combinaciones de variables para mejorar modelos predictivos.  

---
