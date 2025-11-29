# Análisis Exploratorio de Datos (Titanic)

## 🎯 Descripción
En este proyecto apliqué técnicas de **análisis exploratorio de datos (EDA)** sobre el dataset **Titanic de Kaggle**, demostrando mi capacidad para:
1. Analizar la estructura de datasets complejos y comprender sus variables.  
2. Identificar y evaluar el impacto de valores faltantes en los datos.  
3. Explorar la distribución de variables objetivo y detectar patrones.  
4. Descubrir relaciones significativas entre variables predictoras y el resultado.  

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

## 🔧 Metodología Aplicada

### 1. Carga y Exploración de Datos
Realicé la carga de los archivos `train.csv` y `test.csv`, aplicando técnicas de inspección inicial:
  - Análisis dimensional con `.shape` para evaluar el volumen de datos.  
  - Exploración de columnas y tipos de datos con `.columns`, `.info()`.  
  - Análisis estadístico descriptivo con `.describe()` y `.head()`.  

### 2. Análisis de Valores Faltantes
Implementé un análisis sistemático con `.isna().sum()` para identificar columnas con datos incompletos.  
Identifiqué que `Age`, `Cabin` y `Embarked` presentaban los mayores porcentajes de valores faltantes, lo cual requeriría estrategias de imputación posteriores.  

### 3. Exploración de la Variable Objetivo
Analicé la distribución de `Survived`, identificando un desbalance de clases:  
  - **38% de supervivientes** vs **62% de fallecidos**, lo que tiene implicaciones importantes para el modelado predictivo.  

### 4. Visualización y Análisis de Patrones
Generé visualizaciones estratégicas para descubrir patrones clave:
- **Supervivencia por género**: Identifiqué que las mujeres tuvieron una probabilidad significativamente mayor de sobrevivir.  
- **Supervivencia por clase**: Los pasajeros de primera clase mostraron tasas de supervivencia superiores.  
- **Análisis etario**: Exploré la relación entre edad y supervivencia, detectando patrones relevantes.  
- **Mapa de correlaciones**: Utilicé heatmaps para visualizar relaciones entre variables numéricas (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`).  

---

## 📈 Resultados Obtenidos
- Identifiqué valores faltantes críticos en `Age` (20%) y `Cabin` (77%), lo que informó estrategias de preprocesamiento.  
- Descubrí las variables con mayor poder predictivo para la supervivencia:  
  - **Género**: Las mujeres tuvieron una tasa de supervivencia 4 veces mayor.  
  - **Clase socioeconómica**: La primera clase mostró tasas de supervivencia del 63% vs 24% en tercera clase.  
- La variable `Fare` presentó alta correlación con `Pclass`, validando su relevancia como indicador socioeconómico.  

---

## 🔍 Conclusiones y Exploraciones Futuras
- Demostré mi capacidad para realizar análisis exploratorio exhaustivo, identificando patrones y variables clave antes del modelado.  
- Este EDA estableció las bases para el desarrollo de modelos predictivos robustos.  
- Exploraciones que podrían implementarse:  
  - Implementar técnicas de imputación inteligente para `Age` y `Embarked`.  
  - Realizar feature engineering: extraer títulos (`Mr.`, `Mrs.`) desde `Name`, crear variable de tamaño familiar.  
  - Explorar interacciones entre variables para capturar relaciones no lineales.  

---

## 📓 Notebook

**[Abrir en Google Colab](https://colab.research.google.com/github/fedepds/IA-portafolio/blob/main/docs/portfolio/UT1/Practico1/Practica_1Pizarro.ipynb)**
