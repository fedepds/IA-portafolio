# Práctico 6 - Mall Customer Segmentation

## 🎯 Objetivo
Realizar un **análisis exploratorio y segmentación de clientes** a partir del dataset *Mall Customers* con el fin de:
1. Comprender las características principales de los clientes (edad, ingresos, puntaje de gasto).
2. Detectar outliers y analizar distribuciones.
3. Aplicar **clustering (K-Means)** para agrupar clientes en segmentos con patrones de consumo diferenciados.

---

## 📊 Dataset
- **Fuente**: [Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/shwetabh123/mall-customers)  
- **Columnas principales**:  
  - `CustomerID` → identificador único del cliente.  
  - `Genre` → género del cliente.  
  - `Age` → edad.  
  - `Annual Income (k$)` → ingreso anual (en miles de dólares).  
  - `Spending Score (1-100)` → puntaje de gasto asignado por el shopping.  

---

## 🔧 Proceso

### 1. Carga y exploración inicial
- Importación de datos desde **GitHub** para asegurar reproducibilidad.  
- Inspección de dimensiones, tipos de datos y primeras filas.  
- Resumen de uso de memoria y verificación de valores faltantes.

### 2. Análisis Exploratorio
- Estadísticas descriptivas de las variables numéricas.  
- Distribución por género (conteo y porcentajes).  
- Análisis de **rangos y promedios** en edad, ingreso e índice de gasto.  
- Detección de **outliers con IQR** en las variables numéricas.  

### 3. Visualización
- Histogramas de las variables principales (`Age`, `Annual Income`, `Spending Score`).  
- Gráficos de dispersión para observar correlaciones entre variables.  
- Pairplots para explorar relaciones multivariadas.

### 4. Clustering (K-Means)
- Selección de variables de segmentación (`Age`, `Annual Income`, `Spending Score`).  
- Aplicación de **método del codo (Elbow Method)** para determinar el número óptimo de clusters.  
- Entrenamiento del modelo K-Means y asignación de etiquetas de cluster.  
- Visualización de los grupos formados en 2D (por ejemplo, Ingreso vs Spending Score).  

---

## 📈 Resultados
- Se identificaron **segmentos de clientes** con características distintas en función de ingresos y puntaje de gasto.  
- Ejemplo típico de clusters encontrados:  
  - Clientes de **alto ingreso pero bajo gasto**.  
  - Clientes de **bajo ingreso y bajo gasto**.  
  - Clientes de **alto gasto con ingresos medios** (potenciales clientes premium).  
- El clustering permitió detectar perfiles útiles para diseñar **estrategias de marketing personalizadas**.  

---

## 🔍 Reflexión
- El dataset es pequeño (200 clientes), pero suficiente para ilustrar el poder de la segmentación.  
- El análisis de outliers confirmó que algunos clientes presentan valores extremos en edad e ingresos, aunque no afectan gravemente la segmentación.  
- Posibles mejoras:  
  - Usar más variables de comportamiento (por ejemplo, frecuencia de visitas, compras reales).  
  - Comparar con otros algoritmos de clustering (DBSCAN, jerárquico).  
  - Aplicar reducción de dimensionalidad (PCA) para visualizar mejor los grupos.  

---
