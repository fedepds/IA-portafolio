# Clustering con K-Means (Mall Customers)

## 🎯 Descripción
En este proyecto apliqué técnicas de **aprendizaje no supervisado** para segmentación de clientes, demostrando mis habilidades en:
1. Análisis exploratorio exhaustivo de datos de comportamiento de consumidores.
2. Detección y tratamiento de outliers mediante métodos estadísticos (IQR).
3. Implementación de algoritmos de clustering (K-Means) para identificar segmentos.
4. Determinación del número óptimo de clusters mediante el método del codo.
5. Visualización y comunicación de resultados para estrategias de marketing.

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

## 📓 Notebook

[Ver Notebook Completo](UT1/Practico6/TA6Pizarro.ipynb)
