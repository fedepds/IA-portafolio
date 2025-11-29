
# Google Cloud Platform y Vertex AI Pipelines

A continuación presento mi experiencia práctica desarrollando soluciones en la nube, desde la gestión de infraestructura base hasta la orquestación avanzada de modelos de Machine Learning (MLOps).

---

## 1. Fundamentos de Infraestructura en Google Cloud
**Laboratorio:** Google Cloud Self-Paced Labs - Introducción

### Descripción General
Este proyecto consistió en el dominio de la consola de administración de Google Cloud Platform (GCP). El objetivo principal fue comprender la jerarquía de recursos y la gestión de identidades, elementos críticos para cualquier despliegue seguro en la nube.

### Competencias Técnicas
* **Gestión de Recursos:** Administración de entidades jerárquicas, diferenciando entre *Project Name*, *Project Number* y el *Project ID* (identificador global único) para la correcta organización de los servicios.
* **IAM & Seguridad:** Implementación de roles y permisos a través de Cloud Identity and Access Management (IAM) para controlar y auditar el acceso a recursos específicos por parte de usuarios y cuentas de servicio.
* **Navegación de Servicios:** Exploración y activación de APIs clave desde la biblioteca de servicios, utilizando el menú de navegación para acceder rápidamente a funcionalidades de cómputo, almacenamiento y redes.

---

## 2. Orquestación MLOps con Vertex AI Pipelines
**Laboratorio:** Vertex AI Pipelines: Qwik Start

### Descripción General
Diseñé e implementé un flujo de trabajo de Machine Learning "End-to-End" totalmente automatizado y reproducible utilizando **Vertex AI Pipelines**. Este proyecto abordó la necesidad de tratar los flujos de ML no como scripts aislados, sino como sistemas escalables donde cada paso (procesamiento, entrenamiento, evaluación) se encapsula en contenedores independientes.

### Stack Tecnológico
* **Plataforma:** Vertex AI Workbench, Vertex AI Pipelines.
* **SDKs y Librerías:**
    * `Kubeflow Pipelines (KFP) SDK`: Para la definición y compilación del pipeline.
    * `google_cloud_pipeline_components`: Para utilizar componentes preconstruidos optimizados para servicios de Google Cloud.
    * `TensorFlow` & `AutoML`: Para el modelado predictivo y entrenamiento.

### Desarrollo del Proyecto

#### A. Configuración y Componentes Personalizados
Inicialicé el entorno en **Vertex AI Workbench**, instalando las librerías necesarias para interactuar con el motor de compilación de KFP. Como primera etapa, desarrollé componentes personalizados basados en funciones de Python (decorados con `@component`), gestionando dependencias externas (como la librería `emoji`) dentro de la imagen del contenedor base (`python:3.9`) para tareas de preprocesamiento ligero.

#### B. Pipeline de Clasificación con AutoML (Dry Beans Dataset)
La fase central del proyecto consistió en orquestar un pipeline complejo para clasificar tipos de granos utilizando el dataset *UCI Machine Learning Dry Beans*.

**Arquitectura del Pipeline:**

1.  **Ingesta de Datos (`TabularDatasetCreateOp`):** Automaticé la creación de un *TabularDataset* gestionado en Vertex AI, importando los datos directamente desde una tabla de BigQuery (`bq://aju-dev-demos.beans.beans1`).

2.  **Entrenamiento Optimizado (`AutoMLTabularTrainingJobRunOp`):** Configuré un job de entrenamiento con **AutoML**, definiendo un presupuesto de cómputo de 1000 mili-horas de nodo. El componente gestionó automáticamente la ingeniería de características, aplicando transformaciones numéricas y categóricas a columnas como `Area`, `Perimeter` y `Class` (target).

3.  **Evaluación de Modelos con Lógica Personalizada:** Desarrollé un componente específico para la evaluación que extrae las métricas del modelo entrenado (`google-cloud-aiplatform`). Este componente implementa una lógica de decisión crítica: compara métricas clave como el área bajo la curva ROC (**auROC**) contra un umbral de calidad predefinido (0.95) para determinar si el modelo es apto para producción.

#### C. Despliegue Condicional y Linaje
Implementé una estructura de control `dsl.Condition` en el pipeline. Solo si el componente de evaluación retornaba una decisión positiva ("true"), el flujo ejecutaba el componente de despliegue (`ModelDeployOp`).

* **Infraestructura de Inferencia:** El modelo se desplegó en un Endpoint de Vertex AI utilizando máquinas tipo `e2-standard-4`, quedando operativo para predicciones en línea.
* **Trazabilidad:** Utilicé la funcionalidad de **Linaje (Lineage)** de Vertex AI para rastrear el origen de los artefactos generados, auditando visualmente qué dataset específico entrenó al modelo que finalmente fue desplegado.

### Resultados
El pipeline logró automatizar el ciclo de vida completo del modelo. La evaluación automatizada generó visualizaciones detalladas, como la matriz de confusión, accesibles directamente desde la interfaz de Pipelines, validando la precisión del modelo antes de su exposición automática como API REST.
