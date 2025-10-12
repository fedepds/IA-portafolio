# 📚 Práctica 8 – Backpropagation y Optimizadores
En esta práctica exploramos **MLPs aplicados a datasets de imágenes** (MNIST, Fashion-MNIST, CIFAR-10 y CIFAR-100), evaluando cómo distintos hiperparámetros y técnicas afectan el rendimiento.

---

## 🎯 Objetivos
- Aplicar **backpropagation** en redes densas (MLP).
- Explorar **arquitecturas** (profundidad y ancho).
- Comparar **funciones de activación**: ReLU, GELU, tanh.
- Evaluar **regularización**: Dropout, L2, BatchNormalization.
- Probar **inicializadores de pesos**: HeNormal, GlorotUniform.
- Probar distintos **optimizadores**: Adam, SGD, RMSprop, AdamW.
- Usar **callbacks** (EarlyStopping, ReduceLROnPlateau, TensorBoard).
- Analizar resultados y proponer conclusiones.

---

## 🔬 Metodología
1. **Preprocesamiento**:  
   - Normalización de imágenes a rango [-1,1].  
   - Aplanado de 28×28 y 32×32×3 → vectores para MLP.  
   - Split: train, validación (10%), test.  

2. **Arquitecturas probadas**:  
   - 1–3 capas, 32–256 neuronas.  
   - Ejemplos: `[64]`, `[128, 64]`, `[256, 128, 64]`.  

3. **Condiciones controladas**:  
   - Se modificó **un hiperparámetro a la vez**.  
   - Cada entrenamiento: 5–10 épocas, batch size en {32, 64}.  

---

## 📊 Resultados principales

### 🔹 Arquitecturas
- **Redes muy pequeñas** (`[32]`) → baja capacidad, accuracy < 40% en CIFAR-10.  
- **Más capas/neuronas** (ej: `[128,64]`) → mejor rendimiento en validación.  
- A partir de cierto punto, aumentar neuronas generó **sobreajuste**.  

### 🔹 Activaciones
- **ReLU** → más estable y rápida, mejor accuracy en general.  
- **GELU** → resultados similares a ReLU pero más suaves.  
- **tanh** → se quedó atrás en datasets grandes (CIFAR-100).  

### 🔹 Regularización
- **Dropout=0.2** → redujo sobreajuste y mejoró generalización.  
- **BatchNormalization** → ayudó a estabilizar el entrenamiento.  
- **L2 regularization (1e-4)** → útil, pero demasiado fuerte puede frenar el aprendizaje.  

### 🔹 Inicializadores
- **HeNormal** → más adecuado con ReLU, mejor convergencia.  
- **GlorotUniform** → buen rendimiento en general.  

### 🔹 Optimizadores
- **Adam (lr=1e-3)** → buen balance entre rapidez y precisión.  
- **SGD con momentum** → más lento pero estable, sensible al LR.  
- **RMSprop** → funcionó bien en datasets más complejos.  
- **AdamW** → útil cuando se combina con decay (weight decay=1e-4).  

### 🔹 Callbacks
- **EarlyStopping** → evitó entrenamientos innecesarios.  
- **ReduceLROnPlateau** → permitió recuperar accuracy en casos con LR muy alto.  
- **ModelCheckpoint** → guardó siempre el mejor modelo.  

---

## 📝 Conclusiones
- La **arquitectura importa**, pero agregar más neuronas no siempre mejora: llega un punto de sobreajuste.  
- **ReLU + HeNormal + Adam** fue la combinación más robusta.  
- La **regularización ligera** (Dropout 0.2 + BatchNorm) ayudó a mejorar la generalización.  
- En datasets simples (MNIST, Fashion-MNIST), incluso redes pequeñas logran >90% accuracy.  
- En datasets complejos (CIFAR-100), un MLP se queda corto → se justifica pasar a **redes convolucionales (CNNs)**.  

---

## 📌 Reflexión personal
Este ejercicio me permitió entender:
- Cómo cada **hiperparámetro** afecta al entrenamiento.  
- La importancia de **experimentar de manera sistemática** (un cambio a la vez).  
- Que el **MLP es limitado para visión**, pero sirve como base para comprender *backpropagation, optimizadores y regularización*.  

Próximo paso: aplicar las mismas técnicas con **CNNs** para mejorar la performance en datasets de imágenes más complejos.