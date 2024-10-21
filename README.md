
# Predicción de Compras y Clasificación de Imágenes con Redes Neuronales

Este repositorio contiene un proyecto dividido en dos partes principales:
1. **Predicción de compras y clasificación de clientes** usando técnicas de deep learning y embeddings.
2. **Clasificación de imágenes del dataset Fashion MNIST** con redes convolucionales (CNN) y la exploración de 5 variantes de modelos.

## Parte 1: Predicción de Compras en Black Sales

En esta parte, trabajamos con un dataset de compras realizado por diferentes clientes durante una campaña de "Black Sales". El objetivo es crear modelos que puedan predecir cuánto estará dispuesto a gastar un cliente en el futuro. El dataset contiene información detallada sobre las transacciones, como `user_id`, `product_id`, y el monto gastado por el cliente.

### a) Análisis y Preparación del Dataset
Se proporciona un enlace para descargar el dataset [aquí](https://drive.google.com/file/d/1X8_G5BpQMi-Nnbtms2RL8lcWSxzD8ixd/view?usp=sharing). El dataset se analiza y prepara para entrenar varios modelos de deep learning que puedan abordar problemas de clasificación y regresión.

### b) Modelo de Deep Learning Sin Embeddings
Se entrena un modelo de deep learning descartando las columnas de `product_id` y `user_id`. Este modelo trata de predecir el grupo de gasto de un cliente:
- **Grupo 0**: menos de 5000.
- **Grupo 1**: entre 5000 y 10000.
- **Grupo 2**: entre 10000 y 15000.
- **Grupo 3**: más de 15000.

Se grafican las evoluciones de la función de costo y la métrica de validación para evaluar el rendimiento.

### c) Modelo de Deep Learning Con Embeddings
Se entrena un modelo que utiliza dos capas de embeddings: una para los productos y otra para los usuarios. Se grafican las evoluciones de la función de costo y la métrica de validación, y se usan técnicas de regularización para mejorar los resultados.

### d) Función de Sugerencias de Usuarios Similares
Se implementa una función para el modelo del punto c), que recibe un `user_id` y sugiere **n** usuarios con comportamientos de compra similares, basándose en los embeddings de usuarios.

### e) Modelo de Regresión con Embeddings
Se cambia el enfoque a un problema de regresión, donde la salida es el monto que el cliente gasta. Se entrena un modelo con dos capas de embeddings (una para productos y otra para usuarios), y se grafican los resultados.

### f) Ajuste de Hiperparámetros
Para el mejor modelo del punto e), se seleccionan al menos dos hiperparámetros para ajustar. Se explica el método utilizado para el ajuste (como búsqueda en grid o random) y se presentan los resultados obtenidos.

## Parte 2: Clasificación de Imágenes del Dataset Fashion MNIST

En esta parte del proyecto, implementamos una **red neuronal convolucional (CNN)** para clasificar las imágenes del dataset **Fashion MNIST** en 10 clases diferentes. A partir del modelo base, exploramos 5 variantes modificando distintos hiperparámetros para evaluar su impacto en el rendimiento.

### Variaciones de Hiperparámetros
Los modelos propuestos varían en uno o más de los siguientes aspectos:
- Número de capas.
- Número de kernels.
- Tamaño de los kernels.

### Evaluación de los Modelos
Se evalúan los 5 modelos propuestos en términos de:
- **Función de pérdida** y **accuracy** para el conjunto de entrenamiento y prueba.
- **Tiempo de entrenamiento** de cada modelo.
- **Tamaño del modelo** (en parámetros).

### Resultados y Comparaciones
Los resultados obtenidos se presentan en gráficos comparativos:
- **Curva de pérdida** vs. **epochs** para los 5 modelos.
- **Curva de accuracy** vs. **epochs** para los 5 modelos.
- **Gráfico de barras** que muestra el tiempo de entrenamiento y el tamaño de cada modelo.

Finalmente, se justifica la selección del modelo que se considera el mejor, tomando en cuenta métricas como el balance entre rendimiento y complejidad computacional.

## Requisitos

Para ejecutar el código de este proyecto, asegúrate de tener instaladas las siguientes dependencias:
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `pandas`

## Instrucciones

1. Clona este repositorio:
   ```bash
   git clone https://github.com/usuario/proyecto-compras-fashionMNIST.git
   cd proyecto-compras-fashionMNIST
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta cada una de las partes del proyecto:

#### Parte 1: Predicción de Compras
   ```bash
   jupyter notebook
   ```
   Abre y ejecuta el archivo correspondiente a la Parte 1 en tu navegador.

#### Parte 2: Clasificación de Fashion MNIST
   ```bash
   python cnn_fashion_mnist.py
   ```

## Autor
Diego Araujo
