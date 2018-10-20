# Proyecto #1 - Predicción con Kaggle
---
## IC-6200 Inteligencia Artificial
---
Enlace de GitHub: https://github.com/brayanfa07/IA-P1


## Integrantes

- Brayan Fajardo Alvarado - 201157035
- Fabricio Castillo Alvarado - 2014062977
- Gerald Mora Mora - 2014



### II Semestre 2018


---
## 1. Introducción</h2>


## 2. Predicción con Kaggle


## 3. Árbol de Decisión


### Diseño del modelo


### Prueba del modelo


### Análisis de resultados

## 4. Red Neuronal

### Diseño del modelo


### Prueba del modelo


### Análisis de resultados


## 5. División de tareas y calificación de grupo

## 6. Conclusiones



## 7. Apéndice



### 7.1. Manual de Instalación

El manual de instalación es una guía sobre los ajustes y componentes requeridos para instalar los programas que definen los modelos de predicción de inteligencia artificial.

Dichos componentes se utilizan en conjunto con Python, así que se requiere tener instalado la versión 3.5.2 o superior para agregar los demás componentes

A continuación se describen los comandos requeridos para instalar los componentes, los cuales se instalan ejecutando cada comando desde una Terminal de comandos de Linux

#### 7.1.1 Instalación de Scipy

Comando a ejecutar:
`pip3 install scipy --user`

#### 7.1.2 Instalación de scikit

Comandos a ejecutar:

`sudo pip3 install numpy scipy`

`pip3 install -U scikit-learn --user`

#### 7.1.3 Instalación de matplotlib

Comandos a ejecutar:

`pip3 install matplotlib --user`

#### 7.1.4 Instalación de tensorflow

`pip3 install tensorflow --user`

#### 7.1.5 Instalación de Keras

`pip3 install keras`

### 7.2. Manual de usuario

El manual de usuario es una guía para colocar a funcionar el sistema de predicción del voto.

Para realizar la ejecución del programa de Predicción, se deberá contar con tener instalado la versión Python 3.5.2 o superior.

#### 7.2.1 Ejecución de modelos

Se describen los pasos requeridos para poner a funcionar el programa:

Ejecutar el comando `python3 g09.py`


Escribir el comando predecir --prefijo <etiqueta> --poblacion <valor> --porcentaje-pruebas <valor> --red-neuronal --red-neuronal --numero-capas <valor> --unidades-por-capa <valor> --funcion-activacion <valor> si se desea realizar una predicción utilizando la red neuronal.

Escribir el comando predecir --prefijo <etiqueta> --poblacion <valor> --porcentaje-pruebas <valor> --arbol --umbral-poda <valor> si se desea realizar una predicción utilizando el árbol de decisión.





### 7.2. Manual de Usuario




## Referencias











# IA-P1
Proyecto 1 del curso de Inteligencia Artificial


## Tareas por realizar (README TEMPORAL)

### Requerimientos de alta calidad de desarrollo de software
- Implementación de banderas
- Uso de Github
- Pruebas de Pytest y Pytest-cov
- Instanciación de clases

### Preprocesamiento de datos, normalización y codificación
- Transformación de datos de entrada
- Normalizar datos con "z-score" y "standard score"
- "One-hot encoding" para normalizar

### Módulo de entrenamiento y predicción con Random Forest
- Implementación de árboles de decisión
  - Usar random forest wiki
  - Crear archivos de rendimiento con estadísticas
  - Paso de poda
  
### Módulo de entrenamiento y predicción con redes neuronales
- Implementación de red neuronal
  - Utilización de tensor-flow y keras
  - Herencia para pruebas
  - Uso de mocking
  - Análisis de resultados
  
### Módulo de cross-validation y evaluación genérico
- Cross-validation
- Generación de archivos de resultados

### Programa de consola primario que llama a cada algoritmo
- Implementación de main

### Puntos extra
- Regresión lineal
  - Análisis de resultados L1 y L2

### Requerimientos básicos y operativos del programa principal
- Uso de prefijos en archivos generados
- Generación de CSV

### Informe
- Creación de informe



### REFERENCIAS BIBLIOGRÁFICAS

- https://en.wikibooks.org/wiki/A_Beginner%27s_Python_Tutorial/Importing_Modules
- https://docs.python.org/2/tutorial/classes.html
- https://www.python-course.eu/python3_modules_and_modular_programming.php
