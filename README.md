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
## 1. Introducción



## 2. Predicción con Kaggle


## 3. Árbol de Decisión

El aŕbol de decisión es un tipo de algoritmo el cual es utilizado para determinar la predicción de una muestra o arreglo de datos que posea varios atributos. Dicha predicción se realiza de inmediata, con lo cual se van comparando los valores que posee una expresión lógica y se compara con los atributos originados para el árbol de decisión, con el fin de obtener un resultado.

Este modelo necesita analizar datos para determinar como será construida la estructura del árbol 

### Diseño del modelo

La elaboración del diseño del modelo conlleva los siguientes pasos:

- Definición de atributos : Se definen los atributos que poseen los datos que se ingresan como entrada al sistema.
- Definición de valores de atributos : Se comprueba los valores discretos que poseen cada uno de los atributos en los datos. En caso de que sean valores continuos, se aplicará un algorimo de clasificación y separación de datos.
- Definición de la clase de datos : Se define la clase o los resultados a los cuales se espera llegar a tener con una decisión. Estos elementos se irán eliminando conforme se vaya creando el árbol de decisión.
- Cálculo de entropía de la clase de datos : Se aplica el cálculo de la entropía de la clase de datos, el cual indicará que tan cierta es una variable.
- Cálculo de entropía de los valores de los atributos : Se aplica el mismo cálculo de la entropía, y además se logra verificar si la entropía es igual a cero, ya que así se llega a una decisión concreta del árbol.
- Cálculo de la ganancia de la información : El cálculo de la ganancia es un algoritmo en el cual se commprueba, utlizando probabilidades y entropías, el atributo que será mayor determinante para seguir una decisión.
- Comprobación del atributo con mayor ganancia : Después de calcular las ganancias de información, se escoge al primero que posea la mayor ganancia.
- Eliminación de atributos de la lista de la muestra de datos y que poseen la mayor ganancia de información : Se eliminan de la lista los atributos que poseen la mayor ganancia de la información, además que se eliminan aquellas clases de elementos en donde los valores del atributo seleccionados hayan llegado a su fin debido que poseen una entropia igual a cero.
- Insertar en el árbol atributo con mayor ganancia : Del punto anterior, se elige el aributo con la mayor ganancia, y este se inserta en el árbol de decisión, agregando además los valores de dicho atributo que permitirá llegar a un resultado definitivo o seguir agregando más atributos o preguntas de decisión.
- Definición de decisiones que salen de los atributos insertados en el árbol : Después de agregar un atributo al árbol de decisión, se definen cuales son los valores que saldrán de este nodo de árbol. En caso de ser un nodo o elemento que muestre el resultado del árbol, no se agregan más valores o ramas de valores a este nodo.
- Poda del árbol : Es el proceso en el cual se eliminan nodos hoja del árbol, con el fin de que el mismo quede con mucha menos anchura y se mejore la velocidad en la predicción del modelo. 

### Prueba del modelo


### Análisis de resultados

## 4. Red Neuronal

### Diseño del modelo


### Prueba del modelo


### Análisis de resultados


## 5. División de tareas y calificación de grupo

## 6. Conclusiones



## 7. Apéndice
---

### 7.1. Manual de Instalación

El manual de instalación es una guía sobre los ajustes y componentes requeridos para instalar los programas que definen los modelos de predicción de inteligencia artificial.

Dichos componentes se utilizan en conjunto con Python, así que se requiere tener instalado la versión 3.5.2 o superior para agregar los demás componentes

A continuación se describen los comandos requeridos para instalar los componentes, los cuales se instalan ejecutando cada comando desde una Terminal de comandos de Linux

#### 7.1.1 Instalación de pip

`sudo apt-get install python3-pip`

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


## Referencias

- https://en.wikibooks.org/wiki/A_Beginner%27s_Python_Tutorial/Importing_Modules
- https://docs.python.org/2/tutorial/classes.html
- https://www.python-course.eu/python3_modules_and_modular_programming.php
