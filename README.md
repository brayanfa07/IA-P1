# Proyecto #1 - Predicción con Kaggle
---
## IC-6200 Inteligencia Artificial
---
Enlace de GitHub: https://github.com/brayanfa07/IA-P1


## Integrantes

- Brayan Fajardo Alvarado - 201157035
- Fabricio Castillo Alvarado - 2014062977
- Gerald Mora Mora - 2014064955

### II Semestre 2018


---
## 1. Introducción
La predicción de resultados es uno de los más grandes retos que posee la inteligencia artificial, ya que se requiere tener un modelo, de aprendizaje de datos, que logre identificar atributos como entradas y realizar un análisis, específico de cada modelo, para predecir una salida, la cual se espera sea correcta basada en los datos de entrenamiento.

En este proyecto se desarrollan 2 modelos de aprendizaje, los cuales son:
- Redes Neuronales.
- Árboles de decisión.

El objetivo primario del proyecto será enfrentar a los estudiantes con una situación cercana a un proyecto de clasificación real donde existe una fuente de datos cruda, debe procesarse los mismos, comparar algoritmos y reportar los resultados de una manera formal.
Para ello se utiliza un conjunto de datos con ejemplos de muestras de tejidos posiblemente malignos, los modelos creados deberan ser capases de predecir si un cancer de acuerdo a sus caracteristicas es maligno o benigno.


## 2. Predicción de datos. 
Los datos utilizados para formar los diferentes set de datos que utilizan los algoritmos se obtienen desde un repositorio de datos para ciencia de los datos llamado Kaggle. Estos datos corresponden a una coleccion de ejemplos los cuales almacenan informacion de cada caracteristica que presente un diagnostico (ejemplo), por ejemplo, tamaño, axis, entre otros. Todas estas caracteristicas que describen los datos seran indispensables para que cada algoritmo sea capaz de entrenarse con ellos para asi, en caso de ellegar a recibir un nuevo diagnostico o ejemplo poder predecir si el cancer del que se esta tratando es maligno o no.


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
El modelo de Red neuronal utiliza la biblioteca de Tensorflow la cual, es una biblioteca de Machine Learning desarrollada por Google, comúnmente usada en la comunidad. Las redes de neuronas artificiales (denominadas habitualmente como RNA o en inglés como: “ANN”) son un paradigma de aprendizaje y procesamiento automático inspirado en la forma en que funciona el sistema nervioso de los animales. Se trata de un sistema de interconexión de neuronas que colaboran entre sí para producir un estímulo de salida. En inteligencia artificial es frecuente referirse a ellas como redes de neuronas o redes neuronales. Forman parte de los denominados “Sistemas Inteligentes“, dentro de la rama de la Inteligencia Artificial.


### Diseño del modelo
Para la realización del modelo se modificó el archivo de datos para contar con una representación numérica de la predicción para utilizar la red neuronal. Posteriormente los datos son convertidos a un arreglo numpy, el cual es la estructura, que se debe utilizar con Keras. Estos datos son desordenados antes de ser utilizados. Se toma una cantidad de datos para realizar el entrenamiento y las pruebas de la red neuronal. Luego se especifica la estructura de los datos a utilizar por Tensorflow. A continuación se crea la red neuronal, se realiza el entrenamiento y se evalúa la precisión de la red neuronal.

### Prueba del modelo y Análisis de resultados
Para la ejecución del modelo se toman las 569 filas del archivo de datos, las cuales son distintas muestras de cáncer de mama, estos datos son tomados de las distintas características que poseen las muestras del archivo. Para la ejecución del modelo se toman en cuenta las características más importantes y que aportan más información al modelo. Una vez cargados los datos, se ejecuta el modelo para obtener las predicciones del estado de la muestra. Luego de ejecutar el modelo se obtienen varios resultados, uno de los resultados es la precisión, la cual representa que tan eficiente es la predicción para determinar la clasificación del tejido de cada muestra, en este caso si es cáncer maligno o benigno.


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
- Árbol kd. (2016, marzo 14). En Wikipedia, la enciclopedia libre. Recuperado a partir de https://es.wikipedia.org/w/index.php?title=%C3%81rbol_kd&oldid=89814916
- Brownlee, J. (2016a, mayo 24). Develop Your First Neural Network in Python With Keras Step-By-Step. Recuperado 5 de mayo de 2018, a partir de https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
- Brownlee, J. (2016b, junio 2). Multi-Class Classification Tutorial with the Keras Deep Learning Library. Recuperado 5 de mayo de 2018, a partir de https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
- Desarrolla tu Primera Red Neural en Python con Keras Paso a Paso - ® Cursos Python desde 0 a Experto trophy garantizados. (s. f.). Recuperado 5 de mayo de 2018, a partir de https://www.aprenderpython.net/desarrolla-primera-red-neural-python-keras-paso-paso/
- In-Depth: Support Vector Machines | Python Data Science Handbook. (s. f.). Recuperado 5 de mayo de 2018, a partir de https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
- kdtree-in-python: Source Code for K-d tree in Python series. (2018). Python, Tsoding. Recuperado a partir de https://github.com/tsoding/kdtree-in-python (Original work published 2017)
- Logistic regression in Tensorflow for beginner. (s. f.). Recuperado 5 de mayo de 2018, a partir de https://www.kaggle.com/niyamatalmass/logistic-regression-in-tensorflow-for-beginner
- Logistic Regression with TensorFlow. (s. f.). Recuperado 5 de mayo de 2018, a partir de http://www.serrate.net/2018/02/18/logistic-regression-with-tensorflow/index.html
- Polamuri, S. (2017, enero 25). Support vector machine (Svm classifier) implemenation in python with Scikit-learn.
Recuperado 5 de mayo de 2018, a partir de http://dataaspirant.com/2017/01/25/svm-classifier-implemenation-python-scikit-learn/
- Sequential - Keras Documentation. (s. f.). Recuperado 5 de mayo de 2018, a partir de https://keras.io/models/sequential/
- Sikonja, M. R. (1998). Speeding up Relief algorithms with k-d trees.
- TK. (2017, octubre 28). Learning Tree Data Structure. Recuperado 5 de mayo de 2018, a partir de https://medium.com/the-renaissance-developer/learning-tree-data-structure-27c6bb363051
- Understanding Support Vector Machine algorithm from examples (along with code). (2017, septiembre 13). Recuperado 5 de mayo de 2018, a partir de https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
