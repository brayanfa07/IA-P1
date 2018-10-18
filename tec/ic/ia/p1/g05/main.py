print("Creando poblacion de datos inicial...")
import cmd # Para la linea de comandos
# Para encontrar la cantidad de veces que un elemento esta en una lista
import collections
import math
from statistics import mean # Para encontrar el promedio de una lista
import csv # Para manejar archivos csv
import copy # Para hacer deepcopy de variables
import numpy as np # Para crear listas usadas en SVM y regresion lineal
from sklearn import svm # Para crear el modelo de SVM
# Para codificar vectores de manera binaria
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf # Para crear el modelo de regresion lineal
from numpy import argmax # Para decodificar vectores
from keras.models import Sequential
from keras.layers import Dense
#numpy.random.seed(7)
# Para generar la muestra de datos
#from tec.ic.ia.pc1.g09 import generar_muestra_pais, generar_muestra_provincia
from manage_file import read_file, delete_column_datalist, normalize_list
from decisiontree import Decision_tree_model, Decision_tree

# Variables globales generales
prefijo = "" # Prefijo para el archivo de salida
poblacion = 100 # Tamanno de la muestra a generar
# Division de poblacion para pruebas y para entrenamiento
porcentaje_pruebas = 20
modelo = "" # Seleccion de modelo
# knn
k = 0 # Cantidad de vecinos cercanos a considerar
# svm
c = 0 # Parámetro de penalización del término de error.
gamma = 0 # Coeficiente de Kernel para rbf, poly y sigmoid.
kernel = "" # Kernel a utilizar
# Regresion lineal
l1 = 0 # Coeficiente de regularizacion l1
l2 = 0 # Coeficiente de regularizacion l2
# Red neuronal
numero_capas = 0 # Cantidad de capas de la red neuronal
unidades_por_capa = 0 # Cantidad de unidades por cada de la red neuronal
funcion_activacion = "" # Funcion de activacion de la red neuronal
# Arbol de decision
umbral_poda = 0 # Cantidad minima de ganancia para podar un nodo
filename = "cancer.csv"

"""
Entrada: lista es un arreglo con los datos a guardar.
Restriccion: La lista de entrada no debe estar vacia.
Genera un archivo csv con los datos de la lista de entrada.
"""
def generar_csv(lista):
    myFile = open(prefijo + '_datos.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(lista)

"""
Se ejecuta despues de ingresar un comando, comprueba que tipo de modelo es.
Crea una muestra, entrena el modelo seleccionado y luego prueba el modelo
con datos de prueba, al final imprime los resultados de error y llama a la
funcion que guarda los datos en el csv.
"""
def ejecutar():
    #read the data from the 
    data = read_file(filename)
    data = delete_column_datalist(data)
    data = normalize_list(data)
    xdata = copy.deepcopy(data)
    div = int(porcentaje_pruebas * poblacion / 100)
    lista_r1, lista_r2, lista_r21, j = [], [], [], 0

    if (modelo=="rl"):
        lista_modelos_r21, lista_modelos_r2, lista_modelos_r1 = [], [], []
        # 5-Cross validation
        while(j < 5):
            val = ((poblacion - div) / 5)
            lr1 = LR(data[int(val * (j + 1)):] + data[:int(val * j)], "r1", l1, l2)
            lr2 = LR(data[int(val * (j + 1)):] + data[:int(val * j)], "r2", l1, l2)
            lr21 = LR(data[int(val * (j + 1)):] + data[:int(val * j)], "r21", l1, l2)
            test = data[int(val * j):int(val * (j + 1))]
            i, correcto_r21, correcto_r2, correcto_r1, n = 0, 0, 0, 0, len(test)
            while(i < n):
                dr1 = lr1.test([test[i]])
                dr2 = lr2.test([test[i]])
                dr21 = lr21.test([test[i]])
                xdata[int(val * j) + i] += ["Si", dr1, dr2, dr21]
                correcto_r1 += 1 if dr1 == test[i][22] else 0
                correcto_r2 += 1 if dr2 == test[i][23] else 0
                correcto_r21 += 1 if dr21 == test[i][23] else 0
                i += 1
            lista_r1.append(correcto_r1)
            lista_r2.append(correcto_r2)
            lista_r21.append(correcto_r21)
            lista_modelos_r1.append(lr1)
            lista_modelos_r2.append(lr2)
            lista_modelos_r21.append(lr21)
            j += 1
        # Pruebas con el modelo entrenado
        i_r1 = lista_r1.index(max(lista_r1))
        i_r2 = lista_r2.index(max(lista_r2))
        i_r21 = lista_r21.index(max(lista_r21))
        promedio_r1 = mean(lista_r1)
        promedio_r2 = mean(lista_r2)
        promedio_r21 = mean(lista_r21)
        val = poblacion - div
        test = data[val:]
        i, correcto_r21, correcto_r2, correcto_r1, n = 0, 0, 0, 0, len(test)
        while(i < n):
            dr1 = lista_modelos_r1[i_r1].test([test[i]])
            dr2 = lista_modelos_r2[i_r2].test([test[i]])
            dr21 = lista_modelos_r21[i_r21].test([test[i]])
            xdata[val + i] += ["No", dr1, dr2, dr21]
            correcto_r1 += 1 if dr1 == test[i][22] else 0
            correcto_r2 += 1 if dr2 == test[i][23] else 0
            correcto_r21 += 1 if dr21 == test[i][23] else 0
            i += 1
    elif(modelo=="rn"):
        lista_modelos_r21, lista_modelos_r2, lista_modelos_r1 = [], [], []
        # 5-Cross validation
        while(j < 5):
            val = ((poblacion - div) / 5)
            #lr1 = NeuralNet(data[int(val * (j + 1)):] + data[:int(val * j)], "r1", l1, l2)
            lr2 = NeuralNet(data[int(val * (j + 1)):] + data[:int(val * j)], numero_capas,unidades_por_capa,funcion_activacion)
            lr21 = NeuralNet(data[int(val * (j + 1)):] + data[:int(val * j)], numero_capas,unidades_por_capa,funcion_activacion)
            test = data[int(val * j):int(val * (j + 1))]
            i, correcto_r21, correcto_r2, correcto_r1, n = 0, 0, 0, 0, len(test)
            while(i < n):
                #dr1 = lr1.test([test[i]])
                dr2 = lr2.testR2([test[i]])
                dr21 = lr21.testR2_R1([test[i]])
                #xdata[int(val * j) + i] += ["Si", dr1, dr2, dr21]
                xdata[int(val * j) + i] += ["Si", dr2, dr21]
                #correcto_r1 += 1 if dr1 == test[i][22] else 0
                correcto_r2 += 1 if dr2 == test[i][23] else 0
                correcto_r21 += 1 if dr21 == test[i][23] else 0
                i += 1
            #lista_r1.append(correcto_r1)
            lista_r2.append(correcto_r2)
            lista_r21.append(correcto_r21)
            #lista_modelos_r1.append(lr1)
            lista_modelos_r2.append(lr2)
            lista_modelos_r21.append(lr21)
            j += 1
        # Pruebas con el modelo entrenado
        #i_r1 = lista_r1.index(max(lista_r1))
        i_r2 = lista_r2.index(max(lista_r2))
        i_r21 = lista_r21.index(max(lista_r21))
        #promedio_r1 = mean(lista_r1)
        promedio_r2 = mean(lista_r2)
        promedio_r21 = mean(lista_r21)
        val = poblacion - div
        test = data[val:]
        i, correcto_r21, correcto_r2, correcto_r1, n = 0, 0, 0, 0, len(test)
        while(i < n):
            #dr1 = lista_modelos_r1[i_r1].test([test[i]])
            dr2 = lista_modelos_r2[i_r2].testR2([test[i]])
            dr21 = lista_modelos_r21[i_r21].testR2_R1([test[i]])
            #xdata[val + i] += ["No", dr1, dr2, dr21]
            xdata[val + i] += ["No", dr2, dr21]
            #correcto_r1 += 1 if dr1 == test[i][22] else 0
            correcto_r2 += 1 if dr2 == test[i][23] else 0
            correcto_r21 += 1 if dr21 == test[i][23] else 0
            correcto_r1 = 0
            promedio_r1 = 0
            i += 1














    elif(modelo=="ad"):
        lista_modelos_r21, lista_modelos_r2, lista_modelos_r1 = [], [], []
        # 5-Cross validation
        while(j < 5):
            val = ((poblacion - div) / 5)
            x1 = copy.deepcopy(data[int(val * (j + 1)):] + data[:int(val * j)])
            x2 = copy.deepcopy(data[int(val * (j + 1)):] + data[:int(val * j)])
            x3 = copy.deepcopy(data[int(val * (j + 1)):] + data[:int(val * j)])
            lr1 = Decision_tree_model(x1, "r1", umbral_poda)
            lr2 = Decision_tree_model(x2, "r2", umbral_poda)
            lr21 = Decision_tree_model(x3, "r21", umbral_poda)
            test = data[int(val * j):int(val * (j + 1))]
            x1 = copy.deepcopy(test)
            x2 = copy.deepcopy(test)
            x3 = copy.deepcopy(test)
            i, correcto_r21, correcto_r2, correcto_r1, n = 0, 0, 0, 0, len(test)
            while(i < n):
                dr1 = lr1.test(x1[i])
                dr2 = lr2.test(x2[i])
                dr21 = lr21.test(x3[i])
                xdata[int(val * j) + i] += ["Si", dr1, dr2, dr21]
                correcto_r1 += 1 if dr1 == test[i][22] else 0
                correcto_r2 += 1 if dr2 == test[i][23] else 0
                correcto_r21 += 1 if dr21 == test[i][23] else 0
                i += 1
            lista_r1.append(correcto_r1)
            lista_r2.append(correcto_r2)
            lista_r21.append(correcto_r21)
            lista_modelos_r1.append(lr1)
            lista_modelos_r2.append(lr2)
            lista_modelos_r21.append(lr21)
            j += 1
        # Pruebas con el modelo entrenado
        i_r1 = lista_r1.index(max(lista_r1))
        i_r2 = lista_r2.index(max(lista_r2))
        i_r21 = lista_r21.index(max(lista_r21))
        promedio_r1 = mean(lista_r1)
        promedio_r2 = mean(lista_r2)
        promedio_r21 = mean(lista_r21)
        val = poblacion - div
        test = data[val:]
        x1 = copy.deepcopy(test)
        x2 = copy.deepcopy(test)
        x3 = copy.deepcopy(test)
        i, correcto_r21, correcto_r2, correcto_r1, n = 0, 0, 0, 0, len(test)
        while(i < n):
            dr1 = lista_modelos_r1[i_r1].test(x1[i])
            dr2 = lista_modelos_r2[i_r2].test(x2[i])
            dr21 = lista_modelos_r21[i_r21].test(x3[i])
            xdata[val + i] += ["No", dr1, dr2, dr21]
            correcto_r1 += 1 if dr1 == test[i][22] else 0
            correcto_r2 += 1 if dr2 == test[i][23] else 0
            correcto_r21 += 1 if dr21 == test[i][23] else 0
            i += 1

    print("%Error entrenamiento r1", 100 - promedio_r1 * 100 / ((poblacion - div) / 5))
    print("%Error entrenamiento r2", 100 - promedio_r2 * 100 / ((poblacion - div) / 5))
    print("%Error entrenamiento r21", 100 - promedio_r21 * 100 / ((poblacion - div) / 5))
    print("%Error prueba r1", 100 - correcto_r1 * 100 / n)
    print("%Error prueba r2", 100 - correcto_r2 * 100 / n)
    print("%Error prueba r21", 100 - correcto_r21 * 100 / n)
    generar_csv(xdata)














    

"""
Clase que crea una linea de comandos para poder ejecutar los modelos
"""
class Comandos(cmd.Cmd):
    prompt = "Introduzca un comando: "

    def do_predecir(self, args):
        argumentos = args.split()
        try:
            global prefijo
            prefijo = argumentos[argumentos.index("--prefijo")+1]
            if("--poblacion" in argumentos):
                global poblacion
                poblacion = int(argumentos[argumentos.index("--poblacion")+1])
            if("--porcentaje-pruebas" in argumentos):
                global porcentaje_pruebas
                porcentaje_pruebas = int(argumentos[argumentos.index("--porcentaje-pruebas")+1])
            global modelo
            if("--arbol" in argumentos):
                modelo = "ad"
                global umbral_poda
                umbral_poda = int(argumentos[argumentos.index("--umbral-poda")+1])
            elif("--red-neuronal" in argumentos):
                modelo = "rn"
                global numero_capas
                numero_capas = int(argumentos[argumentos.index("--numero-capas")+1])
                global unidades_por_capa
                unidades_por_capa = int(argumentos[argumentos.index("--unidades-por-capa")+1])
                global funcion_activacion
                funcion_activacion = argumentos[argumentos.index("--funcion-activacion")+1]
            elif("--regresion-logistica" in argumentos):
                modelo = "rl"
                global l1
                l1 = float(argumentos[argumentos.index("--l1")+1])
                global l2
                l2 = float(argumentos[argumentos.index("--l2")+1])

        except ValueError:
            print("Error en los argumentos")
        ejecutar()

    def do_salir(self, args):
        return(True)

    def default(self, args):
        print("Error. Comando no reconocido:", args)

    def emptyline(self):
        pass

"""
Clase del modelo regresion logistica hecha con la biblioteca tensorflow.
"""
class LR:
    data = [] # Datos convertidos a numeros
    npData = [] # Datos numericos sin las salidas
    y = [] # Datos numericos que son las salidas
    r = "" # Tipo de prediccion a realizar
    x = 0 # Cantidad de atributos de entrada
    w = 0 # Cantidad de atributos de salidas
    oneHotX = OneHotEncoder() # Codificadores de los datos de entrada
    oneHoty = OneHotEncoder() # Codificadores de los datos de salida
    X = None # Marcador de los datos de entrada
    Y = None # Marcador de los datos de salida
    l1 = 0 # Parametro de regularizacion L1
    l2 = 0 # Parametro de regularizacion L2
    y_ = None # Funcion lineal
    sess = None # Sesion donde se ejecutan los tensores
    x1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [2, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [5, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [6, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [4, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [5, 19, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [2, 23, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [3, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [5, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [6, 27, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 29, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [4, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [5, 33, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 35, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [2, 37, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [3, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 39, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [5, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [6, 41, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 43, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 45, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [4, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [5, 47, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 49, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [2, 51, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [3, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 53, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [5, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [6, 55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 57, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 59, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [4, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [5, 61, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 63, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [2, 65, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [3, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 67, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [5, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [6, 69, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 71, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [3, 73, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [4, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [5, 75, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 77, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [2, 79, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [3, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    x2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                 [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
                 [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                 [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
                 [6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
                 [0, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7],
                 [1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                 [2, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
                 [3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                 [4, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11],
                 [5, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12],
                 [6, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13],
                 [0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14],
                 [1, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [2, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [3, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [4, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                 [5, 19, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
                 [6, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                 [0, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
                 [1, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
                 [2, 23, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7],
                 [3, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                 [4, 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
                 [5, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                 [6, 27, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11],
                 [0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12],
                 [1, 29, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13],
                 [2, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14],
                 [3, 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [4, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [5, 33, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [6, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                 [0, 35, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
                 [1, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                 [2, 37, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
                 [3, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
                 [4, 39, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7],
                 [5, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                 [6, 41, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
                 [0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                 [1, 43, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11],
                 [2, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12],
                 [3, 45, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13],
                 [4, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14],
                 [5, 47, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [6, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 49, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                 [2, 51, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
                 [3, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                 [4, 53, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
                 [5, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
                 [6, 55, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7],
                 [0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                 [1, 57, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
                 [2, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                 [3, 59, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11],
                 [4, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12],
                 [5, 61, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13],
                 [6, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14],
                 [0, 63, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [1, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2, 65, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [3, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                 [4, 67, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
                 [5, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                 [6, 69, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
                 [0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
                 [1, 71, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7],
                 [2, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                 [3, 73, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
                 [4, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                 [5, 75, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11],
                 [6, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12],
                 [0, 77, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13],
                 [1, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14],
                 [2, 79, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                 [3, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # Constructor
    def __init__(self, data, r_, l1, l2):
        self.r = r_
        self.npData = []
        self.y = []
        self.l1 = l1
        self.l2 = l2
        self.data = self.convertData(data)
        self.prepare_training_data()
        self.train()

    # Prueba los datos
    def test(self, data):
        tdata = self.convertData(data)
        xdata = []
        n = len(tdata[0])
        ydata = []
        if(self.r == "r1"):
            xdata = [tdata[0][:n - 2]]
            ydata.append([tdata[0][n - 2]])
        elif(self.r == "r2"):
            xdata = [tdata[0][:n - 2]]
            ydata.append([tdata[0][n - 1]])
        else:
            xdata = [tdata[0][:n - 1]]
            ydata.append([tdata[0][n - 1]])
        x = self.x1
        if(self.r=="r21"):
            x = self.x2
        y = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]]
        self.oneHotX.fit(x)
        self.oneHoty.fit(y)
        xdata = self.oneHotX.transform(xdata).toarray()
        ydata = self.oneHoty.transform(ydata).toarray()
        feed_dict = {self.X: xdata}
        classification = self.y_.eval(feed_dict, session=self.sess)
        i = 0
        n = len(classification[0])
        index = list(classification[0]).index(max(classification[0]))
        classification[0][index] = 1
        while(i < n):
            if(classification[0][i] != 1):
                classification[0][i] = 0
            i += 1
        inverted = argmax(classification[0])
        return convert_party(inverted)

    # Entrenamiento del modelo con los parametros y los datos de ejemplo
    def train(self):
        learning_rate = 0.0001
        num_epochs = 1500
        display_step = 1
        with tf.name_scope("Declaring_placeholder"):
            self.X = tf.placeholder(tf.float32, [None, self.w])
            self.Y = tf.placeholder(tf.float32, [None, self.x])
        with tf.name_scope("Declaring_variables"):
            W = tf.Variable(tf.zeros([self.w, self.x]))
            b = tf.Variable(tf.zeros([self.x]))
        with tf.name_scope("Declaring_functions"):
            self.y_ = tf.nn.softmax(tf.add(tf.matmul(self.X, W), b))
        with tf.name_scope("calculating_cost"):
            cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.y_)
        with tf.name_scope("declaring_gradient_descent"):
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate,
                                                                  l1_regularization_strength=self.l1,
                                                                  l2_regularization_strength=self.l2).minimize(cost)
        self.sess = tf.Session()
        self.sess.as_default()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            cost_in_each_epoch = 0
            _, c = self.sess.run([optimizer, cost], feed_dict = {self.X: self.npData, self.Y: self.y})
            cost_in_each_epoch += c

    # Convierte los datos de texto a numeros
    def convertData(self, data):
        xdata = []
        i = 0
        n = len(data)
        while(i < n):
            tmp = []
            if(data[i][0] == "SAN JOSE"):
                tmp.append(0)
                tmp.append(find_canton_SJ(data[i][1]))
            elif(data[i][0] == "ALAJUELA"):
                tmp.append(1)
                tmp.append(find_canton_AL(data[i][1]))
            elif(data[i][0] == "HEREDIA"):
                tmp.append(2)
                tmp.append(find_canton_HE(data[i][1]))
            elif(data[i][0] == "CARTAGO"):
                tmp.append(3)
                tmp.append(find_canton_CA(data[i][1]))
            elif(data[i][0] == "GUANACASTE"):
                tmp.append(4)
                tmp.append(find_canton_GU(data[i][1]))
            elif(data[i][0] == "PUNTARENAS"):
                tmp.append(5)
                tmp.append(find_canton_PU(data[i][1]))
            elif(data[i][0] == "LIMON"):
                tmp.append(6)
                tmp.append(find_canton_LI(data[i][1]))
            tmp.append(find_location(data[i][5]))
            tmp.append(find_sex(data[i][6]))
            tmp.append(find_house_state(data[i][10]))
            tmp.append(find_overcrowding(data[i][11]))
            tmp.append(find_literacy(data[i][12]))
            tmp.append(find_education(data[i][14]))
            tmp.append(find_work(data[i][15]))
            tmp.append(find_insurance(data[i][16]))
            tmp.append(find_foreign(data[i][17]))
            tmp.append(find_disabled(data[i][18]))
            tmp.append(find_insured(data[i][19]))
            tmp.append(find_female_head(data[i][20]))
            tmp.append(find_shared_head(data[i][21]))
            tmp.append(find_party(data[i][22]))
            tmp.append(find_party(data[i][23]))
            xdata.append(tmp)
            i += 1
        return xdata

    # Prepara los datos para el entrenamiento dividiendolos entre
    # entradas y salidas.
    def prepare_training_data(self):
        n = len(self.data[0])
        if(self.r == "r1"):
            for x in self.data:
                self.npData.append(x[:n - 2])
                self.y.append([x[n - 2]])
        elif(self.r == "r2"):
            for x in self.data:
                self.npData.append(x[:n - 2])
                self.y.append([x[n - 1]])
        else:
            for x in self.data:
                self.npData.append(x[:n - 1])
                self.y.append([x[n - 1]])

        x = self.x1
        if(self.r == "r21"):
            x = self.x2
        y = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]]
        self.oneHotX.fit(x)
        self.oneHoty.fit(y)
        self.npData = self.oneHotX.transform(self.npData).toarray()
        self.w = len(self.npData[0])
        self.y = self.oneHoty.transform(self.y).toarray()
        self.x = len(self.y[0])





class NeuralNet:
    data = []
    dataCompleta = [] 
    #percentageTesting=0
    dataTrainigR1=[]
    dataTrainigR2=[]
    dataTrainigR2_R1=[]
    #dataTesting=[]
    
    #--------------R2-R1
    X_R2_R1=np.array([])
    Y_R2_R1=np.array([])

    # create model  Neural Net =NN
    model_R2_R1 = Sequential()
    rounded_R2_R1=[]

    #--------------R2
    X_R2=np.array([])
    Y_R2=np.array([])

    # create model  Neural Net =NN
    model_R2 = Sequential()
    rounded_R2=[]

    #--------------R1
    X_R1=np.array([])
    Y_R1=np.array([])

    # create model  Neural Net =NN
    model_R1 = Sequential()
    rounded_R1=[]

    numeroCapas=0
    unidadesCapa=0
    funcionActivacion=''

    def __init__(self, data,numeroCapas, unidadesCapa,funcionActivacion):                
        self.data = self.convertData(data)
        self.dataCompleta=np.array(self.data)
        self.numeroCapas=numeroCapas
        self.unidadesCapa=unidadesCapa
        self.funcionActivacion=funcionActivacion
        self.trainNN()
        
        """listCutIndex=(len(self.data)*self.percentageTesting)//100
        self.dataTrainig=np.array(self.data[listCutIndex:])
        self.dataTesting=np.array(self.data[:listCutIndex])
          n=2#len(self.dataCompleta)

        self.dataTrainigR1== [self.data[:n - 2]]
        self.dataTrainigR2== [self.data[:n - 2]]
        self.dataTrainigR2_R1== [self.data[:n - 1]]

        print("R1",self.dataTrainigR1)
        print("R2",self.dataTrainigR2)
        print("R2-R1",self.dataTrainigR2_R1)"""

        #Aca se generarán las 3 tipo de predicciones.

    def trainNN(self):      
        #RED para R2_R1
        self.X_R2_R1 = self.dataCompleta[:,0:23] ##No toma en cuenta partido de segunda ronda
        self.Y_R2_R1 = self.dataCompleta[:,23]  

        #(#neuronas, funcion de activacion ,)
        #Dense=capas conectadas completamente
        self.model_R2_R1.add(Dense(9, input_dim=23, activation='relu'))
        for i in range(self.numeroCapas):  #
            self.model_R2_R1.add(Dense(self.unidadesCapa, activation='relu'))
        self.model_R2_R1.add(Dense(1, activation='sigmoid'))

        # Compile model
        self.model_R2_R1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        self.model_R2_R1.fit(self.X_R2_R1, self.Y_R2_R1, epochs=150, batch_size=10, verbose=0)

        # evaluate the model
        #scores = self.model_R2_R1.evaluate(self.X_R2_R1, self.Y_R2_R1,)
        
        #--------------------------------------------
        #RED para R2
        self.X_R2 = self.dataCompleta[:,0:22] ##No toma en cuenta partido de primera ronda
        self.Y_R2 = self.dataCompleta[:,22]
        
        #(#neuronas, funcion de activacion ,)
        #Dense=capas conectadas completamente
        self.model_R2.add(Dense(9, input_dim=22, activation='relu'))
        for i in range(self.numeroCapas):  #
            self.model_R2.add(Dense(self.unidadesCapa, activation='relu'))
        self.model_R2.add(Dense(1, activation='sigmoid'))

        # Compile model
        self.model_R2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        self.model_R2.fit(self.X_R2, self.Y_R2, epochs=150, batch_size=10, verbose=0)

        # evaluate the model
        #scores = self.model_R2.evaluate(self.X_R2, self.Y_R2,)

        """
        #--------------------------------------------
        #RED para R1
        self.X_R1 = self.dataCompleta[:,0:21] ##No toma en cuenta partido de primera ronda ni partido de segunda ronda
        self.Y_R1 = self.dataCompleta[:,21]  

        #(#neuronas, funcion de activacion ,)
        #Dense=capas conectadas completamente
        self.model_R1.add(Dense(9, input_dim=21, activation='relu'))
        for i in range(self.numeroCapas):  #
            self.model_R1.add(Dense(self.unidadesCapa, activation='relu'))
        self.model_R1.add(Dense(1, activation='softmax'))

        # Compile model
        #self.model_R1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model_R1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        # Fit the model
        self.model_R1.fit(self.X_R1, self.Y_R1, epochs=150, batch_size=10)

        # evaluate the model
        scores = self.model_R1.evaluate(self.X_R1, self.Y_R1,)
        print("\n%s: %.2f%%" % (self.model_R1.metrics_names[1], scores[1]*100)) #Imprime basado en las metricas que puse en model.compile()
        """

    def testR2_R1(self,dataForPrediction): ##R2_R1
        #Filtrar para agarrar solo voto primera ronda
        for persona in range(len(dataForPrediction)):
            dataForPrediction[persona]=dataForPrediction[persona][:23]

        tempDataNum=self.convertData(dataForPrediction)
        aux=np.array(tempDataNum)  ##Tiene 20X elementos predice el 20X+1

        predictions = self.model_R2_R1.predict(aux)
        # round predictions
        rounded = [round(x[0]) for x in predictions]
        tmp = rounded
        return convert_party(tmp[0]) #Imprime basado en la funcion de activacion sigmoide la cual solo devuelve 1 o 0

    def testR2(self,dataForPrediction): #R2
        #Filtrar para NO agarrar voto primera ronda
        for persona in range(len(dataForPrediction)):
            dataForPrediction[persona]=dataForPrediction[persona][:22]

        tempDataNum=self.convertData(dataForPrediction)
        aux=np.array(tempDataNum)  ##Tiene 20X elementos predice el 20X+1

        predictions = self.model_R2.predict(aux)
        # round predictions
        rounded = [round(x[0]) for x in predictions]
        tmp = rounded
        return convert_party(tmp[0]) #Imprime basado en la funcion de activacion sigmoide la cual solo devuelve 1 o 0
    def testR1(self,dataForPrediction):#R1 
        #Filtrar para NO agarrar voto primera ronda
        for persona in range(len(dataForPrediction)):
            dataForPrediction[persona]=dataForPrediction[persona][:21]

        tempDataNum=self.convertData(dataForPrediction)
        aux=np.array(tempDataNum)  ##Tiene 20X elementos predice el 20X+1

        predictions = self.model_R1.predict(aux)
        # round predictions
        rounded = [round(x[0]) for x in predictions]
        tmp = rounded
        return convert_party(tmp[0]) #Imprime basado en la funcion de activacion sigmoide la cual solo devuelve 1 o 0

    def convertData(self, data):
        xdata = []
        i = 0
        n = len(data)
        while(i < n):
            tmp = []
            if(data[i][0]=="SAN JOSE"):
                tmp.append(0)
                tmp.append(find_canton_SJ(data[i][1]))
            elif(data[i][0]=="ALAJUELA"):
                tmp.append(1)
                tmp.append(find_canton_AL(data[i][1]))
            elif(data[i][0]=="HEREDIA"):
                tmp.append(2)
                tmp.append(find_canton_HE(data[i][1]))
            elif(data[i][0]=="CARTAGO"):
                tmp.append(3)
                tmp.append(find_canton_CA(data[i][1]))
            elif(data[i][0]=="GUANACASTE"):
                tmp.append(4)
                tmp.append(find_canton_GU(data[i][1]))
            elif(data[i][0]=="PUNTARENAS"):
                tmp.append(5)
                tmp.append(find_canton_PU(data[i][1]))
            elif(data[i][0]=="LIMON"):
                tmp.append(6)
                tmp.append(find_canton_LI(data[i][1]))
            tmp.append(data[i][2])
            tmp.append(data[i][3])
            tmp.append(data[i][4])
            tmp.append(find_location(data[i][5]))
            tmp.append(find_sex(data[i][6]))
            tmp.append(data[i][7])
            tmp.append(data[i][8])
            tmp.append(data[i][9])
            tmp.append(find_house_state(data[i][10]))
            tmp.append(find_overcrowding(data[i][11]))
            tmp.append(find_literacy(data[i][12]))
            tmp.append(data[i][13])
            tmp.append(find_education(data[i][14]))
            tmp.append(find_work(data[i][15]))
            tmp.append(find_insurance(data[i][16]))
            tmp.append(find_foreign(data[i][17]))
            tmp.append(find_disabled(data[i][18]))
            tmp.append(find_insured(data[i][19]))
            tmp.append(find_female_head(data[i][20]))
            tmp.append(find_shared_head(data[i][21]))
            if len(data[i]) == 23 or len(data[i]) == 24:
                tmp.append(find_party(data[i][22]))
            if len(data[i]) == 24:
                tmp.append(find_party(data[i][23]))
            xdata.append(tmp)
            i += 1
        return xdata0



if __name__ == '__main__':
    Comandos().cmdloop()
