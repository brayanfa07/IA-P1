
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
# Para generar la muestra de datos
#from tec.ic.ia.pc1.g09 import generar_muestra_pais, generar_muestra_provincia
from manage_file import read_file, delete_column_datalist, normalize_list



# Variables globales generales
prefijo = "" # Prefijo para el archivo de salida
# Division de poblacion para pruebas y para entrenamiento
porcentaje_pruebas = 20
modelo = "" # Seleccion de modelo
c = 0 # Parámetro de penalización del término de error.
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


class Decision_tree_model:
    attrib_value_entropy_general_list = []
    r = ""
    dt = None

    def __init__(self, sample_data, r, umbral):
            self.r = r
            self.dt  = self.calc_info_gain(self.convert_to_interval(sample_data), r)
            self.dt = self.prunning(self.dt,umbral,0)

    def test(self,data):
            data = self.convert_to_interval([data])
            return self.testt(data[0],self.dt)

    def testt(self,data,tree):
            try:
                    index = tree.element
                    index2 = tree.decisions.index(data[index])
                    if(type(tree.children[index2]) == str):
                            return tree.children[index2]
                    else:
                            return self.testt(data,tree.children[index2])
            except ValueError:
                    return "No value"

    #Calc the entropy of the attributes
    def calc_info_gain(self, sample_data, r):
            if len(sample_data) == 0:
                    print("Realización de la ejecución del programa con éxito")
            else:
                    attributes_count = 0
                    class_list = []
                    class_type = 0
                    self.attrib_value_entropy_general_list = []
                    if r == "r1":
                            attributes_count = len(sample_data[0]) - 2
                            class_type = len(sample_data[0]) - 2
                            class_list = self.calc_class_data(class_type,sample_data)
                    elif r == "r2":
                            attributes_count = len(sample_data[0]) - 2
                            class_type = len(sample_data[0]) - 1
                            class_list = self.calc_class_data(class_type,sample_data)
                    else:
                            attributes_count = len(sample_data[0]) - 1
                            class_type = len(sample_data[0]) - 1
                            class_list = self.calc_class_data(class_type,sample_data)
                    information_gain_list =[]
                    class_entropy_num = self.class_entropy(class_type, sample_data)
                    position = 0
                    attributes_max = 0
                    index = 0
                    for i in range(attributes_count): #Travel the list of attributes
                            factor = 0
                            attrib_value_list = self.calc_class_data(i, sample_data)
                            attrib_value_entropy_list = [] #Store the entropy of the values of an attribute of the data
                            for j in range(len(attrib_value_list)): #Travel the list of the values of the attributes
                                    attrib_value_entropy_list.append(self.calc_attrib_value_entropy(j,attrib_value_list[j],sample_data,class_list, class_type, i))
                            for k in range(len(attrib_value_entropy_list)):
                                    factor = factor + ((self.calc_attrib_value_found(attrib_value_list[k],sample_data,i)/len(sample_data)) * attrib_value_entropy_list[k])
                            entropy_attribute_rest = class_entropy_num - factor
                            information_gain_list.append(entropy_attribute_rest)
                    if(max(information_gain_list)!= 0):
                            attributes_max = max(information_gain_list)
                            index = information_gain_list.index(attributes_max)
                            tmp = self.calc_attrib_value_class(index,sample_data)
                            tmp2 = self.calc_no_attrib_value_class(index)
                            sample_data = self.calc_del_attrib_value_sd(tmp, index, sample_data)
                            dt = Decision_tree(index)
                            i = 0
                            n = len(tmp)
                            while(i<n):
                                    dt.decisions.append(tmp[i][0])
                                    dt.children.append(tmp[i][1])
                                    i += 1
                            i = 0
                            n = len(tmp2)
                            while(i<n):
                                    dt.decisions.append(tmp2[i])
                                    dt.children.append(self.calc_info_gain(sample_data,self.r))
                                    i += 1
                            return dt
                    else:
                            i = 0
                            n = len(sample_data)
                            tmp = []
                            while(i < n):
                                    if(self.r == "r1"):
                                            tmp.append(sample_data[i][len(sample_data[i])-2])
                                    else:
                                            tmp.append(sample_data[i][len(sample_data[i])-1])
                                    i += 1
                            cuenta1 = collections.Counter(tmp)
                            return cuenta1.most_common(1)[0][0]
    
    def prunning(self,tree,umbral,depth):
            i = 0
            n = len(tree.children)
            while(i<n):
                    if(type(tree.children[i]) != str):
                            tree.children[i] = self.prunning(tree.children[i],umbral,depth+1)
                    i += 1
            if(depth>umbral):
                    cuenta1 = collections.Counter(tree.children)
                    if(len(cuenta1) == 2 or len(cuenta1) == 1):
                            return cuenta1.most_common(1)[0][0]
            return tree

    def calc_attrib_value_class(self, index, sample_data):
            val = self.attrib_value_entropy_general_list[index] #list of the values with a class
            n = len(val)
            i = 0
            result_list = []
            while (i<n):
                    if (val[i][1] == 0):
                            tmp = val[i][0]
                            for x in sample_data:
                                    if (tmp == x[index]):
                                            if (self.r == "r1"):
                                                    result_list.append([tmp, x[len(x) - 2]])
                                            else:
                                                    result_list.append([tmp, x[len(x) - 1]])
                                            break
                    i += 1
            return result_list

    def calc_no_attrib_value_class(self, index):
            val = self.attrib_value_entropy_general_list[index] #list of the values with a class
            n = len(val)
            i = 0
            result_list = []
            while (i<n):
                    if (val[i][1] != 0):
                            result_list.append(val[i][0])
                    i += 1
            return result_list

    def calc_del_attrib_value_sd(self, result_list,index, sample_data):
            i = 0
            n = len(result_list)
            drop = []
            while(i<n):
                    j = 0
                    m = len(sample_data)
                    while(j < m):
                            if (result_list[i][0] == sample_data[j][index]):
                                    drop.append(j)
                            j += 1
                    i += 1
            n = len(drop)
            i = 0
            drop = sorted(drop)
            while(n > i):
                    sample_data.pop(drop[n - 1])
                    n -= 1
            n = len(sample_data)
            while(i < n):
                    sample_data[i][index] = 0
                    i += 1
            return sample_data

    #Calculate the count of a value of a attribute according to a class in a data
    def found_count_in_list(self, class_element,attrib_value,sample_data, class_list,class_type, attrib_value_position):
            count = 0
            class_positions_list = []
            attrib_value_positions_list = []
            for i in range(len(sample_data)):
                    if sample_data[i][class_type] == class_element:
                            class_positions_list.append(i)
            for j in range(len(sample_data)):
                    if sample_data[j][attrib_value_position] == attrib_value:
                            attrib_value_positions_list.append(j)
            for k in range(len(class_positions_list)):
                    for l in range(len(attrib_value_positions_list)):
                            if(class_positions_list[k] == attrib_value_positions_list[l]):
                                    count = count + 1
            return count

    #Calculate the count in which a value of an attribute is found
    def calc_attrib_value_found(self, attrib_value,sample_data,attribute_position):
            count = 0
            for i in range(len(sample_data)):
                    if sample_data[i][attribute_position] == attrib_value:
                            count = count + 1
            return count

    #Calculate the probability of a value inside an attribute of the data
    def calc_value_prob(self, class_element,attrib_value,sample_data,class_list,class_type,attrib_value_position):
            prob_value = self.found_count_in_list(class_element,attrib_value,sample_data, class_list, class_type,attrib_value_position)/self.calc_attrib_value_found(attrib_value,sample_data,attrib_value_position)
            return prob_value

    #Calulate the entropy of a value of an attribute of the data
    def calc_attrib_value_entropy(self, attribute_value_list, attrib_value,sample_data, class_list, class_type, attrib_value_position):
            value_prob_list = [] #List of the probabilities of the every class
            entropy = 0.0

            for i in range(len(class_list)): #class list of the round
                    #append the prob of one value according to a class
                    value_prob_list.append(self.calc_value_prob(class_list[i],attrib_value, sample_data, class_list,class_type,attrib_value_position))
            base = 2.0
            for j in range(len(value_prob_list)):
                    if (value_prob_list[j] == 0):
                            entropy = entropy + 0
                    else:
                            entropy = entropy - (value_prob_list[j] * math.log(value_prob_list[j], base))

            if len(self.attrib_value_entropy_general_list) == attrib_value_position:
                    self.attrib_value_entropy_general_list.append([])
            self.attrib_value_entropy_general_list[attrib_value_position].append([attrib_value, entropy])
            return entropy

    #Calculate the probablistic to be found in a list
    def probability_in_list(self, element, list, position):
            data_count = len(list)
            probabability = self.count_element(element, list, position) / data_count
            return probabability

    def count_element(self, element, sample_data, class_type):
            count = 0
            for i in range(len(sample_data)):
                    if element == sample_data[i][class_type]:
                            count = count + 1
            return count

    #Calculate the probablistic to be found in a class
    def class_entropy(self, class_type, sample_data):
            probabilistic_class_list = [] #List of the probabilities of the every class
            class_list = self.calc_class_data(class_type, sample_data)
            entropy = 0.0
            for i in range(len(class_list)):
                    probabilistic_class_list.append(self.probability_in_list(class_list[i], sample_data, class_type))
            base = 2.0
            for i in range(len(class_list)):
                    if (probabilistic_class_list[i] == 0):
                            entropy = entropy + 0
                    else:
                            entropy = entropy - (probabilistic_class_list[i] * math.log(probabilistic_class_list[i],base))
            return entropy

    #Calc the class of an object
    def calc_class_data(self, position, data_list):
            class_data_list = []
            for i in range(len(data_list)):
                    found = False
                    for j in range(len(class_data_list)):
                            if (class_data_list[j] == data_list[i][position]):
                                    found = True
                    if found == False:
                            class_data_list.append(data_list[i][position])
            return class_data_list        

    def convert_to_interval(self,datalist):
            new_list = []
            height_datalist = len(datalist) 
            weight_datalist = len(datalist[0])
            for i in range(height_datalist):
                row = []
                for j in range(weight_datalist):
                    if datalist[i][j] ==  'M' or datalist[i][j] == 'B':
                        row.append(datalist[i][j])
                    else:
                        if datalist[i][j] > -3 and datalist[i][j] <= 1:
                            row.append(1)
                        if datalist[i][j] > 1 and datalist[i][j] <= 4:
                            row.append(2)
                        if datalist[i][j] > 4 and datalist[i][j] <= 7:
                            row.append(3)
                        if datalist[i][j] > 7 and datalist[i][j] <= 10:
                            row.append(4)
                        if datalist[i][j] > 10 and datalist[i][j] <= 13:
                            row.append(5)
                new_list.append(row)
                print(row)
                print()
            return new_list


class Decision_tree:
    element = None
    children = None
    decisions = None

    def __init__(self, element):
        self.element = element #Index of the attribute in the data
        self.decisions = []
        self.children = []