import csv  # Para abrir el archivo csv y cargar los datos
import os  # Para cargar el archivo de datos desde la ruta de instalacion
from statistics import mean, stdev
import math
   
# Variables globales
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
dataset = []  # Lista donde se guardan todos los individuos creados

"""
Genera la poblacion total de la cual se generaran las muestras por pais y provincia.
Crea un individuo por cada voto recibido y toma en cuenta los indicadores cantonales.
La lista individuo contiene en los indices los siguientes datos:
"""

#Function that red the csv file

def delete_column_datalist(datalist): 
    for i in range(len(datalist)):
        datalist[i].pop(0)
        #print(datalist[i])
        #print()

    return datalist



def read_file(file_name):

    with open(file_name) as csvfile:  ##open(csvURL,encoding="utf8")-- Es para correr en windows
        entry = csv.reader(csvfile)
        for reg in entry:
            dataset.append(reg)
    dataset.pop(0)
    return dataset

#Function that calculates the mean of a list
def mean_function(column_list):
    return mean(column_list)

#Function that calculate the standard deviation
def standard_deviation(column_list):
    return stdev(column_list)

#Function that create a datalist
def calc_datalist(sampledata, column):
    datalist = []
    for i in range(len(sampledata) - 1):
        datalist.append(float(sampledata[i][column]))
    return datalist

#Function that normalize the datalist
def normalize_list(datalist):
    height_datalist = len(datalist) 
    weight_datalist = len(datalist[0])
    new_list = []
    max_list = []
    min_list = []
    for i in range(height_datalist):
        row = []
        for j in range(weight_datalist):
            if datalist[i][j] ==  'M' or datalist[i][j] == 'B':
                row.append(datalist[i][j])
            else:
                element = (float(datalist[i][j]) - mean_function(calc_datalist(datalist, j))) / standard_deviation(calc_datalist(datalist, j))
                row.append(element)
            new_list.append(row)
            #print(row)
            #print()
            #max_list.append(max(row))
            #min_list.append(min(row))
    #print("max: ", max(max_list), "min: ", min(min_list))
    return new_list

#Function that classify the data in categories

def classify_data(datalist):
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