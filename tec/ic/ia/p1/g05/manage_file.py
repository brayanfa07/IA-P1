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

    height_datalist = len(datalist) - 1
    weight_datalist = len(datalist[0]) - 2


    new_list = []
    for i in range(height_datalist):
        row = []
        for j in range(weight_datalist):
            z = (float(datalist[i][j]) - mean_function(calc_datalist(datalist, j))) / standard_deviation(calc_datalist(datalist, j))
            row.append(z)
        new_list.append(row)

    return new_list


#read_file("cancer.csv")