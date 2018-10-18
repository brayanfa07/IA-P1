import csv  # Para abrir el archivo csv y cargar los datos
import os  # Para cargar el archivo de datos desde la ruta de instalacion
from statistics import mean, pstdev
   
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
    new_datalist = []
    for i in range(len(datalist)):
        i = datalist[i].pop(0)
        new_datalist.append(i)

    return new_datalist



def read_file(file_name):
    with open(file_name) as csvfile:  ##open(csvURL,encoding="utf8")-- Es para correr en windows
        entry = csv.reader(csvfile)
        for reg in entry:
            dataset.append(reg)
            print(reg)
            print()
    return 0

#Function that calculates the mean of a list
def mean(datalist):
    return mean(datalist)

#Function that calculate the standard deviation
def standard_deviation(datalist):
    return stdev(datalist)

#Function that create a datalist
def calc_datalist(sampledata, column):
    datalist = []
    for i in range(len(sampledata)):
        datalist.append(sampledata[i][column])


#read_file("cancer.csv")