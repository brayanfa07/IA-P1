import csv  # Para abrir el archivo csv y cargar los datos
import os  # Para cargar el archivo de datos desde la ruta de instalacion

   
# Variables globales
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
dataset = []  # Lista donde se guardan todos los individuos creados

"""
Genera la poblacion total de la cual se generaran las muestras por pais y provincia.
Crea un individuo por cada voto recibido y toma en cuenta los indicadores cantonales.
La lista individuo contiene en los indices los siguientes datos:
"""

def read_file():
    with open("cancer.csv") as csvfile:  ##open(csvURL,encoding="utf8")-- Es para correr en windows
        entry = csv.reader(csvfile)
        for reg in entry:
            dataset.append(reg)
            print(reg)
            print()
    return 0

read_file()