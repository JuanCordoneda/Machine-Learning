#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 2022

@author: Anahí Romo
"""
##############################################################################
#
#      Este código divide el csv gigante en varios archivos más chicos.
#   Usa iteradores para no tener que cargar todo en la memoria.
#   Se invoca a la función split_csv:
#   'filediv' es la cantidad de archivos más pequeños 
#    filename es el nombre del csv gigante
#    header = True si el csv original tiene encabezados
#
#     Gracias por el código a:
#     https://es.stackoverflow.com/questions/117502/
#
##############################################################################

import itertools
import csv
import os

def rows_count(file):                      # cuenta las filas del csv
    with open(file, 'r') as f:
        return sum(1 for row in f)

def get_chucks(it, lines, chucks):    # itera s/ la cant de líneas de c/archivo
    chuck_size = lines // chucks         # total / cantidad de archivos finales
    for i in range(1, chucks):
        yield i, itertools.islice(it, chuck_size)
    yield i + 1, it
    
def write_csv_files(path, data, header = None):
    with open(path, mode='w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        if header:
            spamwriter.writerow(header)      # pone los encabezados
        spamwriter.writerows(data)

def split_csv(file,  filediv,  header = True):
    with open(file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        lines = rows_count(file)
        head = None
        if header:
            lines -= 1
            head = next(spamreader)
        if lines < filediv:
            raise ValueError("The number of rows ({}) is less than the number of output files ({})".format(lines, filediv))
        for i, data in get_chucks(spamreader, lines, filediv):
            path = "{}file_{}.csv".format(os.path.dirname(file), i)
            write_csv_files(path, data, head)      # escribe los subarchivos
            print(f'Listo {path}')




# el csv original tiene 15 808 050 filas, elijo dividirlo en 16 subarchivos
# que tendrán cerca de 1 millón de líneas cada uno

filename = '220430COVID19MEXICO.csv'
split_csv(filename, 16, header = True)