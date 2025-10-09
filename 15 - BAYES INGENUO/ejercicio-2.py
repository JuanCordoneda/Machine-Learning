# -*- coding: utf-8 -*-
# Ejercicio 2 – Naive Bayes Gaussiano con Altura y Peso -> Clase: Hombre/Mujer

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1) ===================== Dataset de ejemplo (reemplazá por el tuyo) =====================

np.random.seed(42)

n_h = 120  # hombres
n_m = 120  # mujeres

# Generamos datos "realistas" (distribuciones normales distintas por clase)
altura_h = np.random.normal(loc=177, scale=7.0, size=n_h)  # cm
peso_h   = np.random.normal(loc=80,  scale=10.0, size=n_h) # kg

altura_m = np.random.normal(loc=164, scale=6.0, size=n_m)  # cm
peso_m   = np.random.normal(loc=63,  scale=8.0,  size=n_m) # kg

df_h = pd.DataFrame({"altura_cm": altura_h, "peso_kg": peso_h, "sexo": "Hombre"})
df_m = pd.DataFrame({"altura_cm": altura_m, "peso_kg": peso_m, "sexo": "Mujer"})
df = pd.concat([df_h, df_m], ignore_index=True)

# Limpieza rápida (por si alguna normal generó valores negativos o extremos)
df = df[(df["altura_cm"] > 120) & (df["altura_cm"] < 210) &
        (df["peso_kg"]   > 35)  & (df["peso_kg"]   < 140)].copy()

# 2) ===================== Train/Test split =====================
X = df[["altura_cm", "peso_kg"]].values
y = df["sexo"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) ===================== Modelo Naive Bayes Gaussiano =====================
nb = GaussianNB()
nb.fit(X_train, y_train)

# 4) ===================== Evaluación =====================
y_pred = nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=["Hombre", "Mujer"])

print(f"Exactitud (accuracy): {acc:.3f}\n")
print("Matriz de confusión [filas=verdadero, columnas=predicho] (Hombre/Mujer):")
print(cm, "\n")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred, digits=3))

# 5) ===================== Predicciones nuevas =====================
# Ejemplo: altura=170 cm, peso=72 kg
nuevos = np.array([[170, 72],
                   [182, 90],
                   [158, 55]])  # podés agregar más filas

pred_nuevos = nb.predict(nuevos)
proba_nuevos = nb.predict_proba(nuevos)  # probabilidades por clase

print("\nCasos nuevos -> Predicción y probabilidades:")
for i, (x, pred, proba) in enumerate(zip(nuevos, pred_nuevos, proba_nuevos)):
    # Orden de clases:
    # nb.classes_ podría ser np.array(["Hombre", "Mujer"]) (verifícalo imprimiendo nb.classes_)
    print(f"Ejemplo {i+1}: Altura={x[0]} cm, Peso={x[1]} kg -> {pred} | Prob={dict(zip(nb.classes_, proba.round(3)))}")

# 6) ===================== (Opcional) Guardar el modelo y el vector de clases =====================
# from joblib import dump
# dump(nb, "modelo_nb_altura_peso.joblib")
# np.save("clases_nb.npy", nb.classes_)


# SALIDA

# Exactitud (accuracy): 0.933

# Matriz de confusión [filas=verdadero, columnas=predicho] (Hombre/Mujer):
# [[27  3]
#  [ 1 29]] 

# Reporte de clasificación:
#               precision    recall  f1-score   support

#       Hombre      0.964     0.900     0.931        30
#        Mujer      0.906     0.967     0.935        30

#     accuracy                          0.933        60
#    macro avg      0.935     0.933     0.933        60
# weighted avg      0.935     0.933     0.933        60


# Casos nuevos -> Predicción y probabilidades:
# Ejemplo 1: Altura=170 cm, Peso=72 kg -> Mujer | Prob={np.str_('Hombre'): np.float64(0.466), np.str_('Mujer'): np.float64(0.534)}
# Ejemplo 2: Altura=182 cm, Peso=90 kg -> Hombre | Prob={np.str_('Hombre'): np.float64(1.0), np.str_('Mujer'): np.float64(0.0)}
# Ejemplo 3: Altura=158 cm, Peso=55 kg -> Mujer | Prob={np.str_('Hombre'): np.float64(0.002), np.str_('Mujer'): np.float64(0.998)}