# ==========================================================
# 📘 Guía práctica de Modelos de Regresión y Funciones Útiles
# ==========================================================
# Esta guía resume los principales modelos de regresión y funciones auxiliares
# que se pueden usar en análisis de datos y machine learning.
# Es un apunte de consulta rápida.

from sklearn.linear_model import (
    HuberRegressor, RANSACRegressor, TheilSenRegressor,
    LinearRegression, RidgeCV
)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# 1️⃣ TIPOS DE REGRESORES
# ----------------------------------------------------------

# 📈 Regresión lineal estándar
modelo = LinearRegression()
# ✔ Ajusta una recta/hiperplano que minimiza el error cuadrático.
# ❌ Muy sensible a outliers.
# ✅ Útil cuando la relación es lineal y los datos son “limpios”.

# 🛡️ Regresores robustos
modelo = HuberRegressor()
# ✔ Combina MSE y MAE → tolera outliers leves.

modelo = RANSACRegressor()
# ✔ Ignora outliers extremos y ajusta solo inliers.

modelo = TheilSenRegressor()
# ✔ Usa medianas de pendientes, muy robusto a outliers.
# 💡 Recomendación:
#   - Muchos outliers   → RANSAC
#   - Pocos pero fuertes → Huber o TheilSen
#   - Sin outliers      → LinearRegression

# 🌲 Modelos adicionales
modelo = ExtraTreesRegressor()
# ✔ Ensamble de árboles aleatorios, captura no linealidades.

modelo = KNeighborsRegressor()
# ✔ Predice con vecinos cercanos (promedio).

modelo = RidgeCV()
# ✔ Regresión lineal con regularización L2 (reduce sobreajuste).

# 🤖 Support Vector Regression (SVR)
modelo = LinearSVR()
# ✔ Ajusta una recta tolerando un margen de error (epsilon).
# ✔ Robusta frente a ruido.
# 💡 Para múltiples salidas: usar MultiOutputRegressor(LinearSVR()).

# ----------------------------------------------------------
# 2️⃣ FUNCIONES AUXILIARES
# ----------------------------------------------------------

# 📊 Correlación de Pearson
def coef_corr(x, y):
    arriba = sum((x - x.mean()) * (y - y.mean()))
    abajo = sum((x - x.mean())**2) * sum((y - y.mean())**2)
    return arriba / np.sqrt(abajo)
# 📌 Mide la relación lineal entre dos variables:
#   1  → correlación perfecta positiva
#  -1  → correlación perfecta negativa
#   0  → no hay relación lineal

# 📏 Métricas de error
# mse = mean_squared_error(y_true, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_true, y_pred)
# 📌 MAE = error absoluto promedio
# 📌 RMSE = penaliza más los errores grandes

# 📐 Criterio de Akaike (AIC)
def AIC(ecm, num_params):
    return -2 * np.log(ecm) + 2 * num_params
# 📌 Compara modelos: menor AIC = mejor ajuste con menos complejidad.

# 📈 Regresión polinómica
pf = PolynomialFeatures(degree=3)
# X_poly = pf.fit_transform(X)
# modelo = LinearRegression()
# modelo.fit(X_poly, y)
# 📌 Convierte relaciones no lineales en un espacio lineal con features polinómicas.

# 🌀 Función sigmoide (curva logística)
def sigmoid(x, Beta_1, Beta_2):
    return 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
# from scipy.optimize import curve_fit
# popt, pcov = curve_fit(sigmoid, xdata, ydata)
# 📌 Útil cuando la curva de crecimiento sigue forma de "S" (ej. población).

# ----------------------------------------------------------
# 3️⃣ VISUALIZACIÓN
# ----------------------------------------------------------

def plot_best_fit(X, y, xaxis, model):
    """Entrena un modelo y dibuja la recta de mejor ajuste."""
    model.fit(X, y)
    yaxis = model.predict(xaxis.reshape((len(xaxis), 1)))
    plt.plot(xaxis, yaxis, label=type(model).__name__)
    plt.scatter(X, y, color='blue', alpha=0.5)
    plt.legend()
    plt.show()
# 📌 Sirve para comparar visualmente distintos modelos.

# ----------------------------------------------------------
# 4️⃣ CUÁNDO USAR CADA MODELO
# ----------------------------------------------------------
# ✅ Datos lineales, sin outliers        → LinearRegression o Ridge
# ✅ Datos lineales, pocos outliers      → Huber o TheilSen
# ✅ Datos lineales, muchos outliers     → RANSAC
# ✅ Relaciones no lineales simples      → PolynomialFeatures + LinearRegression
# ✅ Relaciones complejas                → ExtraTrees, KNN
# ✅ Datos con ruido en regresión lineal → LinearSVR
# ✅ Crecimiento en forma de S           → sigmoid + curve_fit
