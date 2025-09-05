# 📘 DICCIONARIO DE FUNCIONES – VECTORES, GRÁFICOS Y REGRESIÓN LINEAL

# ✅ FUNCIONES BÁSICAS DE NUMPY Y VISUALIZACIÓN
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sympy import symbols, factor, expand, collect

# Crear vectores con NumPy
u = np.array([1, 0])
v = np.array([0, 5])

# Producto punto (dot product)
np.dot(u, v)  # Si es 0 → vectores ortogonales

# Graficar puntos en el plano
x_coords, y_coords = zip(u, v)
plt.scatter(x_coords, y_coords, color=["r", "b"])

# Dibujar flecha (vector 2D)
plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
          head_width=0.2, head_length=0.3, length_includes_head=True)

# Mostrar grilla y gráfico
plt.grid()
plt.show()

# Definir límites de los ejes
plt.axis([0, 9, 0, 6])

# ✅ FUNCIONES PERSONALIZADAS

# Graficar vector 2D
def plot_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
                     head_width=0.2, head_length=0.3, length_includes_head=True,
                     **options)

# Graficar múltiples vectores 3D
from mpl_toolkits.mplot3d import Axes3D

def plot_vectors3d(ax, vectors3d, z0, **options):
    for vec in vectors3d:
        ax.quiver(0, 0, z0, vec[0], vec[1], vec[2], **options)

# Desplazar puntos por un vector
def desplazar_puntos(a, b, c, d):
    desplazamiento = np.array([-7, 5])
    return a + desplazamiento, b + desplazamiento, c + desplazamiento, d + desplazamiento

# ✅ CARGA Y FILTRADO DE DATOS CON PANDAS

import pandas as pd
import urllib.request

# Descargar CSV
url = "https://unket.s3.sa-east-1.amazonaws.com/data/Life%20Expectancy%20Data.csv"
filename = "life_expectancy.csv"
urllib.request.urlretrieve(url, filename)

# Leer CSV
data = pd.read_csv(filename)

# Filtrado por condiciones
data2 = data.query("Year == 2014 and GDP < 80000 and `percentage expenditure` != 0")
data2 = data2[["Country", "percentage expenditure", "GDP"]]

# Graficar relación entre gasto y PBI
gasto = data2["percentage expenditure"]
pbi = data2["GDP"]
plt.scatter(pbi, gasto)
plt.grid()
plt.show()

# ✅ REGRESIÓN LINEAL MANUAL CON NUMPY

# Convertir Series a arrays

# Suponiendo que ya tenés pbi y gasto (ambos como pandas.Series)
x = np.array(pbi).reshape(-1, 1)  # scikit-learn espera 2D para X
y = np.array(gasto)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(x, y)

# Obtener coeficientes
β1 = modelo.coef_[0]
β0 = modelo.intercept_

print(f"Pendiente (β1): {β1}")
print(f"Ordenada al origen (β0): {β0}")

# Predicciones
ŷ = modelo.predict(x)

# Graficar
plt.scatter(x, y, label='Datos')
plt.plot(x, ŷ, color='red', label='Regresión Lineal')
plt.grid()
plt.legend()
plt.xlabel("PBI")
plt.ylabel("Gasto en salud")
plt.title("Regresión Lineal Automática")
plt.show


# ------------------------------------------------------------------------------------------------------------------------
# UNIDAD 2
# ------------------------------------------------------------------------------------------------------------------------

# 📘 DICCIONARIO DE FUNCIONES – FACTORIZACIÓN CON SYMPY Y MATRICES

# ✅ Factorización de un monomio
x, y = symbols('x y')
expr = 3*x*y
factor(expr)  # → 3*x*y (ya está factorizado)

# ✅ Factor común
expr = 3*x + 3*y
factor(expr)  # → 3*(x + y)

# ✅ Binomio al cuadrado
x = symbols('x')
expr = (x + 2)**2
factor(expr)  # → (x + 2)**2
expand(expr)  # → x**2 + 4x + 4

# ✅ Diferencia de cuadrados
expr = x**2 - 4
factor(expr)  # → (x - 2)*(x + 2)

# ✅ Agrupación de términos
x, y, z, a = symbols('x y z a')
expr = x*y + x*z + a*y + a*z
factor(expr)  # → (x + a)*(y + z)

# ✅ Trinomio cuadrado (ax² + bx + c)
expr = x**2 + 3*x + 2
factor(expr)  # → (x + 1)*(x + 2)

# ✅ Sustitución (agrupando por una variable)
u, v, w = symbols('u v w')
expr = 2*u*v + 3*u*w - 4*v*w
collect(expr, u)  # → u*(2*v + 3*w) - 4*v*w

# ✅ División sintética (factorización de polinomio de grado mayor)
expr = x**3 - 6*x**2 + 11*x - 6
factor(expr)  # → (x - 1)*(x - 2)*(x - 3)

# ✅ Factorización simbólica aplicada a un caso empresarial
def factorizar_trinomio_produccion(a, b, c):
    discriminante = b**2 - 4*a*c
    if discriminante < 0:
        return "No tiene raíces reales"
    elif discriminante == 0:
        raiz_a = int(a**0.5)
        raiz_c = int(c**0.5)
        return f"{a}p^2 + {b}p + {c} = ({raiz_a}p + {raiz_c})^2"
    else:
        raiz1 = (-b + discriminante**0.5) / (2 * a)
        raiz2 = (-b - discriminante**0.5) / (2 * a)
        return f"{a}p^2 + {b}p + {c} = (p - {raiz1})(p - {raiz2})"

# ------------------------------------------------------------------------------------------------------------------------
# UNIDAD 3
# ------------------------------------------------------------------------------------------------------------------------

# 📘 FUNCIONES CLAVE – DERIVADAS Y EJERCICIOS (con Derivative y doit)


# Declaración de símbolos
x, h = sp.symbols('x h')

# ✅ DERIVADA POR DEFINICIÓN (Cociente incremental)
f = x**2 + 3*x + 2
limite = sp.limit((f.subs(x, x + h) - f)/h, h, 0)
print("Derivada por definición:", limite)

# ✅ DERIVADA SIMBÓLICA
f = x**3 - 3*x**2 + 2
derivada = sp.diff(f, x)
print("Derivada simbólica:", derivada)

# ✅ USO DE Derivative Y doit
# Se crea el objeto Derivative y luego se resuelve con doit()
f_expr = x**4 + 2*x
d = sp.Derivative(f_expr, x)
print("Objeto Derivative (sin resolver):", d)
print("Resultado con .doit():", d.doit())

# ✅ DERIVADAS DE EXPONENCIALES Y LOGARÍTMICAS
f1 = sp.exp(x)
f2 = sp.log(x)
print("Derivada de e^x:", sp.diff(f1, x))
print("Derivada de ln(x):", sp.diff(f2, x))

# ✅ ECUACIÓN DE LA TANGENTE EN x0
x0 = 1
f = x**2 + 2*x
f_deriv = sp.diff(f, x)
pendiente = f_deriv.subs(x, x0)
tangente = pendiente * (x - x0) + f.subs(x, x0)
print("Ecuación de la tangente:", sp.simplify(tangente))

# ✅ EVALUACIÓN NUMÉRICA DE DERIVADAS
f = x**3
f_deriv = sp.diff(f, x)
valor = f_deriv.subs(x, 2)
print("Derivada evaluada en x=2:", valor)
