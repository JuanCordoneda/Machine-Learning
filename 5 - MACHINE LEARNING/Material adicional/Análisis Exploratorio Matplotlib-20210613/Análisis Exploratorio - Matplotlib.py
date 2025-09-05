#!/usr/bin/env python
# coding: utf-8

# In[1]:


#linea de comando pip install matplotlib
#Anaconda conda install matplotlib
import matplotlib.pyplot as pl
import numpy as np

# Crear una figura de 8x6 puntos de tamaño, 80 puntos por pulgada
pl.figure(figsize=(8, 6), dpi=80)

# Crear una nueva subgráfica en una array de 1x1
pl.subplot(1, 1, 1)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

# En uno de los parámetros puedo especificar el color y el grosor de la línea para la gráfica en este caso elijo Rojo "Red" 
# y Grosor 2, el parámetro LineStyle define el tipo de Linea. En este caso voy a elegir -. para que se diferencie con la 
# segunda gráfica. Agrego el Parámetro Label para luego mostrar en la leyenda

pl.plot(X, C, color="Red", linewidth=2.0, linestyle="-.", label="Función Coseno")

# Voy a cambiar de la segunda gráfica el color a Amarillo, Yellow y el Grosor de la Línea a 4, tipor de línea -. 
#Agrego el parámetro label para luego mostrar la leyenda

pl.plot(X, S, color="Yellow", linewidth=4.0, linestyle="-", label="Función Seno")

#Configuro los Spines de la siguiente manera

ax = pl.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# Puedo poner límites para el Eje x. En el punto anterior la gráfica no tenía límites en este caso va a ir de -4 a 4
pl.xlim(-4.0, 4.0)

# Ticks en x - Con xticks puedo cambiar la graduación del eje
pl.xticks(np.linspace(-4, 4, 9, endpoint=True))

# Puedo poner límites para el Eje y. En el punto anterior la gráfica no tenía límites en este caso va a ir de -1 a 1
pl.ylim(-1.0, 1.0)

# Ticks en y Con yticks puedo cambiar la graduación del eje
pl.yticks(np.linspace(-1, 1, 5, endpoint=True))

pl.legend(loc='upper left')

#Anotaciones en el Gráfico. 
t = 2*np.pi/3
#Trazo la Linea de la función coseno hasta el Eje en Color Azul
pl.plot([t, t], [0, np.cos(t)],
        color='blue', linewidth=1.5, linestyle="--")
#Grafico el punto Azul de la función Coseno
pl.scatter([t, ], [np.cos(t), ], 50, color="blue")
pl.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
            xy=(t, np.sin(t)), xycoords='data',
            xytext=(+10, +30), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
#Trazo la Linea de la función seno hasta el Eje en Color Verde
pl.plot([t, t], [0, np.sin(t)],
        color='Green', linewidth=1.5, linestyle="--")
#Trazo el Punto de la función Seno en Verde
pl.scatter([t, ], [np.sin(t), ], 50, color="Green")
pl.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$', xy=(t, np.cos(t)),
            xycoords='data', xytext=(-90, -50), textcoords='offset points',
            fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# Muestro el Resultado por Pantalla
pl.show()


# Lista de Parámetros y Significado
# ![image.png](attachment:image.png)

# Parámetros de la función fill_between
# 
# fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, hold=None, data=None, **kwargs)
# 
# x : array 
# 
# Una matriz de longitud N de los datos x
# 
# y1 : array
# 
# Una matriz de longitud N (o escalar) de los datos y
# 
# y2 : array
# 
# Una matriz de longitud N (o escalar) de los datos y
# 
# where : array, optional
# 
# Si es Ninguno, por defecto se rellena entre todas partes. Si no es None, es una matriz booleana numérica de longitud N y el relleno solo ocurrirá
# 
# sobre las regiones donde donde == True.
# 
# interpolate : bool, optional
# 
# Si es verdadero, interpole entre las dos líneas para encontrar el punto preciso de intersección. De lo contrario, los puntos inicial y final de
# 
# la región llena solo ocurrirá en valores explícitos en la matriz x.
# 
# step : {‘pre’, ‘post’, ‘mid’}, optional
# 
# Si no es None, completar con el step

# In[2]:


import numpy as np

n = 256
X = np.linspace(-np.pi, np.pi, n, endpoint=True)
Y = np.sin(2 * X)

pl.axes([0.025, 0.025, 0.95, 0.95])

pl.plot(X, Y + 1, color='blue', alpha=1.00)
pl.fill_between(X, 1, Y + 1, color='blue', alpha=.25)

pl.plot(X, Y - 1, color='blue', alpha=1.00)
pl.fill_between(X, -1, Y - 1, (Y - 1) > -1, color='red', alpha=.25)
pl.fill_between(X, -1, Y - 1, (Y - 1) < -1, color='green',  alpha=.25)

pl.xlim(-np.pi, np.pi)
pl.xticks(())
pl.ylim(-2.5, 2.5)
pl.yticks(())

pl.show()


# matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, 
# 
# linewidths=None, verts=<deprecated parameter>, edgecolors=None, *, plotnonfinite=False, data=None, **kwargs)
# 
# x, y scalar o similar a una matriz, forma (n,) Las posiciones de los datos.
# 
# s: escalar o similar a una matriz, forma (n,), opcional
#     
# El tamaño del marcador en puntos ** 2. El valor predeterminado es rcParams ['lines.markersize'] ** 2.
# 
# c: color, secuencia o secuencia de colores, opcional- El color se pone en inglés por ejemplo 'red', 'black'
#     
# marker: MarkerStyle, opcional. El estilo del marcador. El marcador puede ser una instancia de la clase o la abreviatura de 
# 
# texto de un marcador en particular. El valor predeterminado es Ninguno, en cuyo caso toma el valor de rcParams 
#  
# ["scatter.marker"] (predeterminado: 'o') = 'o'. Consulte marcadores para obtener más información sobre los estilos de 
# 
# marcadores.
# 
# cmap: Colormap, opcional, predeterminado: Ninguno
#     
# Una instancia de mapa de colores o un nombre de mapa de colores registrado. cmap solo se usa si c es una matriz de flotantes. 
#     
# Si es Ninguno, el valor predeterminado es rc image.cmap.
# 
# norm: Normalize, opcional, predeterminado: Ninguno
# 
# Una instancia de Normalizar se usa para escalar los datos de luminancia a 0, 1. La norma solo se usa si c es una matriz de 
# 
# flotantes. Si es Ninguno, use los colores predeterminados. Normalizar.
# 
# 
# vmin, vmax :scalar, opcional, predeterminado: Ninguno
# vmin y vmax se utilizan junto con norm para normalizar los datos de luminancia. Si es Ninguno, se utilizan los respectivos
# 
# mínimos y máximos de la matriz de colores. vmin y vmax se ignoran si pasa una instancia de norma.
# 
# alphascalar, opcional, predeterminado: Ninguno
# 
# El valor de fusión alfa, entre 0 (transparente) y 1 (opaco).
# 
# linewidths: scalar o array, opcional, predeterminado: Ninguno
# 
# El ancho de línea de los bordes del marcador. Nota: Los colores de los bordes predeterminados son 'cara'. Es posible que desee 
#     
# cambiar esto también. Si es None, el valor predeterminado es rcParams ["lines.linewidth"] (predeterminado: 1.5).
# 
# edgecolors {'face', 'none', None} o color o secuencia de color, opcional.
# 
# El color del borde del marcador. Valores posibles:
# 
# 'face': el color del borde siempre será el mismo que el color de la cara.
# 'none': no ​​se dibujará ningún límite de parche.
# 
#   Un color o secuencia de color de Matplotlib.
# 
#     El valor predeterminado es Ninguno, en cuyo caso toma el valor de rcParams ["scatter.edgecolors"] (predeterminado: 'face') 
#     
#     = 'face'.
# 
# Para los marcadores sin relleno, el kwarg de los colores de borde se ignora y se fuerza a "encarar" internamente.
# 
# plotnonfiniteboolean, opcional, predeterminado: Falso
# 
# Establecer para trazar puntos con c no finito, junto con set_bad.
#     

# In[3]:


import numpy as np

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)

pl.axes([0.025, 0.025, 0.95, 0.95])
#Invoco al gráfico de dispersión 
#Le estoy pasando como parámetro en color T que lo definí arriba
pl.scatter(X, Y, s=75, c=T, alpha=.5)
#Defino los límites del gráfico
pl.xlim(-1.5, 1.5)
pl.xticks(())
pl.ylim(-1.5, 1.5)
pl.yticks(())

pl.show()


# matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
# 
# x: secuencia de escalares
# 
# Las coordenadas x de las barras. Consulte también alinear para ver la alineación de las barras con las coordenadas.
# 
# height: escalar o secuencia de escalares
# 
# La (s) altura (s) de las barras.
# 
# width: escalar o similar a una matriz, opcional
# 
# El ancho de las barras (predeterminado: 0.8).
# 
# botton: escalar o similar a una matriz, opcional
# 
# La (s) coordenada (s) y de las bases de las barras (predeterminado: 0).
# 
# align: {'center', 'edge'}, opcional, predeterminado: 'center'
# 
# Alineación de las barras a las coordenadas x:
# 
# 'center': Centre la base en las posiciones x.
# 
# 'edge': alinea los bordes izquierdos de las barras con las posiciones x.
# 
# Para alinear las barras en el borde derecho, pase un with negativo y align = 'edge'.

# In[4]:


n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

pl.axes([0.025, 0.025, 0.95, 0.95])
pl.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
pl.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    pl.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va= 'bottom')

for x, y in zip(X, Y2):
    pl.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va= 'top')

pl.xlim(-.5, n)
pl.xticks(())
pl.ylim(-1.25, 1.25)
pl.yticks(())

pl.show()


# contour([X, Y,] Z, [levels], **kwargs)
# 
# X, Y: similar a una matriz, opcional
# 
# Las coordenadas de los valores en Z.
# 
# X e Y deben ser 2-D con la misma forma que Z (por ejemplo, creados a través de numpy.meshgrid), o ambos deben ser 1-D de manera 
# 
# que len (X) == M es el número de columnas en Z y len (Y) == N es el número de filas en Z.
# 
# Si no se dan, se supone que son índices enteros, es decir, X = rango (M), Y = rango (N).
# 
# Z: como una matriz (N, M)
# 
# Los valores de altura sobre los que se dibuja el contorno.
# 
# niveles: int o similar a una matriz, opcional
# 
# Determina el número y las posiciones de las curvas de nivel / regiones.
# 
# Si es int n, use n intervalos de datos; es decir, dibujar n + 1 curvas de nivel. Las alturas de nivel se eligen 
# 
# automáticamente.
# 
# Si tiene forma de matriz, dibuje líneas de contorno en los niveles especificados. Los valores deben estar en orden creciente.

# In[5]:


import numpy as np

def f(x,y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)

pl.axes([0.025, 0.025, 0.95, 0.95])

pl.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=pl.cm.hot)
C = pl.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
pl.clabel(C, inline=1, fontsize=10)

pl.xticks(())
pl.yticks(())
pl.show()


# matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, 
# 
# labeldistance=1.1, startangle=None, radius=None, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), 
# 
# frame=False, rotatelabels=False, *, data=None
# 
# 
# x: como una matriz.
# 
# explode: similar a una matriz, opcional, predeterminado: ninguno
# 
# Si no es Ninguno, es una matriz len (x) que especifica la fracción del radio con la que compensar cada cuña.
# 
# labels: lista, opcional, predeterminado: ninguna
# 
# Una secuencia de cadenas que proporcionan las etiquetas para cada cuña.
# 
# colors: similar a una matriz, opcional, predeterminado: ninguno
# 
# Una secuencia de argumentos de color de matplotlib a través de los cuales se desplazará el gráfico circular. Si es Ninguno, 
# 
# usará los colores en el ciclo activo actualmente.
# 
# autopct: Ninguno (predeterminado), cadena o función, opcional
# 
# Si no es Ninguno, es una cadena o función que se usa para etiquetar las cuñas con su valor numérico. La etiqueta se colocará 
# 
# dentro de la cuña. Si es una cadena de formato, la etiqueta será fmt% pct. Si es una función, se llamará.
# 
# pctdistance: float, opcional, predeterminado: 0.6
# 
# La relación entre el centro de cada sector circular y el inicio del texto generado por autopct. Se ignora si autopct es None.
# 
# shadow: bool, opcional, predeterminado: false. Dibuja una sombra debajo del pastel.
# 
# labeldistance: float o None, opcional, predeterminado: 1.1
# 
# La distancia radial a la que se dibujan las etiquetas circulares. Si se establece en Ninguno, las etiquetas no se dibujan, pero 
# 
# se almacenan para su uso en la leyenda ()

# In[6]:


import numpy as np

n = 10
Z = np.ones(n)
Z[-1] *= 2

pl.axes([0.025, 0.025, 0.95, 0.95])

pl.pie(Z, explode=Z*.05, colors = ['%f' % (i/float(n)) for i in range(n)])
pl.axis('equal')
pl.xticks(())
pl.yticks()

pl.show()


# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

# In[7]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = pl.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=pl.cm.hot)
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=pl.cm.hot)
ax.set_zlim(-2, 2)

pl.show()


# matplotlib.pyplot.text(x, y, s, fontdict=None, withdash=<deprecated parameter>, **kwargs)
#     
# 
# x, y: escalares
# La posición para colocar el texto. De forma predeterminada, está en coordenadas de datos. El sistema de coordenadas se puede 
# 
# cambiar usando el parámetro de transformación.
# 
# s: str El texto.
# 
# fontdict: diccionario, opcional, predeterminado: ninguno
#     
# Un diccionario para anular las propiedades de texto predeterminadas. Si fontdict es None, los valores predeterminados están 
#     
# determinados por sus parámetros rc.
# 
# withdash: booleano, opcional, predeterminado: false
#     
# Crea una instancia de TextWithDash en lugar de una instancia de Text.

# In[8]:


import numpy as np

eqs = []
eqs.append((r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$"))
eqs.append((r"$\frac{d\rho}{d t} + \rho \vec{v}\cdot\nabla\vec{v} = -\nabla p + \mu\nabla^2 \vec{v} + \rho \vec{g}$"))
eqs.append((r"$\int_{-\infty}^\infty e^{-x^2}dx=\sqrt{\pi}$"))
eqs.append((r"$E = mc^2 = \sqrt{{m_0}^2c^4 + p^2c^2}$"))
eqs.append((r"$F_G = G\frac{m_1m_2}{r^2}$"))

pl.axes([0.025, 0.025, 0.95, 0.95])

for i in range(24):
    index = np.random.randint(0, len(eqs))
    eq = eqs[index]
    size = np.random.uniform(12, 32)
    x,y = np.random.uniform(0, 1, 2)
    alpha = np.random.uniform(0.25, .75)
    pl.text(x, y, eq, ha='center', va='center', color="#11557c", alpha=alpha,
         transform=pl.gca().transAxes, fontsize=size, clip_on=True)
pl.xticks(())
pl.yticks(())

pl.show()


# In[ ]:




