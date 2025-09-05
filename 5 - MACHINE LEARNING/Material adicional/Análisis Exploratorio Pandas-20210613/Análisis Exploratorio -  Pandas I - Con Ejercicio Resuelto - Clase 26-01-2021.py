#!/usr/bin/env python
# coding: utf-8

# Librerías a instalar desde la línea de comando 
# 
# pip install -U scikit-learn
# 
# pip install pandas
# 
# ¿Cómo chequeo la instalación desde la línea de comando de windows? 
# 
# pip list
# 
# Librerías a instalar si estoy desde Anaconda
# 
# conda install scikit-learn
# 
# conda install pandas
# 
# conda install pandas=0.20.3
# 
# ¿Cómo chequeo las librerias instaladas de Anaconda?
# 
# conda list
# 

# In[1]:


import os
import pandas as pd
#import numpy as npdiabetes
import matplotlib.pyplot as plt
os.getcwd()
#os.chdir('\')


# In[2]:


os.getcwd()


# 
# 
# 

# In[3]:


iris = pd.read_csv("Iris.csv")


# Para movilizarnos en el Dataframe tenemos las funciones iloc y loc
# 
# La función loc se puede usar de dos formas diferentes: seleccionar filas o columnas en base a una etiqueta o seleccionar filas 
# 
# o columnas en base a una condición.-
# 
# La función iloc me permite desplazarme por el dataframases una función basada en etiquetas para acceder a los datos de un 
# 
# índice y / o columna en particular y devuelve todas las entradas almacenadas para ese índice / columna en particular.

# **Función Columns:**
# Puedo ver las columnas del dataset

# In[4]:


iris.columns


# **Función nunique:**
# Para saber los valores únicos

# In[5]:


iris.nunique()


# **Función dtypes:**
# Para saber el tamaño del dataset - Filas y Columnas

# In[6]:


iris.dtypes


# In[7]:


iris.shape


# **Función Size:**
# Para el tamaño del dataset

# In[8]:


iris.size


# **Ver Valores Nulos - Función isna**

# In[9]:


iris.isna()


# In[10]:


iris.head(10)


# In[11]:


iris.loc[[0,1]]


# In[ ]:





# Ahora si quiero ver una columna específica por ejemplo
# 
# ![image.png](attachment:image.png)

# In[12]:


iris.loc[[0,1],["SepalWidthCm","PetalLengthCm"]]


# Ejercicio:
# 
# Obtener las filas desde la 4, 3 y 9 para SepalLengthCm y SepalWidthCm

# In[ ]:


iris.loc[[4,3,9],["SepalLengthCm","SepalWidthCm"]]


# Ejercicio:
# 
# Mostrar las filas desde la 1 a la 5

# In[ ]:


iris.loc[1:5]


# Ejercicio
# 
# Mostrar las filas desde la 1 a la 5 para SepalLengthCm y SepalWidthCm

# In[ ]:


iris.loc[[1,5],["SepalLengthCm","SepalWidthCm"]]


# In[16]:


iris.iloc[1:3]


# In[17]:


iris.iloc[0:5] # Primeras cinco filas


# In[18]:


iris.iloc[:, 0:5] # Primeras cinco columnas


# In[19]:


iris.iloc[[0,2,1]]  # Primera, tercera y segunda filas


# In[20]:


iris.iloc[:, [0,2,1]]  # Primera, tercera y segunda columnas


# Si quiero filtrar por un valor específico en el dataframe lo puedo hacer de la siguiente manera

# In[21]:


setosa = iris.loc[:, "Species"] == "Iris-setosa"


# In[22]:


df_setosa = iris.loc[setosa]


# In[23]:


df_setosa.head(10)


# In[24]:


iris.describe()


# Combinar Dataframes
# 
# Para combinar dataframes tenemos las opciones de merge
# 
# ![image.png](attachment:image.png)
# 
# 
# 
# Fusionar para combinar observaciones de diferentes marcos de datos utilizando una variable común como referencia
# 
# how = inner
# 
# how = outer
# 
# how = left
# 
# how = right

# In[25]:


df_1 = iris.iloc[[1,51],3:]


# In[26]:


df_1.head()


# In[27]:


df_2 = iris.iloc[[1,51,101],[0,2,4,5]]


# In[28]:


df_2.head()


# In[29]:


result = pd.merge(left = df_1 , right = df_2,on= ["Species"], how = "outer")


# In[30]:


result = pd.merge(left = df_1 , right = df_2,on= ["Species"], how = "left")


# In[31]:


result.head()


# Agrupar - Group By

# In[32]:


df_3 = iris.iloc[[1,2,3,4,51,52,53,54],[0,2,4,5]]


# In[33]:


df_3.head()


# In[34]:


df_3[['PetalWidthCm','Species']].groupby('Species').sum()


# In[35]:


df_3[['PetalWidthCm','Species']].groupby('Species').mean()


# In[36]:


df_3[['PetalWidthCm','Species']].groupby('Species').count()


# In[37]:


df_3[['PetalWidthCm','Species']].groupby('Species').plot(kind='bar')


# In[38]:


df_4 = iris.iloc[1:5,0:3]


# In[39]:


df_4.head()


# In[40]:


df_4["Species"] = ["versicolor", "versicolor", "setosa", "setosa"]


# In[41]:


df_4.head()


# In[42]:


df_4.loc[5] = [4.9, 1.4, 1.3, "setosa"]


# In[43]:


df_4.head()


# Vamos a borrar una columna con la función drop
# 
# ![image.png](attachment:image.png)

# In[44]:


df_4.drop("Species", axis=1)


# Vamos a borrar una fila
# 
# ![image.png](attachment:image.png)

# In[45]:


df_4.drop(2, axis = 0)


# Podriamos querer concatenar dos dataframes lo podemos hacer con la función concat

# In[49]:


pd.concat([df_4,df_4],ignore_index=True)


# Lo que podemos ver que nos conservó los índices, esto podemos corregirlo pasandole como parámetro ignore_index=True
# 
# pd.concat([df_4,df_4], ignore_index=True)

# Reestructurar una tabla usando la función pivot

# In[50]:


df = pd.read_excel("Clima.xlsx")


# In[51]:


df.head(10)


# In[52]:


df.pivot(index = "Ciudad", columns= "Día")


# Vamos a usar la función melt para cambiar el formato de la tabla

# In[53]:


clima_2 = pd.read_excel("Clima_2.xlsx")


# In[54]:


clima_2.head()


# In[55]:


pd.melt(clima_2, id_vars = ['Día'], var_name = ["Cuidad"], value_name = "Temperatura")


# In[ ]:





# In[ ]:




