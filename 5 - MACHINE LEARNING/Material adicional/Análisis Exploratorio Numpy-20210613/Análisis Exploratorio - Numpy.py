#!/usr/bin/env python
# coding: utf-8

# **¿Qué Librería vamos a usar?**
# * Numpy
# 
# **¿Cómo Instalamos la Librería?**
# * Pip install numpy (Línea de Comando CMD)
# * conda install -c anaconda numpy (Anaconda)
# 
# **¿Cómo Chequeo?**
# 
# * CONDA LIST (Anaconda)
# * PIP LIST (Línea de Comando)
# 
# 

# In[ ]:


import numpy
import numpy as np
np.zeros(5)		#crea una matriz de ceros


# In[2]:


np.zeros(5, dtype='int8')	# especifico el tipo de datos


# In[3]:


np.zeros((4, 3)) # crea una matriz de ceros de 4x3


# In[4]:


np.identity(3)


# In[5]:


np.ones((4,3))


# In[6]:


np.ones(5)


# In[9]:


x=np.ones(5, dtype='int8')


# In[12]:


print(x)


# In[10]:


z =  3**x


# In[11]:


print(z)


# In[ ]:




