#!/usr/bin/env python
# coding: utf-8

# In[6]:


#pip install pandas-profiling[notebook]
#conda install -c conda-forge pandas-profiling
#https://pypi.org/project/pandas-profiling/

# Importación de Librerías
from pathlib import Path

import pandas as pd
from ipywidgets import widgets

# Librería de Pandas Profiling
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file


# In[8]:


# DataSet lo tomo de una URL - DataSet Titanic
file_name = cache_file(
    "titanic.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
)
df = pd.read_csv(file_name)


# In[11]:


# Generamos el Reporte de Profiling
profile = ProfileReport(df, title="Titanic Dataset", html={'style': {'full_width': True}}, sort="None")


# In[12]:


# Widget para la interfase de Notebook
#profile.to_widgets()


# In[13]:


# HTML Report
profile


# In[ ]:




