########## LIBRERÍAS A UTILIZAR ##########  #Se importan la librerias a utilizar 
from sklearn import datasets 
########## PREPARAR LA DATA ##########  #Importamos los datos de la misma librería de scikit-learn 
 
dataset = datasets.load_breast_cancer() 
print(dataset) 
########## ENTENDIMIENTO DE LA DATA ########## 
 
#Verifico la información contenida en el dataset 
print('Información en el dataset:') 
print(dataset.keys()) 
print() 
#Verifico las características del dataset 
print('Características del dataset:') 
print(dataset.DESCR) 
#Seleccionamos todas las columnas 
X = dataset.data 
#Defino los datos correspondientes a las etiquetas 
y = dataset.target 
 
########## IMPLEMENTACIÓN DE NAIVE BAYES ########## 
from sklearn.model_selection import train_test_split 
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
#Defino el algoritmo a utilizar 
#Naive Bayes 
from sklearn.naive_bayes import GaussianNB 
algoritmo = GaussianNB() 

#Entreno el modelo 
algoritmo.fit(X_train, y_train) 

#Realizo una predicción 
y_pred = algoritmo.predict(X_test) 

#Verifico la matriz de Confusión 
from sklearn.metrics import confusion_matrix 
matriz = confusion_matrix(y_test, y_pred) 
print('Matriz de Confusión:') 
print(matriz) 

#Calculo la precisión del modelo 
from sklearn.metrics import precision_score 
precision = precision_score(y_test, y_pred) 
print('Precisión del modelo:') 
print(precision)


# Matriz de Confusión:
# [[44  8]
#  [ 3 59]]
# Precisión del modelo:
# 0.8805970149253731