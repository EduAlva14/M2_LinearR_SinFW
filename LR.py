#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np             # Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression  # Paquete Regresión linear


# In[20]:


df = pd.read_csv('Salaries.csv')          # Lectura datos
df_bi = df[['YearsExperience', 'Salary']]

df_bi.columns = ['Years', 'Sal']
df_bi.head()


# In[21]:



sns.lmplot(x ="Years", y ="Sal", data = df_bi, order = 2, ci = None)    # Visualización datos


# In[22]:


X = np.array(df_bi['Years']).reshape(-1, 1)
y = np.array(df_bi['Sal']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) # Train y Test separación

model = LinearRegression()  # Llamamos a la librería

model.fit(X_train, y_train)  # Train


print(model.score(X_test, y_test))


# In[23]:


y_pred = model.predict(X_test)     # Predicción
print(model.score(X_test, y_test)) # Score

plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()  # Visualización LR


# In[ ]:




