#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %reset    #limpiar variables


# In[19]:


#Librerías


import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# In[20]:


class LinearR() :

    def __init__( self, learn_r, itera ) :

        self.learn_r = learn_r

        self.itera = itera
        return None

    
    def fit( self, X, Y ) :     # Model training
        
        self.m, self.n = X.shape
        
        
        
        self.W = np.zeros( self.n )  # Inicialización variables pesos
                                     # y = w*x + b
        self.b = 0
        
        self.X = X
        
        self.Y = Y
        
        
        
        for i in range( self.itera ) :    # Gradiente descendente
        
            self.update_w()
            
        return self

   
    

    def update_w( self ):       # Ajuste pesos: gradiente descendente
            
        Y_pred = self.predict( self.X )
        
    
        dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) / self.m    # Cálculo gradiente
    
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m
        

        self.W = self.W - self.learn_r * dW     # Ajuste pesos

        self.b = self.b - self.learn_r * db
        
        return self


    
    
    def predict( self, X ) :     # Función H
    
        return X.dot(self.W) + self.b


# In[21]:


df = pd.read_csv( "Salaries.csv" )    # Lectura datos

X = df.iloc[:,:-1].values

Y = df.iloc[:,1].values



X_train, X_test, Y_train, Y_test = train_test_split(    # Train & test set
X, Y, test_size = 1/3, random_state = 0 )

plt.scatter( X_train, Y_train, color = 'blue' )



model = LinearR( learn_r = 0.01,itera = 1000 )    # Training
model.fit(X_train,Y_train)


# In[22]:


Y_pred = model.predict(X_test)   # Predicción


# In[23]:


print( "Predicción ", np.round( Y_pred[:3], 2 ) )  # Resultados
    
print( "Reales ", Y_test[:3] )
    
print( "W ", round( model.W[0], 2 ) )
    
print( "b ", round( model.b, 2 ) )
    

    
plt.scatter( X_test, Y_test, color = 'red' )   # Figuras de visualización
    
plt.plot( X_test, Y_pred, color = 'yellow' )
    
plt.title( 'Salario vs Experiencia' )
    
plt.xlabel( 'Experiencia(años)' )
    
plt.ylabel( 'Salario' )
    
plt.show()
    


# In[ ]:





# In[ ]:




