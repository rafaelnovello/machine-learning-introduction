#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette="muted")
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
from sklearn import datasets
import sklearn.model_selection
from sklearn.linear_model import LinearRegression


# In[2]:


boston = datasets.load_boston()
print(boston.DESCR)


# In[3]:


plt.hist(boston.target,bins=50)
plt.xlabel('Preço em $1000s')
plt.ylabel('Num Casas')


# In[4]:


plt.scatter(boston.data[:,5],boston.target)
plt.ylabel('Preço em $1000s')
plt.xlabel('Num Quartos')


# In[5]:


boston_df = pd.DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['Price'] = boston.target
boston_df.head()


# In[6]:


sns.lmplot('RM','Price',data = boston_df)


# In[7]:


X = boston_df.drop('Price', 1)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, boston_df.Price, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[8]:


lreg = LinearRegression()
lreg.fit(X_train,Y_train)

pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

print("MSE com Y_train: %.2f" % sklearn.metrics.mean_squared_error(Y_train, pred_train))

print("MSE com X_test e Y_test: %.2f" % sklearn.metrics.mean_squared_error(Y_test, pred_test))


# Para entender o MSE (Erro Quadrado Médio) [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
# 
# ![](http://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/220px-Linear_least_squares_example2.svg.png)
# 
# > $ MSE = \frac{1}{n}\sum_{i=1}^n(Y_i-\hat{Y_i})^2$
# 
# Onde:
# - *n* é a quantidade de amostras
# - *Y* é a variavel target real
# - $\hat{Y}$ é o target predito pelo modelo
# 
# Uma métrica mais clara para os casos de regressão pé o [R2 score](http://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)

# In[9]:


from sklearn.metrics import r2_score

print("R2 score no conjunto de testes: %.2f" % r2_score(Y_test, pred_test))

print("R2 score no conjunto de treinamento: %.2f" % r2_score(Y_train, pred_train))


# In[ ]:




