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


# In[2]:


iris_data = sklearn.datasets.load_iris()
print(iris_data.DESCR)


# In[3]:


X = iris_data.data
y = iris_data.target

iris = pd.DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
iris['Species'] = y
iris.Species.astype(int, inplace=True)
iris.sample(5)


# In[4]:


def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolour'
    else:
        return 'Virginica'

iris['Species'] = iris['Species'].apply(flower)
iris.sample(5)


# In[6]:


sns.pairplot(iris,hue='Species',size=2)


# In[8]:


sns.factorplot('Petal Length',data=iris,hue='Species', kind='count', size=8)


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[11]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[15]:


from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, average='macro')

print("Precisão: ", precision)


# In[16]:


from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred, average='macro')
print("Recall: ", recall)


# In[17]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia: ", accuracy)


# In[18]:


from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)

print("Matriz de Confusão: ", matrix)


# In[ ]:




