#!/usr/bin/env python
# coding: utf-8

# ## Dataset Titanic
# 
# Para saber mais sobre os dados acesse o desafio na plataforma Kaggle no link abaixo:
# 
# [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

# In[1]:


import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

titanic_df = sns.load_dataset('titanic')

titanic_df.head()


# In[2]:


titanic_df.info()


# In[3]:


sns.factorplot('sex', data=titanic_df, kind='count')


# In[4]:


sns.barplot(y='age', x='pclass', data=titanic_df)


# In[5]:


sns.factorplot('pclass', data=titanic_df, kind='count')


# In[6]:


titanic_df['age'].hist(bins=70)


# In[7]:


sns.factorplot('sex', data=titanic_df, hue='pclass', kind='count')


# In[8]:


def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex


# In[9]:


titanic_df['person'] = titanic_df[['age', 'sex']].apply(male_female_child, axis=1)
titanic_df.head()


# In[10]:


sns.factorplot('pclass', data=titanic_df, hue='person', kind='count')


# In[13]:


titanic_df['survivor'] = titanic_df.survived.map({0:'no', 1:'yes'})
sns.factorplot('survivor', data=titanic_df, kind='count', palette='Set1')


# In[14]:


sns.factorplot('pclass', data=titanic_df, hue='survivor', kind='count')


# In[15]:


sns.factorplot('pclass', 'survived', hue='person', data=titanic_df)


# In[16]:


sns.lmplot('age', 'survived', hue='pclass', data=titanic_df)


# In[17]:


sns.lmplot('age', 'survived', hue='sex', data=titanic_df)


# In[20]:


titanic_df.info()


# In[27]:


data = titanic_df[['survived', 'pclass', 'sibsp', 'parch', 'fare', 'adult_male']]
data.head()


# In[28]:


X = data.drop('survived', axis=1)
y = data.survived


# In[29]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)


# In[31]:


model.score(X, y)


# In[32]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[33]:


model.score(X_test, y_test)


# In[ ]:




