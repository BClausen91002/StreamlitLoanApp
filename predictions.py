#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# ![image-2.png](attachment:image-2.png)

# # Import Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Import the Dataset

# In[2]:


data=pd.read_csv('train.csv')


# # Summarizing the dataset

# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.describe()


# ## Categorical variables 

# In[9]:


categorical_variables=['Gender','Married','Dependents','Education','Self_Employed','Credit_History', 'Property_Area', 'Loan_Status']
print('Categorical variable are :\n')
for i in range (len(categorical_variables)):
    print(categorical_variables[i])


# # Married or not 

# # Is there some missing values in my dataset ?

# In[10]:


data.isnull().sum()


# # Replace the missing values

# Categorical variables

# In[11]:


data.Gender.fillna(data.Gender.mode()[0],inplace=True)
data.Married.fillna(data.Married.mode()[0],inplace=True)
data.Dependents.fillna(data.Dependents.mode()[0],inplace=True)
data.Self_Employed.fillna(data.Self_Employed.mode()[0],inplace=True)
data.Credit_History.fillna(data.Credit_History.mode()[0],inplace=True)


# Numerical variables

# In[12]:


data.LoanAmount.fillna(data.LoanAmount.median(), inplace=True)
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.median(), inplace=True)


# In[13]:


data.isnull().sum()


# no more missing values

# # Get_dummies for Categorical Variables

# In[14]:


dummies1=pd.get_dummies(data['Gender'])
dummies2=pd.get_dummies(data['Married'])
dummies2=dummies2.rename(columns = {'Yes':'Married_yes','No':'NotMarried'})
dummies3=pd.get_dummies(data['Dependents'])
dummies3=dummies3.rename(columns = {'0': 'Dependents_0','1': 'Dependents_1','2': 'Dependents_2','3+': 'Dependents_3+'})
dummies4=pd.get_dummies(data['Education'])
dummies5=pd.get_dummies(data['Self_Employed'])
dummies5=dummies5.rename(columns = {'Yes':'Self_Employed_yes','No':'NotSelf_Employed'})
dummies6=pd.get_dummies(data['Property_Area'])


# In[15]:


data= pd.concat([data,dummies1],axis=1)
data= pd.concat([data,dummies2],axis=1)
data= pd.concat([data,dummies3],axis=1)
data= pd.concat([data,dummies4],axis=1)
data= pd.concat([data,dummies5],axis=1)
data= pd.concat([data,dummies6],axis=1)


data=data.drop(['Loan_ID','Gender','Married','Property_Area','Dependents','Self_Employed','Education','NotMarried','NotSelf_Employed','Not Graduate','Female'] , axis=1)


# # Replace 'Y' and 'N' in the target variable to 1 and 0

# In[16]:


data['Loan_Status']= data['Loan_Status'].replace(['Y'],1)
data['Loan_Status']= data['Loan_Status'].replace(['N'],0)


# # My dataset after all the modifications

# In[17]:


data.head(10)


# # Splitting the data-set into Training and Test Set

# In[18]:


X=data.drop(columns='Loan_Status')
Y=pd.DataFrame(data['Loan_Status'])


# In[19]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=3)


# In[20]:


data.info()


# In[21]:


from pycaret.classification import *


# In[ ]:


data


# # Try the Classifiers

# ![image.png](attachment:image.png)

# In[ ]:


classifier = RandomForestClassifier(n_estimators=1000,max_features=15,max_depth=5,bootstrap=True)
classifier.fit(X_train,Y_train)
predictions = classifier.predict(X_test)
accuracyScores = accuracy_score(predictions, Y_test)
print(accuracyScores)


# In[ ]:


predictions


# In[ ]:


import pickle


# In[ ]:


filename = 'RF.sav'
pickle.dump(classifier, open(filename, 'wb'))

