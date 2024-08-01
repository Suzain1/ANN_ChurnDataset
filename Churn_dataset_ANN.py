#!/usr/bin/env python
# coding: utf-8

# #**Importing Necessary Libraries**

# In[1]:


import numpy as np
import pandas as pd


# #**Loading the Churn Dataset**
# 
# This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.
# 
# **Binary flag 1 if the customer closed account with bank and 0 if the customer is retained.**

# In[2]:


churn_data = pd.read_csv("D:\\Suzain\\Churn_Modelling.csv")
churn_data.head(5)


# #**Accessing the Column Names in the Dataset**

# In[3]:


churn_data.columns


# #**Setting Column as a Index**

# In[4]:


churn_data = churn_data.set_index('RowNumber')
churn_data.head()


# #**Finding the Shape of the Dataset**

# In[5]:


churn_data.shape


# In[6]:


churn_data.info()


# #**Checking Missing Values**

# In[7]:


churn_data.isna().sum()


# In[8]:


churn_data.nunique()


# In[9]:


churn_data.drop(['CustomerId','Surname'],axis=1,inplace=True)


# In[10]:


churn_data.head()


# In[11]:


churn_data.shape


# # **Label Encoding of Categorical Variables**
# 
# Label Encoding means converting categorical features into numerical values. So that they can be fitted by machine learning models which only take numerical data.
# 
# **Example:** Suppose we have a column Height in some dataset that has elements as Tall, Medium, and short. To convert this categorical column into a numerical column we will apply label encoding to this column. After applying label encoding, the Height column is converted into a numerical column having elements 0,1, and 2 where 0 is the label for tall, 1 is the label for medium, and 2 is the label for short height.

# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
churn_data[['Geography', 'Gender']] = churn_data[['Geography', 'Gender']].apply(le.fit_transform)


# In[21]:


churn_data.head()


# #**Seperating Label from Data**

# In[22]:


y = churn_data.Exited
X = churn_data.drop(['Exited'],axis=1)


# In[23]:


X.columns


# In[24]:


y


# #**Splitting the Data into Training and Testing**

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)


# In[26]:


print("Shape of the X_train", X_train.shape)
print("Shape of the X_test", X_test.shape)
print("Shape of the y_train", y_train.shape)
print("Shape of the y_test", y_test.shape)


# #**Need for Normalization**
# For example, consider a data set containing two features, age(x1), and income(x2). Where age ranges from 0–100, while income ranges from 0–20,000 and higher. Income is about 1,000 times larger than age and ranges from 20,000–500,000. So, these two features are in very different ranges. When we do further analysis, like multivariate linear regression, for example, the attributed income will intrinsically influence the result more due to its larger value. But this doesn’t necessarily mean it is more important as a predictor.

# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# #**Building the ANN Model**

# In[28]:


from keras.models import Sequential
from keras.layers import Dense


# In[29]:


classifier = Sequential()
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# # **Compiling and Fitting the Model**

# In[30]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose = 1)


# #**Testing the Model**

# In[31]:


score, acc = classifier.evaluate(X_train, y_train,
                            batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print('*'*20)
score, acc = classifier.evaluate(X_test, y_test,
                            batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc)


# In[32]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
target_names = ['Retained', 'Closed']
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


p = sns.heatmap(pd.DataFrame(cm), annot=True, xticklabels=target_names, yticklabels=target_names, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

