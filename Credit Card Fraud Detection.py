#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd

credit_card=pd.read_csv("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\creditcard.csv")
df=pd.DataFrame(credit_card)
df.head()


# In[46]:


df.isnull().sum()


# In[47]:


pd.value_counts(df["Class"])


# In[48]:


df.info()


# In[49]:


df.columns


# In[50]:


df.shape


# In[51]:


import seaborn as sns
sns.countplot(x="Class",hue="Class",data=credit_card)


# In[52]:


features=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

x=df[features]
y=df["Class"]


# In[53]:


from imblearn.over_sampling import RandomOverSampler

oversampler=RandomOverSampler()
x_res,y_res=oversampler.fit_resample(x,y)


# In[54]:


print(x.shape)
print(y.shape)


# In[55]:


print(x_res.shape)
print(y_res.shape)


# In[56]:


from collections import Counter


# In[57]:


print(f'before over sampling: {Counter(y)}')
print(f'after over sampling: {Counter(y_res)}')


# In[58]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.3,random_state=20,stratify=y_res)


# In[59]:


print(x_train.shape)
print(x_test.shape)


# In[60]:


print(y_train.shape)
print(y_test.shape)


# In[61]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(random_state=19)
classifier.fit(x_train,y_train)


# In[62]:


y_pred=classifier.predict(x_test)
print("y_pred:",y_pred)


# In[63]:


n_errors=(y_pred!=y_test).sum()
print("n_errors:",n_errors)


# In[64]:


from sklearn.metrics import accuracy_score

score=accuracy_score(y_pred,y_test)
print("accuracy_score:",score)


# In[65]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print("confusion matrix")
print(cm)


# In[66]:


sns.heatmap(cm,annot=True)


# In[67]:


from sklearn.metrics import classification_report

cr = classification_report(y_pred, y_test)
print("Classification report:\n", cr)


# In[ ]:





# In[23]:


import pandas as pd

credit_card=pd.read_csv("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\creditcard.csv")
df=pd.DataFrame(credit_card)

features=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

x=df[features]
y=df["Class"]

from imblearn.over_sampling import RandomOverSampler

oversampler=RandomOverSampler()
x_res,y_res=oversampler.fit_resample(x,y)


from collections import Counter

print(f'before under sampling: {Counter(y)}')
print(f'after under sampling: {Counter(y_res)}')


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.3,random_state=20)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(random_state=19)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print("y_pred:",y_pred)

from sklearn.metrics import accuracy_score

score=accuracy_score(y_pred,y_test)
print("accuracy_score:",score)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print("confusion matrix")
print(cm)




# In[ ]:





# In[44]:


import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
credit_card = pd.read_csv("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\creditcard.csv")
df = pd.DataFrame(credit_card)

# Define features and target variable
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
x = df[features]
y = df["Class"]

# Perform Random Oversampling
oversampler = RandomOverSampler()
x_res, y_res = oversampler.fit_resample(x, y)

# Print class distribution before and after oversampling
print(f'Before oversampling: {Counter(y)}')
print(f'After oversampling: {Counter(y_res)}')

# Split the data into training and testing sets with stratified sampling
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=20, stratify=y_res)

# Create a Decision Tree classifier and fit it to the training data
classifier = DecisionTreeClassifier(random_state=19)
classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(x_test)
print("Predicted labels (y_pred):", y_pred)

# Calculate and print the accuracy score
score = accuracy_score(y_pred, y_test)
print("Accuracy Score:", score)

# Calculate and print the confusion matrix
cm = confusion_matrix(y_pred, y_test)
print("Confusion Matrix:\n", cm)


# In[ ]:




