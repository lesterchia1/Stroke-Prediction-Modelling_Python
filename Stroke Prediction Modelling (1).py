#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
import plotly.express as px
import plotly
plotly.offline.init_notebook_mode(connected = True)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as tts,RandomizedSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,plot_confusion_matrix
#data loading


# In[6]:


st=pd.read_csv('C:/Users/User/Documents/upskill in data science/Self_learning/Python for Data Science/Project/healthcare-dataset-stroke-data.csv')
st.head()


# In[7]:


#checking number of rows and columns of dataset
st.shape


# In[8]:


st.info()


# In[9]:


st.isnull().sum()


# In[10]:


#creating a copy of original dataset for treating missing values
st_copy=st.copy(deep=True)


# In[11]:


st_copy['ever_married']=st_copy['ever_married'].replace({'Yes':1,'No':0})
st_copy=pd.get_dummies(st_copy,drop_first=True)


# In[18]:


plt.figure(figsize=(20,20))
sns.heatmap(st_copy.corr(),annot=True)


# In[13]:


st['gender'].value_counts(normalize=True)


# In[36]:


st.loc['gender']='Other'
st['gender'].unique()


# In[40]:


st['gender']=st['gender'].replace('Other','Female')
st['gender'].iloc[:]


# In[33]:


#dropping missing data 
st=st.dropna()


# In[34]:


#dropping unnecessary columns
st.drop(columns='id',inplace=True)


# In[35]:


#function to observe values in each categorical feature
def value_viz(feature,title):
    return px.pie(st,feature,title=title)


# In[36]:


value_viz('gender','Distribution Of Gender')


# In[38]:


value_viz('hypertension','Distribution of people with High Blood Pressure')


# In[39]:


value_viz('heart_disease','Distribution of People having Heart Disease')


# In[40]:


value_viz('ever_married','Distribution of people who are married')


# In[41]:


value_viz('work_type','Distribution of people\'s work type')


# In[42]:


value_viz('Residence_type','Distribution of where people live')


# In[43]:


value_viz('smoking_status','Distribution of people who smoke')


# In[44]:


value_viz('stroke','Distribution of people having stroke')


# In[45]:


plt.figure(figsize=(20,5))
sns.histplot(st['age'])
plt.xticks(range(0,100,10))
plt.title("Distribution of Age")


# In[46]:


plt.figure(figsize=(20,5))
sns.histplot(st['avg_glucose_level'])
plt.title('Distribution of Average Glucose Level')
plt.xticks(range(0,300,25))


# In[47]:


plt.figure(figsize=(20,5))
sns.histplot(st['bmi'])
plt.title('Distribution of Body Mass Index')
plt.xlabel('BMI in kg/m2')
plt.xticks(range(0,100,10))


# In[48]:


plt.figure(figsize=(20,5))
sns.boxplot(x='age',data=st)


# In[49]:


plt.figure(figsize=(20,5))
sns.boxplot(x='bmi',data=st)


# In[50]:


plt.figure(figsize=(20,5))
sns.boxplot(x='avg_glucose_level',data=st)


# In[51]:


#function to find outliers
def iqr_outliers(df):
    out=[]
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr
    for i in df:
        if i > Upper_tail or i < Lower_tail:
            out.append(i)
    return out


# In[52]:


d=iqr_outliers(st['bmi'])


# In[53]:


#finding minimum of outliers in bmi
d.sort()
d[0]


# In[54]:


e=iqr_outliers(st['avg_glucose_level'])


# In[55]:


#finding minimum of outliers in avg_glucose_level
e.sort()
e[0]


# In[56]:


#median imputation in bmi
med=st.bmi.median()
for i in st.bmi:
    if i>=47.6:
        st.bmi=st.bmi.replace(i,med)


# In[57]:


#median imputation in avg_glucose_level
med=st.avg_glucose_level.median()
for i in st.avg_glucose_level:
    if i>=168.68:
        st.avg_glucose_level=st.avg_glucose_level.replace(i,med)


# In[58]:


st.describe()


# In[59]:


#hard encoding of feature which have yes or no as values and rest of the values are one hot encoded
st['ever_married']=st['ever_married'].replace({'Yes':1,'No':0})
st=pd.get_dummies(st)


# In[60]:


st.head()


# In[61]:


#splitting the original dataset
y=st.stroke
X=st.drop('stroke',axis=1)
X_train_or,X_test_or,Y_train_or,Y_test_or=tts(X,y,test_size=0.25,random_state=27)


# In[62]:


#using standard scaler to scale training data and applying it to testing data
sc=StandardScaler()
X_train_scaled_or=sc.fit_transform(X_train_or)
X_test_scaled_or=sc.transform(X_test_or)


# In[88]:


rf=RandomForestClassifier(random_state=25)
rf.fit(X_train_or,Y_train_or)
pred=rf.predict(X_test_scaled_or)
plot_confusion_matrix(rf,X_test_scaled_or,Y_test_or,cmap=plt.cm.Blues,normalize='all')
print(classification_report(pred,Y_test_or))


# In[64]:


st['stroke'].value_counts()


# In[65]:


#using SMOTE to generate synthetic examples in target variables 
sm = SMOTE(random_state=27)
X, Y = sm.fit_resample(X, y)


# In[66]:


Y.value_counts()


# In[67]:


#splitting the transformed dataset
X_train,X_test,Y_train,Y_test=tts(X,Y,test_size=0.25,random_state=27)


# In[68]:


sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)


# In[69]:


#function to fit models
def model(model):
    mod=model
    mod.fit(X_train_scaled,Y_train)
    mod_pred=mod.predict(X_test_scaled)
    plot_confusion_matrix(mod,X_test_scaled,Y_test,cmap=plt.cm.Blues,normalize='all')
    print(classification_report(mod_pred,Y_test))


# In[70]:


model(LogisticRegression(random_state=25)) 


# In[90]:


model(DecisionTreeClassifier(random_state=25))


# In[72]:


model(KNeighborsClassifier())


# In[73]:


model(XGBClassifier(use_label_encoder=False,random_state=25))


# In[74]:


model(RandomForestClassifier(random_state=25)) 


# In[ ]:


#https://www.kaggle.com/tmchls/stroke-prediction-modelling
#INSIGHTS-

#Random Forest Classifier had outperformed other classification models.

#With precision=0.99,recall=0.96,f1-score=0.97 for classifying non-stroke cases and precision=0.96,recall=0.99,f1-score=0.97 for classifying stroke cases.

#As False Negative > False Positive for most of the models so recall factor is most important in this classification.

#Random Forest Classifier has correctly predicted 96% of the actual cases where people don't suffer a stroke and 99% of cases where people actually suffer a stroke.

#50% of the positive class data points were correctly classified by the model(True Positive).

#47% of the negative class data points were correctly classified by the model(True Negative).

#0.68% of the negative class data points were incorrectly classified as belonging to the positive class by the model
#(False Positive).
#
#2% of the positive class data points were incorrectly classified as belonging to the negative class by the model(False Negative).

