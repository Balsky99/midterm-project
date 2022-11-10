#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text


# In[2]:


df = pd.read_csv(r'C:\Users\acer\Downloads\archive\train.csv')


# In[3]:


df.head(15)


# In[4]:


df.price_range.unique()


# In[5]:


df


# In[48]:


#EDA


# In[17]:


#check missing value
df.isnull().sum()


# In[50]:


#there are not missing value in dataset for each column


# In[18]:


df.describe()


# # description of price_range in dataset
# 0 ---->low 
# 1 ---->medium
# 0 ---->high
# 1 ---->very high

# In[19]:


#check distribution of target value

val = df.price_range.value_counts()
label = ['low','medium','high', 'very high']

plt.pie(val, labels = label, startangle=90, autopct='%.1f%%')
plt.show()


# As you can see, distribution of target value is similar for each value, its means that the dataset is a balanced data

# In[54]:


#check several features with target


# In[20]:


#correlation heatmap
sns.heatmap(df.corr())


# In[ ]:


#EDA use kdeplot for check data distribution for each features


# In[ ]:


#features int_memory vs price_range


# In[21]:


plt.figure(figsize=(8,6))

sns.kdeplot(df.loc[df['price_range']==0, 'int_memory'], label='Low')
sns.kdeplot(df.loc[df['price_range']==1, 'int_memory'], label='Medium')
sns.kdeplot(df.loc[df['price_range']==2, 'int_memory'], label='High')
sns.kdeplot(df.loc[df['price_range']==3, 'int_memory'], label='Very High')

plt.xlabel('price range')
plt.ylabel('density')
plt.legend()


# In[ ]:


#features battery_power vs price_range


# In[22]:


plt.figure(figsize=(8,6))
sns.kdeplot(df.loc[df['price_range']==0, 'battery_power'], label='Low')
sns.kdeplot(df.loc[df['price_range']==1, 'battery_power'], label='Medium')
sns.kdeplot(df.loc[df['price_range']==2, 'battery_power'], label='High')
sns.kdeplot(df.loc[df['price_range']==3, 'battery_power'], label='Very High')

plt.xlabel('price range')
plt.ylabel('density')
plt.legend()


# In[ ]:


#features ram vs price_range


# In[23]:


plt.figure(figsize=(8,6))
sns.kdeplot(df.loc[df['price_range']==0, 'ram'], label='Low')
sns.kdeplot(df.loc[df['price_range']==1, 'ram'], label='Medium')
sns.kdeplot(df.loc[df['price_range']==2, 'ram'], label='High')
sns.kdeplot(df.loc[df['price_range']==3, 'ram'], label='Very High')

plt.xlabel('price range')
plt.ylabel('density')
plt.legend()


# In[ ]:





# based on kdeplot above
# some features like ram, battery power, internal memory is high, price range will increase also.

# In[24]:


cs_em = pd.crosstab(df['dual_sim'], df['price_range'])
cs_em = cs_em.div(cs_em.sum())
ax = cs_em.T.plot(kind='bar', stacked=False, rot=1, figsize=(12,8), title='Dual sim across price range')


# total mobile phone is approximately similar both dual sim or not. so this features is not main features to define price range for mobile phone

# based on kdeplot above, more px width will increase price rangebased

# In[ ]:





# In[ ]:


#train the model


# In[25]:


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)


# In[26]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[27]:


y_train = df_train['price_range'].values
y_val = df_val['price_range'].values
y_test = df_test['price_range'].values

del df_train['price_range']
del df_val['price_range']
del df_test['price_range']


# In[ ]:


#Decision Tree


# In[32]:


scores = []

for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s, criterion='gini')
        dt.fit(df_train, y_train)

        y_pred = dt.predict(df_val)#[:, 1]
        
        from sklearn import preprocessing

        lb = preprocessing.LabelBinarizer()
        lb.fit(y_val)
        y_val = lb.transform(y_val)
        y_pred = lb.transform(y_pred)
        
        au = roc_auc_score(y_val, y_pred, multi_class='ovo')
        
        scores.append((depth, s, au))


# In[33]:


columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores


# In[34]:


df_scores.sort_values(by=['auc'])


# In[ ]:


#Random Forest


# In[30]:


from sklearn.ensemble import RandomForestClassifier

scores = []

for d in [5, 10, 15]:
    for n in [10,20,30,40,50]:
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(df_train, y_train)

        y_pred = rf.predict(df_val)#[:, 1]
        
        from sklearn import preprocessing

        lb = preprocessing.LabelBinarizer()
        lb.fit(y_val)
        y_val = lb.transform(y_val)
        y_pred = lb.transform(y_pred)
        
        
        auc = roc_auc_score(y_val, y_pred,multi_class='ovo')

        scores.append((d, n, auc))


# In[31]:


columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores


# In[ ]:


#Select the best model


# In[ ]:


from sklearn import preprocessing


# In[35]:


dt = DecisionTreeClassifier(max_depth=7, min_samples_leaf=1)
dt.fit(df_train, y_train)

y_pred = rf.predict(df_val)#[:, 1]
        
lb = preprocessing.LabelBinarizer()
lb.fit(y_val)
y_val = lb.transform(y_val)
y_pred = lb.transform(y_pred)
roc_auc_score(y_val, y_pred,multi_class='ovo')


# In[37]:


rf = RandomForestClassifier(n_estimators=50,
                            max_depth=10,
                            random_state=1)
rf.fit(df_train, y_train)
y_pred = rf.predict(df_val)#[:, 1]
        
lb = preprocessing.LabelBinarizer()
lb.fit(y_val)
y_val = lb.transform(y_val)
y_pred = lb.transform(y_pred)
roc_auc_score(y_val, y_pred,multi_class='ovo')


# In[39]:


rf.feature_importances_


# In[44]:


#feature importance analysis
feature_scores = pd.Series(rf.feature_importances_, index=df_train.columns).sort_values(ascending=False)
feature_scores


# In[45]:


#as you can see, The feature importance that correlated with the target is RAM.
#because feature_scores is highest. This features also have proved by kdeplot analysis before


# In[ ]:




