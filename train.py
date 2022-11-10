#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\acer\Downloads\archive\train.csv')

X = df
y = df.price_range.values
del X['price_range']


output_file = f'midterm_model.bin'
rf = RandomForestClassifier(n_estimators=50,
                            max_depth=10,
                            random_state=1).fit(X, y)

with open(output_file, 'wb') as f_out:
    pickle.dump((df, rf), f_out)


# In[ ]:




