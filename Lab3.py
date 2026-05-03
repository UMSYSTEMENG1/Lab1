#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[3]:


df = pd.read_csv("WorldEnergy.csv")
df.head()


# In[4]:


df = df[df['country'].isin(['China', 'France'])]
df['country'].unique()


# In[5]:


def year_group(y):
    if 1995 <= y <= 2004:
        return 'Early'
    elif 2005 <= y <= 2014:
        return 'Mid'
    elif 2015 <= y <= 2024:
        return 'Recent'

df['year_group'] = df['year'].apply(year_group)
df[['year', 'year_group']].head()


# In[6]:


df = df.dropna(subset=['renewables_electricity'])
df.isna().sum()


# In[7]:


model = ols(
    'renewables_electricity ~ C(country) + C(year_group) + C(country):C(year_group)', 
    data=df
).fit()


# In[8]:


anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# In[9]:


print("=== Two-way ANOVA Result ===")
print(anova_table)


# In[10]:


get_ipython().system('jupyter nbconvert --to script Lab3.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:




