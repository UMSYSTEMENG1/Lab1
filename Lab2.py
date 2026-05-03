#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd              
import numpy as np               
import matplotlib.pyplot as plt  
import seaborn as sns            


# In[2]:


df = pd.read_csv("WorldEnergy.csv")

df.head()


# In[3]:


df = df[df["country"] == "Bolivia"]

df.head()


# In[4]:


df = df[[
    "country", "year", "population", "gdp",
    "energy_cons_change_pct", "energy_cons_change_twh",
    "energy_per_capita", "energy_per_gdp"
]]


# In[5]:


df = df[(df["year"] >= 2000) & (df["year"] <= 2024)]

df = df.sort_values("year").reset_index(drop=True)

print(df.head())


print("The earliest year:", df["year"].min())
print("The maximum year:", df["year"].max())


# In[6]:


df = df.sort_values("year")

df.head()


# In[7]:


variables = [
    "population", "gdp", "energy_cons_change_pct",
    "energy_cons_change_twh", "energy_per_capita", "energy_per_gdp"
]

for var in variables:
    plt.figure()
    plt.scatter(df["year"], df[var])
    plt.title(f"{var} Over Time (Outlier Detection)")
    plt.xlabel("Year")
    plt.ylabel(var)
    plt.show()


# In[8]:


print("The situation of missing values：")
print(df.isnull().sum())


# In[9]:


df["gdp"] = df["gdp"].interpolate(method='linear')

mask = df["year"] >= 2020

df.loc[mask, "gdp"] = df.loc[mask, "gdp"].interpolate(method="linear")


# In[10]:


avg_pct = df.loc[df["year"].between(2020, 2023), "energy_cons_change_pct"].mean()
df.loc[df["year"] == 2024, "energy_cons_change_pct"] = avg_pct

for i in range(1, len(df)):

    # energy_per_capita
    if pd.isna(df.loc[i, "energy_per_capita"]):
        df.loc[i, "energy_per_capita"] = (
            df.loc[i-1, "energy_per_capita"]
            * df.loc[i-1, "population"]
            * (1 + df.loc[i, "energy_cons_change_pct"] / 100)
            / df.loc[i, "population"]
        )

    # energy_cons_change_twh
    if pd.isna(df.loc[i, "energy_cons_change_twh"]):
        df.loc[i, "energy_cons_change_twh"] = (
            df.loc[i-1, "energy_per_capita"]
            * df.loc[i-1, "population"]
            * (df.loc[i, "energy_cons_change_pct"] / 100/ 1e9 )
        )

    # energy_per_gdp
    if pd.isna(df.loc[i, "energy_per_gdp"]):
        df.loc[i, "energy_per_gdp"] = (
            df.loc[i, "population"]
            * df.loc[i, "energy_per_capita"]
            / df.loc[i, "gdp"]
        )


# In[11]:


print("Post-processing missing values：")
print(df.isnull().sum())


# In[12]:


# Population
plt.figure()
plt.plot(df["year"], df["population"])
plt.title("Population Trend (Bolivia)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.show()


# GDP
plt.figure()
plt.plot(df["year"], df["gdp"])
plt.title("GDP Trend (Bolivia)")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()


# Energy Consumption Change %
plt.figure()
plt.plot(df["year"], df["energy_cons_change_pct"])
plt.title("Energy Consumption Change (%)")
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.show()


# Energy Consumption Change (TWh)
plt.figure()
plt.plot(df["year"], df["energy_cons_change_twh"])
plt.title("Energy Consumption Change (TWh)")
plt.xlabel("Year")
plt.ylabel("TWh")
plt.show()


# Energy per Capita
plt.figure()
plt.plot(df["year"], df["energy_per_capita"])
plt.title("Energy per Capita")
plt.xlabel("Year")
plt.ylabel("Energy")
plt.show()


# Energy per GDP
plt.figure()
plt.plot(df["year"], df["energy_per_gdp"])
plt.title("Energy per GDP")
plt.xlabel("Year")
plt.ylabel("Energy Efficiency")
plt.show()


# In[13]:


fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# Population
axes[0,0].plot(df["year"], df["population"])
axes[0,0].set_title("Population")

# GDP
axes[0,1].plot(df["year"], df["gdp"])
axes[0,1].set_title("GDP")

# Energy %
axes[0,2].plot(df["year"], df["energy_cons_change_pct"])
axes[0,2].set_title("Energy Change %")

# Energy TWh
axes[1,0].plot(df["year"], df["energy_cons_change_twh"])
axes[1,0].set_title("Energy Change TWh")

# Per Capita
axes[1,1].plot(df["year"], df["energy_per_capita"])
axes[1,1].set_title("Energy per Capita")

# Per GDP
axes[1,2].plot(df["year"], df["energy_per_gdp"])
axes[1,2].set_title("Energy per GDP")

plt.tight_layout()
plt.show()


# In[14]:


plt.scatter(df["population"], df["energy_per_capita"])
plt.title("Population vs Energy per Capita")
plt.xlabel("Population")
plt.ylabel("Energy per Capita")
plt.show()


# GDP vs Energy per Capita

# In[15]:


plt.figure()
plt.scatter(df["gdp"], df["energy_per_capita"])

plt.title("GDP vs Energy per Capita (Bolivia)")
plt.xlabel("GDP")
plt.ylabel("Energy per Capita")

plt.show()


# In[16]:


sns.regplot(x=df["gdp"], y=df["energy_per_capita"])

plt.title("GDP vs Energy per Capita (Trend)")
plt.show()


# In[17]:


corr = df["gdp"].corr(df["energy_per_capita"])
print("Correlation:", corr)


# In[18]:


corr_matrix = df[[
    "population", "gdp",
    "energy_cons_change_pct", "energy_cons_change_twh",
    "energy_per_capita", "energy_per_gdp"
]].corr()

print(corr_matrix)


# In[19]:


plt.figure(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,        
    cmap="coolwarm",   
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap (Bolivia)")
plt.show()


# In[20]:


get_ipython().system('jupyter nbconvert --to script Lab2.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




