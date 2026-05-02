#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

from IPython.display import Image, display

from datasets import load_dataset


# In[2]:


URL = "https://huggingface.co/datasets?search=autonomous%20driving"
headers = {"User-Agent": "Mozilla/5.0"}

page = requests.get(URL, headers=headers)
soup = BeautifulSoup(page.content, "html.parser")

articles = soup.find_all("article")

dataset_list = []

for a in articles[:2]:
    
    title_tag = a.find("h4")
    name = title_tag.text.strip() if title_tag else "N/A"
    
    link_tag = a.find("a")
    link = "https://huggingface.co" + link_tag["href"]
    
    dataset_list.append({
        "Dataset Name": name,
        "Link": link
    })

df = pd.DataFrame(dataset_list)

print("Dataset List:")
display(df)


# In[3]:


def get_dataset_details(url):
    
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    
    desc_tag = soup.find("p")
    description = desc_tag.text.strip() if desc_tag else "N/A"
    
    tags = []
    for t in soup.find_all("span"):
        text = t.text.strip()
        if text != "":
            tags.append(text)
    
    return description, tags


# In[4]:


def get_dataset_rows(dataset_name):
    
    api_url = "https://datasets-server.huggingface.co/rows"
    
    params = {
        "dataset": dataset_name,
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 10
    }
    
    response = requests.get(api_url, params=params)
    data = response.json()
    
    if "rows" not in data:
        return None
    
    return [item["row"] for item in data["rows"]]


# In[5]:


def process_text(rows, dataset_name):
    
    folder = dataset_name.replace("/", "_")
    os.makedirs(folder, exist_ok=True)
    
    print("Saving text data to:", folder)
    
    for i, row in enumerate(rows):
        print("Processing row", i)
        
        with open(f"{folder}/data.txt", "a", encoding="utf-8") as f:
            for k, v in row.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
    
    print("Text data saved")


# In[6]:


def process_image(dataset_name):
    
    print("Loading image dataset (streaming mode)...")
    
    try:
        ds = load_dataset(dataset_name, streaming=True)
    except:
        print("Unable to load dataset")
        return
    
    folder = dataset_name.replace("/", "_")
    os.makedirs(folder, exist_ok=True)
    
    count = 0
    
    for i, row in enumerate(ds["train"]):
        
        if "image" in row:
            
            img = row["image"]
            
            path = f"{folder}/img_{i}.jpg"
            img.save(path)
            
            img.show()
            
            count += 1
        
        if count == 2:
            break
    
    print("Number of images saved:", count)


# In[ ]:


for dataset in dataset_list:
    
    print("\n-----------------------------------")
    print("Dataset Name:", dataset["Dataset Name"])
    print("Dataset URL:", dataset["Link"])
    
    desc, tags = get_dataset_details(dataset["Link"])
    
    print("\nDescription:")
    print(desc[:150])
    
    print("\nTags:")
    print(tags[:8])
    
    rows = get_dataset_rows(dataset["Dataset Name"])
    
    if rows is not None:
        
        print("\nDataset Type: Text Data")
        print("Rows fetched:", len(rows))
        
        process_text(rows, dataset["Dataset Name"])
    
    else:
        
        print("\nDataset Type: Image Data")
        
        process_image(dataset["Dataset Name"])
    
    print("\nFinished processing\n")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Lab1.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




