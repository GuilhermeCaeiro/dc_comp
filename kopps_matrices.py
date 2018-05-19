
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import time
from datetime import datetime
from Wisard import Wisard


# In[2]:


math_science_data = pd.read_csv("matriz_Math_Science.csv", sep=';')
health_sports_data = pd.read_csv("matriz_Health_Sports.csv", sep=';')

train_data = pd.read_csv("train.csv", sep=',')
#test_data = pd.read_csv(test_file_path, sep=',')
resources_data = pd.read_csv("resources.csv", sep=',')


# In[3]:


math_science_data.head()


# In[4]:


health_sports_data.head()


# In[5]:


print(len(math_science_data), len(health_sports_data))
print(len(math_science_data["id"].unique()))


# In[6]:


concatenated = pd.concat([math_science_data, health_sports_data], ignore_index=True)


# In[7]:


del math_science_data
del health_sports_data


# In[8]:


concatenated.head()


# In[9]:


concatenated = concatenated.fillna(int(0))


# In[10]:


concatenated.head()


# In[12]:


#len()


# In[13]:


expected_values = concatenated["approved"]
concatenated = concatenated.drop(["approved", "id"], axis=1)
concatenated.head()

