#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
A=pd.read_csv("D:/DS_Recordings/DataSets/Cars93.csv")


# In[3]:


A.head()


# In[9]:


B=A[["Price","MPG.city"]]


# In[10]:


B.head(2)


# In[26]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=4)
model=km.fit(B)


# In[27]:


model.labels_


# In[28]:


B['Cluster']=model.labels_   #adding column cluster in B data frame and assigning label values to it


# In[31]:


B


# In[36]:


Q = pd.DataFrame(model.cluster_centers_,columns=["Q1","Q2","Q3","Q4"])


# In[39]:


import matplotlib.pyplot as plt
plt.scatter(B.Price,B['MPG.city'],c=B.Cluster)
plt.scatter(Q.Q1,Q.Q2,c="red",alpha=1,marker="+")


# In[ ]:




