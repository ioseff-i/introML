#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
vals = np.random.normal(27,2,300)
print(vals)


# In[2]:


np.mean(vals)


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.hist(vals,300)
plt.show()


# In[7]:


np.median(vals)


# In[10]:


ages=np.random.randint(18,high=90,size=500)


# In[11]:


ages


# In[12]:


from scipy import stats
stats.mode(ages)


# In[ ]:




