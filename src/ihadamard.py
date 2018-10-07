#!/usr/bin/env python
# coding: utf-8

# # Function ihadamard
# 
# ## Synopse
# 
# Inverse Hadamard Transform.
# 
#  - **F = iaihadamard(f)**
#      - Output:
#          - **F**: Image.
#      - Input:
#          - **f**: Image.
# 

# ## Function code

# In[1]:


import numpy as np

def ihadamard(f):
    import ia898.src as ia
    f = np.asarray(f).astype(np.float64)
    if len(f.shape) == 1: f = f[:, newaxis]
    (m, n) = f.shape
    A = ia.hadamardmatrix(m)
    if (n == 1):
        F = np.dot(np.transpose(A), f)
    else:
        B = ia.hadamardmatrix(n)
        F = np.dot(np.dot(np.transpose(A), f), B)
    return F


# ## Examples

# In[2]:


testing = (__name__ == "__main__")
if testing:
    get_ipython().system(' jupyter nbconvert --to python ihadamard.ipynb')
    import sys
    import os
    ia898path = os.path.abspath('../../')
    if ia898path not in sys.path:
        sys.path.append(ia898path)
    import ia898.src as ia
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    get_ipython().run_line_magic('matplotlib', 'inline')


# ### Example 1

# In[3]:


if testing:
    f = mpimg.imread('../data/cameraman.tif')
    F = ia.hadamard(f)
    g = ia.ihadamard(F)
    print(sum(sum(abs(f.astype(float)-g.astype(float)))))
    
    nb = ia.nbshow(3)
    nb.nbshow(f, 'Original')
    nb.nbshow(ia.normalize(F), 'hadamard of f')
    nb.nbshow(ia.normalize(g), 'ihadamard of F')
    nb.nbshow()


# ## Measuring time:

# In[4]:


if testing:
    f = mpimg.imread('../data/cameraman.tif')
    F = ia.hadamard(f)
    print('Computational time is:')
    get_ipython().run_line_magic('timeit', 'ia.ihadamard(F)')


# In[ ]:




