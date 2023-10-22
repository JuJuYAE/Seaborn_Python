#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


tips = sns.load_dataset("tips")


# In[4]:


tips.head()


# In[8]:


sns.displot(tips["total_bill"],bins = 30,kde = True)


# In[15]:


sns.jointplot(x = "total_bill", y ="tip" , data = tips, kind = "kde", color = "black")


# In[18]:


sns.pairplot(data = tips, hue = "sex", palette = "coolwarm")


# In[19]:


sns.rugplot(tips["total_bill"])


# In[20]:


# Don't worry about understanding this code!
# It's just for the diagram below
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Create dataset
dataset = np.random.randn(25)

# Create another rugplot
sns.rugplot(dataset);

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min,x_max,100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2


# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)

plt.ylim(0,1)


# In[21]:


# To get the kde plot we can sum these basis functions.

# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list,axis=0)

# Plot figure
fig = plt.plot(x_axis,sum_of_kde,color='indianred')

# Add the initial rugplot
sns.rugplot(dataset,c = 'indianred')

# Get rid of y-tick marks
plt.yticks([])

# Set title
plt.suptitle("Sum of the Basis Functions")


# In[22]:


#sns.displot()
#sns.joinplot()
#sns.pairplot()
#sns.rugplot()


# In[23]:


sns.kdeplot(tips["total_bill"])


# In[26]:


tips = sns.load_dataset('tips')
tips.head()


# In[30]:


import numpy as np


# In[ ]:





# In[31]:


sns.barplot(data = tips, x = "sex",y = "total_bill",estimator = np.std)


# In[32]:


sns.countplot(x = "sex", data = tips)


# In[35]:


sns.boxplot(x = "day",y ="total_bill", data = tips, hue = "sex")


# In[38]:


sns.violinplot(x = "day", y = "total_bill", data = tips, hue= "sex", split = True)


# In[43]:


sns.stripplot(x = "day", y = "total_bill", data = tips, jitter = True, hue = "sex")


# In[45]:


sns.swarmplot(x = "day", y = "total_bill", data = tips)
sns.violinplot(x = "day", y = "total_bill", data = tips, color = "yellow")


# In[58]:


sns.catplot(x = "day", y = "total_bill", data = tips, kind = "swarm")


# In[59]:


flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')


# In[60]:


tips


# In[62]:


tc = tips.corr()


# In[65]:


sns.heatmap(tc,annot = True, cmap = "coolwarm")


# In[69]:


flights.pivot_table(index = "month", columns = "year", values = "passengers")
fp = flights.pivot_table(index = "month", columns = "year", values = "passengers")


# In[74]:


sns.heatmap(fp, cmap ="coolwarm", linecolor = "white",lw = 1)


# In[76]:


sns.clustermap(fp,cmap = "coolwarm", standard_scale = 1)


# In[77]:


iris = sns.load_dataset('iris')


# In[78]:


iris.head()


# In[89]:


g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_lower(sns.kdeplot)
g.map_upper(plt.scatter)


# In[90]:


tips = sns.load_dataset("tips")


# In[91]:


tips.head()


# In[99]:


g = sns.FacetGrid(data = tips, col = "time" , row ="smoker" )

g.map(plt.scatter, "total_bill", "tips")


# In[93]:





# In[100]:


tips.head()


# In[102]:


sns.lmplot(x = "total_bill", data = tips, y ="tip", hue = "sex", markers = ["o","v"], scatter_kws = {"s":100})


# In[109]:


sns.lmplot(x = "total_bill", data = tips, y ="tip", col = "day",hue= "sex", aspect = 0.6, height = 8)


# In[120]:


#ns.set_style("ticks")
#plt.figure(figsize = (12,3))
sns.set_context("poster", font_scale = 0.5)
sns.countplot(x= "sex", data = tips)
sns.despine()


# In[138]:


sns.lmplot(x = "total_bill", y = "tip", data = tips, palette = "coolwarm",)

