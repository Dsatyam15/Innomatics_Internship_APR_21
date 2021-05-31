#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, norm


# In[2]:


def t_score(sample_size, sample_mean, pop_mean, pop_std):
    numerator = sample_mean - pop_mean
    denomenator = pop_std / sample_size**0.5
    return numerator / denomenator


# **Q1)**
# Known Variance of Population
# Q-1: Pista House selling Hyderabadi Chicken Dum biryani claims that each parcel packet has 500 grams of biryani (also mentioned on the label of packet). You are sceptic of their claims and believe that on average each packet does not contain 500 grams of biryani. How do you prove your claim? 
# 
# Step - 1:
# Alternate Hypothesis (Bold claim):$$ H_1 \neq 500 $$
# Null Hypothesis (Status Quo):$$ H_0= 500g $$
# 
# Step - 2:
# 
# Collect a sample of size n = 10$$ [490, 220, 470, 500, 495, 496, 496, 498, 508, 480] $$
# Compute sample mean$$ \bar{x} \ = 465.3 $$
# Step - 3: Compute Test Statistic:$$ t = \frac{\bar{x} - \mu}{S/\sqrt[2]{n}}$$
# 
# Step - 4: Decide $ \alpha $ or significance level
# 
# Step - 5.1: 2 tailed t-test:$$ reject \ H_0 : if\; \;p-value < \alpha $$
# 
# Step - 5.2: Compute p-value
# 

# In[3]:


l = [490, 220, 470, 500, 495, 496, 496, 498, 508, 480]

sum(l)/len(l)


# In[43]:


# Two tail
from scipy.stats import t
alpha = 1 - 0.95
t_critical = t.ppf(1-alpha/2, df = 9)
print(t_critical)


# In[6]:


sample_size = 10
sample_mean = 465.3
pop_mean = 500
pop_std = 89.40


# In[9]:


# t_score for sampling distributions

def t_score(sample_size, sample_mean, pop_mean, pop_std):
    numerator = sample_mean - pop_mean
    denomenator = pop_std / sample_size**0.5
    return numerator / denomenator
sample_size = 10
sample_mean = 465.3
pop_mean = 500
pop_std = 89.40
t = t_score(sample_size, sample_mean, pop_mean, pop_std)

print(t) 


# In[41]:


# Ploting the sampling distribution with rejection regions

# Defining the x minimum and x maximum
x_min = 400
x_max = 600


# Defining the sampling distribution mean and sampling distribution std
mean = pop_mean
std = pop_std / sample_size**0.5


# Ploting the graph and setting the x limits
x = np.linspace(x_min, x_max, 100)
y = norm.pdf(x, mean, std)
plt.xlim(x_min, x_max)
plt.plot(x, y)


# Computing the left and right critical values (Two tailed Test)
t_critical_left = pop_mean + (-t_critical * std)
t_critical_right = pop_mean + (t_critical * std)


# Shading the left rejection region
x1 = np.linspace(x_min, t_critical_left, 100)
y1 = norm.pdf(x1, mean, std)
plt.fill_between(x1, y1, color='orange')

# Shading the right rejection region
x2 = np.linspace(t_critical_right, x_max, 100)
y2 = norm.pdf(x2, mean, std)
plt.fill_between(x2, y2, color='orange')


# Ploting the sample mean and concluding the results 
plt.scatter(sample_mean, 0)
plt.annotate("x_bar", (sample_mean, 0.0007))


# In this case sample mean falls in the rejection region
# i.e. here we reject the Null Hypothesis


# In[12]:


# Conclusion using t test

if(np.abs(t) > t_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# In[34]:


# Conclusion using p test

p_value = 2 * (1.0 - norm.cdf(np.abs(t)))

print("p_value = ", p_value)

if(p_value < alpha):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# **Q2)**
# You have developed a new Natural Language Processing Algorithms and done a user study. You claim that the average rating given by the users is greater than 4 on a scale of 1 to 5. How do you prove this to your client?
# 
# Step - 1:
# Alternate Hypothesis (Bold Claim):$$ H_1: \mu >4 $$
# Null Hypothesis (Status Quo):$$ H_0:\mu \leq 4 $$
# 
# Step - 2:
# 
# Collect a sample of size n = 10$$ [4, 3, 5, 4, 5, 3, 5, 5, 4, 2, 4, 5, 5, 4, 4, 5, 4, 5, 4, 5] $$
# Compute sample mean$$ \bar{x} \ = 4.25 $$
# Step - 3: Compute Test Statistic:$$ z = \frac{\bar{x} - \mu}{\sigma/\sqrt[2]{n}}$$
# 
# Step - 4: Decide $ \alpha $
# 
# Step - 5.1: 1 right tailed t-test:$$ reject \ H_0 : if\; \;p-value < \alpha  $$
# 
# Step - 5.2: Compute p-value

# In[16]:


l = [4, 3, 5, 4, 5, 3, 5, 5, 4, 2, 4, 5, 5, 4, 4, 5, 4, 5, 4, 5]
sum(l)/len(l)


# In[17]:


# One tail
from scipy.stats import t
alpha = 1 - 0.95
t_critical = t.ppf(1-alpha, df = 19)
print(t_critical)


# In[18]:


sample_size = 20
sample_mean = 4.5
pop_mean = 4
pop_std = 0.88


# In[19]:


# t_score for sampling distributions

def t_score(sample_size, sample_mean, pop_mean, pop_std):
    numerator = sample_mean - pop_mean
    denomenator = pop_std / sample_size**0.5
    return numerator / denomenator
sample_size = 20
sample_mean = 4.5
pop_mean = 4
pop_std = 0.88
t = t_score(sample_size, sample_mean, pop_mean, pop_std)

print(t) 


# In[20]:


# Ploting the sampling distribution with rejection regions

# Defining the x minimum and x maximum
x_min = 3.5
x_max = 4.5


# Defining the sampling distribution mean and sampling distribution std
mean = pop_mean
std = pop_std / (sample_size**0.5)


# Ploting the graph and setting the x limits
x = np.linspace(x_min, x_max, 100)
y = norm.pdf(x, mean, std)
plt.xlim(x_min, x_max)
plt.plot(x, y)


# Computing the right critical value (Right tailed Test)
t_critical_right = pop_mean + (t_critical * std)


# Shading the right rejection region
x1 = np.linspace(t_critical_right, x_max, 100)
y1 = norm.pdf(x1, mean, std)
plt.fill_between(x1, y1, color='orange')

# Ploting the sample mean and concluding the results 
plt.scatter(sample_mean, 0)
plt.annotate("x_bar", (sample_mean, 0.1))

# In this case sample mean falls in the rejection region
# i.e. Reject the Null Hypothesis


# In[21]:


# Conclusion using t test

if(np.abs(t) > t_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# In[22]:


# Conclusion using p test

p_value = 2 * (1.0 - norm.cdf(np.abs(t)))

print("p_value = ", p_value)

if(p_value < alpha):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# **Q3)**
# TATA has developed a better fuel management system for the SUV segment. They claim that with this system, on average the SUV's mileage is at least 15 km/litre?
# Step - 1:
# Alternate Hypothesis (Bold Claim):$$ H_1: \mu > 15 $$
# Null Hypothesis (Status Quo):$$ H_0: \mu \leq 15  $$
# 
# Step - 2:
# 
# Collect a sample of size n = 10$$ [14.08, 14.13, 15.65, 13.78, 16.26, 14.97, 15.36, 15.81, 14.53, 16.79, 15.78, 16.98, 13.23, 15.43, 15.46, 13.88, 14.31, 14.41, 15.76, 15.38] $$
# Compute sample mean$$ \bar{x} \ = 15.099 $$
# Step - 3: Compute Test Statistic:$$ z = \frac{\bar{x} - \mu}{\sigma/\sqrt[2]{n}}$$
# 
# Step - 4: Decide $ \alpha $
# 
# Step - 5.1: 1 right-tailed z-test:$$ reject \ H_0 : if\; \;p-value < \alpha $$
# 
# Step - 5.2: Compute p-value

# In[27]:


import numpy
l = [14.08, 14.13, 15.65, 13.78, 16.26, 14.97, 15.36, 15.81, 14.53, 16.79, 15.78, 16.98, 13.23, 15.43, 15.46, 13.88, 14.31, 14.41, 15.76, 15.38]
a = sum(l)/len(l)
b = numpy.std(l)
print(a)
print(b)


# In[25]:


# One tail
from scipy.stats import t
alpha = 1 - 0.95
t_critical = t.ppf(1-alpha, df = 19)
print(t_critical)


# In[28]:


# t_score for sampling distributions

def t_score(sample_size, sample_mean, pop_mean, pop_std):
    numerator = sample_mean - pop_mean
    denomenator = pop_std / sample_size**0.5
    return numerator / denomenator
sample_size = 20
sample_mean = 15.099
mil_mean = 15
mil_std = 1
t = t_score(sample_size, sample_mean, pop_mean, pop_std)

print(t) 


# In[29]:


x_min = 13
x_max = 17

mean = mil_mean
std = mil_std / (sample_size**0.5)

x = np.linspace(x_min, x_max, 100)
y = norm.pdf(x, mean, std)

plt.xlim(x_min, x_max)
# plt.ylim(0, 0.03)

plt.plot(x, y)

t_critical_right = pop_mean + (t_critical * std)

x1 = np.linspace(t_critical_right, x_max, 100)
y1 = norm.pdf(x1, mean, std)
plt.fill_between(x1, y1, color='orange')

plt.scatter(sample_mean, 0)
plt.annotate("x_bar", (sample_mean, 0.1))

# In this case sample mean falls in the acceptance region
# i.e. Fail to Reject the Null Hypothesis


# In[30]:


# Conclusion using t test

if(np.abs(t) > t_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# In[31]:


# Conclusion using p test

p_value = 2 * (1.0 - norm.cdf(np.abs(t)))

print("p_value = ", p_value)

if(p_value < alpha):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# **Q4)**
# You have developed a new Machine Learning Application and claim that on average it takes less than 100 ms to predict for any future datapoint. How do you convince your client about this claim?
# 
# Step - 1:
# Alternate Hypothesis (Bold Claim):$$ H_1: \mu < 100ms $$
# Null Hypothesis (Status Quo):$$ H_0: \mu \geq 100ms $$
# 
# Step - 2:
# 
# Collect a sample of size n = 100
# Compute sample mean$$ \bar{x} \ = \ 97.5 $$
# Step - 3: Compute Test Statistic:$$ z = \frac{\bar{x} - \mu}{\sigma/\sqrt[2]{n}}$$
# 
# Step - 4: Decide $ \alpha $
# 
# Step - 5.1: 1 left-tailed t-test:$$ reject \ H_0  : if\; \;p-value < \alpha $$
# 
# Step - 5.2: Compute p-value

# In[32]:


# One Tail

alpha = 1 - 0.99

t_critical = norm.ppf(1 - alpha)

print(t_critical)


# In[44]:


sample_size = 100
sample_mean = 97.5
pop_mean = 100
pop_std = 10


# In[45]:


z = z_score(sample_size, sample_mean, pop_mean, pop_std)

print(z)


# In[47]:


x_min = 95
x_max = 105

mean = pop_mean
std = pop_std / (sample_size**0.5)

x = np.linspace(x_min, x_max, 100)
y = norm.pdf(x, mean, std)

plt.xlim(x_min, x_max)
# plt.ylim(0, 0.03)

plt.plot(x, y)

t_critical_left = pop_mean + (-t_critical * std)

x1 = np.linspace(x_min, t_critical_left, 100)
y1 = norm.pdf(x1, mean, std)
plt.fill_between(x1, y1, color='orange')

plt.scatter(sample_mean, 0)
plt.annotate("x_bar", (sample_mean, 0.02))

# In this case sample mean falls in the rejection region

# i.e. Reject Null Hypothesis


# In[50]:



if(z < -t_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# In[51]:


p_value = 1.0 - norm.cdf(np.abs(z))

print("p_value = ", p_value)

if(p_value < alpha):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# In[ ]:




