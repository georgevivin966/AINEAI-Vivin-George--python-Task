#!/usr/bin/env python
# coding: utf-8

# # Project 7: Statistical Analysis and Hypothesis Testing

# ## Packages and setup

# In[2]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

from scipy.stats import shapiro
import scipy.stats as stats

#parameter settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# ### Reading data and initial processing

# In[3]:


#Read data using pandas
user_df=pd.read_csv("C:/Users/Vivin/Desktop/AINE AI WIP/Project 7/cookie_cats.csv")

#Check data types of each column using "dtypes" function
print("Data types for the data set:")
user_df.dtypes

#Check dimension of data i.e. # of rows and #column using pandas "shape" funtion
print("Shape of the data i.e. no. of rows and columns")
user_df.shape

#display first 5 rows of the data using "head" function
print("First 5 rows of the raw data:")
user_df.head(5)


# ## Exercise

# ## Q1. Detect and resolve problems in the data (Missing value, Outliers, etc.)

# ### Q1.1 Identify missing value

# In[4]:


#Check for any missing values in the data using isnull() function
user_df.isnull().sum().sum()


# ### Q1.2 Identify outliers

# In[5]:


#Check for outlier values in sum_gamerounds column
plt.title("Total gamerounds played")
plt.xlabel("Index")
plt.ylabel("sum_gamerounds")
plt.plot(user_df.sum_gamerounds)


# In[6]:


user_df['sum_gamerounds'].max()


# In[7]:


user_df[user_df['sum_gamerounds']>=49584]


# In[8]:


#Based on the plot, filter out the outlier from sum_gamerounds played; Use max() fucntion to find the index of the outlier
#Check for outlier values in sum_gamerounds column
user_df.drop(57702, inplace=True)


# In[9]:


user_df[user_df['sum_gamerounds']>=49584]


# In[10]:


#Plot the graph for sum_gamerounds player after removing the outlier
#Check for outlier values in sum_gamerounds column
plt.title("Total gamerounds played")
plt.xlabel("Index")
plt.ylabel("sum_gamerounds")
plt.plot(user_df.sum_gamerounds)


# ## Q2. Plot summary statistics and identify trends to answer basis business questions

# ### Q2.1 What is the overall 7-day retention rate of the game?

# In[11]:


#Insert calculation for 7-day retention rate

retention_rate_7=round((user_df["retention_7"].sum()/user_df["retention_7"].count())*100)
print("Overal 7 days retention rate of the game for both versions is: " ,retention_rate_7,"%")


# ### Q2.2 How many players never played the game after installing? 

# In[12]:


# Find number of customers with sum_gamerounds is equal to zero

(user_df["sum_gamerounds"]==0).sum()


# ### Q2.3 Does the number of users decrease as the level progresses highlighting the difficulty of the game?

# In[14]:


#Group by sum_gamerounds and count the number of users for the first 200 gamerounds
#Use plot() function on the summarized stats to visualize the chart

graph=user_df.groupby("sum_gamerounds").userid.count()[:200]
graph.plot()


# ## Q3. Generate crosstab for two groups of players to understand if there is a difference in 7 days retention rate & total number of game rounds played

# ### Q3.1 Seven days retention rate summary for different game versions

# In[14]:


#Create cross tab for game version and retention_7 flag counting number of users for each possible categories

pd.crosstab(user_df.version, user_df.retention_7).apply(lambda r: r/r.sum(), axis=1)


# <mark>__Analsysis Results:__
#     
# Type your interpretation here from the crosstab generated above   

#  The data clearly shows that gate_30 has slighlty higher retention as comapred to gate_40.Both gates also show that the amount of retention is very less adn the rate is around 80% in both the gates.

# ### Q3.2 Gamerounds summary for different game versions

# In[9]:


#use pandas group by to calculate average game rounds played summarized by different versions

user_df.groupby('version').mean()['sum_gamerounds']
print('/n')
user_df.groupby('version').count()['sum_gamerounds']


# <mark>__Analsysis Results:__
#     
# Do total number of gamerounds played in total by each player differ based on  different versions of the game? 

# There is a very slight difference between the total number of players in the different verison.

# ## Q4. Perform two-sample test for groups A and B to test statistical significance amongst the groups in the sum of game rounds played i.e., if groups A and B are statistically different

# ### Initial data processing

# In[12]:


#Define A/B groups for hypothesis testing
user_df["version"] = user_df["version"].replace(['gate_30','gate_40'],[ "A", "B"])
group_A=pd.DataFrame(user_df[user_df.version=="A"]['sum_gamerounds'])
group_B=pd.DataFrame(user_df[user_df.version=="B"]['sum_gamerounds'])


# ### Q4.1 Shapiro test of Normality

# In[13]:


#---------------------- Shapiro Test ----------------------
# NULL Hypothesis H0: Distribution is normal
# ALTERNATE Hypothesis H1: Distribution is not normal    

#test for group_A
stats.shapiro(group_A)

#test for group_B
stats.shapiro(group_B)


# <mark>__Analsysis Results:__
#     
# __Type your answer here:__ Analyze and interpret the results of shapiro test of normality i.e. are the two groups normally distributed?

# From the above value of shapiro test we can say that basedon the data GroupA and GroupB are not normally distributed for 0.05 level of signifciance.

# ### Q4.2 Test of homegienity of variance

# In[15]:


#---------------------- Leven's Test ----------------------
# NULL Hypothesis H0: Two groups have equal variances
# ALTERNATE Hypothesis H1: Two groups do not have equal variances

#perform levene's test and accept or reject the null hypothesis based on the results


stats.levene(group_A.sum_gamerounds,group_B.sum_gamerounds)


# <mark>__Analsysis Results:__
#     
# __Type your answer here:__ Write your final recommendation from the results of Levene's test

# As p- vaue greater than 0.05 .Hence we do no have enough evidence to reject the null hypothesis at 0.05 significance leveland we can say that the two groups have different variances.

# ### Q4.3 Test of significance: Two sample test

# In[17]:


#---------------------- Two samples test ----------------------
# NULL Hypothesis H0: Two samples are equal
# ALTERNATE Hypothesis H1: Two samples are different

#Apply relevant two sample test to accept or reject the NULL hypothesis

stats.mannwhitneyu(group_A, group_B)


# <mark>__Analsysis Results:__
#     
# __Type your answer here:__ Write your final recommendation from the results of two sample hyothesis testing

# As the P-value is less than significance , hence there is enough statistical veidence to reject the null hypothesis, the two sample sum_gamerounds are different fro Group_A and Group_B.

# ## Q5. Based on significance testing results, if groups A and B are statistically different, which level has more advantage in terms of player retention and number of game rounds played

# In[58]:


#Analyze the 1 day and 7 days retention rate for two different groups using group by function

user_df.groupby('version')['retention_1'].mean()
user_df.groupby('version')['retention_7'].mean()


# <mark>__Analsysis Results:__
#     
# __Type your answer here:__ Write your final recommendation to the company regarding which level works best as the first gate  - Level 30 or Level 40

# After analysis of retetnion rate of two groups we can see that the eman of gate_30 is hgiher in terms of retention as comapre to gate 40.

# The retention rate for both 1 day and 7 day retention is slightly greater for Gate 30 than for Gate 40.

# ## Q6. [Bonus Question]  Using bootstrap resampling, plot the retention rate distribution for both the groups inorder to visualize effect of different version of the game on retention.

# In[32]:


#Hint: Plot density function
import pandas as pd

retent_A=[]
retent_B=[]
for i in range(500):
    boot_A=user_df.sample(frac=0.8,replace=True).groupby('version')['retention_1'].mean()
    retent_A.append(boot_A)
    boot_B=user_df.sample(frac=0.8,replace=True).groupby('version')['retention_7'].mean()
    retent_B.append(boot_B)
    
retention_1=pd.DataFrame(retent_A,columns=['gate_30','gate_40'])
retention_2=pd.DataFrame(retent_B,columns=['gate_30','gate_40'])

