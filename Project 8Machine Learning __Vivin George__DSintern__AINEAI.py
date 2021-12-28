#!/usr/bin/env python
# coding: utf-8

# # Project 8: Machine Learning for Predictive Analytics

# # By Vivin George __DS & Business Intelligence intern AINE AI

# ## Packages and setup

# In[ ]:


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


# ## Reading data and initial processing

# In[5]:



cust_df=pd.read_csv("C:/Users/Vivin/Desktop/AINE AI WIP/Project 8/Telecom Data - Machine Learning/Telecom Data.csv")


# In[6]:


cust_df.shape


# ## Exercise

# **## Placeholder for your analysis**

# In[ ]:



cust_df.size


# In[ ]:


cust_df.describe


# In[ ]:


# Type your code here to explore and understand the data better
cust_df.info()


# In[ ]:




#e.g., identify the dimension of cust_df
cust_df.shape


# ## Q1. Detect and resolve problems in the data (Missing value, Outliers, Unexpected value, etc.)

# In[ ]:


cust_df[cust_df['MonthlyRevenue']==0]


# ### Q1.1 How many customers had zero monthly revenue?
# There are 6 customers with monthly revenue 0

# In[ ]:


#How many customers had zero monthly revenue?
#Exclude the customers with zero monthly revenue from cust_df and from any subsequent analysis
cust_df = cust_df[cust_df['MonthlyRevenue']!=0]


# ### Q1.2 How many columns has missing values percentage > 5%

# In[ ]:


#Calculate for each column % of missing value in the data
#How many columns has missing values percentage > 5%
#What strategy should be used for imputation?
percent_missing = cust_df.isnull().sum() * 100 / len(cust_df)
missing_value_df = pd.DataFrame({'column_name': cust_df.columns,
                                 'percent_missing': percent_missing}) 
missing_value_df


# <mark>__Result:__
#     
# No Columns are Having Missing Values more than 5%. So we can drop those rows which has missing values as the number of missing values are less

# In[ ]:


cust_df.dropna(axis=0, inplace=True)


# In[ ]:


cust_df.shape


# ### Q1.3 For columns, "UniqueSubs" and "DirectorAssistedCalls" remove outliers, if any

# In[ ]:


#plot box plot using pandas for columns "UniqueSubs" and "DirectorAssistedCalls"
cols=["UniqueSubs","DirectorAssistedCalls"]
cust_df.boxplot(column=cols)
plt.title('Boxplot')


# In[ ]:


#Remove top 1% outliers from the above mentioned two columns and plot the box plot again
#Use "quantile" function to identify top 1% outliers

print(cust_df[cols].quantile(0.99))

cust_df = cust_df[(cust_df['UniqueSubs']<=5) & (cust_df['DirectorAssistedCalls']<=9.65)]

cust_df.boxplot(column=cols)


# ## Q2. Perform exploratory analysis to analyze customer churn

# ### Q2.1 Does customers with high overage minutes also have high revenue?

# In[ ]:


#use scatter plot to find the correlation between monthly revenue and overage minutes
sns.scatterplot(x = 'MonthlyRevenue', y = 'OverageMinutes', data = cust_df )


# In[ ]:


column_1 = cust_df ['MonthlyRevenue']
column_2 = cust_df ['OverageMinutes']
correlation = column_1.corr(column_2)
print("The Correlation Between MonthlyRevenue and OverageMinutes is = "+ str(correlation))


# <mark>__Result:__
#     
# MonthlyRevenue and OverageMinutes have a strong positive correlation which is 0.787 and it tends to +1, also both the variables move in the same direction, therefore customers with high overage minutes also have high revenue

# ### Q2.2 Does high number of active subscribers lead to low monthly revenue?

# In[ ]:


print(cust_df['ActiveSubs'].value_counts())


# In[ ]:


#use category plot to plot monthly revenue for each active subs category
ax = sns.catplot(x="ActiveSubs", y="MonthlyRevenue", data= cust_df) 


# <mark>__Result:__
#     
# It seems with the Higher Number of Active Subs the Monthly Revenue Decreases

# ### Q2.3 Does credit rating have an impact in churn rate?

# In[ ]:


sns.catplot(x="CreditRating",kind='count', data= cust_df, hue='Churn')


# <mark>__Result:__
#     
# Customers falling under the category of High CreditRating has recorded to be maximum in churning as wells as not churning among all categories. Customers falling under all categories has the higher count of not churning compared to the customers who churned in every category of credit rating

# ### Placeholder for additional exploratory analysis

# In[ ]:


#type your code here for any additional exploratory analysis (if any)


# ## Q3. Create additional features to help predict churn

# In[ ]:


#wrapper function to create additional features for churn prediction
def create_features(cust_df):
    
    #3.1 Percent of current active subs over total subs
    cust_df['perc_active_subs'] = cust_df['ActiveSubs'] / cust_df['UniqueSubs']
    
    #3.2 Percent of recurrent charge to monthly charge
    #type your code here to create a new column in cust_df
    cust_df['perc_recurrent_charge']= cust_df['TotalRecurringCharge'] / cust_df['MonthlyRevenue']
    
    #3.3 Percent of overage minutes over total monthly minutes
    #type your code here to create a new column in cust_df
    cust_df['perc_overage_mins'] = cust_df['OverageMinutes'] / cust_df['MonthlyMinutes']
    
    
    #type your code here to creat any other additional features which you think will help improve your model accuracy
    
    
    return cust_df  


# In[ ]:


#Make a call to the feature engineering wrapper function for churn prediction
cust_df=create_features(cust_df)

#Checking For Missing Values
cust_df.isna().sum()
print

#Investigating Missing Values
rows_with_nan = [ index for index, value in cust_df['perc_overage_mins'].iteritems() if np.isnan(value)]
cust_df.loc[rows_with_nan, ['OverageMinutes', 'MonthlyMinutes']]

#Filling Missing Values with  0
cust_df.fillna(value=0, axis=0, inplace=True)


# In[ ]:


#Seperating categorical and continuous variables
categorical_cols = list(cust_df.select_dtypes('object').columns.values)
continuous_cols = list(cust_df.select_dtypes('number').columns.values)

#Removing unnecssary columns and target variable
categorical_cols = [i for i in categorical_cols if i not in ('Churn','CustomerID','ServiceArea')]
continuous_cols.remove('CustomerID')


# In[ ]:


def check_categorical_imp(cust_df, categorical_cols):
  new_categorical_cols = []
  for i in categorical_cols:
    if stats.chi2_contingency(pd.crosstab(cust_df.Churn, cust_df[i]))[1] > 0.05:
      pass
    else:
      new_categorical_cols.append(i)

  return new_categorical_cols


# In[ ]:


#Checking For Class imbalance
cust_df['Churn'].value_counts(normalize=True)*100


# In[ ]:


#Encoding dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cust_df['Churn'] = le.fit_transform(cust_df['Churn'])

class_count_0, class_count_1 = cust_df['Churn'].value_counts()

class_0 = cust_df[cust_df['Churn']== 0]
class_1 = cust_df[cust_df['Churn']== 1]#print the shape of the Class

class_1_over = class_1.sample(class_count_0, replace=True)

test_over = pd.concat([class_1_over, class_0], axis=0)

#Assigning oversampled dataframe back to cust_df
cust_df = test_over
cust_df['Churn'].value_counts(normalize=True)*100


# ## Q4. Build classification model to predict customer churn month in advance

# ### Initial data processing for model building exercise

# In[ ]:


#Train - test split to train and test model accuracy
from sklearn.model_selection import train_test_split

#Define columns to be included in X and y
X = cust_df[check_categorical_imp(cust_df, categorical_cols)+ continuous_cols]
Y = cust_df['Churn']
#Create dummy variables for all categorical variables
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Feature scaling for all continuous variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()




# In[ ]:


#Scaling testing and training data
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ### Q4.1 Build a simple logistic regression model to predict churn and evaluate model accuracy on test data set
# 

# In[ ]:


#-------------------------------- Model training -------------------------------#

#type your code to build logistic regression model on training data set
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,f1_score,plot_confusion_matrix,average_precision_score,roc_curve,plot_roc_curve

#creating an instance of logistic regression model
lr = LogisticRegression(random_state=0)

#Training model on training data
lr.fit(X_train, y_train)

# Predicting Probabilities of Customer Churning in Test Data
y_prediction = lr.predict_proba(X_test)[:,1]


# In[ ]:


#Finding Optimal Threshold Value
fpr, tpr, thresholds = roc_curve(y_test, y_prediction)

accuracy_lr=[]
for i in thresholds:
  y_pred = np.where(y_prediction > i, 1, 0)
  accuracy_lr.append(accuracy_score(y_test, y_pred))

accuracy_lr = pd.concat([pd.Series(thresholds), pd.Series(accuracy_lr)], axis=1)
accuracy_lr.columns=['thresholds','accuracy']
accuracy_lr.sort_values('accuracy', ascending=False, inplace=True)
display(accuracy_lr[:5])


best_threshold_lr = accuracy_lr.iloc[0,0]


# In[ ]:


#-------------------------------- Model testing ---------------------------------#

#type your code here to predict for X_test data set using the fitted model
y_prediction_lr = lr.predict_proba(X_test)[:,1]


#Use a probability threshold to classify customers as churners and non churners (Default = 0.5)

y_prediction_lr[y_prediction_lr >= best_threshold_lr] = 1

y_prediction_lr[y_prediction_lr < best_threshold_lr] = 0



#Evaluate model accuracy using fitted y value and actual y_test
#Evaluatio metrics to be used - GINI, Precision, Recall, confusion matrix
print(classification_report(y_test, y_prediction_lr))
test_auc = roc_auc_score(y_test, y_prediction_lr)
print("Logistic Regression's roc-auc is: {}".format(test_auc))

print("\nConfusion Matrix: \n{}".format(confusion_matrix(y_test, y_prediction_lr)))

tn,fp, fn, tp = confusion_matrix(y_test, y_prediction_lr).ravel()
print("\nTrue Negatives: {} \nFalse Positives: {} \nFalse Negatives: {}  \nTrue Positives: {}".format(tn, fp, fn, tp))


# In[ ]:


#Plotting a ROC Curve
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC-AUC Curve for Logistic Regression')
plt.show()


# ### Q4.2 Build Random Forest classifier to compare model accuracy over the logistic regression model
# 

# In[ ]:


#-------------------------------- Model training -------------------------------#

#type your code to build random forest classifier model on training data set
from sklearn.ensemble import RandomForestClassifier

#Creating an instance of Random Classifier Model
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

#Training the Model on training data
rf.fit(X_train, y_train)

#Predicting probabilities of customer churning in Test Data
rf_test_output = rf.predict_proba(X_test)[:,1]




# In[ ]:


#indicating the optimal threshold value
pr, tpr, thresholds = roc_curve(y_test, rf_test_output)

accuracy_rf=[]
for i in thresholds:
  y_pred = np.where(rf_test_output > i, 1, 0)
  accuracy_rf.append(accuracy_score(y_test, y_pred))

accuracy_rf = pd.concat([pd.Series(thresholds), pd.Series(accuracy_rf)], axis=1)
accuracy_rf.columns=['thresholds','accuracy']
accuracy_rf.sort_values('accuracy', ascending=False, inplace=True)
display(accuracy_rf[:5])


best_threshold_rf = accuracy_rf.iloc[0,0]


# In[1]:


#-------------------------------- Model testing ---------------------------------#

#type your code here to predict for X_test data set using the fitted RF model
y_prediction_rf = rf.predict_proba(X_test)[:,1]


#Use a probability threshold to classify customers as churners and non churners (Default = 0.5)
y_prediction_rf[y_prediction_rf >= best_threshold_rf] = 1

y_prediction_rf[y_prediction_rf < best_threshold_rf] = 0



#Evaluate model accuracy using fitted y value and actual y_test
#Evaluatio metrics to be used - GINI, Precision, Recall, confusion matrix
print(classification_report(y_test, y_prediction_rf))
test_auc = roc_auc_score(y_test, y_prediction_rf)
print("Random forest's roc-auc is: {}".format(test_auc))

print("\nConfusion Matrix: \n{}".format(confusion_matrix(y_test, y_prediction_rf)))

tn_n,fp_n, fn_n, tp_n = confusion_matrix(y_test, y_prediction_rf).ravel()
print("\nTrue Negatives: {} \nFalse Positives: {} \nFalse Negatives: {}  \nTrue Positives: {}".format(tn, fp, fn, tp))


# <mark>__Result:__
#     
# The Random Forest Classifier Model is having a better accuracy than the Logistic Regression model, it can be seen from accuracy_score, recall and f1_score being more in Random Forest Classifier

# ### Q4.3 Identify most important features impacting churn

# In[ ]:


# Type your code here to plot the top 20 important predictor features impacting churn using the Random Forest model created

top_20_features = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20)
plt.figure(figsize=(5,8))
top_20_features.sort_values(ascending=True).plot.barh(title='Top 20 Importnat Features Impacting Ch')


# ## Q5. Use the hold out data provided to predict churners using the best model identified in step 4 

# In[ ]:


#Type your code here to predict churners based on the hold out data set provided
predict = pd.read_csv('/content/drive/MyDrive/Telecom - Prediction Data.csv')

#Note #1: use "create_features(cust_df)" functions to create the additional list of features in the hold out data set
create_features(cust_df=predict)
#Note #2: Also, perform feature scaling and dummy variables creation as performed during the initial stages of step #4

customer_id = predict['CustomerID']

#Use "predict" function on the transformend data using the best fitted model object
predict = predict[check_categorical_imp(cust_df,categorical_cols)+ continuous_cols]

predict['perc_overage_mins'].fillna(0, inplace=True)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#Checking for Missing Values
missing_cols = []
for i, v in (predict.isna().sum() / predict.shape[0] * 100).iteritems():
  if v > 0:
    missing_cols.append(i)

#filling missing values with median of each column
predict[missing_cols] = predict[missing_cols].fillna(predict[missing_cols].median(), axis=0)


# In[ ]:


#Creating Dummy Variables
predict = pd.get_dummies(predict, drop_first=True)


# In[ ]:


#Feature Scaling
X = sc.fit_transform(predict)


# In[ ]:


churn_pred = rf.predict(X)

#Adding a New Column to check Customers who are going to churn
predict['Churn'] = churn_pred

#Adding CustomerID column back to the predict dataframe
predict['CustomerID'] = customer_id
print(predict['Churn'].value_counts())

#Customers who are going to churn
predict.loc[predict['Churn'] == 1, ['CustomerID','Churn']]


# ## Q6. [Bonus Question] Calculate lift chart and total monthly revenue saved by targeting top 10-20% of the customers using your best predictive model

# In[ ]:


#Type your code here to plot the lift chart from the best model
#Identify the lift i.e. 3x of capturing churn with and without using the model 
#(Assume: only top 20% high risk customers are to be targetted)





# <mark>__Result:__
#     
# 1. What is the % actual churn captured by targeting top 20% (top 2 deciles) of the customers sorted by their churn probability?
# 2. What is the total monthly revenue of actual churn customers identified in the top 20% of the customers?
#    
