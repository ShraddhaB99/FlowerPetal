#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:



# Suppress warning messages
from warnings import filterwarnings
filterwarnings(action='ignore')


# In[4]:



# Define the path to your dataset CSV file
dataset_path = r"\Users\shraddha\Downloads\iris (1).csv"
# Load the dataset
iris = pd.read_csv(dataset_path)


# In[5]:


# Display the dataset
print(iris)


# In[6]:



# Check for missing values
print(iris.isna().sum())


# In[7]:


# Display statistical summary
print(iris.describe())


# In[8]:


# Display the first few rows of the dataset
iris.head()

# Calculate and display counts of 'versicolor', 'virginica', and 'setosa'
n = len(iris[iris['Species'] == 'versicolor'])
print("No of Versicolor in Dataset:", n)


# In[9]:



n1 = len(iris[iris['Species'] == 'virginica'])
print("No of Virginica in Dataset:", n1)

n2 = len(iris[iris['Species'] == 'setosa'])
print("No of Setosa in Dataset:", n2)


# In[10]:



# Create a pie chart to visualize species distribution
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50, 50, 50]
ax.pie(s, labels=l, autopct='%1.2f%%')
plt.show()


# In[11]:



# Visualize outliers using box plots
plt.figure(1)
plt.boxplot([iris['Sepal.Length']])
plt.figure(2)
plt.boxplot([iris['Sepal.Width']])
plt.show()


# In[12]:


# Display histograms
iris.hist()
plt.show()


# In[13]:



# Display density plots for features
iris.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
iris.plot(kind='box', subplots=True, layout=(2, 5), sharex=False)


# In[14]:



# Create a 2x2 grid of violin plots
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.violinplot(x='Species', y='Petal.Length', data=iris)


# In[15]:



# Create a heatmap to visualize feature correlations
fig = plt.gcf()
fig.set_size_inches(10, 7)
fig = sns.heatmap(iris.corr(), annot=True, cmap='cubehelix', linewidths=1, linecolor='k', square=True,
                  mask=False, vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True)


# In[16]:


# Prepare data for scatter plot
X = iris['Sepal.Length'].values.reshape(-1, 1)
Y = iris['Sepal.Width'].values.reshape(-1, 1)

# Create scatter plot
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X, Y, color='b')
plt.show()


# In[17]:



# Display correlation matrix
corr_mat = iris.corr()
print(corr_mat)


# In[18]:



# Import machine learning modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[19]:


# Split data into train and test sets
train, test = train_test_split(iris, test_size=0.25)

# Extract features and labels for training and testing
train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
train_y = train.Species
test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
test_y = test.Species


# In[21]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[22]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))


# In[23]:


#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# In[24]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))


# In[25]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))


# In[28]:


# Create a DataFrame to compare model scores
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Naive Bayes', 'KNN', 'Decision Tree'],
    'Score': [ 0.973, 0.947, 0.973, 0.947, 0.973]
})

# Sort and display model scores
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[29]:


# Display confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
confusion_mat = confusion_matrix(test_y, prediction)
print("Confusion matrix: \n", confusion_mat)
print(classification_report(test_y, prediction))


# In[ ]:


#Title: Flower Species Classification Project

#Maximized Accuracy:
#Utilized five classification algorithms (LR, SVM, KNN, Naive Bayes, Decision Tree) achieving high accuracy—topping at 0.973 .
#Visualized Patterns:
#Utilized Seaborn and Matplotlib for insightful analysis, uncovering correlations and distributions.
#Precision and Recall Evaluation:
#Examined models using confusion matrices and precision-recall metrics for in-depth model assessment.


# In[ ]:




