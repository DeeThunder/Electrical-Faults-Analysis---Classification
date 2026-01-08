#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Transmission lines are used to transfer electrical energy from one point to another, typically over long distances. These lines are made up of conductors that are supported by towers or poles, and they are designed to minimize energy losses due to resistance and other factors. Transmission lines can carry AC or DC currents and are typically used to connect power plants to substations or to distribute electricity to industrial, commercial, and residential customers.
# 
# The design and operation of transmission lines are critical to ensuring reliable and efficient electricity delivery. The length, voltage, and current capacity of a transmission line are all factors that must be carefully considered in its design. The condition of the line, including the level of insulation, is also important in preventing power outages and ensuring public safety. There are different types of transmission lines, including overhead lines, underground lines, and submarine cables. Each type has its own advantages and disadvantages, depending on the specific application. Overall, transmission lines play a vital role in modern society, enabling the efficient and reliable transfer of electrical energy across long distances.

# <div class="alert alert-success" role="alert">
# Therefore, it is important to know what factors can cause problems in these transmission lines and predict outages to ensure smooth operations. In this project, the idea is to analyse the voltage and current patterns across the three phases of the transmission lines to detect faults, and if a fault exists, we futher classify its type to fix the problem.
# </div>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# In[ ]:


sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (13,9)
plt.rcParams['font.size'] = 20
warnings.filterwarnings('ignore')


# # Data and Overview
# 
# ### Binary and Multiclass Classification
# 
# The dataset contains the possibility for both types of classification. Firstly we predict whether the electrical relays have a fault or not. Futhermore, we predict where the fault is. Based on where the fault is found, there are different permutations. <br>
# 
# **Starting with Binary Classification**, we detect whether a fault is detected or not. <br>
# 
# **For Multiclass Classification**  <br>
# Output: [G C B A] <br>
# * [0 0 0 0] - No Fault <br>
# * [1 0 0 1] - LG fault (Between Phase A and Ground) <br>
# * [0 0 1 1] - LL fault (Between Phase A and Phase B) <br>
# * [1 0 1 1] - LLG Fault (Between Phases A,B and Ground) <br>
# * [0 1 1 1] - LLL Fault (Between all three phases) <br>
# * [1 1 1 1] - LLLG fault (Three phase symmetrical fault) <br>
# 
# Here, we conclude that there are 6 types of faults, hence 6 output classes.
# 

# In[ ]:


binary_data = pd.read_csv('detect_dataset.csv')
multi_data = pd.read_csv('classData.csv')


# In[ ]:


binary_data.info()


# In[ ]:


binary_data.head()


# In[ ]:


any(binary_data.isna().sum() > 0)


# In[ ]:


multi_data.info()


# In[ ]:


multi_data.head()


# In[ ]:


any(multi_data.isna().sum() > 0)


# # Binary Classification - Fault Detection

# #### Columns 7 and 8 contain no information, so we drop them

# In[ ]:


binary_data.drop(binary_data.iloc[:,[7,8]], axis=1, inplace=True)


# In[ ]:


print(f'Number of Samples: {binary_data.shape[0]}\nNumber of Features: {binary_data.shape[1]}')


# In[ ]:


sns.heatmap(binary_data.corr(), annot=True, cmap='Blues')
plt.show()


# ### Voltage vs Current

# In[ ]:


plt.figure(figsize=(25,6))

a1 = plt.subplot2grid((1,3),(0,0))
a1.scatter(binary_data['Ia'], binary_data['Va'])
a1.set_title('Line a')
a1.set_xlabel('Ia')
a1.set_ylabel('Va')

a2 = plt.subplot2grid((1,3),(0,1))
a2.scatter(binary_data['Ib'], binary_data['Vb'])
a2.set_title('Line b')
a2.set_xlabel('Ib')
a2.set_ylabel('Vb')

a3 = plt.subplot2grid((1,3),(0,2))
a3.scatter(binary_data['Ic'], binary_data['Vc'])
a3.set_title('Line c')
a3.set_xlabel('Ic')
a3.set_ylabel('Vc')

plt.show()


# ### Composition of Target variable

# In[ ]:


plt.pie(x=binary_data['Output (S)'].value_counts(), labels=['No Fault', 'Fault'],
        explode = [0, 0.2], autopct= '%1.1f%%', labeldistance=1.15,
       colors=['#0c06c7', '#05daed'])
plt.show()


# In[ ]:


def dist(cola,colb):

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(18,10))

    sns.distplot(binary_data[cola], label='Line Current', hist=True, color='#fc0328', ax=axs[0])
    sns.distplot(binary_data[colb], label='Line Voltage', hist=True, color='#0c06c7', ax=axs[1])

    axs[0].legend(loc='upper right', prop={'size': 12})
    axs[1].legend(loc='upper right', prop={'size': 12})

    plt.show()


# In[ ]:


lines = [
    ('Ia', 'Va'),
    ('Ib', 'Vb'),
    ('Ic', 'Vc')
]

for cola,colb in lines:
    dist(cola,colb)
    print('\n')


# # Model Building - Binary Classifier

# In[ ]:


y = binary_data.iloc[:,0]
X = binary_data.iloc[:,1:7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


model_1 = RandomForestClassifier()
model_1.fit(X_train, y_train)
y_pred = model_1.predict(X_test)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='.4g')
plt.show()


# In[ ]:


print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:.03f}%')
print(f'Precision Score: {precision_score(y_test, y_pred)*100:.03f}%')
print(f'Recall Score: {recall_score(y_test, y_pred)*100:.03f}%')


# In[ ]:


print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


# # Multiclass Classification

# In[ ]:


print(f'Number of Samples: {multi_data.shape[0]}\nNumber of Features: {multi_data.shape[1]}')


# In[ ]:


plt.figure(figsize=(18,12))
sns.heatmap(multi_data.corr(), annot=True, cmap='Blues')
plt.show()


# ### Voltage vs Current

# In[ ]:


plt.figure(figsize=(25,6))

a1 = plt.subplot2grid((1,3),(0,0))
a1.scatter(multi_data['Ia'], multi_data['Va'])
a1.set_title('Line a')
a1.set_xlabel('Ia')
a1.set_ylabel('Va')

a2 = plt.subplot2grid((1,3),(0,1))
a2.scatter(multi_data['Ib'], multi_data['Vb'])
a2.set_title('Line b')
a2.set_xlabel('Ib')
a2.set_ylabel('Vb')

a3 = plt.subplot2grid((1,3),(0,2))
a3.scatter(multi_data['Ic'], multi_data['Vc'])
a3.set_title('Line c')
a3.set_xlabel('Ic')
a3.set_ylabel('Vc')

plt.show()


# In[ ]:


def dist(cola,colb):

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(18,10))

    sns.distplot(multi_data[cola], label='Line Current', hist=True, color='#fc0328', ax=axs[0])
    sns.distplot(multi_data[colb], label='Line Voltage', hist=True, color='#0c06c7', ax=axs[1])

    axs[0].legend(loc='upper right', prop={'size': 12})
    axs[1].legend(loc='upper right', prop={'size': 12})

    plt.show()


# In[ ]:


for cola, colb in lines:
    dist(cola,colb)
    print('\n')


# # Model Building - Multiclass Classifier

# #### Since output is of four different types, we put them together in one column and there permutations become different output classes

# In[ ]:


multi_data['faultType'] = multi_data['G'].astype(str) + multi_data['C'].astype(str) + multi_data['B'].astype(str) + multi_data['A'].astype(str)
multi_data.head()


# In[ ]:


plt.pie(multi_data['faultType'].value_counts(), autopct='%1.1f%%',
       labels=['No Fault', 'LLG Fault', 'LLLG Fault', 'LG Fault', 'LLL Fault', 'LL Fault'])
plt.show()


# In[ ]:


X = multi_data.drop(['G','C','B','A','faultType'], axis=1)
y = multi_data['faultType']


# In[ ]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


model_2 = RandomForestClassifier()
model_2.fit(X_train, y_train)
y_pred = model_2.predict(X_test)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='.4g')
plt.show()


# In[ ]:


print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:.03f}%')


# In[ ]:


print(classification_report(y_test, y_pred))


# # Class-wise Error Analysis
# 
# Class-wise error analysis is a method used to evaluate the performance of a machine learning classifier in multiclass classification tasks. It involves computing different metrics for each class separately, rather than aggregating them across all classes as in overall accuracy. The main advantage of class-wise error analysis is that it provides a more detailed understanding of the strengths and weaknesses of the classifier for each class, which can help in identifying areas for improvement.
# 
# By computing metrics such as accuracy and F1-score for each class, we can identify which classes are being misclassified the most and which ones are being classified correctly. This information can be used to fine-tune the classifier and improve its performance.

# In[ ]:


cr = classification_report(y_test, y_pred, output_dict=True)
cr.keys()


# #### Extract Class wise F1 scores from Classification report

# In[ ]:


f1_scores = {}
for key, val in cr.items():
    if key == 'accuracy':
        break
    class_name = le.inverse_transform([int(key)])[0]
    f1_scores[class_name] = val["f1-score"]
sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


f1_scores_df = pd.DataFrame({'class': f1_scores.keys(),
                             'f1': f1_scores.values()})
f1_scores_df


# In[ ]:


sns.barplot(data=f1_scores_df, x='f1', y='class')
plt.title("Class-wise F1 score")
plt.show()


# ### Inferences
# 
# * Two classes show particularly lower F1 score while other classes are classified very well
# * These classes are `0111` (Fault between all three phases) and `1111` (Symmetrical Fault between all three phases)
# * We can see how this can be confusing for our model. We can further analyze this observation to improve the classifier.
