
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for data visualization

# Load the dataset
data_path = r"D:\Data_Science&AI\ClassRoomMaterial\September\28th - KNN\28th - KNN\projects\KNN\brest cancer.txt"
df = pd.read_csv(data_path, header=None)
# Display the first few rows of the DataFrame
print(df.head())

#Rename the column names
col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
df.columns = col_names


# drop Id column from dataset
df.drop('Id',axis=1,inplace=True)

#Frequency distribution of values in variables
for freq in df.columns:
    print("The frequency of  attributes ",df[freq].value_counts())
    
    
#Bare_Nuclei is Object type
print(df['Bare_Nuclei'].dtype)
    
#Convert data Object type to Bare_Nuclei to integer
df['Bare_Nuclei']=pd.to_numeric(df['Bare_Nuclei'],errors='coerce')
print(df['Bare_Nuclei'].dtype)

 
# view percentage of frequency distribution of values in `Class` variable
Bare_Nuclei_freq_percentage= df['Class'].value_counts()/np.float64(len(df))

# view summary statistics in numerical variables
round(df.describe(),2).T  #floating point -after Decimal 2 numbers

#Check 'na' or null values in dataframe
df.isna().sum()
#We can see that the Bare_Nuclei column contains 16 nan values.

# check frequency distribution of `Bare_Nuclei` column
df['Bare_Nuclei'].value_counts()

#check unique values
df['Bare_Nuclei'].unique()
#We can see that there are nan values in the Bare_Nuclei column.
df['Bare_Nuclei'].isna().sum()
#We can see that there are 16 nan values in the dataset. I will impute missing values after dividing the dataset into training and test set.
#============================================================================
#check frequency distribution of target variable Class
df['Class'].value_counts()
df['Class'].isna().sum()

df.to_csv('BreastCancerDataset.csv',index=False)

# view percentage of frequency distribution of values in `Class` variable
class_freq_Percentage = df['Class'].value_counts()/np.float64(len(df))
#We can see that the Class variable contains 2 class labels - 2 and 4.
# 2 stands for begin and 4 stands for malignant cancer.

#Outliers in numerical variables
# view summary statistics in numerical variables
outlier_describe= round(df.describe(),2).T
#============================================================================
# Data Visualization
#Univariate plots
# plot histograms of the variables
plt.rcParams['figure.figsize']=(30,25)
df.plot(kind='hist', bins=10, subplots=True, layout=(5,2), sharex=False, sharey=False)
plt.show()
#We can see that all the variables in the dataset are positively skewed.
#----------------------------------------------------------
#Multivariate plots
#Estimating correlation coefficients
correlation = df.corr()
#Our target variable is Class. So, we should check how each attribute correlates with the Class variable. We can do it as follows:-
correlation['Class'].sort_values(ascending=False)

#Correlation Heat Map
plt.figure(figsize=(10,8))
plt.title('Correlation of Attributes with Class variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()
#==========================================================
#model training
#Assign X and y variable
#Declare feature vector and target variable 
X = df.drop(['Class'],axis=1)
y = df['Class']
df1 = df.copy()
data_set = df.to_csv('breast_cancer_data.csv',index=False)
#Split data into separate training and test set 
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)



# check the shape of X_train and X_test
X_train.shape, X_test.shape
#===================================================
#Feature Engineering
#Engineering missing values in variables
X_train.isnull().sum()

#13 missing values [Bare_Nuclei]
# check missing values in numerical variables in X_test
X_test.isnull().sum()
#No missing values in X_test

# print percentage of missing values in the numerical variables in training set
for col in X_train.columns:
    if X_train[col].isnull().mean()>0:
        print(col,round(X_train[col].isnull().mean(),4))
# Bare_Nuclei 0.0233
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in X_train.columns:
        col_median=X_train[col].median()
        df1[col]= df1[col].fillna(col_median)      
        
  # check again missing values in numerical variables in X_train
df1.to_csv('BreastCancerDataset.csv',index=False)

X_train.isnull().sum()    
# check missing values in numerical variables in X_test

X_test.isnull().sum()
X_train.head()

X_train1 = X_train.copy()#Before standardscaler X_train
y_train1 = y_train.copy()#Before standardscaler X_train
#======================================================
#Feature Scaling
cols = X_train.columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.head()

#Fit K Neighbours Classifier to the training Set
#import KNeighbors classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier
#instance the model
knn = KNeighborsClassifier(n_neighbors=3)

#fit the model to the training set
knn.fit(X_train, y_train)

knn_train = X_train.copy()


# Optional: save to CSV

#predict test-set results
y_pred_knn = knn.predict(X_test)
y_pred_knn

#predict_proba method
#predict_proba method gives the probabilities for the target variable(2 and 4) in this case, in array form.
#2 is for probability of begining cancer and 4 is for probability of malignant cancer.
#==========================================================================
# probability of getting output as 2 - benign cancer
predict_proba_0 = knn.predict_proba(X_test)[:,0]

# probability of getting output as  4 - malignant cancer
predict_proba_1 = knn.predict_proba(X_test)[:,1]
#===========================================================================
# Check accuracy score
from sklearn.metrics import accuracy_score
y_pred_knn_acc = print('Model accuracy score:{0:0.4f}'.format(accuracy_score(y_test,y_pred_knn)))
#y_test are the true class labels and y_pred_knn are the predicted class labels in the test_set.
#Model accuracy score:0.9714

#Compare the train-set and test-set accuracy
y_pred_train = knn.predict(X_train)
y_pred_train_acc = print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#Training-set accuracy score: 0.9821

#Check for overfitting and underfitting
# print the scores on training and test set
X_train_y_train_acc = print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))
X_test_y_test_acc = print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))
#Training set score: 0.9821
#Test set score: 0.9714
#The training-set accuracy score is 0.9821 while the test-set accuracy to be 0.9714. These two values are quite comparable. So, there is no question of overfitting.

#==========================================================
#Compare model accuracy with null accuracy
# check class distribution in test set
y_test.value_counts()
#We can see that the occurences of most frequent class is 85. So, we can calculate null accuracy by dividing 85 by total number of occurences.
# check null accuracy score
null_accuracy = (85/(85+55))
print('Null accuracy score:{0:0.4f}'.format(null_accuracy))
#Null accuracy score:0.6071
#We can see that our model accuracy score is 0.9714 but null accuracy score is 0.6071. So, we can conclude that our K Nearest Neighbors model is doing a very good job in predicting the class labels.
#=================================================================================
#Rebuild kNN Classification model using different values of k 
# instantiate the model with k=5
knn_5 = KNeighborsClassifier(n_neighbors=5)
# fit the model to the training set
knn_5.fit(X_train, y_train)

knn5_train = X_train.copy()
# predict on the test-set
y_pred_5 = knn_5.predict(X_test)
print('Model accuracy score with k=5 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_5)))
y_pred_knn5_acc = accuracy_score(y_test, y_pred_5)
#Model accuracy score with k=5 : 0.9714
#===============================================================================
# instantiate the model with k=6
knn_6 = KNeighborsClassifier(n_neighbors=6)
# fit the model to the training set
knn_6.fit(X_train, y_train)
# predict on the test-set
y_pred_6 = knn_6.predict(X_test)
print('Model accuracy score with k=6 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_6)))
y_pred_knn6_acc = accuracy_score(y_test, y_pred_6)
#Model accuracy score with k=6 : 0.9786
#============================================================================
# instantiate the model with k=7
knn_7 = KNeighborsClassifier(n_neighbors=7)
# fit the model to the training set
knn_7.fit(X_train, y_train)
# predict on the test-set
y_pred_7 = knn_7.predict(X_test)
y_pred_train_knn7 =  knn_7.predict(X_train)

print('Model accuracy score with k=7 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_7)))
y_pred_knn7_acc = accuracy_score(y_test, y_pred_7)

knn7model = X_train.to_csv('knn7_model1.csv',index=False)
#y_pred_train_knn7.to_csv('KNN_7_Model.csv',index=False)
#Model accuracy score with k=7 : 0.9786

#===========================================================================
# instantiate the model with k=8
knn_8 = KNeighborsClassifier(n_neighbors=8)
# fit the model to the training set
knn_8.fit(X_train, y_train)
# predict on the test-set
y_pred_8 = knn_8.predict(X_test)
# Predict on the training set


y_pred_knn8_acc = accuracy_score(y_test, y_pred_8)
print('Model accuracy score with k=8 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_8)))
df1.to_csv('scaled_trained_data_knn7.csv',index=False)
#Model accuracy score with k=8 : 0.9786
#===========================================================================
# instantiate the model with k=9
knn_9 = KNeighborsClassifier(n_neighbors=9)
# fit the model to the training set
knn_9.fit(X_train, y_train)
knn9_model =X_train.copy()
# predict on the test-set
y_pred_9 = knn_9.predict(X_test)
print('Model accuracy score with k=9 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_9)))
y_pred_knn9_acc = accuracy_score(y_test, y_pred_9)
#Model accuracy score with k=9 : 0.9714
#============================================================================
# Confusion matrix 
# Print the Confusion Matrix with k =3 and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_knn)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

#========================================================================
# Print the Confusion Matrix with k =7 and slice it into four pieces

cm_7 = confusion_matrix(y_test, y_pred_7)

print('Confusion matrix\n\n', cm_7)

print('\nTrue Positives(TP) = ', cm_7[0,0])

print('\nTrue Negatives(TN) = ', cm_7[1,1])

print('\nFalse Positives(FP) = ', cm_7[0,1])

print('\nFalse Negatives(FN) = ', cm_7[1,0])

#======================================================================
#visualization for Knn with k =7
# visualize confusion matrix with seaborn heatmap

plt.figure(figsize=(6,4))

cm_matrix = pd.DataFrame(data=cm_7, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#==================================================================
#Classification metrices 
from sklearn.metrics import classification_report
class_report = print(classification_report(y_test, y_pred_7))

#Classification accuracy
TP = cm_7[0,0]
TN = cm_7[1,1]
FP = cm_7[0,1]
FN = cm_7[1,0]

# print classification accuracy
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

#Classification error
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

#Percision
# print precision score
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))

#Recall
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

#True Positive Rate
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

#False Positive Rate
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

#Specificity
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))

#=========================================================================
#Adjusting the classification threshold level
# print the first 10 predicted probabilities of two classes- 2 and 4
y_pred_prob = knn.predict_proba(X_test)[0:10]
y_pred_prob

# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - benign cancer (2)', 'Prob of - malignant cancer (4)'])
y_pred_prob_df

# print the first 10 predicted probabilities for class 4 - Probability of malignant cancer
knn.predict_proba(X_test)[0:10, 1]

# store the predicted probabilities for class 4 - Probability of malignant cancer
y_pred_1 = knn.predict_proba(X_test)[:, 1]
#=====================================================================
#Visualization
# plot histogram of predicted probabilities
# adjust figure size
plt.figure(figsize=(6,4))
# adjust the font size 
plt.rcParams['font.size'] = 12
# plot histogram with 10 bins
plt.hist(y_pred_1, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of malignant cancer')
#set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of malignant cancer')
plt.ylabel('Frequency')

#Observations
#We can see that the above histogram is positively skewed.
#The first column tell us that there are approximately 80 observations with 0 probability of malignant cancer.
#There are few observations with probability > 0.5.
#So, these few observations predict that there will be malignant cancer.

#======================================================================
#ROC-AUC
#ROC Curve
#Another tool to measure the classification model performance visually is ROC Curve. ROC Curve stands for Receiver Operating 
#Characteristic Curve. An ROC Curve is a plot which shows the performance of a classification model at various classification threshold levels.
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_1, pos_label=4)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Breast Cancer kNN classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
#==================================================

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_1, pos_label=4)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([-1,1], [1,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Breast Cancer kNN classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
#==========================================================
#ROC AUC
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred_1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
#ROC AUC : 0.9825
# calculate cross-validated ROC AUC 
from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(knn_7, X_train, y_train, cv=5, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
#Cross validated ROC AUC : 0.9910

#================================================================
#k-fold Cross Validation 
# Applying 10-Fold Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_7, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))

# compute Average cross-validation score
cross_validation_mean=print('Average cross-validation score: {:.4f}'.format(scores.mean()))
#============================================================
#Visualization
import matplotlib.pyplot as plt
df.boxplot(column=['Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 
                     'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 
                     'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses'])


#===================================================================
#knn_7 = pd.DataFrame(knn_7)
#knn_7.to_csv('knn_7.csv', index=False)

# Example DataFrame creation
model_trained_X = pd.DataFrame(X_train)
#model_trained_knn8.to_csv("model_trained_knn8.csv", index=False)
model_trained_X.to_csv("X_train.csv",index=False)

#=================================================================================
#acuuracy dataframe

# Create a dictionary with k values and their corresponding accuracy scores
accuracy_scores = {
    'Accuracy': [ y_pred_knn5_acc, y_pred_knn6_acc, y_pred_knn7_acc, y_pred_knn8_acc, y_pred_knn9_acc]
}
# Convert the dictionary into a DataFrame
accuracy_df = pd.DataFrame(accuracy_scores)
# Display the DataFrame
print(accuracy_df)
accuracy_df.to_csv('Knn_Accuracy.csv',index=False)
#=========================================================================
predictions = pd.DataFrame({
    'knn':[5,6,7,8,9],
   'predictions':[y_pred_5, y_pred_6, y_pred_7,y_pred_8,y_pred_9]
})
prediction = pd.DataFrame(predictions)
print(prediction)
#prediction.to_csv('Knn5-9_prediction.csv',index=False)
#=============================================
pred_knn_all = pd.DataFrame({
    'knn':[5,6,7,8,9], 
})
pred_knn_all = pd.DataFrame(pred_knn_all)
print(pred_knn_all)
pred_knn_all.to_csv('pred_knn_all.csv', index=False)
#====================================================================
pred = pd.DataFrame({
    'knn5':y_pred_5,
    'knn6':y_pred_6,
    'knn7':y_pred_7,
    'knn8':y_pred_8,
    'knn9':y_pred_9
})
pred.to_csv('Knn5-9_pred.csv',index=False)
print(pred)
#===========================================================
# Combine DataFrames vertically
combined_df = pd.concat([pred_knn_all,accuracy_df, pred], axis=1)
print(combined_df)
# Optional: save to CSV
combined_df.to_csv('Combined_Knn_Predictions.csv', index=False)

combined_df.to_csv('Combined_Knn_Predictions.csv', index=False)

y_pred_train_knn7.to_csv('Knn7.csv',index=False,axis=1)

import pickle

# Dumping the model to a file
with open('BreastCancer.pkl', 'wb') as file:
    pickle.dump(KNeighborsClassifier, file)

# Load the saved model
with open('BreastCancer.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
# Dumping the model to a file
with open('knn_7.pkl', 'wb') as file:
    pickle.dump(KNeighborsClassifier, file)

# Load the saved model
with open('knn_7.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


import os
print(os.getcwd())