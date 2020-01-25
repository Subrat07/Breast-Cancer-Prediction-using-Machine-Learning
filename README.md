import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

df = pd.read_csv("/content/data.csv",index_col= 0)

df.info()

df.describe().T

df.head()

df.tail()

df.isnull().any()

df["Unnamed: 32"].values

df.drop("Unnamed: 32",axis =1,inplace = True)

df.isnull().any()

df[df["diagnosis"]== "M"].shape[0]/df.shape[0]*100

df[df["diagnosis"]=="B"].shape[0]/df.shape[0]*100

df.diagnosis.value_counts()

df.shape

df.columns

x = df.copy()
print(x.shape)
x.drop("diagnosis",axis =1, inplace = True)
print(x.shape)

y =df.diagnosis 
type(y)

y = y.values
type(y)

print(y)

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

x = x.values
type(x)

x.shape

sc = StandardScaler()
x = sc.fit_transform(x)
x

print(x)

#EDA-Exploratory Data Analysis

df.columns

df.head()

sns.countplot(x= df.diagnosis, data = df)
plt.show()

melted_data = pd.melt(df,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()

f,ax=plt.subplots(1,2,figsize=(14,7))
df['diagnosis'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow = True)
ax[0].set_title('cancer type', fontSize = 20)
ax[0].set_ylabel('')
sns.countplot('diagnosis',data=df,ax=ax[1])
ax[1].set_title('cancer type', fontSize = 20)
ax[1].set_ylabel("Count",fontSize =15)
ax[1].set_xlabel("Diagnosis", fontSize = 15)
plt.show()

plt.figure(figsize=(15,8))
m = plt.hist(df[df["diagnosis"] == "M"].radius_mean,bins=30,color = "red", label = "Malignant")
b = plt.hist(df[df["diagnosis"] == "B"].radius_mean,bins=30,color = "green",label = "Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors", fontSize = 20)
plt.show()


def dist_plot(data_select):
  plt.figure(figsize = (15, 8))
  sns.distplot(df[df["diagnosis"] == "M"][str(data_select)], hist = True, color = "red", label = "Maligant")
  sns.distplot(df[df["diagnosis"] == "B"][str(data_select)], hist = True, color = "green", label = "Bening")
  plt.legend(fontsize = 10)
  plt.xlabel("Values {}".format(data_select))
  plt.ylabel("Frequency")
  plt.title("Histogram of Radius Mean for Bening and Malignant Tumors", fontsize = 12)
  plt.show()
  
  
 #plot distribution 'mean'
dist_plot('radius_mean')
dist_plot('texture_mean')
dist_plot('perimeter_mean')
dist_plot('area_mean')
dist_plot('smoothness_mean')
dist_plot('compactness_mean')
dist_plot('concavity_mean',)
dist_plot('concave points_mean')
dist_plot('symmetry_mean')
dist_plot('fractal_dimension_mean')


#plot distribution 'se'
dist_plot('radius_se')
dist_plot('texture_se')
dist_plot('perimeter_se')
dist_plot('area_se')
dist_plot('smoothness_se')
dist_plot('compactness_se')
dist_plot('concavity_se',)
dist_plot('concave points_se')
dist_plot('symmetry_se')
dist_plot('fractal_dimension_se')

#plot distribution 'worst'
dist_plot('radius_worst')
dist_plot('texture_worst')
dist_plot('perimeter_worst')
dist_plot('area_worst')
dist_plot('smoothness_worst')
dist_plot('compactness_worst')
dist_plot('concavity_worst',)
dist_plot('concave points_worst')
dist_plot('symmetry_worst')
dist_plot('fractal_dimension_worst')

correlation = df.corr()
correlation

plt.figure(figsize=(20,20))
sns.heatmap(correlation,annot=True, fmt= '.2f',annot_kws={'size': 10}, xticklabels = "auto", center =0,linewidths=.5, square= True)
plt.show()

def plot_feat1_feat2(first,second):
  plt.figure(figsize=(7,7))
  sns.scatterplot(x = first, y = second, hue = "diagnosis", data = df, palette=['red','green'], legend='full')
  plt.show()
  
#Positive correlated features

plot_feat1_feat2('perimeter_mean','radius_worst')
plot_feat1_feat2('area_mean','radius_worst')
plot_feat1_feat2('texture_mean','texture_worst')
plot_feat1_feat2('area_worst','radius_worst')

plot_feat1_feat2('smoothness_mean','texture_mean')
plot_feat1_feat2('radius_mean','fractal_dimension_worst')
plot_feat1_feat2('texture_mean','symmetry_mean')
plot_feat1_feat2('texture_mean','symmetry_se')

plot_feat1_feat2('area_mean','fractal_dimension_mean')
plot_feat1_feat2('radius_mean','fractal_dimension_mean')
plot_feat1_feat2('area_mean','smoothness_se')
plot_feat1_feat2('smoothness_se','perimeter_mean')

means = [col for col in df.columns if col.endswith('_mean')]
se = [col for col in df.columns if col.endswith('_se')]
worst = [col for col in df.columns if col.endswith('_worst')]

def plot_violinplot(feat_list):
    scaler = StandardScaler()
    feat_scaled = pd.DataFrame(scaler.fit_transform(df[feat_list]),columns=feat_list, index = df.index)
    data = pd.concat([df['diagnosis'],feat_scaled],axis=1)
    df_melt = pd.melt(frame=data, value_vars=feat_list, id_vars=['diagnosis'])
    fig, ax = plt.subplots(1, 1, figsize = (7, 7),dpi = 100)
    sns.violinplot(x="variable",y="value",hue = "diagnosis",data=df_melt,split = True, inner="quart",palette='Set2').set_title('Distribution of features among malignant and benign tumours'.format(feat_list))
    plt.xticks(rotation=45)
    L=plt.legend()
    L.get_texts()[0].set_text('Benign')
    L.get_texts()[1].set_text('Malignant')
    
plot_violinplot(means)

plot_violinplot(se)

plot_violinplot(worst)

means_and_worst = pd.merge(df[means],df[worst],left_index= True,right_index= True, how = "left")
sns.pairplot(means_and_worst)

# Classification Models

#Predictive Modeling :

#We have gained some insights from the EDA part. But with that, we cannot accurately predict or tell whether a passenger will #survive or die. So now we will predict the whether the Passenger will survive or not using some great Classification #Algorithms.Following are the algorithms I will use to make the model:

#1. Logistic Regression

#2. Random Forest

#3. Support Vector Machines(Linear and radial)

#4. Decision Tree

#5. K-Nearest Neighbours

#6. Naive Bayes

random_state =123
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = random_state, stratify = y, shuffle = True)

# Logistic Regression

log_clf = LogisticRegression(random_state = random_state)
param_grid = {
            'penalty' : ['l2','l1'],  
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }

CV_log_clf = GridSearchCV(estimator = log_clf, param_grid = param_grid , scoring = 'accuracy', verbose = 1)
CV_log_clf.fit(x_train, y_train)

best_parameters = CV_log_clf.best_params_
print('The best parameters for using this model is', best_parameters)

CV_log_clf = LogisticRegression(C = best_parameters['C'], 
                                penalty = best_parameters['penalty'], 
                                random_state = random_state)

CV_log_clf.fit(x_train, y_train)
y_pred = CV_log_clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot =True, fmt = "d")

print("Accuracy_score    :  ",accuracy_score(y_test, y_pred))
print("precision_score   :  ",precision_score(y_test, y_pred))
print("Recall_score      :  ",recall_score(y_test, y_pred))
print("F1_score          :  ",f1_score(y_test, y_pred))

classification_report(y_test, y_pred,target_names=["B","M"])

# Support Vector Machine Classifier

svc = svm.SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot =True, fmt = "d")

print("Accuracy_score    :  ",accuracy_score(y_test, y_pred))
print("precision_score   :  ",precision_score(y_test, y_pred))
print("Recall_score      :  ",recall_score(y_test, y_pred))
print("F1_score          :  ",f1_score(y_test, y_pred))

classification_report(y_test, y_pred,target_names=["B","M"])

# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot =True, fmt = "d")

print("Accuracy_score    :  ",accuracy_score(y_test, y_pred))
print("precision_score   :  ",precision_score(y_test, y_pred))
print("Recall_score      :  ",recall_score(y_test, y_pred))
print("F1_score          :  ",f1_score(y_test, y_pred))

classification_report(y_test, y_pred,target_names=["B","M"])

# K Neighbour Classifier

knc = KNeighborsClassifier()
knc.fit(x_train,y_train)

y_pred = knc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot =True, fmt = "d")

print("Accuracy_score    :  ",accuracy_score(y_test, y_pred))
print("precision_score   :  ",precision_score(y_test, y_pred))
print("Recall_score      :  ",recall_score(y_test, y_pred))
print("F1_score          :  ",f1_score(y_test, y_pred))

classification_report(y_test, y_pred,target_names=["B","M"])

# Naive Bayes

nb=GaussianNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot =True, fmt = "d")

print("Accuracy_score    :  ",accuracy_score(y_test, y_pred))
print("precision_score   :  ",precision_score(y_test, y_pred))
print("Recall_score      :  ",recall_score(y_test, y_pred))
print("F1_score          :  ",f1_score(y_test, y_pred))

classification_report(y_test, y_pred,target_names=["B","M"])

# **Grid** **Search**  for hyper parameter optimization

#The confusion matrix, also known as the error matrix, allows visualization of the performance of an algorithm :

   #true positive (TP) : Malignant tumour correctly identified as malignant
   #true negative (TN) : Benign tumour correctly identified as benign
   #false positive (FP) : Benign tumour incorrectly identified as malignant
   #false negative (FN) : Malignant tumour incorrectly identified as benign

#Metrics :

   #Accuracy : (TP +TN) / (TP + TN + FP +FN)
   #Precision : TP / (TP + FP)
   #Recall : TP / (TP + FN)
   
def Classification_model_gridsearchCV(model,param_grid):
    clf = GridSearchCV(model,param_grid,cv=10,scoring="accuracy") 
    clf.fit(x_train,y_train)
    print("\nThe best parameter found on development set is :")
    print(clf.best_params_)
    print("\nThe bset estimator is ")
    print(clf.best_estimator_)
    print("\nThe best score is ")
    print(clf.best_score_)
   
 # support vector machine
model=svc

C=[0.001,0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
param_grid={'kernel':kernel,'C':C,'gamma':gamma}

Classification_model_gridsearchCV(model,param_grid)

# knearest neighbour
model = knc

k_range = list(range(1, 30))
leaf_size = list(range(1,30))
weight_options = ['uniform', 'distance']

param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}

Classification_model_gridsearchCV(model,param_grid)

# decision tree
model = dt
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10], 
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }

Classification_model_gridsearchCV(model,param_grid)

# XGBoost

import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,x_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())

xgboost.fit(x_train,y_train)

y_pred = xgboost.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot =True, fmt = "d")

print("Accuracy_score    :  ",accuracy_score(y_test, y_pred))
print("precision_score   :  ",precision_score(y_test, y_pred))
print("Recall_score      :  ",recall_score(y_test, y_pred))
print("F1_score          :  ",f1_score(y_test, y_pred))

classification_report(y_test, y_pred,target_names=["B","M"])

# ANN

import keras
from keras import backend as k
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import rmsprop, Adam

y_test.shape

classifier = Sequential()

classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dropout(0.1))
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(x_train, y_train, batch_size=100, nb_epoch=150)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
cm

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100)))

sns.heatmap(cm, annot = True)

def build_classifier(opt,inty,l):
    model = Sequential()
    model.add(Dense(output_dim = 16, kernel_initializer = inty, activation = 'relu', input_dim = 30))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim=16, kernel_initializer = inty, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim=1, kernel_initializer = inty, activation='sigmoid'))
    model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.optimizer.lr=l
    return model

classifier=KerasClassifier(build_fn=build_classifier)
# parameters tuning grid search
parameters={
        "batch_size":[25,28,32],
        "epochs":[100,500],
        "opt":["adam","rmsprop"],
        "inty":["uniform","glorot_uniform"],
         "l":[0.0001,0.001,0.00001]
        }

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring="accuracy",
                         cv=10)
grid_search=grid_search.fit(x_train,y_train)

best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_
print(best_parameters)
print(best_accuracy)

def build_classifier():
    model=Sequential()
    model.add(Dense(units=16,kernel_initializer="uniform",activation="relu",input_shape=(30,)))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=16,kernel_initializer="uniform",activation="relu"))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return model

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,n_jobs=1)

print(accuracies)
print(accuracies.mean())
print(accuracies.std())













