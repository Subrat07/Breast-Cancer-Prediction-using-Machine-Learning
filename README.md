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
