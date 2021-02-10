#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

def clean_data(df):
    print("Cleaning data - ")
    least = 50 # At least 60 out of 98 non nan values are required.
    print("\n Dropping rows with more than 30nans-")
    df = df.dropna(axis=0,trh=least)
    df = df.reset_index(drop=True)
    return df

def get_count(df):
    pattern = ['AA', 'A', 'BBB', 'BB']
    ctr = [0,0,0,0,0]
    a = df['rating'].values.tolist()
    for i in a:
        fcnt=0
        for p in range(len(pattern)):
            if fcnt==0 and pattern[p] in i:
                ctr[p]=ctr[p]+1
                fcnt=1
            if fcnt==1:
                break
        if fcnt==0:
            ctr[-1]+=1
    return ctr

def make_y(df):
    l = list()
    for i in range(len(df['rating'])):
        if 'AA' in df['rating'].iloc[i]:
            l.append(0)
        elif 'A' in df['rating'].iloc[i]:
            l.append(1)
        elif 'BBB' in df['rating'].iloc[i]:
            l.append(2)
        elif 'BB' in df['rating'].iloc[i]:
            l.append(3)
        else:
            l.append(4)
    return l

def nan_by_col(df):
    dff = df.isnull().sum()
    y = dff.index.values.tolist()
    x = dff.values.tolist()
    return (x,y)
   

df = pd.read_csv("CR_data.csv")
df = clean_data(df)
ctr = get_count(df)
print(ctr)
print(sum(ctr))
for i in ctr:
    i = (i / sum(ctr)) * 100
    print(i)


# In[6]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import operator
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif

def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

# X
df = pd.read_csv('CR_data.csv')
X = df.drop('rating',axis=1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X.values)

# Y
l = list()
for i in range(len(df['rating'])):
    if df['rating'][i][0]=='A':
        l.append(1)
    elif df['rating'][i][:2]=='BB':
        l.append(2)
    else:
        l.append(3)
Y = l

X_indices = np.arange(X.shape[-1])
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X,Y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
data = sorted(enumerate(scores), key=operator.itemgetter(1))
nf = 98
k = list()
print(f"Top {nf} features by percentile are -")
for i in range(nf):
    k.append(df.columns[data[-i-1][0]])
    print(f"{df.columns[data[-i-1][0]]} -> {data[-i-1][1]}")

plt.show()


# In[44]:


import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.interpolate import UnivariateSpline, splrep
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report


# In[68]:


data = pd.read_csv('CR_data.csv')


# In[69]:


data


# In[18]:


data.rating


# In[19]:


data.rating.value_counts()


# In[20]:


data.shape


# In[21]:


random_state = 42

def shuffling(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


# In[22]:


def cleaning(df, trh):
    
    df = df.dropna(axis=0,trh=trh)
    # df = df.fillna(0)
    df = df.reset_index(drop=True)
    return df


# In[23]:


def convert_rating(df):
    l = list()
    nc = 5
    for i in range(len(df['rating'])):
        if 'AA' in df['rating'].iloc[i]:
            l.append(0)
        elif 'A' in df['rating'].iloc[i]:
            l.append(1)
        elif 'BBB' in df['rating'].iloc[i]:
            l.append(2)
        elif 'BB' in df['rating'].iloc[i]:
            l.append(3)
        else:
            l.append(4)
    return l


# In[24]:


def pca_(X, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    print("Explained variance after PCA = ", sum(pca.explained_variance_ratio_))
    print("Transforming ...")
    X = pca.transform(X)
    return X


# In[33]:


def select_nf_features(df, nf):
    with open('variance_features.pickle', 'rb') as handle:
        data = pickle.load(handle)
    l = list()
    for i in data:
        l.append(i[0])
    l = l[-nf:] # Top nf features. >0.5 var
    cols = df.columns[l]
    df = df[cols]
    return df


# In[26]:


df = cleaning(data, 60)


# In[27]:


df.shape


# In[28]:


X = df.drop('rating',axis=1)


# In[29]:


X.shape

X.info()


# In[31]:


y = convert_rating(df)
y


# In[36]:



def svc(nf, trh, n_comp=0):
    X = df.drop('rating',axis=1)
    print("Length of X = ", len(X))
    X = select_nf_features(X,nf)
    print("Columns = ", len(X.columns))
    imputer = SimpleImputer()
    X = imputer.fit_transform(X.values)
   
    scaler = StandardScaler()
    scaler.fit(X)
    x = scaler.transform(X)
    y = convert_rating(df)

    # Split data into training and test set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, stratify=y,test_size = 0.20, random_state=random_state)

    print('SVC')
    C = 100000000
    g = 0.0000001
    clf = svm.SVC(kernel='sigmoid', C=C, verbose=0, gamma=g, coef0=0.001)
    print(f"C = {C} and gamma = {g}")
    clf.fit(x_train, y_train)
    score = clf.score(x_dev, y_dev)
    y_pred = clf.predict(x_dev)
    target_names = ['0', '1', '2', '3', '4']
    y_true = y_dev
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("Accuracy = ", score)
    return score


s = svc(70,60)


# In[67]:


score = list()
a = range(1,30,3)
for i in a:
    s = run_svc(70, 60, i)
    score.append(s)


# In[52]:



random_state = 42

def shuffle(df, n=1, axis=0):
    print("Shuffling data-")
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

def clean_data(df, trh=60):
    print("Cleaning data-")
    print(f"Dropping rows with more than {98-trh} nans")
    df = df.dropna(axis=0,trh=trh)
    df = df.reset_index(drop=True)
    return df

def make_y(df):
    l = list()
    nc = 5
    for i in range(len(df['rating'])):
        if 'AA' in df['rating'].iloc[i]:
            l.append(0)
        elif 'A' in df['rating'].iloc[i]:
            l.append(1)
        elif 'BBB' in df['rating'].iloc[i]:
            l.append(2)
        elif 'BB' in df['rating'].iloc[i]:
            l.append(3)
        else:
            l.append(4)
    return l

def select_nf_features(df, nf):
    with open('variance_features.pickle', 'rb') as handle:
        data = pickle.load(handle)
    l = list()
    for i in data:
        l.append(i[0])
    l = l[-nf:] # Top nf features. >0.5 var
    cols = df.columns[l]
    df = df[cols]
    return df

def pca_(X, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    print("Explained variance after PCA = ", sum(pca.explained_variance_ratio_))
    print("Transforming ...")
    X = pca.transform(X)
    return X

def rfc(nf, trh, n_comp=0):
    df = pd.read_csv('dataset.csv')
    df = clean_data(df, trh=trh)
    X = df.drop('rating',axis=1)
    print("Columns = ", len(X.columns))
    imputer = SimpleImputer()
    X = imputer.fit_transform(X.values)
    if(n_comp>0):
        X = pca_(X,n_comp)
    scaler = StandardScaler()
    scaler.fit(X)
    x = scaler.transform(X)
    y = make_y(df)
    ya = array(y)
    kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=20)
    a = kfold.split(x, ya)
    cvscores = []
    for t in a:
        n_estimators=32
        max_features=5
        clf = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features=50)
        clf.fit(x[t[0]], ya[t[0]])
        y_pred = clf.predict(x[t[1]])
        scores = clf.score(x[t[1]], ya[t[1]])
        target_names = ['0', '1', '2', '3', '4']
        y_true = ya[t[1]]
        print(confusion_matrix(list(ya[t[1]]), list(y_pred)))
        print(classification_report(y_true, y_pred, target_names=target_names))
        print("acc: %.2f%%" % (scores*100))
        cvscores.append(scores * 100)
    mean_score = np.mean(cvscores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return mean_score

rfc(77,85)


# In[54]:



l = range(5,50,3)
scores = list()
for i in l:
    scores.append(run(i,60))
plt.figure()
xnew = np.linspace(l[0],l[-1],100)
power_smooth = splrep(l,scores,xnew)
plt.plot(xnew, power_smooth)
plt.plot(l,scores)
plt.title('No of PCA components vs. Accuracy for RandomForestClassifier')
plt.xlabel('n_comp')
plt.ylabel('acc')
plt.show()
print("MAX: ", max(scores))
dt = sorted(enumerate(scores), key=operator.itemgetter(1))
print(dt)


# In[59]:


#knc
def knc(nf, trh,n_comp=0):
    df = pd.read_csv('dataset.csv')
    df = clean_data(df, trh=trh)
    # X
    X = df.drop('rating',axis=1)
    print("Length of X = ", len(X))
    X = select_nf_features(X,nf)
    print("Columns = ", len(X.columns))
    imputer = SimpleImputer()
    X = imputer.fit_transform(X.values)
    if n_comp>0:
        X = pca_(X,n_comp)
    scaler = StandardScaler()
    scaler.fit(X)
    x = scaler.transform(X)
    y = make_y(df)

    # Split data into training and test set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, stratify=y,test_size = 0.25, random_state=random_state)

    print('Nearest neighbors')
    n = 2
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance',algorithm='kd_tree', p=1)

    knn.fit(x_train, y_train)
    score = knn.score(x_dev, y_dev)
    preds = knn.predict(x_dev)
    print(type(preds))
    print(type(y_dev))
    print(confusion_matrix(y_dev, preds))
    print("Accuracy = ", score)
    return score

knc = knc(41,85)

