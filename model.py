#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:31:25 2019

@author: Junhong Shen
"""
"""""""""""""""""""""""""""""""""
Import Libraries
"""""""""""""""""""""""""""""""""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, metrics, preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB

"""""""""""""""""""""""""""""""""
General Idea
"""""""""""""""""""""""""""""""""
# Our goal is to train a classfier that classifies romance movies into two categories:
# recommended (those with rating >= 6.5) and unrecommended (those with rating < 6.5). After
# training, the classfier can tell the user whether a romance movie deserves watching
# or not given some basic information of the movie as input.
# To train the classifer, the features we decide to use are: region, year, type and 
# director. The target value is a boolean variable associated with rating.
# To obatin the best result, we have trained multiple classifiers and would like to
# get one with the best performance.

"""""""""""""""""""""""""""""""""
Helper Functions
"""""""""""""""""""""""""""""""""
def clean_year(y):
    """
    Helper function for preprocessing data, turn 'year' into integers that are 
    recentered to 1980 and get rid of '\N'
    """  
    try:
        return int(y) - 1980
    except:
        return 0

def clean_rating(y):
    """
    Helper function for preprocessing data, turn 'rating' into integers and get 
    rid of '\N' by giving it a default rating of 7
    """
    try:
        return int(y)
    except:
        return 7

def clean_string(y):
    """
    Helper function for preprocessing data, turn strings into list of words 
    and get rid of '\N'
    """
    y = str(y)
    if y == '\\N':
        return ''
    return y.split(',')[0].strip()

def clean_genre(y):
    """
    Helper function for preprocessing data, select data entries that are classified
    as Romance and filter out those that are not of interest
    """
    y = str(y)
    if "Romance" not in y:
        return ''
    return 'Romance'

def extract_dic(arr):
    """
    Helper function for getting the feature matrix using one-hot encoding.
    Use 'bag of words' model, create a dictionary that maps a catagory to a 
    unique index.
    """
    alist = {}
    size = 0
    for a in arr:
        if a == '\N' or a == '':
            alist[a] = -1
        if a in alist:
            continue
        else:
            alist[a] = size
            size += 1  
    return alist, size

def extract_mat(a, dic, size, n):
    """
    Helper function for getting the feature matrix using one-hot encoding.
    Use 'bag of words' model, map a categorical value to a vector containing a
    1 at the corresponding index specified in the dictionary and 0 at all the other
    places.
    In the returned matrix, each row represents a feature for a training sample. 
    """
    mat = np.zeros((n, size))
    for i in range(n):
        if a[i] == '\N':
            continue
        j = dic[a[i]]
        mat[i][j] = 1
    return mat


#sys.stdout.flush()
"""""""""""""""""""""""""""""""""
Preprocess Data
"""""""""""""""""""""""""""""""""
# read data from files
yearAndGenre = pd.read_csv('title.basics.tsv', sep='\t')
regionAndType = pd.read_csv('title.akas.tsv', sep='\t')
directors = pd.read_csv('title.crew.tsv', sep='\t')
rating = pd.read_csv('title.ratings.tsv', sep='\t')
regionAndType.rename(columns={'titleId':'tconst'}, inplace=True)

# preprocess data by applying different helper functions according to the format
# of the features
yearAndGenre['genres'] = yearAndGenre['genres'].apply(clean_genre)
regionAndType['region'] = regionAndType['region'].apply(clean_string)
regionAndType['types'] = regionAndType['types'].apply(clean_string)
directors['directors'] = directors['directors'].apply(clean_string)
yearAndGenre['startYear'] = yearAndGenre['startYear'].apply(clean_year)
rating['averageRating'] = rating['averageRating'].apply(clean_rating)

# create a mask that selects romance movie
mask = ((yearAndGenre['titleType'] == 'movie') &
        (yearAndGenre['genres'] == 'Romance'))

# merge all data that we want to use into a single table called "movies"
movies = yearAndGenre[mask].merge(rating, on='tconst')
movies = movies.merge(regionAndType, on='tconst')
movies = movies.merge(directors, on='tconst')


# histogram of ratings
#import seaborn as sns
#ratings = movies['averageRating']
#sns.set()
#plt.hist([ratings], color=['green'])
#plt.xlabel("Average Rating")
#plt.ylabel("Frequency")
#plt.xticks(range(1, 11))
#plt.title('Average Rating')
#plt.show()
#sns.distplot([ratings], norm_hist=True)


"""""""""""""""""""""""""""""""""
Get Feature Matrix and Label Array for Training and Test Set
"""""""""""""""""""""""""""""""""
# find features
n_samples = 10000
regions = movies['region'][:n_samples].values.tolist()
types = movies['types'][:n_samples].values.tolist()
directors = movies['directors'][:n_samples].values.tolist()
years = movies['startYear'][:n_samples].values.tolist()
years = np.array(years).reshape((n_samples, 1))
ratings = movies['averageRating'][:n_samples].values.tolist()
ratings = np.array(ratings)

# for categorical features, create dictionaries and then apply one-hot encoding
region_dic, rsize = extract_dic(regions)
region_mat = extract_mat(regions, region_dic, rsize, n_samples)
director_dic, dsize = extract_dic(directors)
director_mat = extract_mat(directors, director_dic, dsize, n_samples)
type_dic, tsize = extract_dic(types)
type_mat = extract_mat(types, type_dic, tsize, n_samples)

# concatenate all features into a single input feature matrix
feature = np.concatenate((region_mat, director_mat, type_mat, years), axis=1)

# create the label array for the data
label = np.zeros((n_samples,))
mask2 = ratings >= 6.5
label[mask2] = 1

# apply singular value decomposition to reduce 
# dimensionality for the input
tsvd = TruncatedSVD(n_components=100)
feature = tsvd.fit(feature).transform(feature)

permutation = np.random.permutation(feature.shape[0])
# shuffle the arrays by giving the permutation in the square brackets.
feature = feature[permutation]
label = label[permutation]

# separate training set and test set
train_size = 8500
train_data = feature[:train_size,:]
train_label = label[:train_size]
test_data = feature[train_size:,:]
test_label = label[train_size:]


"""""""""""""""""""""""""""""""""
Train Classifiers
"""""""""""""""""""""""""""""""""
print "Sample size: ", n_samples, " Training size: ", train_size
# apply random classifier to find out the baseline
dummy = DummyClassifier(strategy='uniform', random_state=1234)
dummy.fit(train_data, train_label)
print "Baseline for accuracy: ", dummy.score(test_data, test_label)

"""""""""""""""""""""""""""""""""
1. SVM with Gaussian Kernel
"""""""""""""""""""""""""""""""""
print "1. SVM with Gaussian Kernel: "

# define variables used for grid search
kf = StratifiedKFold(n_splits=3, random_state=1234)
C_range = 10. ** np.arange(0, 3)
gamma_range = 10. ** np.arange(-3, 0)

# do grid search for support vector machine to find the best parameters
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=kf, 
                    scoring='accuracy')

#grid = SVC(C=10, gamma=0.001, kernel='rbf')

# fit and test the model
grid.fit(train_data,train_label)
pred_label = grid.predict(test_data)
score = metrics.accuracy_score(test_label, pred_label)
print "  Best parameters: ", grid.best_params_
print "  Accuracy: ", score

"""""""""""""""""""""""""""""""""
2. Bernoulli Naive Bayes Classifier
"""""""""""""""""""""""""""""""""
print "2. Bernoulli Naive Bayes: "
clf = BernoulliNB()
model = clf.fit(train_data, train_label)
pred_label = clf.predict(test_data)
score = metrics.accuracy_score(test_label, pred_label)
print "  Accuracy: ", score

"""""""""""""""""""""""""""""""""
3. KNN
"""""""""""""""""""""""""""""""""
print "3. KNN: "
k_list = range(14)[1::2]
train_scores_knn = []
test_scores_knn = []
for k in k_list:
    clf = neighbors.KNeighborsClassifier(k, weights = 'uniform')
    trained_model = clf.fit(train_data, train_label)
    train_scores_knn.append(trained_model.score(train_data, train_label))
    pred_label = clf.predict(test_data)
    test_scores_knn.append(metrics.accuracy_score(test_label, pred_label))
idx = np.argmax(np.array(test_scores_knn))
print "  Best k: ", k_list[idx]
print "  Train score: ", train_scores_knn[idx]
print "  Accuracy: ", test_scores_knn[idx]

'''
"""""""""""""""""""""""""""""""""
Visualization
"""""""""""""""""""""""""""""""""
# Parameter selection for SVM
C_range = 2. ** np.arange(-10, 10)
gamma_range = 2. ** np.arange(-10, 10)
scores_svm_c = []

for c in C_range:
    svm = SVC(C=c, gamma=0.001, kernel='rbf')
    svm.fit(train_data,train_label)
    pred_label = svm.predict(test_data)
    scores_svm_c.append(metrics.accuracy_score(test_label, pred_label))

scores_svm_g = []
for g in gamma_range:
    svm = SVC(C=1, gamma=g, kernel='rbf')
    svm.fit(train_data,train_label)
    pred_label = svm.predict(test_data)
    scores_svm_g.append(metrics.accuracy_score(test_label, pred_label))

#plt.subplot(131)
plt.xlabel("C: 2^c")
plt.xticks(np.arange(-10, 10))
plt.ylabel("Accuracy")
plt.plot(np.arange(-10, 10), scores_svm_c, 'bs-', markersize=4)
plt.title('Parameter C Selection for Gaussian SVM')
plt.show()


#plt.subplot(132)
plt.xlabel("gamma : 2^g")
plt.xticks(np.arange(-10, 10))
plt.ylabel("Accuracy")
plt.plot(np.arange(-10, 10), scores_svm_g, 'g^-', markersize=4)
plt.title('Parameter gamma Selection for Gaussian SVM')
plt.show()


# k for KNN
#plt.subplot(133)
plt.xlabel("k")
plt.xticks(k_list)
plt.ylabel("Accuracy")
plt.plot(k_list, test_scores_knn, 'bs-', markersize=4)
plt.title('Parameter k Selection for KNN')
plt.show()

# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(grid, 
                                                        feature, 
                                                        label,
                                                        # Number of folds in cross-validation
                                                        cv=5
                                                        # Evaluation metric
                                                        
                                                        # Use all computer cores
                                            
                                                        # 50 different sizes of the training set
                                                        )

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
'''
