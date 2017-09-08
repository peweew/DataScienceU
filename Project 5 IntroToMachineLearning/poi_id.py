#!/usr/bin/python

import sys
import pickle
import numpy as np
from operator import itemgetter, attrgetter, methodcaller, add
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix 
from copy import copy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from random import randint
from sklearn.grid_search import GridSearchCV 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.cross_validation import train_test_split

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from operator import itemgetter, attrgetter, methodcaller
from tester import test_classifier
 

 

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = []  

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


data_keys = data_dict['METTS MARK'].keys()

# Get the name of features in dataset
# The number of record
# The number of features
# The number of poi persons in dataset.
print "\nThe name of features: \n", data_keys
print "The number of records: ", len(data_dict)
print "The length of features: ", len(data_keys) 
print "The number of poi: ", sum([1 for key, record in data_dict.iteritems() if record['poi'] == 1])

# Get the number of NaN in each features
feature_count_nan = {}
for feature in data_keys:
    feature_count_nan[feature] = sum([1 for key, record in data_dict.iteritems() if record[feature] == 'NaN'])

feature_count_nan = sorted(feature_count_nan.iteritems(), key=lambda (k,v): (v,k), reverse = True)
print feature_count_nan

### Task 2: Remove outliers
 
# Assisting function to get the number of NaN in a record
def numnan(record):
    num_nan = 0
    for rkey, rrecord in record.iteritems():
        if rrecord == 'NaN':
            num_nan += 1;
    return num_nan

# print the record with the most NaN value

print '\nThe record has the most NaN value in record and its content: \n', sorted(list([list([key, numnan(record)]) for key, record in data_dict.iteritems()]), key=itemgetter(1), reverse=True)[0]

# remove two outliers: 1). 'TOTAL' record are meaningless. This record aggregate all the records information
# 2). 'LOCKHART EUGENE E' record has 20 NaN in features except 'poi'. This record does not help for predict poi

data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

# The number of record after removing outliers.
print "\nThe number of records after remove outlier: ", len(data_dict)





### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = copy(data_dict)
features_list = copy(data_keys)

# Remove email_address feature. This feature does not relate to poi.
features_list.remove('email_address')

# Move the poi feature as the first feature
features_list.remove('poi')  
features_list = ['poi'] + features_list

 
#Create two new features: 
#1). from_poi_message_rate, the ratio between the number of emails from poi email to this person and total number of emails to this person.
#2). to_poi_message_rate, the ratio between the number of emails from this person to poi and total number of emails from this person.
for k, record in my_dataset.iteritems():
    if record['to_messages'] == 'NaN' or record['from_poi_to_this_person'] == 'NaN':
        record['from_poi_message_rate'] = 0
    else:
        record['from_poi_message_rate'] = float(record['from_poi_to_this_person']) / float(record['to_messages'])
    if record['from_messages'] == 'NaN' or record['from_this_person_to_poi'] == 'NaN':
        record['to_poi_message_rate'] = 0
    else:
        record['to_poi_message_rate'] = float(record['from_this_person_to_poi']) / float(record['from_messages'])
 
features_list = features_list + ['from_poi_message_rate'] + ['to_poi_message_rate']
 

# Calculate the score of the features using SelectKBest function.
# Sort the score of features in descend order
# Select the highest features
# let the features_list equal to the highest k features
# Pick the feature size k = 2, 4, 6, 8, 10, 12, 14
# Test the best feature size by selected machine learning algorithm.

FEATURE_SIZE = 4

temp_data = featureFormat(my_dataset, features_list) 
labels, features = targetFeatureSplit(temp_data)

selector = SelectKBest(f_classif, k = FEATURE_SIZE)
selector.fit(features, labels)
features = selector.transform(features) 

scores = np.around(selector.scores_, decimals=3) 
score_rank = sorted(zip(features_list[1:], scores), key = itemgetter(1), reverse = True)

print '\n The score of features: \n',  score_rank
 
features_list = ['poi'] + [record[0] for record in score_rank][:FEATURE_SIZE]  



### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
 
### Scale the features 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)
 


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

   

clf = DecisionTreeClassifier() 
test_classifier(clf, my_dataset, features_list)

#clf = RandomForestClassifier() 
#test_classifier(clf, my_dataset, features_list)

#clf = AdaBoostClassifier() 
#test_classifier(clf, my_dataset, features_list)

clf = GaussianNB() 
test_classifier(clf, my_dataset, features_list)  

#clf = KMeans(n_clusters=2) 
#test_classifier(clf, my_dataset, features_list)  

clf = LogisticRegression() 
test_classifier(clf, my_dataset, features_list)   



 

 


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# find the best parameter of KMeans and LogisticRegression using GridSearchCV

param_grid_kmean = { 'tol': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8,  1e-7, 1e-6, 1e-5, 1e-4, 1e-3], 'max_iter': [300, 600, 1000], 'n_init': [3, 6, 10, 15],}  
param_grid_dt = {'max_depth': [None, 2, 4, 6, 8, 10], 'min_samples_split' : [2, 4, 6, 10], 'min_samples_leaf' : [1, 2, 4, 8], }
 
# Test the parameter of the selected three machine learning algorithm.
 
clf = GridSearchCV(KMeans(n_clusters=2), param_grid_kmean)
clf = clf.fit(features, labels) 
print clf.best_estimator_

clf = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_clusters=2, n_init=6, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=1e-07, verbose=0)
test_classifier(clf, my_dataset, features_list) 



clf = GridSearchCV(DecisionTreeClassifier(), param_grid_dt)
clf = clf.fit(features, labels) 
print clf.best_estimator_

clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,
        max_features=None, max_leaf_nodes=None,
        min_impurity_split=1e-07, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        presort=False, random_state=None, splitter='best')
test_classifier(clf, my_dataset, features_list) 





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)