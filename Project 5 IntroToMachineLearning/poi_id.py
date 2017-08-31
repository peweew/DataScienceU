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

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from operator import itemgetter, attrgetter, methodcaller


 

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_keys = data_dict['METTS MARK'].keys()

print data_dict['METTS MARK']
print "\nThe name of features: \n", data_keys
print "The number of records: ", len(data_dict)
print "The length of features: ", len(data_keys) 
print "The number of poi: ", sum([1 for key, record in data_dict.iteritems() if record['poi'] == 1])

print "\nAll person names: \n", data_dict.keys()

###assist function for number of NaN in record
def numnan(record):
    num_nan = 0
    for rkey, rrecord in record.iteritems():
        if rrecord == 'NaN':
            num_nan += 1;
    return num_nan

print '\nThe record has the most NaN value in record and its content: \n', sorted(list([list([key, numnan(record)]) for key, record in data_dict.iteritems()]), key=itemgetter(1), reverse=True)[0]

print data_dict['LOCKHART EUGENE E']


### Task 2: Remove outliers

data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
print "\nThe number of records after remove outlier: ", len(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.



 
 

my_dataset = copy(data_dict)

features_list = copy(data_keys)
features_list.remove('email_address')
features_list.remove('poi') 
features_list = ['poi'] + features_list + ['poi_email_communication']

for k, record in my_dataset.iteritems():
    if record['from_this_person_to_poi'] == 'NaN':
        record['from_this_person_to_poi'] = 0
    if record['from_poi_to_this_person'] == 'NaN':
        record['from_poi_to_this_person'] = 0
    record['poi_email_communication'] = record['from_this_person_to_poi'] + record['from_poi_to_this_person']

 

temp_data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(temp_data) 

selector = SelectKBest(f_classif)
selector.fit_transform(features, labels)

scores = np.around(selector.scores_, decimals = 3) 
score_rank = sorted(zip(features_list[1:], scores), key = itemgetter(1), reverse = True)

print '\n The score of features: \n',  score_rank

print '\n Selected features: '
features_list = ['poi'] + [record[0] for record in score_rank][:9] + ['poi_email_communication']
print features_list



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
 
### Scale teh features 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)
 


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.cross_validation import train_test_split



def evaluation_score(labels_test, pred):
    tp = len([(t, p)  for t, p in zip(labels_test, pred) if t == 1 and p == 1]) 
    tn = len([(t, p)  for t, p in zip(labels_test, pred) if t == 0 and p == 0]) 
    fp = len([(t, p)  for t, p in zip(labels_test, pred) if t == 0 and p == 1]) 
    fn = len([(t, p)  for t, p in zip(labels_test, pred) if t == 1 and p == 0]) 
    return round(accuracy_score(labels_test, pred), 3), 0 if (tp + fp) == 0 else tp / float(tp + fp), 0 if (tp + fn) == 0 else tp / float(tp + fn)
    

n_test = 1000

def evaluate_score(clf):
    final_score = 0., 0., 0.
    for i in range(n_test):
        seed = randint(0, n_test)
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=seed)
        clf.fit(features_train, labels_train) 
        score = evaluation_score(labels_test, clf.predict(features_test))
        final_score = map(add, final_score, score)
    return [num / n_test for num in final_score]

clf = SVC()
print "SVC accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf)) 

clf = DecisionTreeClassifier() 
print "DecisionTreeClassifier accuracy score: : \n", np.around(evaluate_score(clf), decimals = 3) , sum(evaluate_score(clf)) 

clf = RandomForestClassifier() 
print "RandomForestClassifier accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf))  

clf = AdaBoostClassifier() 
print "AdaBoostClassifier accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf))  

clf = GaussianNB() 
print "GaussianNB accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf))  

clf = KMeans(n_clusters=2) 
print "KMeans accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf))  

clf = LogisticRegression() 
print "LogisticRegression accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf))  

# find the best parameter of KMeans and LogisticRegression

param_grid_kmean = { 'tol': [1e-8,  1e-7, 1e-6, 1e-5, 1e-4, 1e-3], } 
param_grid_logistic = {'C' : [.0001, .001, .01, .1, 1., 10.0], 'tol': [1e-8,  1e-7, 1e-6, 1e-5, 1e-4, 1e-3], 'class_weight' : [None, 'balanced'], }



clf = GridSearchCV(LogisticRegression(), param_grid_logistic)
clf = clf.fit(features, labels) 
print clf.best_estimator_

clf = LogisticRegression(C=0.0001, class_weight='balanced', dual=False,
                    fit_intercept=True, intercept_scaling=1, max_iter=100,
                    multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                    solver='liblinear', tol=1e-08, verbose=0, warm_start=False)
print "LogisticRegression optimized accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf))  


clf = GaussianNB() 

clf = GridSearchCV(KMeans(n_clusters=2), param_grid_kmean)
clf = clf.fit(features, labels) 
print clf.best_estimator_

clf = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=1e-06, verbose=0)
print "KMeans optimized accuracy score: \n", np.around(evaluate_score(clf), decimals = 3), sum(evaluate_score(clf))  



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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)