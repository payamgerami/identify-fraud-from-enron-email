#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")

from sklearn.feature_selection import f_regression,SelectKBest
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from poi_utils import fraction

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'bonus',
                 'total_stock_value',
                 'exercised_stock_options',
                 'fraction_to_poi',
                 'shared_receipt_with_poi'
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop("FREVERT MARK A",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for name in my_dataset:
    from_poi_to_this_person = my_dataset[name]["from_poi_to_this_person"]
    to_messages = my_dataset[name]["to_messages"]
    fraction_from_poi = fraction( from_poi_to_this_person, to_messages )
    
    my_dataset[name]["fraction_from_poi"] = fraction_from_poi
   
    from_this_person_to_poi = my_dataset[name]["from_this_person_to_poi"]
    from_messages = my_dataset[name]["from_messages"]
    fraction_to_poi = fraction( from_this_person_to_poi, from_messages )
   
    my_dataset[name]["fraction_to_poi"] = fraction_to_poi


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

select = SelectKBest()
dtc = DecisionTreeClassifier()

# Load pipeline steps into list
steps = [('feature_selection', select),
         ('dtc', dtc)
        ]

# Create pipeline
pipeline = Pipeline(steps)

parameters = dict(
                  feature_selection__k=[2, 3, 5, 6], 
                  dtc__criterion=['gini', 'entropy'],
                  dtc__max_depth=[None, 1, 2, 3, 4],
                  dtc__min_samples_split=[1, 2, 3, 4, 25],
                  dtc__class_weight=[None, 'balanced'],
                  dtc__random_state=[42]
                  )

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Split features and labels into train and test data.
#There shouldn't be any need to partition the data in this line because it will be done behind the scenes during the grid search process in the lines below. 
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Cross-validation for parameter tuning in grid search 
sss = StratifiedShuffleSplit(labels_train,n_iter = 20,test_size = 0.5,random_state = 0)

# Create, fit, and make predictions with grid search
gs = GridSearchCV(pipeline,param_grid=parameters,scoring="f1",cv=sss,error_score=0)
gs.fit(features, labels)
labels_predictions = gs.predict(features_test)

# Pick the classifier with the best tuned parameters
clf = gs.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)