Identify Fraud from Enron emails
===============================================================================================

### Enron Submission Free-Response Questions ###

__1.__	Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it.  As part of your answer, give some background on the dataset and how it can be used to answer the project question.  Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

__Goal__
The goal of is to identify the persons of interest in the Enron fraud case using machine learning methods along with financial and email data. The persons of interest means individuals who were indicted, reached a settlement or plea
deal with the government, or testified in exchange for prosecution immunity.
__Dataset__
The Enron data set is comprised of email and financial data. The dataset contains 146 entries and 21 features. Financial features:  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']. email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']. 18 of these points is labeled as a POI and 128 as non-POI.
__Outliers__
The following was eliminated from the data set.
-	TOTAL:  This is not a person
-	THE TRAVEL AGENCY IN THE PARK: because it is a company and does not represent a person
-   FREVERT MARK A: Not a POI but has a very high salary

__2.__	What features did you end up using in your POI identifier, and what selection process did you use to pick them?  Did you have to do any scaling?  Why or why not?  As part of the assignment, you should attempt to engineer your own feature that doesn’t come ready-made in the dataset--explain what feature you tried to make, and the rationale behind it.  (You do not necessarily have to use it in the final analysis, only engineer and test it.)  If you used an algorithm like a decision tree, please also give the feature importances of the features that you use.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

__Added Features__
I add three features to the dataset:
* fraction_from_poi
* fraction_to_poi  
* bonus_ratio

Scaling the 'fraction_from_poi' and 'fraction_to_poi' by the total number of emails sent and received, respectively, might help us identify those have low amounts of email activity overall, but a high percentage of email activity with POIs. Also, bonus to total payment is added since if the ration is high it could lead us to identify POI


I used SelectKBest algorithm to score the features. Here is the result:

| FeatureName| Score |
| ------------- | ---------|
| total_stock_value             | 24.182898678566879  |
| bonus                         | 20.792252047181535  |
| bonus_ratio                   | 20.715596247559954  |
| salary                        | 18.289684043404513  |
| deferred_income               | 11.458476579280369  |
| long_term_incentive           | 9.9221860131898225  |
| restricted_stock              | 9.2128106219771002  |
| total_payments                | 8.7727777300916756  |
| shared_receipt_with_poi       | 8.589420731682381   |
| loan_advances                 | 7.1840556582887247  |
| expenses                      | 6.0941733106389453  |
| from_poi_to_this_person       | 5.2434497133749582  |
| other                         | 4.1874775069953749  |
| from_this_person_to_poi       | 2.3826121082276739  |
| director_fees                 | 2.1263278020077054  |
| to_messages                   | 1.6463411294420076  |
| fraction_to_poi               | 1.2565738314129471  |
| fraction_from_poi             | 0.23029068522964966 |
| deferral_payments             | 0.22461127473600989 |
| from_messages                 | 0.16970094762175533 |
| restricted_stock_deferred     | 0.065499652909942141|

__Selected Features__
In order to choose the best K value for feature selection, I tested different k values (2, 3, 5, 6, 9). The best result is produced by K=6 that has relatively the best precision and recall.

the selected features:
 * 'exercised_stock_options'
 * 'total_stock_value'
 * 'bonus'
 * 'bonus_ratio'
 * 'salary'
 * 'deferred_income'  
 
Hence, only one of the three engnineerd features (bonus_ratio) is in the selected list.

__3.__ What algorithm did you end up using?  What other one(s) did you try? [relevant rubric item: “pick an algorithm”]

__Algorithms__
I focused on three following  algorithms:
* decision tree classifier (The decision tree classifier had the best precision and a recall)
* SVM (The SVM classifier had a high precision of but a poor recall)
* k-nearest neighbors (The k-nearest neighbors classifier had low precision and a recall)

Among them, decision tree classifier was chosen with the following performance result:
* Accuracy: 0.79907
* Precision: 0.30466
* Recall: 0.31700

__4.__ What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?  (Some algorithms don’t have parameters that you need to tune--if this is the case for the one you picked, identify and briefly explain how you would have done it if you used, say, a decision tree classifier). [relevant rubric item: “tune the algorithm”]

__Parameters Tuning__
Tuning is the process of altering algorithm parameters to effect performance.  Poor tuning can result in poor algorithm performance.  It can also result in an algorithm that gives reasonable performance but is not efficient. The goal is to set the parameters to optimal values so that it gives the best performance in terms of accuracy, precision and recall.

I used GridSearchCV for parameter tuning. GridSearchCV allows us to construct a grid of all the combinations of parameters, tries each combination, and then reports back the best combination/model.

For the chosen decision tree classifier for example, I tried out multiple different parameter values and here is the result:

* Accuracy: 0.82114
* Precision: 0.31250
* Recall: 0.21000
* F1: 0.25120
* Best Parameters: {'min_samples_split': 10, 'max_depth': 2}

 In other words, it is important to tune the algorithm parameters because setting those parameters wronge could result in overfitting or underfitting, poor performance (accuracy, percision and recalls) and slow running algorithms . Se we should perform parameter tuning to control how much detail the model learns once we train it.

__5.__ What is validation, and what’s a classic mistake you can make if you do it wrong?  How did you validate your analysis?  [relevant rubric item: “validation strategy”]

__Validation__
Validation is process of determining how well your model performs or fits, using a specific set of criteria.  The idea is that you break the data set into a test set (often 20-30% of the data) and a training set (the remainder of the data).  You fit on the training set and predict on the test set.  Metrics from the test set determine your performance.  When using cross-validation on small data set, it is helpful to perform this process multiple times, randomly splitting each time.  

In my poi_id.py file, my data was separated into training and testing sets. The test size was 30% of the data, while the training set was 70%.

Considering the fact that in this project we are dealing with a small and imbalance dataset I use StratifiedShuffleSplit.

__6.__ Give at least 2 evaluation metrics, and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [Relevant rubric item: “usage of evaluation metrics”]

__Evaluation__
The two notable evaluation metrics for this POI identifier are precision and recall.

* Precision is how often our class prediction (POI vs. non-POI) is right when we guess that class
* Recall is how often we guess the class (POI vs. non-POI) when the class actually occurred

It is arguably more important to make sure we don't miss any POIs, so we don't care so much about precision. The decision tree algorithm performed best in recall. In other words,  high rate of recall means that high number of "poi" is correctly classified as "poi" and small number of "poi" is misclassified as "not-poi". Low rate of recall means that small number of "poi" is correctly classified as "poi" and high number of "poi" is misclassified as "not-poi".