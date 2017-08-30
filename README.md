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
I add two features to the dataset:
* fraction_from_poi
* fraction_to_poi  

Scaling the 'fraction_from_poi' and 'fraction_to_poi' by the total number of emails sent and received, respectively, might help us identify those have low amounts of email activity overall, but a high percentage of email activity with POIs.

__Selected Features__
I used a univariate feature selection process, select k-best, in a pipeline with grid search to select the features. Select k-best removes all but the k highest scoring features. The number of features, 'k', was chosen through an exhaustive grid search driven by the 'f1' scoring estimator, intending to maximize precision and recall.  
The following six features was chosen in POI identifier, which was a decision tree classifier.
* Feature no. 1: salary  
* Feature no. 2: bonus
* Feature no. 3: total_stock_value
* Feature no. 4: exercised_stock_options
* Feature no. 5: fraction_to_poi
* Feature no. 6: shared_receipt_with_poi

__3.__ What algorithm did you end up using?  What other one(s) did you try? [relevant rubric item: “pick an algorithm”]

__Algorithms__
I focused on three following  algorithms:
* decision tree classifier (The decision tree classifier had the best precision and a recall)
* SVM (The SVM classifier had a high precision of but a poor recall)
* k-nearest neighbors (The k-nearest neighbors classifier had low precision and a recall)

Among them, decision tree classifier was chosen.

__4.__ What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?  (Some algorithms don’t have parameters that you need to tune--if this is the case for the one you picked, identify and briefly explain how you would have done it if you used, say, a decision tree classifier). [relevant rubric item: “tune the algorithm”]

__Parameters Tuning__
Tuning is the process of altering algorithm parameters to effect performance.  Poor tuning can result in poor algorithm performance.  It can also result in an algorithm that gives reasonable performance but is not efficient.

I used GridSearchCV for parameter tuning. GridSearchCV allows us to construct a grid of all the combinations of parameters, tries each combination, and then reports back the best combination/model.

For the chosen decision tree classifier for example, I tried out multiple different parameter values for each of the following parameters

* criterion: gini and entropy
* splitter: best and random
* max_depth: None, 1, 2, 3 and 4
* min_samples_split: 1, 2, 3, 4 and 25
* class_weight: None and balanced

__5.__ What is validation, and what’s a classic mistake you can make if you do it wrong?  How did you validate your analysis?  [relevant rubric item: “validation strategy”]

__Validation__
Validation is process of determining how well your model performs or fits, using a specific set of criteria.  The idea is that you break the data set into a test set (often 20-30% of the data) and a training set (the remainder of the data).  You fit on the training set and predict on the test set.  Metrics from the test set determine your performance.  When using cross-validation on small data set, it is helpful to perform this process multiple times, randomly splitting each time.  

In my poi_id.py file, my data was separated into training and testing sets. The test size was 30% of the data, while the training set was 70%.

__6.__ Give at least 2 evaluation metrics, and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [Relevant rubric item: “usage of evaluation metrics”]

__Evaluation__
The two notable evaluation metrics for this POI identifier are precision and recall. The average precision for my decision tree classifier was 0.31336 and the average recall was 0.59100. 

* Precision is how often our class prediction (POI vs. non-POI) is right when we guess that class
* Recall is how often we guess the class (POI vs. non-POI) when the class actually occurred

It is arguably more important to make sure we don't miss any POIs, so we don't care so much about precision. The decision tree algorithm performed best in recall (0.59). In other words,  high rate of recall means that high number of "poi" is correctly classified as "poi" and small number of "poi" is misclassified as "not-poi". Low rate of recall means that small number of "poi" is correctly classified as "poi" and high number of "poi" is misclassified as "not-poi".