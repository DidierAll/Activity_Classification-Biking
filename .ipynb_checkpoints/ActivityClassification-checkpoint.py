# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:45:24 2022

@author: d_all

I believe the approach proposed by the instructor was problematic. Indeed
by looping through the hyperparameters for each validation/tests subsets, 
you end up with different best hyperparameters for each set

Instead, it seems more logical to loop through each validation test and each 
hyperparameters pair, and then compute the average accuracy accross validation sets
to determine which is best

I need to read this more
https://stats.stackexchange.com/questions/65128/nested-cross-validation-for-model-selection

Then run the prediction on test set to compute final accuracy
"""

import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

import activity_classifier_utils


fs = 256
data = activity_classifier_utils.LoadWristPPGDataset()
labels, subjects, features = activity_classifier_utils.GenerateFeatures(data,
                                                                        fs,
                                                                        window_length_s=10,
                                                                        window_shift_s=10)

n_estimators_opt = [2, 10, 20, 50, 100, 150, 300]
max_tree_depth_opt = range(1, 7) # added 1 to the list provided by instructor

class_names = np.array(['bike', 'run', 'walk'])
logo = LeaveOneGroupOut()
accuracy_table = []

import itertools


#%% PARAMETER TUNING BASED ON TRAIN, VALIDATION, TEST SETS USING NESTED LOOP

class_names = ['bike', 'run', 'walk']

# Store the confusion matrix for the outer CV fold.

splits = 0

accuracy_table = []

indx1 = 0
Ntest = logo.get_n_splits(features, labels, subjects)
# same as Ntest = len(np.unique(subjects))

# create a split list
splitlist = list(logo.split(features, labels, subjects))
classification_accuracy = np.zeros((Ntest, len(n_estimators_opt),len(max_tree_depth_opt)))

for indx1, (train_val_ind, test_ind) in enumerate(splitlist):
    # Split the dataset into a test set and a training + validation set.
    # Model parameters (the random forest tree nodes) will be trained on the training set.
    # Hyperparameters (how many trees and the max depth) will be trained on the validation set.
    # Generalization error will be computed on the test set.        
    X_train_val, y_train_val = features[train_val_ind], labels[train_val_ind]
    subjects_train_val = subjects[train_val_ind]


    
    # Keep track of the best hyperparameters for this training + validation set.
    best_hyper_params = None  ## THERE WAS A TYPO : best_hyper_parames = None
    best_accuracy = 0
    print('Runing split {}'.format(indx1+1))
    
    for indx2, n_estimators in enumerate(n_estimators_opt):
        for indx3, max_tree_depth in enumerate(max_tree_depth_opt):
    #for n_estimators, max_tree_depth in itertools.product(n_estimators_opt,
    #                                                      max_tree_depth_opt):

            #print(str(indx1+1) + '/' + str(Ntest) + ' : (' + str(n_estimators) + ',' 
            #      + str(max_tree_depth) + ')'  )  

            inner_cm = np.zeros((3, 3), dtype='int')
            clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_tree_depth,
                                     random_state=42,
                                     class_weight='balanced')
            for train_ind, validation_ind in logo.split(X_train_val, y_train_val,
                                                    subjects_train_val):
                X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
                X_val, y_val = X_train_val[validation_ind], y_train_val[validation_ind]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                c = confusion_matrix(y_val, y_pred, labels=class_names)
                inner_cm += c
        
            classification_accuracy[indx1,indx2,indx3] = np.sum(np.diag(inner_cm)) / np.sum(np.sum((inner_cm)))
 
   
    
 # Find the best pair of hyperparameters by averaging classification accuracy
 # across test runs for each pair of parameters
accuracy_average = np.mean(classification_accuracy, axis=0)
(n1max, n2max) = np.unravel_index(accuracy_average.argmax(),np.shape(accuracy_average))
best_hyper_params = (n_estimators_opt[n1max], max_tree_depth_opt[n2max])
best_accuracy = accuracy_average[n1max, n2max]
    
      
# Create a model with the best pair of hyperparameters for this training + validation set.
best_clf = RandomForestClassifier(n_estimators=best_hyper_params[0],
                                      max_depth=best_hyper_params[1],
                                      class_weight='balanced')
    
#%% Finally, train this model and test it on the test set.
nested_cv_cm = np.zeros((3, 3), dtype='int')
nested_cv_cm_val = np.zeros((3, 3), dtype='int')

for indx1, (train_val_ind, test_ind) in enumerate(splitlist):

    X_train_val, y_train_val = features[train_val_ind], labels[train_val_ind]
    subjects_train_val = subjects[train_val_ind]
    X_test, y_test = features[test_ind], labels[test_ind]   
    
    best_clf.fit(X_train_val, y_train_val)
    y_pred = best_clf.predict(X_test)
    
    # Aggregate confusion matrices for each CV fold.
    c = confusion_matrix(y_test, y_pred, labels=class_names)
    nested_cv_cm += c
    splits += 1

    yval_pred = best_clf.predict(X_train_val)
    c2 = confusion_matrix(y_train_val, yval_pred, labels=class_names) 
    nested_cv_cm_val += c2
    
    
    print('Done split {}'.format(splits))

    # store parameters and performances for outerloop
    accuracy_table.append(( best_clf.score(X_train_val, y_train_val),best_clf.score(X_test, y_test),
                           np.sum(y_test==class_names[0]),sum(y_test==class_names[1]),sum(y_test==class_names[2]) ))

accuracy_table_df = pd.DataFrame(accuracy_table,
                                 columns=['Validation acc.', 'Test acc.','N_bike','N_run','N_walk'])
accuracy_validation_table_df = pd.DataFrame(accuracy_average,
                                            index=[str(a) for a in n_estimators_opt],columns=[str(a) for a in max_tree_depth_opt])

print(accuracy_table_df)
print(accuracy_validation_table_df)
print('Total Test Accuracy= {:0.2f}'.format(np.sum(np.diag(nested_cv_cm)) / np.sum(np.sum(nested_cv_cm))))

activity_classifier_utils.PlotConfusionMatrix(nested_cv_cm, class_names,normalize="True",title="Random Tree Classifier Performance (Test)")
activity_classifier_utils.PlotConfusionMatrix(nested_cv_cm_val, class_names,normalize="True",title="Random Tree Classifier Performance (Validation)")

#%% Feature Importance: reducing the number of features


clf = RandomForestClassifier(n_estimators=100,
                             max_depth=4,
                             random_state=42,
                             class_weight='balanced')
activity_classifier_utils.LOSOCVPerformance(features, labels, subjects, clf)
# this gives the features with best performance
clf.feature_importances_

sorted(list(zip(clf.feature_importances_, activity_classifier_utils.FeatureNames())), reverse=True)[:10]

sorted_features = sorted(zip(clf.feature_importances_, np.arange(len(clf.feature_importances_))), reverse=True)
best_feature_indices = list(zip(*sorted_features))[1]
X = features[:, best_feature_indices[:10]]

X.shape

cm = activity_classifier_utils.LOSOCVPerformance(X, labels, subjects, clf)
activity_classifier_utils.PlotConfusionMatrix(cm, class_names, normalize=True)
print('Classification accuracy [10 features] = {:0.2f}'.format(np.sum(np.diag(cm)) / np.sum(np.sum(cm))))
