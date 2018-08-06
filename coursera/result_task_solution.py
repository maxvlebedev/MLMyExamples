#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:11:46 2017

@author: maximus
"""

from sklearn import datasets

import pandas as pds
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

import datetime

RANDOM_STATE = 42

def _preprocessing_skip_data(data):
    sz = data.shape
    skip_data = []
    for el in data.columns:
        if data[el].count() != sz[0]:
            skip_data.append(el)
            data[el] = data[el].fillna(0)

    with open('col_names_skip_data.txt', 'w') as f:
        for el in skip_data:
            f.write(el + '\n') 
    
    return data
    
def _get_target(data):
    sz = data.shape
    target_col_name = 'radiant_win'
    if (target_col_name in set(data.columns)):
        Y = np.zeros(sz[0])
        Y[:] = data['radiant_win'][:]
        #Y = Y[Y != 1]
    else:
        Y = None    
    return Y

def _logistic_regression(X, Y ,cv, regular_coeffs_arr):
    scaler = StandardScaler()
    scaler.fit(X)
    X_new = scaler.transform(X)
    res = []
    for c in regular_coeffs_arr:
        start_time = datetime.datetime.now()
        lgr = LogisticRegression(penalty='l2',C = c, random_state=RANDOM_STATE)
        v = cross_val_score(lgr, X_new, Y, cv=cv, \
                                          scoring='roc_auc') 
        res.append([c,np.min(v)])
        print '=======Question 2.1-4 Logistic regression======='
        print 'Time elapsed:', datetime.datetime.now() - start_time
        print 'Min roc_auc: ',np.min(v)
        print 'Regular. coeff: ', c

    return res

def _tree_boosting(X, Y, cv, learning_rates_arr, max_depthes_arr, n_tree_arr):
    res = []
    for lr in learning_rates_arr:
        for md in max_depthes_arr:
            for n_est in n_tree_arr:
                start_time = datetime.datetime.now()
                gbc = GradientBoostingClassifier(n_estimators=n_est, \
                                     verbose=False, random_state=RANDOM_STATE,\
                                     learning_rate=lr, max_depth=md)
                v = cross_val_score(gbc, X, Y, cv=cv, \
                                          scoring='roc_auc') 
                print v
                res.append([lr, md, n_est, np.min(v)])
                print '==========Question 1.3-4 Boosting(Tree)==========='
                print 'Time elapsed:', datetime.datetime.now() - start_time
                print 'Min roc_auc: ',np.min(v)
                print 'Learning rate: ',lr
                print 'Number of estimators: ',n_est
                print 'Max tree depth: ', md
    return res

def _delete_feature(data, list_of_feature):
    for el in list_of_feature:
        data = data.drop(el, 1)
    return data

def _nag_of_word_for_horoes(data, N):
    X_pick = np.zeros([data.shape[0], N])
    
    for i, match_id in enumerate(data.index):
        for p in xrange(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    return X_pick
    
def _add_bag_of_word(X_bag, X):
    X_new = np.zeros([X.shape[0],X_bag.shape[1] + X.shape[1]])
    X_new[:,:X.shape[1]] = X[:,:]
    X_new[:,X.shape[1]:] = X_bag[:,:]
    return X_new

def _preprocessing_cols(data, with_bag, is_delete_categor_cols):
    categor_feature_list = []
    if (with_bag or is_delete_categor_cols):
        for i in range(1,6):
            categor_feature_list.append('r%i_hero' % (i))
            categor_feature_list.append('d%i_hero' % (i))
    
        if (with_bag):
            x = np.unique(data[categor_feature_list].values)
            X_bag = _nag_of_word_for_horoes(data, max(x))
            print "Number of heroes type: ", max(x)
            with open('number_of_heroes.txt','w') as f:
                f.write('max_id: %i, min_id: %i, number of id: %i'% \
                                                 (max(x), min(x), x.shape[0]))

                     
        categor_feature_list.append('lobby_type')
    data = _delete_feature(data, categor_feature_list)
    
    X = np.zeros(data.shape)    
    X[:,:] = data.as_matrix()[:,:]    
    if (with_bag):
        X = _add_bag_of_word(X_bag, X)    
    return X

def preprocessing_data(data, with_bag=0, is_delete_categor_cols=0):
    Y = _get_target(data)
    delete_col = ['radiant_win','duration', 'tower_status_radiant', \
                  'tower_status_dire', 'barracks_status_radiant', \
                  'barracks_status_dire']

    data = _delete_feature(data, delete_col)
    data = _preprocessing_skip_data(data)
    X = _preprocessing_cols(data, with_bag, is_delete_categor_cols)
    return X,Y
    
def boosting_task(data, cv, n_est_arr=[10,20,30,40], \
                  learning_rates_arr=[1, 0.5, 0.3, 0.2, 0.1], \
                  max_depthes_arr=[2,3], with_bag=0, \
                  is_delete_categor_cols=0):
    X,Y = preprocessing_data(data, with_bag, is_delete_categor_cols)
    res = _tree_boosting(X, Y, cv, learning_rates_arr, \
                                          max_depthes_arr,n_est_arr) 
    return res

def log_regression_task(data, cv, regular_coeffs_arr=[0.1, 0.5,10], \
                        with_bag=1, \
                        is_delete_categor_cols=1):
    X,Y = preprocessing_data(data, with_bag, is_delete_categor_cols)
    res = _logistic_regression(X, Y, cv, regular_coeffs_arr)    
    return res

def best_predict_test(data, test_data, cv, with_bag=1, \
                              is_delete_categor_cols=1, \
                              c=10):
    
    X,Y = preprocessing_data(data, with_bag, is_delete_categor_cols)

    scaler = StandardScaler()
    scaler.fit(X)
    X_new = scaler.transform(X)
    lgr = LogisticRegression(penalty='l2',C=c, random_state=RANDOM_STATE)
    v = cross_val_score(lgr, X_new, Y, cv=cv, \
                                          scoring='roc_auc')
    print "=======Best predict (Logistic regression)========="
    print 'Optimum roc_auc for trainig data:', v
    
    lgr.fit(X_new,Y)
    test_data = _preprocessing_skip_data(test_data)
    X_test = _preprocessing_cols(test_data, with_bag, is_delete_categor_cols)
    X_test_new = scaler.transform(X_test)
    res = lgr.predict_proba(X_test_new)
    np.savetxt('Y_test_prob',res)
    print "==============Question 2.5==================="
    print 'Max. proba for test data:',np.max(res[:,1])
    print 'Min. proba for test data:',np.min(res[:,1])
    return res

if __name__ == '__main__':
    data = pds.read_csv('features.zip',  compression='zip', index_col=False)
    n_splits = 5 
    cv = KFold(n_splits = n_splits, shuffle = True, \
                                                   random_state = RANDOM_STATE)
    print "================Question 1.1================="
    print "View in file 'col_names_skip_data.txt'"
    target_col_name = 'radiant_win'
    print "================Question 1.2================="
    print "Column name:", target_col_name
    
    boosting_task(data, cv, n_est_arr = [20, 30, 40], \
                  max_depthes_arr=[3, 2],\
                  learning_rates_arr=[0.5])
    
    print "===========With all categorial features=============="
    log_regression_task(data, cv, regular_coeffs_arr=[0.1, 0.5, 10],\
                                     with_bag=0, is_delete_categor_cols=0)
    print "==========Delete all categorial features============="
    log_regression_task(data, cv, regular_coeffs_arr=[0.1, 0.5, 10], \
                                     with_bag=0, is_delete_categor_cols=1)
    print "=============With bag of words======================="
    log_regression_task(data, cv, regular_coeffs_arr=[0.1, 0.5, 10], \
                                     with_bag=1, is_delete_categor_cols=1)
    
    test_data = pds.read_csv('features_test.zip',  compression='zip', \
                                                               index_col=False)    

    res = best_predict_test(data, test_data, cv, with_bag=1, \
                                                      is_delete_categor_cols=1)

    


    
    
    
            
        
                
