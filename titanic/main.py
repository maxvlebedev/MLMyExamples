#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:02:47 2018

@author: maximus
"""
import re
import pandas as pds
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import string as modstr
import pylab as plt

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.svm
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

import time
import datetime

import xgboost as xgb

def prepopressing_(df):
    df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[.?/?]', '', x) )
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )
        
    # create binary features for each prefix
    prefixes = pds.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
    df = pds.concat([df, prefixes], axis=1)
    
    # factorize the prefix to create a numerical categorical variable
    df['TicketPrefixId'] = pds.factorize(df['TicketPrefix'])[0]
    
    # extract the ticket number
    df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )
    
    # create a feature for the number of digits in the ticket number
    df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)
    
    # create a feature for the starting number of the ticket number
    df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)
    
    # The prefix and (probably) number themselves aren't useful
    #df.drop(['TicketPrefix', 'TicketNumber'], axis=1, inplace=True)
    
    return df, prefixes.columns.values.tolist()
    

def processing_from_ultraviolet(df):
    df['Sex'] = df['Sex'] == 'male'
    
    age = df.loc[np.isnan(df['Age']) == False] 
    val = np.mean(age['Age'])
    df.loc[np.isnan(df.Age),'Age'] = val
    
    fare = df.loc[np.isnan(df['Fare']) == False] 
    val = np.mean(fare['Fare'])
    df.loc[np.isnan(df.Fare),'Fare'] = val
    
    # extract and massage the ticket prefix
    df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[.?/?]', '', x) )
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )
        
    # create binary features for each prefix
    prefixes = pds.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
    df = pds.concat([df, prefixes], axis=1)
    
    # factorize the prefix to create a numerical categorical variable
    df['TicketPrefixId'] = pds.factorize(df['TicketPrefix'])[0]
    
    # extract the ticket number
    df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )
    
    # create a feature for the number of digits in the ticket number
    df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)
    
    # create a feature for the starting number of the ticket number
    df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)
    
    # The prefix and (probably) number themselves aren't useful
    df.drop(['TicketPrefix', 'TicketNumber'], axis=1, inplace=True)
    
    df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))
    
    # What is each person's title?
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?).").findall(x)[0])

    # Group low-occuring, related titles together
    df['Title'].loc[df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

    # Build binary features
    df = pds.concat([df, pds.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1) 
    
    # Replace missing values with "U0"
    df['Cabin'][df.Cabin.isnull()] = 'U0'

    # Create a feature for the deck
    df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = pds.factorize(df['Deck'])[0]

    # Create binary features for each deck
    decks = pds.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
    df = pds.concat([df, decks], axis=1)

    # Create feature for the room number
    #df['Room'] = df['Cabin'].map(lambda x: re.compile("([0-9]+)").search(x).group()).astype(int) + 1
    return df

def getTicketPrefix(ticket):
    match = re.compile(r"([a-zA-Z./]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'

def getTicketNumber(ticket):
    match = re.compile(r"\b\d+\b").search(ticket)
    if match:
        return match.group()
    else:
        return '0'

def find_same_ticket(x,unique_tickets):
    number= x['Ticket']
    return unique_tickets.tolist().index(number)
    
def find_children(x):
    age = x['Age']
    if age>3:
        return 1
    else:
        return 0

def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
def replace_one(x,str_in):
    if x == str_in:
        return 1
    return 0

def substrings_in_string(big_string, substrings):
    if (type(big_string) == str):
        for substring in substrings:
            if modstr.find(big_string, substring) != -1:
                return substring
    #print big_string
    return 'U'
    
def preprocessing(data):
    data['Sex'] = data['Sex'] == 'male'
    age = data.loc[np.isnan(data['Age']) == False] 
    val = np.median(age['Age'])
    data.loc[np.isnan(data.Age),'Age'] = val
    
    fare = data.loc[np.isnan(data['Fare']) == False] 
    val = np.median(fare['Fare'])
    data.loc[np.isnan(data.Fare),'Fare'] = val
    #Decks
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    
    data['Deck'] = \
               data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    #replacing all titles with mr, mrs, miss, master
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
    arr_title = ['Mr', 'Mrs', 'Miss', 'Master']
    #arr_title = ['Mrs', 'Miss', 'Master']
    
   
    data['Title'] = \
               data['Name'].map(lambda x: substrings_in_string(x, title_list))
    data['Title'] = \
              data.apply(replace_titles, axis=1)
    
    #for el in arr_title:
    #    age = data.loc[ \
    #                   np.logical_and(np.isnan(data['Age']) == False, \
    #                   data['Title'].eq(el))] 
    #    print age['Age'].shape               
    #    val = np.mean(age['Age'])
    #    data.loc[\
    #             np.logical_and(np.isnan(data.Age),data.Title.eq(el)),\
    #                            'Age'] = val
    #data.loc[np.isnan(data.Age),'Age'] = val
              
    data['Children'] = \
              data.apply(find_children,axis=1)          
    for el in  arr_title:         
        data[el] = data['Title'].map(lambda x: replace_one(x,el))
    
    cabin_list1 =  ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'U']
    #cabin_list1 =  ['U']
    
    for el in  cabin_list1:         
        data[el] = data['Deck'].map(lambda x: replace_one(x,el))
        
    #family size          
    data['Family_Size'] = data['SibSp']+data['Parch']
    #
    data['Age*Class'] = data['Age']*data['Pclass']
    #
    data['Fare_Per_Person'] = data['Fare']/(data['Family_Size']+1)
    
    arr_embarked = ['Cemb','Qemb','Semb']
    
    data['Cemb'] = data['Embarked'].map(lambda x: replace_one(x,'C'))
    data['Qemb'] = data['Embarked'].map(lambda x: replace_one(x,'Q'))
    data['Semb'] = data['Embarked'].map(lambda x: replace_one(x,'S'))
    
    unique_tickets = data.Ticket.unique()
    data['TicketCount'] = \
              data.apply(lambda x: find_same_ticket(x,unique_tickets),axis=1) 
    
    return data,arr_title, cabin_list1, arr_embarked

def svc(X, Y, regular_coeffs_arr, scoring_str='accuracy'):
    n_splits = 5 
    cv = KFold(n_splits = n_splits, shuffle = True, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X)
    X_new = scaler.transform(X)
    res = []
    for c in regular_coeffs_arr:
        start_time = datetime.datetime.now()
        svc = SVC(kernel='poly', C = c, random_state=241) 
        v = cross_val_score(svc, X_new, Y, cv=cv, \
                                          scoring=scoring_str) 
        print v
        res.append([c,np.min(v)])
        print '======================================='
        print 'Time elapsed:', datetime.datetime.now() - start_time
        print 'Min ' + scoring_str +':',np.min(v)
        print 'Regular. coeff: ', c
        
    return res, scaler

def logistic_regression(X, Y, regular_coeffs_arr, scoring_str='accuracy'):
    n_splits = 5 
    cv = KFold(n_splits = n_splits, shuffle = True, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X)
    X_new = scaler.transform(X)
    res = []
    for c in regular_coeffs_arr:
        start_time = datetime.datetime.now()
        lgr = LogisticRegression(penalty='l1', C = c, random_state=42)
        v = cross_val_score(lgr, X_new, Y, cv=cv, \
                                          scoring=scoring_str) 
        print v
        res.append([c,np.min(v)])
        print '======================================='
        print 'Time elapsed:', datetime.datetime.now() - start_time
        print 'Min ' + scoring_str +':',np.min(v)
        print 'Regular. coeff: ', c
    
    X_new = 0

    return res, scaler

def random_frorest(X, Y, n_tree_arr, scoring_str='accuracy'):
    n_splits = 5 
    cv = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    
    res = []
    for n_est in n_tree_arr:
        start_time = datetime.datetime.now()
        clf = RandomForestClassifier(n_estimators=n_est, \
                                     verbose=False, random_state=42)
        v = cross_val_score(clf, X, Y, cv=cv, \
                                          scoring=scoring_str) 
        res.append([n_est, np.min(v)])
        print '======================================='
        print 'Time elapsed:', datetime.datetime.now() - start_time
        print 'Min ' + scoring_str +':',np.min(v)
        print 'Number of estimator: ',n_est
    return res
    

def tree_boosting(X, Y, learning_rates_arr, max_depthes_arr, n_tree_arr, \
                  scoring_str='accuracy'):
    n_splits = 5 
    cv = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    
    res = []
    for lr in learning_rates_arr:
        for md in max_depthes_arr:
            for n_est in n_tree_arr:
                start_time = datetime.datetime.now()
                #gbc = GradientBoostingClassifier(n_estimators=n_est, \
                #                     verbose=False, random_state=42, \
                #                     learning_rate=lr, max_depth=md)
                gbc = xgb.XGBClassifier(max_depth=md,\
                                       n_estimators=n_est,\
                                       nthread=4, random_state=42)
                v = cross_val_score(gbc, X, Y, cv=cv, \
                                          scoring=scoring_str) 
                res.append([lr, md, n_est, np.min(v)])
                print '======================================='
                print 'Time elapsed:', datetime.datetime.now() - start_time
                print 'Min ' + scoring_str +':',np.min(v)
                print 'Learning rate: ',lr
                print 'Number of estimator: ',n_est
                print 'Max tree depth: ', md
    return res

def mashine_learning(X,X_test,Y):
    regular_coeffs_arr = [0.1, 0.38, 0.39, 0.395, 0.4, 0.41, 0.5, 0.6, 0.89, 0.9,0.95, 1,2,2.9, 3,3.1, 4,5,6]
    res_logreg, scale = logistic_regression(X,Y,regular_coeffs_arr)
    
    regular_coeffs_arr = [0.1, 10,100,1000,10000]
    res_svc, scale = svc(X, Y, regular_coeffs_arr) 


    max_depthes_arr = [2, 3, 4,10]
    n_est_arr = [200,250,300]
    learning_rates_arr = [0.5,0.3,0.1] 
    n_tree_arr = [1,10,15,20,25,30,40]
    
    res_boosting = tree_boosting(X, Y, learning_rates_arr, \
                                 max_depthes_arr,n_est_arr)
    res_random_forest = random_frorest(X,Y,n_tree_arr)
    
    indb = np.argmax(np.array(res_boosting)[:,3])
    indl = np.argmax(np.array(res_logreg)[:,1])
    indsvc = np.argmax(np.array(res_svc)[:,1])
    indrf = np.argmax(np.array(res_random_forest)[:,1])
    
    

   
    
    print 'Opt logregresssion: ', res_logreg[indl]
    print 'Opt Svc: ', res_logreg[indsvc]
    print 'Opt boosting: ', res_boosting[indb]
    print 'Opt random forest: ', res_boosting[indrf]
    
    #[lr, md, n_est, np.min(v)]
    gbc = GradientBoostingClassifier(n_estimators=res_boosting[indb][2], \
                                     verbose=False, random_state=42, \
                                     learning_rate=res_boosting[indb][0],\
                                     max_depth=res_boosting[indb][1])
    gbc.fit(X,Y)
    Y_test_boosting_predict = gbc.predict(X_test)
    
    lgr = LogisticRegression(penalty='l2',C = res_logreg[indl][1],\
                             random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scale_train = scaler.transform(X)
    lgr.fit(X_scale_train,Y)
    
    scaler_test = StandardScaler()
    scaler_test.fit(X_test)
    X_scale_test = scale.transform(X_test)
    #X_scale_test = scaler_test.transform(X_test)
    Y_test_logreg_predict = lgr.predict(X_scale_test)
    #print np.argwhere(np.isnan(X_test))
    #print np.sum(np.isnan(X_test))
    
    
    return Y_test_boosting_predict, Y_test_logreg_predict
    
    #print Y_test
    
def predict_data(Y_test_bst_predict, Y_test_lgr_predict, test_data):
    predictiondf = pds.DataFrame(test_data['PassengerId'])
    #print predictiondf.size
    predictiondf['Survived']= Y_test_bst_predict
    predictiondf.to_csv('prediction_boosting.csv',\
                  index=False)
    
        #print predictiondf.size
    predictiondf['Survived']= Y_test_lgr_predict
    predictiondf.to_csv('prediction_logreg.csv',\
                  index=False)
    
    x = np.arange(Y_test_bst_predict.shape[0])
    plt.plot(x,Y_test_bst_predict,'go',x,Y_test_lgr_predict,'b^')
    plt.legend(['Boosting', 'LogRegression'])
    plt.show()
    
    print 'Number of values no eq(logreg vs boosting):', np.sum(Y_test_lgr_predict <> Y_test_bst_predict)
    last_prediction = pds.read_csv('./7198881/prediction_logreg.csv') 
    print 'Number of Values no eq in last pred(logreg):', \
    np.sum(last_prediction['Survived'].as_matrix() <> Y_test_lgr_predict)
    
    last_prediction = pds.read_csv('./7036998/prediction.csv') 
    print 'Number of Values no eq in last pred(boosting):', \
    np.sum(last_prediction['Survived'].as_matrix() <> Y_test_bst_predict)
    
    
def problem_solver1(train_data, test_data):

    
    #train_data = train_data.loc[np.isnan(train_data['Age']) == False] 
    
    train, arr_title_names, arr_deck, arr_embarked =  preprocessing(train_data)
    test, arr_title_names, arr_deck, arr_embarked =  preprocessing(test_data)
    test, prefix_title = prepopressing_(test)
    train, prefix_title = prepopressing_(train)
    

   # for el in test.columns:
   #     print test.loc[np.isnan(test[el]) == True] 
    
    #feature_list = ['Pclass', 'Fare', 'Age', 'Sex']
    #feature_list = ['Pclass','Age','SibSp','Parch'] + arr_title_names
    feature_list = ['Pclass', 'Fare_Per_Person', \
                    'Sex', 'Age*Class', 'Family_Size'] \
              + arr_title_names + ['TicketNumberDigits',  'TicketNumberStart']
             
    Y = train['Survived'].as_matrix()
    X = train[feature_list].as_matrix()
    X_test = test[feature_list].as_matrix()
    
    Y_test_bst_predict, Y_test_lgr_predict = mashine_learning(X,X_test,Y)
    predict_data(Y_test_bst_predict, Y_test_lgr_predict, test)
    
    print 'Number of true values in train:', np.sum(Y)
    print 'Size of train data:', Y.shape[0]
    print '==============================='
    print 'Size of tes data:', Y_test_lgr_predict.shape[0]
    print 'Number of true values(logregress) in predict:'\
    , np.sum(Y_test_lgr_predict)
    
    
    
    return train, test
    
def problem_solver2(train_data, test_data):
    
    feature_list =\
    ['Age','Pclass','SibSp','Parch'] 
    
    train = processing_from_ultraviolet(train_data)
    test =  processing_from_ultraviolet(test_data)
    Y = train['Survived'].as_matrix()
    X = train[feature_list].as_matrix()
    X_test = test[feature_list].as_matrix()
    
    Y_test_bst, Y_test_lgr = mashine_learning(X,X_test,Y)
    predict_data(Y_test_bst, Y_test_lgr,test)
    
    return train, test

def comapare_files():
    d1 = pds.read_csv('./7036998/prediction.csv')
    d2 = pds.read_csv('./7048685/prediction_logreg.csv')
    d3 = pds.read_csv('./7132879/prediction_logreg.csv')  
    d4 = pds.read_csv('./7198881/prediction_logreg.csv') 
    Y4 = d4['Survived'].as_matrix()
    Y3 = d3['Survived'].as_matrix()
    Y2 = d2['Survived'].as_matrix()
    Y1 = d1['Survived'].as_matrix()
    
    print 'Y4 != Y3', np.sum(Y4 <> Y3)
    print Y4.shape
    print 6./418 + 0.78947
    
def test_script():
    train_data = pds.read_csv('train.csv')
    test_data = pds.read_csv('test.csv')
    train, test = problem_solver1(train_data, test_data)

    n_test = test_data.Ticket.shape[0]
    unique_test_ticket = set(test_data.Ticket)
    nu_test = len(unique_test_ticket)

    print 'Number of unique el in test:', nu_test                 
    print 'Size - Unique size (for test):', n_test - nu_test                  
    print 'Size of test ticket:', n_test
    print "==================================="
    n_tr = train_data.Ticket.shape[0]
    unique_tr_ticket = set(train_data.Ticket)
    nu_tr = len(unique_tr_ticket)
    
    print 'Number of unique el in train:', nu_tr                 
    print 'Size - Unique size (for train):', n_tr - nu_tr                  
    print 'Size of train ticket:', n_tr
    common = \
    set.intersection(set(train_data.Ticket), \
                     set(test_data.Ticket))
    print "==================================="
    #print common
    print 'Number of common elements in train and test set:',\
                     len(common)
    
    #print set.intersection(set(['666','1fd','fds2','3sd','5','6']),\
    #                       set(['4fs','5fs','666']))
    

if __name__ == "__main__":
    #comapare_files()
    test_script()