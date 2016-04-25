# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:55:18 2016

@author: JIN-E
"""


import numpy as np
import xgboost as xgb
#import pandas as pd

class xgwrap():
    def __init__(self, params, xgb_num_rounds, early_stopping_rounds, x_test = None, y_test = None):
        self.xgtest = None
        self.params = params
        self.xgb_num_rounds = xgb_num_rounds
        self.early_stopping_rounds = early_stopping_rounds  
#        print type(x_test)
#        a =  x_test is not None
#        print a
        if x_test is not None:
            self.xgtest = xgb.DMatrix(x_test, y_test, missing = -1) 
        
    
    def fit(self,x_train, y_train, x_test = None, y_test = None):
        self.xgtrain = xgb.DMatrix(x_train, y_train, missing = -1)     
        
        if self.xgtest is not None:
            self.algo = xgb.train(self.params, self.xgtrain, self.xgb_num_rounds
                , evals= [(self.xgtrain,'train'),(self.xgtest,'test')], early_stopping_rounds=self.early_stopping_rounds) 
        else:
            self.algo = xgb.train(self.params, self.xgtrain, xgb_num_rounds = self.xgb_num_rounds
                , evals= [(self.xgtrain,'train')], early_stopping_rounds=self.early_stopping_rounds) 
                
        if hasattr(self.algo, 'best_score'): # this for early stopping
            return {'best_score':self.algo.best_score, 'best_iteration':self.algo.best_iteration, 'best_ntree_limit':self.algo.best_ntree_limit}
        
    def predict(self,x_test):
        
        self.xgfinaltest = xgb.DMatrix(x_test, missing = -1) 
        
        if hasattr(self.algo, 'best_score'): # this for early stopping
            return self.algo.predict(self.xgfinaltest, ntree_limit= self.algo.best_ntree_limit)
        else:
            return self.algo.predict(self.xgfinaltest)
            
        '''
        if hasattr(self.algo, 'best_score'): # this for early stopping
            logger(['Best score:',algo.best_score])
            logger(['best_iteration:',algo.best_iteration])
            logger(['best_ntree_limit:',algo.best_ntree_limit])
    #        algo.predict(xgtest,ntree_limit=algo.best_ntree_limit)    
            y_train_true, y_train_pred = y_train.values, algo.predict(xgtrain, ntree_limit=algo.best_ntree_limit)#.astype('float64')
            y_test_true, y_test_pred = y_test.values, algo.predict(xgtest, ntree_limit=algo.best_ntree_limit)#.astype('float64')
        else:
            y_train_true, y_train_pred = y_train.values, algo.predict(xgtrain)#.astype('float64')
            y_test_true, y_test_pred = y_test.values, algo.predict(xgtest)#.astype('float64')
        '''
    def predict_proba(self,x_test):
        self.xgfinaltest = xgb.DMatrix(x_test, missing = -1) 
        
        if hasattr(self.algo, 'best_score'): # this for early stopping
            self.a  = self.algo.predict(self.xgfinaltest, ntree_limit= self.algo.best_ntree_limit)
        else:
            self.a  = self.algo.predict(self.xgfinaltest)
            
        self.b = np.zeros((self.a.shape[0],2))
        self.b[:,1] = self.a
        self.b[:,0] = 1 - self.a
        return self.b
    
    def save_model(self, filepath):
        self.algo.save_model(filepath)
            
            
            