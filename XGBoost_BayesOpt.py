#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:45:16 2018

@author: jeremywatkins
"""



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.cross_validation import train_test_split
from bayes_opt import BayesianOptimization
from operator import itemgetter
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
from sklearn import preprocessing
from sklearn.metrics import average_precision_score

droughts=pd.read_csv('droughts.csv')
water_usage=pd.read_csv('water_usage.csv')


droughts['date']=pd.to_datetime(droughts['valid_end'])
droughts['year']=droughts.date.dt.year

merge=droughts.merge(water_usage,left_on=['fips'],right_on=['fips'],how='inner')
merge['state']=merge['state_x']
merge['county']=merge['county_x']
merge['year']=merge['year_x']
del(merge['state_y'])
del(merge['county_y'])
del(merge['state_x'])
del(merge['county_x'])
del(merge['year_x'])
del(merge['year_y'])


merge['ground_ratio'] = merge['gro_wat_3']/merge['total_withdrawal_3'] 
merge['fresh_ratio'] = merge['total_withdrawal_1']/merge['total_withdrawal_3']
merge['industry_ratio'] = merge['ind_9']/merge['total_withdrawal_3']
merge['irrigation_ratio'] = merge['irrigation_3']/merge['total_withdrawal_3']
merge['livestock_ratio'] = merge['livestock_3']/merge['total_withdrawal_3']
merge['aqua_ratio'] = merge['aqua_9']/merge['total_withdrawal_3']
merge['mining_ratio'] = merge['mining_9']/merge['total_withdrawal_3']
merge['thermoelectric_ratio'] = merge['thermoelectric_9']/merge['total_withdrawal_3']
merge['dom_per_cap'] = merge['dom_sup_5']+merge['dom_sup_7']

#tempDT$dom_per_cap <- as.numeric(tempDT$dom_sup_5)+as.numeric(tempDT$dom_sup_7)

merge=merge[(merge['year']>2010)&(merge['year']<=2015)]
merge=merge[merge['state']=='CA']

#state_le = preprocessing.LabelEncoder()
#county_le = preprocessing.LabelEncoder()

#merge['state_enc']=state_le.fit_transform(merge['state'])
#merge['county_enc']=county_le.fit_transform(merge['county'])

col_list=list(merge.columns.values)
lags=[]
resp_list=['none','d0','d1','d2','d3','d4']
for each in resp_list:
    for i in range(4,11):
        merge[each+'_'+str(i+1)+'_Week_lag']=merge.groupby("fips")[each].shift(i)
        lags.append(each+'_'+str(i+1)+'_Week_lag')
    
features=col_list[:]
merge['d0_pred'] = np.where(merge['d0']>0, 1, 0)

features.remove('year')
features.remove('valid_start')
features.remove('valid_end')
features.remove('date')
features.remove('state')
features.remove('county')
features.remove('d0')
features.remove('d1')
features.remove('d2')
features.remove('d3')
features.remove('d4')
features.remove('none')
features+=lags
#features.append('state_enc')
#features.append('county_enc')

########################
### Global Variables ###
########################
train = merge[merge['year']<2015]
test = merge[merge['year']==2015]
#features=list(train.columns.values)
#train['none']=merge['none']
target='d0_pred'
num_models=2
RANDOM_SEED=184

########################
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true+.0000001))) * 100

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance
    
def run_single(train, test, params, features, target, random_state=0):

    num_boost_round = 1000
    early_stopping_rounds = 20
    test_size = 0.125
    verbosity=False
    #verbosity=True
   
    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    #print('Length train:', len(X_train.index))
    #print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbosity)
    
 
    
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)
    

    return test_prediction

def run_single_plot(train, test, params, features, target, random_state=0):

    num_boost_round = 1000
    early_stopping_rounds = 20
    test_size = 0.125
    verbosity=True
    #verbosity=True
   
    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    #print('Length train:', len(X_train.index))
    #print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbosity)
    xgb.plot_importance(gbm)
    plt.show()

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)
    
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)
    ############################################ ROC Curve
    check=test_prediction
    #area under the precision-recall curve
    score = average_precision_score(test[target].values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

 
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(test[target].values, check)
    roc_auc = auc(fpr, tpr)
    #xgb.plot_importance(gbm)
    #plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    ##################################################


    return test_prediction

  
def optim_run_single(train, features, target, params, random_state=0):   
    num_boost_round = 1000
    early_stopping_rounds = 20
    test_size = 0.125
    verbosity=True
   
    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    #print('Length train:', len(X_train.index))
    #print('Length valid:', len(X_valid.index))
    y2_train = X_train[target]
    y2_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y2_train)
    dvalid = xgb.DMatrix(X_valid[features], y2_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbosity)
    return gbm.best_score
    #return -1.0 * gbm['test-rmse-mean'].iloc[-1]

def multi_model(train, test, params, features, target,num_models, random_state=0):
    all_preds=[]
    for i in range(num_models):
        preds =run_single(train, test,params, features, target, random_state)
        all_preds.append(preds)
        random_state=random_state+1
    avg_pred=np.mean(np.array(all_preds),axis=0)
    return avg_pred

def xgb_eval_single(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha):
    
    random_state=42
    eta=.05
    xtrain=train
    xfeatures=features

    
    params = {
        "objective": "reg:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "silent": 1,
        "seed": random_state,
        #"num_class" : 22,
    }
    
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    
    

    score =optim_run_single(xtrain, xfeatures, target, params, random_state)

    
    return -1*score
  
def xgb_eval_multi(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha):
    
    random_state=42
    eta=.05
    xtrain=train
    xfeatures=features
    
    params = {
        "objective": "reg:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "silent": 1,
        "seed": random_state,
        #"num_class" : 22,
    }
    
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    
    
    all_scores=[]
    for i in range(num_models):
        score =optim_run_single(xtrain, xfeatures, target, params, random_state)
        all_scores.append(score)
        random_state=random_state+1

    avg_score=np.mean(all_scores)
    
    return -1*avg_score

#####################################################################
### bayesian optimization with either single model or multi model ###
#####################################################################
""" 
xgb_bo = BayesianOptimization(xgb_eval_multi, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (3, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })
        
"""   
"""    
xgb_bo = BayesianOptimization(xgb_eval_single, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (3, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })


# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations

xgb_bo.maximize(init_points=1, n_iter=1, acq='ei')

params = xgb_bo.res['max']['max_params']
params['eval_metric']='auc'
params["objective"]="reg:logistic"
params["eta"]= .05
params["tree_method"] = 'exact'
params['max_depth'] = int(params['max_depth'])
"""       
        
#preds=multi_model(train, test,params, features, target,num_models,RANDOM_SEED)
preds=run_single_plot(train, test,params, features, target,RANDOM_SEED)
#print('mean absolute percentage error:',mean_absolute_percentage_error(test['none'],preds))