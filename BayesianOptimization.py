
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from data_sampling import X_test, X_train, y_test, y_train, df, X, y, X_train_rebal, X_test_rebal, y_train_rebal, y_test_rebal
from metrics import general_model_metrics
import xgboost as xgb


def gbm_bopt(max_depth, learning_rate, n_estimators):
    params_gbm = {}
    params_gbm['max_depth'] = int(max_depth)
    params_gbm['learning_rate'] = learning_rate
    params_gbm['n_estimators'] = int(n_estimators)
    
    estimator_xgb = xgb.XGBClassifier(objective='binary:logistic'
                                          , booster='gbtree'
                                          , eval_metric='auc')
    
    # Fit the estimator
    estimator_xgb.fit(X_train_rebal,y_train_rebal)
    
    # calculate out-of-the-box roc_score using validation set 1
    prob_xgb = estimator_xgb.predict_proba(X_test)
    prob_xgb = prob_xgb[:,1]
    gini_classifier_object = 2 * roc_auc_score(y_test,prob_xgb) - 1
    
    # return the mean validation score to be maximized 
    return np.array([gini_classifier_object]).mean()
    
hyperparameter_space ={
    'max_depth':(1, 10),
    'learning_rate': (0, 1),
    'n_estimators':(40, 300),
}
gbm_bo = BayesianOptimization(gbm_bopt, hyperparameter_space)
gbm_bo.maximize(init_points=3, n_iter=5)