import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix, classification_report, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def additional_metrics(input_confusion_matrix):
    
    TN, FP, FN, TP = input_confusion_matrix.ravel()
    binary_cl_metrics = {}
    binary_cl_metrics['TPR'] = TP/(TP+FN)
    binary_cl_metrics['TNR'] = TN/(TN+FP) 
    binary_cl_metrics['PPV'] = TP/(TP+FP)
    binary_cl_metrics['NPV'] = TN/(TN+FN)
    binary_cl_metrics['FPR'] = FP/(FP+TN)
    binary_cl_metrics['FNR'] = FN/(TP+FN)
    binary_cl_metrics['FDR'] = FP/(TP+FP)
    binary_cl_metrics['ACC'] = (TP+TN)/(TP+FP+FN+TN)
    binary_cl_metrics['F1'] = (2 * TP)/(2 * TP+FP+FN)
    
    return binary_cl_metrics

def general_model_metrics(classifier_object, sample_y, sample_x, model_comment = ''):
    
    classifier_object_prob = classifier_object.predict_proba(sample_x)
    auc_classifier_object = roc_auc_score(sample_y,classifier_object_prob[:,1])
    gini_classifier_object = 2 * auc_classifier_object - 1

    # Predicted values and confusion matrix
    pred_classifier_object = classifier_object.predict(sample_x)
    cl_report = classification_report(sample_y,pred_classifier_object, output_dict=True, digits=3)

    cm_pred_classifier_object = confusion_matrix(sample_y, pred_classifier_object)
    plt.figure(figsize = (5,5))
    plt.title(f'Confusion matrix {model_comment}')
    sns.heatmap(cm_pred_classifier_object, annot=True, cmap = 'Blues', fmt = '', cbar=False)
    plt.show()
    

    print(f'Predicted AUC: {auc_classifier_object:.3%}, Gini coefficient: {gini_classifier_object:.3%}')
    print (classification_report(sample_y,pred_classifier_object, digits=2))
    print(cm_pred_classifier_object)
    
    return additional_metrics(cm_pred_classifier_object)


def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

def gini_coefficient(dependent,independent):
    auc = roc_auc_score(dependent,independent)
    return abs(2 * auc - 1)
    