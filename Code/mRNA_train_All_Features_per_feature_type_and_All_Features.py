import numpy as np
import pandas as pd
import re
from pandas import MultiIndex, Int64Index
from pandas import MultiIndex, Int16Dtype
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import  precision_recall_curve, auc
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# Evaluation Metrics Functions
def compute_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    mcc = (tp*tn - fp*fn) / (np.sqrt(  (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)  ) + 1e-8)
    return round(mcc,3)
    
def compute_sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn)
    return round(sensitivity,3)
    
def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn/(fp+tn)
    return round(specificity,3)
    
def compute_accuracy(y_true, y_pred):
    accuracy = (y_true==y_pred).sum()/len(y_true)
    return round(accuracy,3)

def compute_precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tn/(tn+fp)
    return round(precision,3)

def compute_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro" , zero_division=1)
    return round(f1,3)

GLOBAL_RANDOM_STATE = 42


# #########################################################################################

classifiers = [GaussianNB(),
               DecisionTreeClassifier(),
               RandomForestClassifier(random_state=42),
               XGBClassifier(eval_metric='rmse', use_label_encoder=False, random_state=42),
               CatBoostClassifier(verbose = False),
               SVC( max_iter = -1)]

train_datasets = ["../All_Feature_Selection/train/Cytoplasm_all_fea_ovr_dataset.csv",
                  "../All_Feature_Selection/train/Nucleus_all_fea_ovr_dataset.csv",
                  "../All_Feature_Selection/train/Endoplasmic_Reticulum_all_fea_ovr_dataset.csv",
                  "../All_Feature_Selection/train/Extracellular_all_fea_ovr_dataset.csv",
                  "../All_Feature_Selection/train/Mitochondria_all_fea_ovr_dataset.csv"]

Localization = ["Cytoplasm", "Nucleus", "Endoplasmic_Reticulum", "Extracellular", "Mitochondria"]

with open('./ablation_results/train_on_80_test_on_20_All_Fea_CLASSIFIERS_RUN_RESULTS.txt', 'w') as cvresults:
    cvresults.write("Location,"+"Classifer,"+"Sen(%)," + "Spe(%)," + "ACC(%)," + "Pre(%)," + "F1(%)," + "MCC," + "\n")
    loc = 0
    for ds_train in train_datasets:  
        df_train = pd.read_csv(ds_train)
        print(df_train.shape)
        #df_train = df_train.sample(frac = 1)  # Shuffle dataframe
        df_train["label"] = df_train["label"].replace({Localization[loc]:1, "AllRest":0})
        X = df_train.drop(["SampleName","label"], axis=1)
        y = df_train["label"]
        
        print(X.shape)
        print(y.shape)


        print("\n--------------------------------------------------------")
        print("Training Dataset is: ", Localization[loc], df_train.shape)
        print("--------------------------------------------------------\n")

        #scaler = StandardScaler()
        #X = scaler.fit_transform(X_train)
        #Splitting the dataset into training and validation dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, shuffle=True,random_state=GLOBAL_RANDOM_STATE)
        for clf in classifiers:
            print(clf,"\n")
            model = clf
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            #print(classification_report(y_test, pred_values))
            mcc = compute_mcc(y_test, pred_values)
            sen = compute_sensitivity(y_test, pred_values)
            spe = compute_specificity(y_test, pred_values)
            acc = compute_accuracy(y_test, pred_values)
            pre = compute_precision(y_test, pred_values)
            F1  = compute_f1(y_test, pred_values)

            print('Sensitivity : {}'.format(round(sen,3)))
            print('Specifity : {}'.format(round(spe,3)))
            print('accuracy : {}'.format(round(acc,3)))
            print('Precision : {}'.format(round(pre,3)))
            print('F1 : {}'.format(round(F1,3)))
            print('mcc : {}'.format(round(mcc,3)))

            ResultsSummary = (str(round(sen,3)) + "," + 
                              str(round(spe,3)) + "," + 
                              str(round(acc,3)) + "," + 
                              str(round(pre,3)) + "," +
                              str(round(F1,3))  + "," +
                              str(round(mcc,3 )) 
                             )
            cvresults.write(Localization[loc] + "," + str(clf).split("(")[0] + "," + str(ResultsSummary + "\n"))
            print("========================================================")
        loc+=1
