from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd

def F1(A_true, A_pred):
    return f1_score(A_true.ravel(), A_pred.ravel())

def Precision(A_true, A_pred):
    return precision_score(A_true.ravel(), A_pred.ravel())

def Recall(A_true, A_pred):
    return recall_score(A_true.ravel(), A_pred.ravel())

def scores(df):
    y_pred = df['Pred']
    y_true = df['GT']
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, df['P_value'])
    cm = confusion_matrix(y_true, y_pred)
    typeI = cm[1, 0] / cm[1].sum()
    typeII = cm[0, 1] / cm[0].sum()
    res = {
        'F1': f1 * 100,
        'AUC': auc * 100,
        'Type I': typeI * 100,
        'Type II': typeII * 100,
    }
    return pd.Series(res)