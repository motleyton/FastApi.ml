
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
import pandas as pd
import os
import numpy as np



class data():
    """Class for prepare data to experiments"""
    
    def prep_data_baf(path) -> pd.DataFrame:
        """INPUT: str path to dir with pandas.DataFrame
            OUTPUT: pandas.DataFrame TRAIN (X, y), VAL (X, y), TEST (X, y)

            (splited by time 5/1/1)"""

        # Check
        assert type(path) == str, "Not string format of arg, need path - string"
        assert 'Base.csv' in os.listdir(path), "The specified folder does not contain Base data"

        tmp = pd.read_csv(f'{path}/Base.csv')

        #object_cols = tmp.dtypes[tmp.dtypes == 'object'].index.tolist()   
        #num_cols = tmp.dtypes[tmp.dtypes != 'object'].index.tolist()

        target = 'fraud_bool'

        new_num = ['income',
                     'name_email_similarity',
                     'prev_address_months_count',
                     'current_address_months_count',
                     'customer_age',
                     'days_since_request',
                     'intended_balcon_amount',
                     'zip_count_4w',
                     'velocity_6h',
                     'velocity_24h',
                     'velocity_4w',
                     'bank_branch_count_8w',
                     'date_of_birth_distinct_emaiprep_data_bafls_4w',
                     'credit_risk_score',
                     'bank_months_count',
                     'proposed_credit_limit',
                     'session_length_in_minutes',
                     'device_fraud_count',
                     'month']

        need_lg_columns = ['prev_address_months_count',
                     'current_address_months_count',
                     'days_since_request',
                     'zip_count_4w',
                     'bank_branch_count_8w',
                     'date_of_birth_distinct_emails_4w',
                     'proposed_credit_limit',
                     'session_length_in_minutes']

        tmp.loc[:, need_lg_columns] = tmp.loc[:, need_lg_columns].apply(lambda x: np.log1p(x+10))

        new_cat = ['payment_type',
                     'employment_status',
                     'housing_status',
                     'source',
                     'device_os',
                     'email_is_free',
                     'phone_home_valid',
                     'phone_mobile_valid',
                     'has_other_cards',
                     'foreign_request',
                     'keep_alive_session',
                     'device_distinct_emails_8w']

        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        # Get one-hot-encoded columns
        ohe_cat = pd.DataFrame(ohe.fit_transform(tmp[new_cat]), index=tmp.index)
        result = pd.concat([tmp[new_num], ohe_cat, tmp[target]], axis=1)

        # Split data
        train = result[result.month < 5]
        valid = result[(result.month >= 5) & (result.month < 6)]
        test = result[result.month >= 6]

        # Normalize  
        X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
        X_val, y_val = valid.iloc[:, :-1], valid.iloc[:, -1]
        X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

        transformer = Normalizer().fit(X_train.values)

        # Transform
        X_train_n = transformer.transform(X_train.values)
        X_val_n = transformer.transform(X_val.values)
        X_test_n = transformer.transform(X_test.values)

        return X_train_n, y_train, X_val_n, y_val, X_test_n, y_test
    
    
    
    
    def prep_data_ccf(path) -> pd.DataFrame:
        """INPUT: str path to dir with pandas.DataFrame
            OUTPUT: pandas.DataFrame TRAIN (X, y), VAL (X, y), TEST (X, y)

            (splited by day 90 days / 25 days / 35 days)"""

        # Check
        assert type(path) == str, "Not string format of arg, need path - string"
        assert 'creditcard.csv' in os.listdir(path), "The specified folder does not contain creditcard data"

        tmp = pd.read_csv(f'{path}/creditcard.csv')
        t = tmp.copy()
        # Скорее всего в этом столбце минуты
        t['day'] = (t.Time/60/24).astype(int)
        
        # Split data
        train = tmp[t.day < 90]
        valid = tmp[(t.day >= 90)&((t.day < 105))]
        test = tmp[t.day >= 105]
  
        X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
        X_val, y_val = valid.iloc[:, :-1], valid.iloc[:, -1]
        X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

        return X_train, y_train, X_val, y_val, X_test, y_test

class metrics():
    """Класс для расчета метрик"""

    def show_metrics(y_true, y_score):
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        UTPUT:
                print: tpr
                        fpr
                        precision
                        recall
                        tnr
                        auc
                        f1
                        mcc
                        g_mean
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # True positive rate (sensitivity or recall)
        tpr = tp / (tp + fn)
        # False positive rate (fall-out)
        fpr = fp / (fp + tn)
        # Precision
        precision = tp / (tp + fp)
        # Recall
        recall = tp / (tp + fn)
        # True negatvie rate (specificity)
        tnr = 1 - fpr
        # F1 score
        f1 = 2*tp / (2*tp + fp + fn)
        # ROC-AUC for binary classification
        auc = (tpr+tnr) / 2

        # MCC
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # G-mean
        g_mean = np.sqrt((tp/(tp+fn)) * (tn/(fp+tn)))

        #print("True positive: ", tp)
        #print("False positive: ", fp)
        #print("True negative: ", tn)
        #print("False negative: ", fn)

        print("True positive rate (recall): ", tpr)
        print("False positive rate: ", fpr)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("True negative rate: ", tnr)
        print("ROC-AUC: ", auc)
        print("F1: ", f1)
        print("MCC: ", mcc)
        print("G-mean: ", g_mean)

    def write_metrics(y_true, y_score):
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        
        UTPUT:          Recall_not_fraud
                        Recall_fraud
                        auc
                        f1
                        mcc
                        g_mean
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # True positive rate (sensitivity or recall)
        tpr = tp / (tp + fn)
        # False positive rate (fall-out)
        fpr = fp / (fp + tn)
        # Precision
        precision = tp / (tp + fp)
        
        # Recall_fraud
        recall_f = tpr #tp / (tp + fn)
        # Recall_not_fraud
        recall_nf = 1 - fpr
        
        # True negatvie rate (specificity)
        tnr = 1 - fpr
        # F1 score
        f1 = 2*tp / (2*tp + fp + fn)
        # ROC-AUC for binary classification
        auc = (tpr+tnr) / 2

        # MCC
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # G-mean
        g_mean = np.sqrt((tp/(tp+fn)) * (tn/(fp+tn)))


        #print("True positive rate (recall): ", tpr)
        #print("False positive rate: ", fpr)
        #print("Precision: ", precision)
        #print("Recall_f: ", recall_f)
        #print("Recall_nf: ", recall_nf)
        #print("True negative rate: ", tnr)
        #print("ROC-AUC: ", auc)
        #print("F1: ", f1)
        #print("MCC: ", mcc)
        #print("G-mean: ", g_mean)
        
        return recall_nf, recall_f, auc, f1, mcc, g_mean

    # Metrics
    def mcc(y_true, y_score)-> float:
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        OUTPUT:  float -> mcc (коэффициент корреляции Мэтьюса)
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # MCC
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        return mcc
    
    def g_mean(y_true, y_score)-> float:
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        OUTPUT:  float -> mcc (коэффициент корреляции Мэтьюса)
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # g_mean
        g_mean = np.sqrt((tp/(tp+fn)) * (tn/(fp+tn)))
        
        return g_mean


if __name__ == "__main__":
     output = data.prep_data_ccf('/data/raw')
     print(output)
