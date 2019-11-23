import os
import pickle
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

path = '/home/srgrace/genericContest_data/ZS_hiring/'


def training():
    train = pd.read_csv(os.path.join(path, 'train_file.csv'))
    category = {
        'SINGLE FAMILY / DUPLEX': 1,
        'COMMERCIAL': 2,
        'MULTIFAMILY': 3,
        'INSTITUTIONAL': 4,
        'INDUSTRIAL': 5
    }
    y = train['Category'].apply(lambda a: category[a])

    tfv = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
    x = tfv.fit_transform(train.Description.values.astype('U'))
    pickle.dump(tfv, open('./data/tfv.pickle', 'wb'))

    # smote = SMOTE(random_state=42)
    # x_sm, y_sm = smote.fit_sample(x, y)
    # print(Counter(y_sm))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=37)

    # clf1 = RandomForestClassifier()
    # clf1.fit(x_train, y_train)
    # pickle.dump(clf1, open('./data/rf_tfidf.pickle', 'wb'))

    clf1 = XGBClassifier()
    clf1.fit(x_train, y_train)
    pickle.dump(clf1, open('./data/xgb_tfidf.pickle', 'wb'))

    pred1 = clf1.predict(x_test)

    print(confusion_matrix(y_test, pred1))
    print(classification_report(y_test, pred1, target_names=['1', '2', '3', '4', '5']))


def prediction():
    test = pd.read_csv(os.path.join(path, 'test_file.csv'))
    permit_no = test['Application/Permit Number']

    tfv = pickle.load(open('./data/tfv.pickle', 'rb'))
    clf1 = pickle.load(open('./data/rf_tfidf.pickle', 'rb'))

    pred1 = clf1.predict(tfv.transform(test.Description.values.astype('U')))
    # pred1 = np.argmax(pred1, axis=1)
    test['Category'] = pred1

    category = {
        1: 'SINGLE FAMILY / DUPLEX',
        2: 'COMMERCIAL',
        3: 'MULTIFAMILY',
        4: 'INSTITUTIONAL',
        5: 'INDUSTRIAL'
    }
    predict = test['Category'].apply(lambda a: category[a])

    submission = pd.DataFrame(
        {'Application/Permit Number': permit_no,
         'Category': predict,
         })
    submission.to_csv('./data/rf_submission.csv', index=False)


def pre_process():
    train = pd.read_csv(os.path.join(path, 'train_file.csv'))
    # print(train.columns)
    x = train.iloc[:, 0:18]
    y = train.iloc[:, -1]
    corrmat = train.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(train[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


training()
# prediction()
# pre_process()

'''
RF-TFIDF
[[3469   82   66    1    0]
 [ 163 1774   45    5    4]
 [ 233  134  462    1    1]
 [  29  133    1   39    0]
 [  10   53    1    0    2]]
              precision    recall  f1-score   support

           1       0.89      0.96      0.92      3618
           2       0.82      0.89      0.85      1991
           3       0.80      0.56      0.66       831
           4       0.85      0.19      0.31       202
           5       0.29      0.03      0.05        66

   micro avg       0.86      0.86      0.86      6708
   macro avg       0.73      0.53      0.56      6708
weighted avg       0.85      0.86      0.84      6708


XGB-TFIDF
[[3476   60   79    3    0]
 [ 204 1718   49   17    3]
 [ 241  107  483    0    0]
 [  30   98    2   72    0]
 [  12   51    0    0    3]]
              precision    recall  f1-score   support

           1       0.88      0.96      0.92      3618
           2       0.84      0.86      0.85      1991
           3       0.79      0.58      0.67       831
           4       0.78      0.36      0.49       202
           5       0.50      0.05      0.08        66

   micro avg       0.86      0.86      0.86      6708
   macro avg       0.76      0.56      0.60      6708
weighted avg       0.85      0.86      0.85      6708


'''



