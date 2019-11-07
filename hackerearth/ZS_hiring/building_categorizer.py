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

    clf1 = RandomForestClassifier()
    clf1.fit(x_train, y_train)
    pickle.dump(clf1, open('./data/rf_tfidf.pickle', 'wb'))

    # clf2 = XGBClassifier()
    # clf2.fit(x_train, y_train)
    # pickle.dump(clf2, open('./data/xgb_tfidf.pickle', 'wb'))

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


# training()
prediction()
# pre_process()





