import os
import pickle
from math import ceil

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

data_path = '/home/srgrace/genericContest_data/Sentiment_analysis_for_drug_medicines'
sample_sub = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
# print(sample_sub.head())


def remove_stopwords(input_text):
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)


def training():
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train = train[['text', 'drug', 'sentiment']]
    # train['text_comb'] = train['text'] + ' ||| ' + train['drug']
    # train.text_comb = train.text_comb.apply(remove_stopwords)
    train.text = train.text.apply(remove_stopwords)

    # print(Counter(train.sentiment))

    tfv = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3))
    x1 = tfv.fit_transform(train.text)
    pickle.dump(tfv, open('./data/tfv.pickle', 'wb'))

    smote = SMOTE(random_state=42)
    x_sm, y_sm = smote.fit_sample(x1, train.sentiment)
    # print(Counter(y_sm))
    # exit()
    x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.2, random_state=37)

    # Fitting Random Forest Classifier on TF-IDF
    clf1 = RandomForestClassifier()
    clf1.fit(x_train, y_train)
    pickle.dump(clf1, open('./data/rf_tfidf.pickle', 'wb'))

    clf2 = XGBClassifier()
    clf2.fit(x_train, y_train)
    pickle.dump(clf2, open('./data/xgb_tfidf.pickle', 'wb'))

    clf3 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf3.fit(x_train, y_train)
    pickle.dump(clf3, open('./data/lr_tfidf.pickle', 'wb'))

    pred1 = clf1.predict_proba(x_test)
    pred2 = clf2.predict_proba(x_test)
    pred3 = clf3.predict_proba(x_test)

    print(classification_report(y_test, (pred1 + pred2 + pred3)/3))
    print(confusion_matrix(y_test, (pred1 + pred2 + pred3)/3))


def predict():
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_hash = test['unique_hash']
    test = test[['text', 'drug']]

    # test['text_comb'] = test['text'] + ' ||| ' + test['drug']
    # test.text_comb = test.text_comb.apply(remove_stopwords)
    test.text = test.text.apply(remove_stopwords)

    tfv = pickle.load(open('./data/tfv.pickle', 'rb'))

    clf1 = pickle.load(open('./data/rf_tfidf.pickle', 'rb'))
    clf2 = pickle.load(open('./data/xgb_tfidf.pickle', 'rb'))
    clf3 = pickle.load(open('./data/lr_tfidf.pickle', 'rb'))
    pred1 = clf1.predict_proba(tfv.transform(test.text))
    pred1 = np.argmax(pred1, axis=1)

    pred2 = clf2.predict_proba(tfv.transform(test.text))
    pred2 = np.argmax(pred2, axis=1)

    pred3 = clf3.predict_proba(tfv.transform(test.text))
    pred3 = np.argmax(pred3, axis=1)

    prediction = np.array([])
    for i in range(0, len(test_hash)):
        prediction = np.append(prediction, int(pred1[i]*0.3 + pred2[i]*0.3 + pred3[i]*0.4))
    submission = pd.DataFrame(
        {'unique_hash': test_hash,
         'sentiment': prediction,
         })
    submission.to_csv('./data/ensemble_submission.csv', index=False)


# training()
predict()



