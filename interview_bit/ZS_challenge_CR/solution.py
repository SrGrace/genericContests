import pickle

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


def data_prep(df):
    df.drop(['match_event_id', 'location_x', 'location_y', 'game_season', 'team_name', 'date_of_game',
             'home/away', 'lat/lng', 'match_id', 'team_id'], axis=1, inplace=True)
    df.dropna(subset=['shot_id_number'], inplace=True)
    # print(df.head())
    # print(df.columns)

    df["remaining_min"].fillna(df["remaining_min"].mean(), inplace=True)
    df["remaining_min.1"].fillna(df["remaining_min.1"].mean(), inplace=True)
    df["remaining_sec"].fillna(df["remaining_sec"].mean(), inplace=True)
    df["remaining_sec.1"].fillna(df["remaining_sec.1"].mean(), inplace=True)
    df["power_of_shot"].fillna(df["power_of_shot"].mean(), inplace=True)
    df["power_of_shot.1"].fillna(df["power_of_shot.1"].mean(), inplace=True)
    df["knockout_match"].fillna(df["knockout_match"].mean(), inplace=True)
    df["knockout_match.1"].fillna(df["knockout_match.1"].mean(), inplace=True)
    df["distance_of_shot"].fillna(df["distance_of_shot"].mean(), inplace=True)
    df["distance_of_shot.1"].fillna(df["distance_of_shot.1"].mean(), inplace=True)

    df['area_of_shot'].fillna('empty', inplace=True)
    area_of_shot = {
        'Left Side(L)': 1,
        'Left Side Center(LC)': 2,
        'Right Side Center(RC)': 3,
        'Center(C)': 4,
        'Right Side(R)': 5,
        'Mid Ground(MG)': 6,
        'empty': 4
    }
    df['area_of_shot'] = df['area_of_shot'].apply(lambda a: area_of_shot[a])

    df['shot_basics'].fillna('empty', inplace=True)
    shot_basics = {
        'Mid Range': 1,
        'Goal Area': 2,
        'Goal Line': 3,
        'Penalty Spot': 4,
        'Right Corner': 5,
        'Mid Ground Line': 6,
        'Left Corner': 7,
        'empty': 4
    }
    df['shot_basics'] = df['shot_basics'].apply(lambda a: shot_basics[a])

    df['type_of_combined_shot'].fillna('empty', inplace=True)
    type_of_combined_shot = {
        'shot - 0': 1,
        'shot - 1': 2,
        'shot - 2': 3,
        'shot - 3': 4,
        'shot - 4': 5,
        'shot - 5': 6,
        'empty': 4
    }
    df['type_of_combined_shot'] = df['type_of_combined_shot'].apply(lambda a: type_of_combined_shot[a])

    df['type_of_shot'].fillna('empty', inplace=True)
    type_of_shot = dict()
    for ty in df['type_of_shot'].unique():
        try:
            typ = int(ty.split(' - ')[1] + 1)
            type_of_shot[ty] = typ
        except:
            type_of_shot[ty] = 29
    df['type_of_shot'] = df['type_of_shot'].apply(lambda a: type_of_shot[a])

    df['range_of_shot'].fillna('empty', inplace=True)
    range_of_shot = {
        '8-16 ft.': 1,
        '16-24 ft.': 2,
        'Less Than 8 ft.': 3,
        '24+ ft.': 4,
        'Back Court Shot': 5,
        'empty': 3
    }
    df['range_of_shot'] = df['range_of_shot'].apply(lambda a: range_of_shot[a])

    df['finesse'] = (df['distance_of_shot'] * df['power_of_shot']) / df['area_of_shot']


def train():
    data_path = '/home/srgrace/Downloads/Cristano_Ronaldo_Final_v1/train.csv'

    df = pd.read_csv(data_path, index_col=False)
    data_prep(df)
    df.drop('shot_id_number', axis=1, inplace=True)

    # print(df.columns)
    # print(df.head())
    # exit()

    # Split-out validation data-set
    x = df.drop('is_goal', axis=1).values
    y = df['is_goal'].values

    pca = PCA(n_components=3, random_state=0)
    reduced_data = pca.fit_transform(x)
    pickle.dump(pca, open('data/pca.pickle', "wb"))

    validation_size = 0.20
    seed = 7
    x_train, x_test, y_train, y_test = train_test_split(reduced_data, y, test_size=validation_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=seed)

    # Test options and evaluation metric
    # scoring = 'accuracy'
    #
    # Spot Check Algorithms
    # models = []
    # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('RF', RandomForestClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC(gamma='auto')))
    #
    # # evaluate each model in turn
    # results = []
    # names = []
    # for name, model in models:
    #     kfold = KFold(n_splits=10, random_state=seed)
    #     cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)

    # clf = RandomForestClassifier(random_state=0)
    # clf.fit(x_train, y_train)
    #
    # # clf_path = 'data/rf_clf.pickle'
    # # pickle.dump(clf, open(clf_path, 'wb'))
    #
    # clf = XGBClassifier(objective="binary:logistic", random_state=42)
    # clf.fit(x_train, y_train)
    # clf_path = 'data/xgb_clf.pickle'
    # pickle.dump(clf, open(clf_path, 'wb'))

    # Blending
    model1 = RandomForestClassifier(random_state=0)
    model1.fit(x_train, y_train)
    pickle.dump(model1, open('data/rf.pickle', 'wb'))
    val_pred1 = model1.predict(x_val)
    test_pred1 = model1.predict(x_test)

    val_pred1 = pd.DataFrame(val_pred1)
    test_pred1 = pd.DataFrame(test_pred1)

    model2 = LogisticRegression(solver='liblinear', multi_class='ovr')
    model2.fit(x_train, y_train)
    pickle.dump(model2, open('data/lgr.pickle', 'wb'))
    val_pred2 = model2.predict(x_val)
    test_pred2 = model2.predict(x_test)

    val_pred2 = pd.DataFrame(val_pred2)
    test_pred2 = pd.DataFrame(test_pred2)

    df_val = pd.concat([pd.DataFrame(x_val), val_pred1, val_pred2], axis=1)
    df_test = pd.concat([pd.DataFrame(x_test), test_pred1, test_pred2], axis=1)

    df_val = df_val.loc[:, ~df_val.columns.duplicated()]
    df_test = df_test.loc[:, ~df_test.columns.duplicated()]

    model = XGBClassifier(objective="binary:logistic", random_state=42)
    model.fit(df_val.values, y_val)
    pickle.dump(model, open('data/xgb.pickle', 'wb'))

    print(model.score(df_test.values, y_test))

    predictions = model.predict(x_test)

    print(classification_report(y_test, predictions))


def predict():
    data_path = '/home/srgrace/Downloads/Cristano_Ronaldo_Final_v1/val.csv'

    df = pd.read_csv(data_path, index_col=False)
    data_prep(df)
    df.drop('is_goal', axis=1, inplace=True)

    test_fet1 = df.drop('shot_id_number', axis=1).values
    pca_model = pickle.load(open("data/pca.pickle", 'rb'))
    test_fet = pca_model.fit_transform(test_fet1)

    test_id = df['shot_id_number']

    loaded_model = pickle.load(open("data/xgb.pickle", 'rb'))

    pred_dict = {
        'shot_id_number': [],
        'is_goal': []
    }
    for i in range(len(test_fet)):
        prediction = loaded_model.predict_proba([test_fet[i]])
        max_prob = np.max(prediction[0])
        # print(max_prob)

        if max_prob:
            pred_dict['shot_id_number'].append(int(test_id.iloc[i]))
            pred_dict['is_goal'].append(float(max_prob))

    df1 = pd.DataFrame.from_dict(pred_dict, orient='index')
    df1 = df1.T
    df1.to_csv('data/predict.csv', index=False)


train()
# predict()


'''
XGB with new feature
              precision    recall  f1-score   support

         0.0       0.60      0.77      0.68      2724
         1.0       0.55      0.35      0.43      2162

   micro avg       0.59      0.59      0.59      4886
   macro avg       0.58      0.56      0.55      4886
weighted avg       0.58      0.59      0.57      4886


Blend rf, lrg with XGB with new features
0.5697912402783463
              precision    recall  f1-score   support

         0.0       0.59      0.72      0.65      2724
         1.0       0.52      0.38      0.44      2162

   micro avg       0.57      0.57      0.57      4886
   macro avg       0.56      0.55      0.55      4886
weighted avg       0.56      0.57      0.56      4886



'''

