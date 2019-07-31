import collections
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pd.set_option('display.max_colwidth', -1)
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from keras import models
from keras import layers
from keras import regularizers

sns.set(style="darkgrid")
sns.set(font_scale=1.3)

data_path = '/home/srgrace/genericContest_data/Sentiment_analysis_for_drug_medicines'
sample_sub = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
# print(sample_sub.head())

NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
NB_START_EPOCHS = 20  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent


def remove_stopwords(input_text):
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)


def one_hot_seq(seqs, nb_features=NB_WORDS):
    ohs = np.zeros((len(seqs), nb_features))
    for i, s in enumerate(seqs):
        ohs[i, s] = 1.
    return ohs


def deep_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='rmsprop'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])

    history = model.fit(x_train
                        , y_train
                        , epochs=NB_START_EPOCHS
                        , batch_size=BATCH_SIZE
                        , validation_data=(x_test, y_test)
                        , verbose=0)

    return history


def eval_metric(history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()


def compare_loss_with_baseline(h, model_name, base_history):
    loss_base_model = base_history.history['val_loss']
    loss_model = h.history['val_loss']

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, loss_base_model, 'bo', label='Validation Loss Baseline Model')
    plt.plot(e, loss_model, 'b', label='Validation Loss ' + model_name)
    plt.legend()
    plt.show()


def test_model(model, epoch_stop, x_train, y_train, x_test, y_test):
    model.fit(x_train
              , y_train
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0)
    results = model.evaluate(x_test, y_test)

    return results


def training():
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train = train.reindex(np.random.permutation(train.index))
    train = train[['text', 'drug', 'sentiment']]
    # print(train.head())

    train['text_comb'] = train['text'] + train['drug']

    # sns.factorplot(x="sentiment", data=train, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")
    # plt.show()

    train.text_comb = train.text_comb.apply(remove_stopwords)
    # print(train.head())

    x_train, x_test, y_train, y_test = train_test_split(train.text_comb, train.sentiment, test_size=0.2, random_state=37)
    # print('# Train data samples:', x_train.shape[0])
    # print('# Test data samples:', x_test.shape[0])
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    # Converting words to numbers
    tk = Tokenizer(num_words=NB_WORDS,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                   lower=True,
                   split=" ")
    tk.fit_on_texts(x_train)

    # print('Fitted tokenizer on {} documents'.format(tk.document_count))
    # print('{} words in dictionary'.format(tk.num_words))
    # print('Top 5 most common words are:', collections.Counter(tk.word_counts).most_common(5))
    x_train_seq = tk.texts_to_sequences(x_train)
    x_test_seq = tk.texts_to_sequences(x_test)

    # print('"{}" is converted into {}'.format(x_train[0], x_train_seq[0]))
    x_train_oh = one_hot_seq(x_train_seq)
    x_test_oh = one_hot_seq(x_test_seq)

    # print('"{}" is converted into {}'.format(x_train_seq[0], x_train_oh[0]))
    # print('For this example we have {} features with a value of 1.'.format(x_train_oh[0].sum()))

    y_train_oh = to_categorical(y_train)
    y_test_oh = to_categorical(y_test)
    # print('"{}" is converted into {}'.format(y_train[0], y_train_oh[0]))

    # Splitting of a validation set
    x_train_rest, x_valid, y_train_rest, y_valid = train_test_split(x_train_oh, y_train_oh,
                                                                    test_size=0.2, random_state=37)

    assert x_valid.shape[0] == y_valid.shape[0]
    assert x_train_rest.shape[0] == y_train_rest.shape[0]

    # print('Shape of validation set:', x_valid.shape)

    # Baseline model
    base_model = models.Sequential()
    base_model.add(layers.Dense(64, activation='relu', input_shape=(NB_WORDS,)))
    base_model.add(layers.Dense(64, activation='relu'))
    base_model.add(layers.Dense(3, activation='softmax'))
    base_model.summary()

    base_history = deep_model(base_model, x_train_rest, y_train_rest, x_valid, y_valid)

    # eval_metric(base_history, 'loss')
    # eval_metric(base_history, 'acc')

    # Handling over-fitting
    reduced_model = models.Sequential()
    reduced_model.add(layers.Dense(32, activation='relu', input_shape=(NB_WORDS,)))
    reduced_model.add(layers.Dense(3, activation='softmax'))
    reduced_model.summary()

    reduced_history = deep_model(reduced_model, x_train_rest, y_train_rest, x_valid, y_valid)

    # compare_loss_with_baseline(reduced_history, 'Reduced Model', base_history)

    # Adding regularization
    reg_model = models.Sequential()
    reg_model.add(
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(NB_WORDS,)))
    reg_model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    reg_model.add(layers.Dense(3, activation='softmax'))
    reg_model.summary()

    reg_history = deep_model(reg_model, x_train_rest, y_train_rest, x_valid, y_valid)

    # compare_loss_with_baseline(reg_history, 'Regularized Model', base_history)

    # Adding dropout layers
    drop_model = models.Sequential()
    drop_model.add(layers.Dense(64, activation='relu', input_shape=(NB_WORDS,)))
    drop_model.add(layers.Dropout(0.5))
    drop_model.add(layers.Dense(64, activation='relu'))
    drop_model.add(layers.Dropout(0.5))
    drop_model.add(layers.Dense(3, activation='softmax'))
    drop_model.summary()

    drop_history = deep_model(drop_model, x_train_rest, y_train_rest, x_valid, y_valid)

    # compare_loss_with_baseline(drop_history, 'Dropout Model', base_history)

    # Training on the full train data and evaluation on test data
    base_results = test_model(base_model, NB_START_EPOCHS, x_train_oh, y_train_oh, x_test_oh, y_test_oh)
    print('Test accuracy of baseline model: {0:.2f}%\n'.format(base_results[1] * 100))

    reduced_results = test_model(reduced_model, NB_START_EPOCHS, x_train_oh, y_train_oh, x_test_oh, y_test_oh)
    print('Test accuracy of reduced model: {0:.2f}%\n'.format(reduced_results[1] * 100))

    reg_results = test_model(reg_model, NB_START_EPOCHS, x_train_oh, y_train_oh, x_test_oh, y_test_oh)
    print('Test accuracy of regularized model: {0:.2f}%\n'.format(reg_results[1] * 100))

    drop_results = test_model(drop_model, NB_START_EPOCHS, x_train_oh, y_train_oh, x_test_oh, y_test_oh)
    print('Test accuracy of dropout model: {0:.2f}%\n'.format(drop_results[1] * 100))

    base_model.save(os.path.join('./data/', 'base_model.h5'))
    reduced_model.save(os.path.join('./data/', 'reduced_model.h5'))
    reg_model.save(os.path.join('./data/', 'reg_model.h5'))
    drop_model.save(os.path.join('./data/', 'drop_model.h5'))


def predict():
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_hash = test['unique_hash']
    # print(test.head())
    test = test.reindex(np.random.permutation(test.index))
    test = test[['text', 'drug']]

    test['text_comb'] = test['text'] + test['drug']
    test.text_comb = test.text_comb.apply(remove_stopwords)
    # print(test.head())

    tk = Tokenizer(num_words=NB_WORDS,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                   lower=True,
                   split=" ")
    tk.fit_on_texts(test.text_comb)

    test_seq = tk.texts_to_sequences(test.text_comb)
    test_oh = one_hot_seq(test_seq)

    reg_model = models.load_model('./data/reg_model.h5')
    prediction = reg_model.predict_classes(test_oh)

    submission = pd.DataFrame(
        {'unique_hash': test_hash,
         'sentiment': prediction,
         })
    submission.to_csv('./data/dl_submission.csv', index=False)


# training()
predict()


