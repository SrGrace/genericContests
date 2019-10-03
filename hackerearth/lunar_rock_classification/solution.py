import os

import cv2
import numpy as np
import pandas as pd
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import Adam
from tqdm import tqdm


data_dir = '/home/srgrace/genericContest_data/lunar_rock'


def pre_process(df, folder):
    processed_image = []
    for i in tqdm(range(df.shape[0])):
        # img = os.path.join(data_dir, folder) + df['image_id'][i].astype('str') + '.jpg'
        #
        # img = cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # # perform transformations on image
        # # EuclideanDistanceTransform = DIST_L2
        # # LinearDistanceTransform = DIST_L1, distance = |x1-x2| + |y1-y2|
        # # MaxDistanceTransform = DIST_C, distance = max(|x1-x2|,|y1-y2|)
        # b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
        # g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
        # r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
        #
        # transformed_image = cv2.merge((b, g, r))
        #
        # img = image.img_to_array(transformed_image)
        # processed_image.append(img)
        img = image.load_img(os.path.join(data_dir, folder) + df['Image_File'][i],
                             target_size=(28, 28, 1), grayscale=True)
        img = image.img_to_array(img)
        img = img / 255
        processed_image.append(img)

    return processed_image


def training():
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    # print(train.head())
    # exit()

    train_image = pre_process(train, 'train/')
    x = np.array(train_image)

    y1 = train['Class'].values
    # y = to_categorical(y)
    y = []
    for i in y1:
        if i == 'Small':
            y.append(0)
        else:
            y.append(1)
    y = to_categorical(y)
    # print(y, y.shape, len(y))
    # exit()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(BatchNormalization())
    # model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    # model.add(BatchNormalization())
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(BatchNormalization())
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(103, activation='softmax'))

    # data_gen = ImageDataGenerator(zoom_range=0.1, height_shift_range=0.1,
    #                               width_shift_range=0.1, rotation_range=10)

    # optimizer = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # annealer = LearningRateScheduler(lambda a: 1e-3 * 0.9 ** a)
    #
    # hist = model.fit_generator(data_gen.flow(x_train, y_train, batch_size=16),
    #                            steps_per_epoch=500,
    #                            epochs=100,
    #                            verbose=2,  # 1 for ETA, 0 for silent
    #                            validation_data=(x_test[:400, :], y_test[:400, :]),  # For speed
    #                            callbacks=[annealer])
    # print(hist)
    model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))

    model.save(os.path.join('./data/', 'seq_200_epochs.h5'))


def predict():
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    # print(train.head())
    # exit()

    test_image = pre_process(test, 'test/')

    x = np.array(test_image)
    model = load_model('./data/seq_200_epochs.h5')
    prediction = model.predict_classes(x)

    pred = []
    for i in prediction:
        if i == 0:
            pred.append('Small')
        else:
            pred.append('Large')
    # print(prediction)
    # exit()

    test['Class'] = pred
    # test = test.sort_values(by=['Image_File'], ascending=[True])
    test.to_csv('./data/predict2.csv', header=True, index=False)


# training()
predict()



