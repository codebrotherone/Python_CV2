import numpy as np
import scipy.io
import keras
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import History
from keras.preprocessing.image import ImageDataGenerator


def pull_and_resize_dataset(size):
    '''resizes images from X (dataset) to new size
    returns training set resized to size, and test set resized to size

    returns: (np array, np array)
    '''
    # load our train and test data
    train_data = scipy.io.loadmat('src/train_32x32.mat')
    test_data = scipy.io.loadmat('src/test_32x32.mat')
    train_imgs = train_data['X'][:, :, :, :]
    train_labels = train_data['y']
    test_imgs = test_data['X'][:, :, :, :]
    test_labels = test_data['y']
    #categorize our labels to binary class matrices
    y_train = train_labels.flatten()
    y_test = test_labels.flatten()
    #set 10 to 0
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    y_train = np_utils.to_categorical(y_train.reshape(1, -1)[0], num_classes=10)
    y_test = np_utils.to_categorical(y_test.reshape(1, -1)[0], num_classes=10)
    #amount to train/test
    num_train = train_imgs[:, :, :, :].shape[3]
    num_test = test_imgs[:, :, :, :].shape[3]

    tmp_arr_tr = []
    tmp_arr_te = []
    print('Working on Train Images')
    for i in range(0, num_train):
        print('we are working on {}/{}'.format(i, num_train))
        tmp = cv2.resize(train_imgs[:, :, :, i], size)
        tmp_arr_tr.append(tmp)
    tmp_arr_tr = np.array(tmp_arr_tr)
    np.save('train_48_48.npy', tmp_arr_tr)

    print('Working on Test Images')
    for i in range(0, num_test):
        print('we are working on {}/{}'.format(i, num_test))
        tmp = cv2.resize(train_imgs[:, :, :, i], size)
        tmp_arr_te.append(tmp)
    tmp_arr_te = np.array(tmp_arr_te)
    np.save('src/test_48_48.npy', tmp_arr_te)

    np.save('src/test_y.npy', y_test)
    np.save('src/train_y.npy', y_train)

    return tmp_arr_tr, y_train, tmp_arr_te, y_test

def preprocess():
    '''
    This will return preprocessed inputs for our DNN. It will take .mat files from memory and return them as a tuple
    of needed data formats for training our model

    :return: tuple of np.arrays like (X_train, y_train, X_test, y_test)
    '''
    #load our train and test data
    train_data = scipy.io.loadmat('src/train_32x32.mat')
    test_data = scipy.io.loadmat('src/test_32x32.mat')
    train_imgs = train_data['X'][:, :, :, :].astype(np.float32)
    train_labels = train_data['y']
    test_imgs = test_data['X'][:, :, :, :].astype(np.float32)
    test_labels = test_data['y']


    # print('min/max of train is {} and {}'.format(np.max(train_imgs), np.min(train_imgs)))
    # print('min/max of test is {} and {}'.format(np.max(test_imgs), np.min(test_imgs)))

    train_imgs -= int(np.mean(train_imgs))
    # train_imgs /= np.max(train_imgs)
    test_imgs -= int(np.mean(test_imgs))
    # test_imgs /= np.max(test_imgs)

    # print('min/max of train is {} and {}'.format(np.max(train_imgs), np.min(train_imgs)))
    # print('min/max of test is {} and {}'.format(np.max(test_imgs), np.min(test_imgs)))
    # print('shape is {};'.format(train_imgs.shape))
    # print('shape is {};'.format(test_imgs.shape))
    # cv2.imshow('img', train_imgs[:, :, :, 0].astype(np.int32))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #isolate our images from labels
    (X_train, y_train) = np.moveaxis(train_imgs, 3, 0), train_labels.flatten()
    (X_test, y_test) = np.moveaxis(test_imgs, 3, 0), test_labels.flatten()

    # print(X_train.shape)
    # cv2.imshow('img', X_train[0, :, :, :])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    y_train = np_utils.to_categorical(y_train.reshape(1, -1)[0], num_classes=10)
    y_test = np_utils.to_categorical(y_test.reshape(1, -1)[0], num_classes=10)

    return (X_train, y_train, X_test, y_test)


def createModel(X_train, y_train, X_test, y_test, input_shape):
    '''
    This creates a sequential model from the Keras Library to try a simple approach to creating DNNs
    This will not be needed for the TAs grading this project. The below Code is only for the 13 Layer Network
    Which is already trained and stored as an h5 file in the models directory. The configuration for every
    network is in models/configs/

    This function will rely heavily on Keras -- specficially it's sequential model class.
    '''

    #This is an example code for my Custom 10 Layer Network
    vgg13model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu', data_format='channels_last'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Dropout(.25),

        # # Conv2D(16, (3, 3), activation='relu', padding='same'),
        # # Conv2D(16, (3, 3), activation='relu', padding='same', ),
        # # Conv2D(16, (3, 3), activation='relu', padding='same', ),
        # # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Conv2D(6, (3, 3), activation='relu', padding='same', ),
        # Conv2D(6, (3, 3), activation='relu', padding='same', ),
        # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Dropout(.25),
        #
        # Conv2D(8, (3, 3), activation='relu', padding='same', ),
        # Conv2D(8, (3, 3), activation='relu', padding='same', ),
        # Conv2D(8, (3, 3), activation='relu', padding='same', ),
        # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Dropout(.5),
        Conv2D(64, (3, 3), activation='relu', padding='same', ),
        Conv2D(64, (3, 3), activation='relu', padding='same', ),
        Conv2D(64, (3, 3), activation='relu', padding='same', ),
        Conv2D(64, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dense(4096, activation='relu', name='fc1'),
        Dense(4096, activation='relu', name='fc2'),
        Dense(10, activation='softmax', name='predictions')
    ])

    #This is example code for my vgg16 style implementation
    # vgg16model = Sequential([
    #     Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu', data_format='channels_last'),
    #     Conv2D(32, (3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    #     Dropout(.25),
    #
    #     Conv2D(64, (3, 3), activation='relu', padding='same'),
    #     Conv2D(64, (3, 3), activation='relu', padding='same', ),
    #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    #     Dropout(.5),
    #
        # Conv2D(128, (3, 3), activation='relu', padding='same', ),
        # Conv2D(128, (3, 3), activation='relu', padding='same', ),
        # Conv2D(64, (3, 3), activation='relu', padding='same', ),
        # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # # Dropout(.5),
        # Conv2D(256, (3, 3), activation='relu', padding='same', ),
        # Conv2D(256, (3, 3), activation='relu', padding='same', ),
        # Conv2D(256, (3, 3), activation='relu', padding='same', ),
        # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Conv2D(256, (3, 3), activation='relu', padding='same', ),
        # Conv2D(256, (3, 3), activation='relu', padding='same', ),
        # Conv2D(256, (3, 3), activation='relu', padding='same', ),
        # MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Dropout(.5),
    #     Flatten(),
    #     Dense(256, activation='relu', name='fc1'),
    #     Dense(256, activation='relu', name='fc2'),
    #     Dense(10, activation='softmax', name='predictions')
    # ])

    sgd = SGD(lr=0.15, decay=1e-6, momentum=0.9, nesterov=True)

    vgg13model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        data_format='channels_last')

    print('METRICS NAMES')
    print(vgg13model.metrics_names)
    print('-'*100)
    print('TRAINING MODEL')
    history = History()
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=0,
                                               patience=2,
                                               verbose=0,
                                               mode='auto')

    #model settings
    callbacks = [history, early_stop]
    epochs = 30
    batch_size = 128
    print('X_Train shape {}'.format(X_train.shape[0]))
    # #
    model_history = vgg13model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                             steps_per_epoch=int(X_train.shape[0]/batch_size),
                                             verbose=1,
                                             epochs=epochs,
                                             callbacks=callbacks,
                                             shuffle=True,)

    # model_history = vgg13model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    #saving model
    filename = 'models/vgg13_adam_datagen_' + str(epochs) + '_' + str(batch_size)
    vgg13model.save(filename)

    print('-'*100)
    print('SCORING MODEL')
    score = vgg13model.evaluate(X_test, y_test, batch_size=64)
    print(vgg13model.metrics_names)
    print(score)

    return vgg13model, model_history, score


if __name__ == '__main__':
    (X_train, y_train, X_test, y_test) = preprocess()
    #Create and Save the Model
    model, model_history, score = createModel(X_train, y_train, X_test, y_test, input_shape=(32, 32, 3))
    print(model_history)

    # print('-' * 100)
    # print('This is the history of the model')
    # print(model_history.history)
    # print('-' * 100)
    # print('This is the score')
    # print(score)

    # model = load_model('models/vgg16_custom_dnn_adam_1.h5')
    # score = model.evaluate(X_test, y_test, batch_size=32)
    # print(model.summary())
    # print(score)
    # print(model.history.history)
    # print(keras.callbacks.History())



