import keras
import keras.layers
import keras.preprocessing
import dnn
import config

from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History

def create_datagens(X_train):
    '''
    This will create data generators for the training and test dataset using keras ImageDataGenerator and .flow_from_directory()
    :return: 2 datagenerators: training, and test
    '''

    print(X_train.shape)
    print(X_test.shape)

    train_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format='channels_last'
    )

    train_datagen.fit(X_train)

    return train_datagen

def add_last_layers(pretrained_model, nb_classes):
    '''
    Adding the last layer to the pretrained imagenet model

    :param pretrained_model:keras.aplications.V3_Inception

    :return: keras Model
    '''
    print('CURRENT LAYERS')
    print(pretrained_model.summary())
    x = pretrained_model.output
    x = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dense(4096, activation='relu')(x)
    # x = Dense(1024, activation='relu')(x)
    pred = Dense(nb_classes, activation='softmax')(x)
    model_changed = Model(input=pretrained_model.input, output=pred)

    return model_changed

def freeze(pretrained_model, changed_model):
    '''Freeze all other layers except the last two that we add, and compile

    '''

    loss = config.pt_global_settings['loss']
    optimizer = config.pt_global_settings['optimizer']
    metrics = config.pt_global_settings['metrics']

    for l in pretrained_model.layers:
        print('Layer frozen')
        l.trainable = False
    print(changed_model)

    #Fine TUNING for INCEPTION V3
    # for layer in changed_model.layers[:172]:
    #     layer.trainable = False
    # for layer in changed_model.layers[172:]:
    #     layer.trainable = True
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
    #               loss='categorical_crossentropy')

    changed_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return changed_model

def train(train_datagen, changed_model):
    '''
    Trains the pre-trained network using our datagen for SVHN dataset (2nd)

    :param train_datagen: datagenerator keras.applications
    :param changed_model: modified pretrained network Inception_V3
    :return: history of model, Model.history.history()
    '''
    epochs = config.pt_global_settings['nb_epoch']
    samples_per_epoch = config.pt_global_settings['samples_per_epoch']
    batch_size = config.pt_global_settings['batch_size']

    history = changed_model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                          steps_per_epoch=samples_per_epoch/batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks()
                                          )
    changed_model.save('models/pretrained_vgg16_datagen_2_ep20_32.h5')

    return history


def callbacks():
    '''return callbacks for dnn '''

    history = History()
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=0,
                                               patience=2,
                                               verbose=0,
                                               mode='auto')

    # model settings
    callbacks = [history, early_stop]
    return callbacks


if __name__ == '__main__':

    #This is the main portion of the script that calls the functions above. It is not necessary for grading.
    #This is merely the work involved in training a pretrained VGG16 Network
    X_train, y_train, X_test, y_test = dnn.preprocess()
    pretrained_model = keras.applications.VGG16(weights='imagenet', include_top=False)
    train_datagen = create_datagens(X_train)
    tl_model = add_last_layers(pretrained_model, 10)
    print(tl_model)
    changed_model = freeze(pretrained_model, changed_model=tl_model)
    print(changed_model)
    hist = train(train_datagen, changed_model)
    print('-'*100)
    print('This is the history')
    print(hist)
    print('Score')
    score = changed_model.evaluate(X_test, y_test)
    print(score)