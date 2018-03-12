'''
This file is a configuration file for hyperparameters/configurations for different VGG16 inspired CNNs.
This file is not necessary, unless a model needs to be trained again. It was only a reference. All configurations
for ALL FILES are in the models/config directory.
'''

pt_global_settings = {
    'nb_epoch': 20,
    'samples_per_epoch': 73257,
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
    'batch_size': 256,
    'metrics': ['accuracy']
}

# layers_13_adam = {
#     'optimizer': 'adam',
#     'loss': 'categorical_crossentropy',
#     'metrics': ['loss', 'accuracy'],
#     'epochs': 20,
#     'batch_size': 32,
#     'layers': {'conv2': 10, 'fclayers': 3},
#     'model_config_filename': 'models/vgg13_adam_1_ep20.h5'
# }
#
# vgg16_adam = {
#     'optimizer': 'adam',
#     'loss': 'categorical_crossentropy',
#     'metrics': ['loss', 'accuracy'],
#     'epochs': 20,
#     'batch_size': 32,
#     'layers': {'conv2': 10, 'fclayers':3}
# }
#

'''
model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
'''