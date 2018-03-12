import pprint as pp
import dnn
import config
import cv2
import pickle
import numpy as np
import os

from keras.models import load_model

#This is the run file that loads each model, and it's summary and configuration so that the graders can see
#how the networks were trained and designed.

def saveConfig(config, filename):
    '''Saves the configuration (model.get_config()) as a pickle file
    '''

    tmp = config
    cwd = os.getcwd()
    path = cwd + '/models/configs/' + filename.strip('.h5') + '.pickle'

    print('Saving Config File...')
    try:

        with open(path, 'xb') as handle:
            pickle.dump(tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path, 'rb') as handle:
            tmp2 = pickle.load(handle)
    except FileExistsError:
        print('Config File Exists, moving on...')
        return None
    #check to see if what we wrote == what we read
    if tmp == tmp2:
        print('Done saving the config file!! ')
        return None
    else:
        print('We had issues saving the config file....\n No worries, you can always\
              load the model then call model.get_config()')

def getModelsInfo(filename):
    '''
    Get's the Models and returns Model/Configuration. This will also print the input as well.. while saving the configuration files.

    :return: keras.Model, keras.Model.get_config()
    '''

    path = 'models/' + filename
    model = load_model(path)
    config = model.get_config()
    saveConfig(config, filename)

    return model, config

# def testImages(model):
#     '''
#     This is for testing images we have against whichever model is passed into this function
#
#     :param images: 4d array containing image shape [num_images, 32, 32, 3]
#     :param model: keras.Model() class, pre-trained
#
#     :return: binary class matrix representing probabilities for 0-9
#     '''
#
#     im = cv2.imread('graded_images/1.png')
#     # h, w = im.shape[0], im.shape[1]
#     # print(h, w)
#     # im = cv2.resize(im, (32, 32), fx=32./float(w), fy=32./float(h), interpolation=cv2.INTER_CUBIC)
#     im = np.array([im])
#     im -= np.mean(im)
#
#     print(im.shape)
#     print(im[0, :, :, :])
#     print(im.shape)
#     cv2.imshow('img', im[0, :, :, :])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def main():
    print('Loading images first for evaluation of model accuracy later...')
    print('We will print the configuration, summary, and score')
    X_train, y_train, X_test, y_test = dnn.preprocess()
    print('Done! \n Iterating through Models Now... \n')

    # Print the Summary and Configuration for All the Trained Models
    for filename in os.listdir('models/'):
        if filename == 'configs' or filename == 'experimental_models':
            continue
        else:
            model, config = getModelsInfo(filename)
            print('-' * 200)
            print(filename)
            print('This is the configuration for {}'.format(filename.strip('.h5')))
            pp.pprint(config)
            print('-' * 200)
            print('This is the summary for {}'.format(filename.strip('.h5')))
            print(model.summary())
            print('-' * 200)
            print('\n')
            print('Scoring Model Now...')
            score = model.evaluate(X_test, y_test, batch_size=32)
            saved_scores = {'network': filename, 'score': score}
            print('This is the score \n Loss: {} and Accuracy: {}'.format(score[0], score[1]))
            print('-' * 200)
            print('NEXT MODEL')
            print('-' * 200)

    print('Done! Hope you like it!')

if __name__ == '__main__':
    main()