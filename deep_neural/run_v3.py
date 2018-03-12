# import dnn
import os
import cv2
import numpy as np
import load_models
from keras.models import load_model
from collections import Counter

cwd = os.getcwd()
IMAGE_DIR = cwd + '/graded_images'


def gauss_pyr():
    tmp = []
    image_labels = []
    for obj in os.listdir(IMAGE_DIR):
        if obj != 'orig' and '.png' in str(obj):
            # print(obj)
            im = cv2.imread('graded_images/' + obj)
            tmp.append(im)
            image_labels.append(obj)
        else:
            continue

    print('WE HAVE {} IMAGES TO TEST in /graded_images...'.format(len(tmp)))


    gauss_imgs = np.array([])
    for im in tmp:
        tmp2 = np.array([])
        im_3 = cv2.pyrDown(im, borderType=cv2.BORDER_DEFAULT)
        im_2 = cv2.pyrDown(im_3, borderType=cv2.BORDER_DEFAULT)
        im_1 = cv2.pyrDown(im_2, borderType=cv2.BORDER_DEFAULT)
        im_0 = cv2.pyrDown(im_1, borderType=cv2.BORDER_DEFAULT)
        tmp2 = np.append(tmp2, [im_3, im_2, im_1, im_0])
        gauss_imgs = np.append(gauss_imgs, tmp2)

    gauss_imgs = gauss_imgs.reshape(5, 4)

    return gauss_imgs, image_labels

def slidingWindow(image, step, window):
    '''This returns a sliding window object which is a 4-tuple.

    This 4 tuple represents, y position (h), x position (w), and the image slice given the window size
    '''

    for h in range(0, image.shape[0], step):
        for w in range(0, image.shape[1], step):
            yield (h, w, image[h:h+window[1], w:w+window[0]])
            # cv2.imshow('img', image[h:h + window[1], w:w + window[0]])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


def testModels():
    '''
    This function will test each model against the /graded_images directory (A.K.A. the images we are using for our final results)

    :return: None
    '''
    # This returns a shape (5,4) np array representing 5 different images, with 4 pyramid levels each.
    gauss_imgs, image_labels = gauss_pyr()
    print('IMAGE LABELS')
    print(image_labels)
    cwd = os.getcwd()
    for filename in os.listdir(cwd + '/models'):
        if filename == 'configs' or filename == 'experimental_models':
            continue
        else:
            print('USING MODEL {}'.format(filename))
            model = load_model('models/' + filename)
            i = 0
            for im_list in gauss_imgs:
                preds = np.array([])
                for img in im_list:
                    for (x, y, window) in slidingWindow(img, step=32, window=(32, 32)):
                        w = window
                        if w.shape[0] != 32 or w.shape[1] != 32:
                            break
                        w = np.array(w).reshape(1, 32, 32, 3)
                        prediction = model.predict(w, batch_size=32)
                        preds = np.append(preds, np.argmax(prediction))
                counter = Counter(preds.flatten())
                nums = counter.most_common(3)
                print('For Image: {} These are the numbers we found {}'.format(image_labels[i], [int(tup[0]) for tup in nums]))
                i += 1
                print('\n')
    print('DONE!')

if __name__ == '__main__':
    viewModels = None
    while viewModels not in ['yes', 'no']:
        viewModels = input('Would you like to see the models and their configurations/summaries? [yes, no]/'
                           ' \nWarning This will take a few minutes per Model View (since it will score the images) \n:')
    if viewModels == 'yes':
        print('-' * 100)
        print('VIEWING MODELS NOW')
        print('-' * 100)
        load_models.main()
        print('-' * 100)
        print('TESTING MODELS NOW')
        print('-' * 100)
        testModels()
    else:
        print('-' * 100)
        print('TESTING MODELS NOW')
        print('-' * 100)
        testModels()
