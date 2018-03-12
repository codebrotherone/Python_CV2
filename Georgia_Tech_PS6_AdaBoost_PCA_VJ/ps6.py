"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os
import scipy.linalg
from helper_classes import WeakClassifier, VJ_Classifier
import numpy as np
import re
import itertools

# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   ([int]): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    X = []
    Y = []
    for img in images_files:
        f_img = cv2.imread(folder + '/' + img)
        f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
        f_img = cv2.resize(f_img, tuple(size))
        X.append(f_img.flatten().astype(float))
        num_list = re.findall(r'\d+', img)
        num = int(num_list[0])
        Y.append(num)

    X = np.array(X).astype(np.float64)
    Y = np.array(Y)


    return (X, Y)




def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    M = X.shape[0]
    rand_rows = np.random.permutation(X)
    n = int(p*M)
    Xtrain = rand_rows[:n]
    Xtest = rand_rows[n:]
    ytrain = []
    ytest = []
    for img in Xtrain:
        idx = np.where(np.all(X == img, axis=1))
        ytrain.append(y[idx[0][0]])

    for img in Xtest:
        idx = np.where(np.all(X == img, axis=1))
        ytest.append(y[idx[0][0]])
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)

    return (Xtrain, ytrain, Xtest, ytest)


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    arr = x.copy()
    mean_x = np.mean(arr, axis=0, dtype=float)

    return mean_x



def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    m = get_mean_face(X)
    X = X - m.astype(np.float64)

    X = np.dot(X.transpose(), X)
    N = X.shape[0]
    eigval, eigvec = scipy.linalg.eigh(X, eigvals=(N-k, N-1))

    return (np.array(np.array([arr[::-1] for arr in eigvec])), np.array(list(reversed(eigval))))



class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = np.array([])
        self.alphas = np.array([])
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""

        # self.weights = np.array([(1. / self.num_obs) for wt in self.weights])

        for i in range(0, self.num_iterations):
            h = WeakClassifier(X=self.Xtrain, y=self.ytrain, weights=self.weights)
            h.train()
            h_j = h.predict(np.transpose(self.Xtrain))
            errors = np.array([self.ytrain[i] != h_j[i] for i in range(0, self.num_obs)])
            err_sum = np.sum(self.weights[errors])/np.sum(self.weights)
            alpha = .5*np.log((1.-err_sum)/err_sum)
            self.weakClassifiers = np.append(self.weakClassifiers, h)
            self.alphas = np.append(self.alphas, alpha)
            if err_sum > self.eps:
                self.weights[errors] = self.weights[errors] * np.exp(-alpha * h_j[errors] * self.ytrain[errors])
            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        N = self.num_obs
        labels = self.ytrain
        train_set = self.predict(self.Xtrain)
        corr = np.array([train_set[i] == labels[i] for i in range(0, N)])
        corr = np.array([int(x) for x in corr])
        num_corr = corr.sum()
        num_wrong = N-num_corr

        return (num_corr, num_wrong)

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.alphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """

        predictions = []
        for j in range(0, len(self.weakClassifiers)):
            pred_hx = [self.weakClassifiers[j].predict(np.transpose(X))]
            predictions.append(pred_hx)

        for i in range(0, len(self.alphas)):
            predictions[i] = np.array(predictions[i]) * self.alphas[i]
        predictions = np.sum(predictions, axis=0)
        predictions = predictions[0]
        return np.sign(predictions)




class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size

        features = np.zeros(shape, dtype=np.uint8)
        features[y:y+h//2, x:x+w] = 255
        features[y+h//2:y+h, x:x+w] = 126

        return features

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        features = np.zeros(shape, dtype=np.uint8)
        features[y:y+h, x:x+w//2] = 255
        features[y:y+h, x+w//2: x+w] = 126

        return features

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        print h//3

        features=np.zeros(shape, dtype=np.uint8)
        features[y:y+h//3, x:x+w] = 255
        features[y+h//3:y+2*h//3, x:x+w] = 126
        features[y+2*h//3:y+h, x:x+w] = 255


        return features


    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        features=np.zeros(shape, dtype=np.uint8)
        features[y:y+h, x:x+w//3] = 255
        features[y:y+h, x+w//3:x+2*w//3] = 126
        features[y:y+h, x+2*w//3:x+w] = 255
        return features


    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size

        features = np.zeros(shape, dtype=np.uint8)
        features[y: y+h//2, x: x+w//2]=126
        features[y: y+h//2, x+w//2:x+w] = 255
        features[y + h // 2: y + h, x:x + w//2] = 255
        features[y + h // 2: y + h, x + w // 2:x + w] = 126

        return features

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        # print 'THIS IS THE POSITION'
        # print self.position
        y, x = self.position
        h, w = self.size

        x -= 1
        y -= 1
        if x < 0:
            x = 0
        if y < 0:
            y = 0


        if self.feat_type == (1, 2):
            #for left vertical square
            p1, p2, p3, p4 = (x, y), (x+w//2, y), (x, y+h), (x+w//2, y+h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            white = int(d) - int(b) - int(c) + int(a)
            #for right vertical square
            p1, p2, p3, p4 = (x+w//2, y), (x+w, y), (x+w//2, y+h), (x+w, y+h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            gray = -(int(d) - int(b) - int(c) + int(a))
            score = white + gray
            return score
        elif self.feat_type == (2, 1):
            #for top square

            p1, p2, p3, p4 = (x, y), (x+w, y), (x, y+h//2), (x+w, y+h//2)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            white = int(d) - int(b) - int(c) + int(a)
            #for bottom square
            p1, p2, p3, p4 = (x, y+h//2), (x+w, y+h//2), (x, y+h), (x+w, y+h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            gray = -(int(d) - int(b) - int(c) + int(a))
            score = white + gray
            return score
        elif self.feat_type == (3,1):
            #top, white
            p1, p2, p3, p4 = (x, y), (x+w, y), (x, y+h//3), (x+w, y+h//3)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            top_white = int(d) - int(b) - int(c) + int(a)
            #middle, gray
            p1, p2, p3, p4 = (x, y+h//3), (x + w, y+h//3), (x, y + 2*h // 3), (x + w, y + 2*h // 3)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            middle_gr = -(int(d) - int(b) - int(c) + int(a))
            #bottom, white
            p1, p2, p3, p4 = (x, y+2*h//3), (x + w, y+2*h//3), (x, y + h), (x + w, y + h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            bottom_white = int(d) - int(b) - int(c) + int(a)
            score = top_white+middle_gr+bottom_white

            return score

        elif self.feat_type == (1,3):
            #left, white
            p1, p2, p3, p4 = (x, y), (x + w//3, y), (x, y + h), (x + w//3, y + h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            left_wh = int(d) - int(b) - int(c) + int(a)
            #middle, gray
            p1, p2, p3, p4 = (x+w//3, y), (x + 2*w//3, y), (x+w//3, y + h), (x + 2*w//3, y + h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            middle_gr = -(int(d) - int(b) - int(c) + int(a))
            #right, white
            p1, p2, p3, p4 = (x+2*w//3, y), (x + w, y), (x+2*w//3, y + h), (x + w, y + h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            right_wh = int(d) - int(b) - int(c) + int(a)
            score = left_wh + middle_gr + right_wh
            return score

        else:
            #top left, gray
            p1, p2, p3, p4 = (x, y), (x + w//2, y), (x, y + h // 2), (x + w//2, y + h // 2)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            tl_gr = -(int(d) - int(b) - int(c) + int(a))
            #top right, white
            p1, p2, p3, p4 = (x+w//2, y), (x + w, y), (x+w//2, y + h // 2), (x + w, y + h // 2)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            tr_wh = int(d) - int(b) - int(c) + int(a)
            #bottom left, white
            p1, p2, p3, p4 = (x, y+h//2), (x + w//2, y+h//2), (x, y + h), (x + w//2, y + h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            bl_wh = int(d) - int(b) - int(c) + int(a)
            #bottom right, gray
            p1, p2, p3, p4 = (x+w//2, y+h//2), (x + w, y+h//2), (x+w//2, y + h), (x + w, y + h)
            a, b, c, d = ii[p1[1], p1[0]], ii[p2[1], p2[0]], ii[p3[1], p3[0]], ii[p4[1], p4[0]]
            br_gr = -(int(d) - int(b) - int(c) + int(a))

            score = tl_gr+tr_wh+bl_wh+br_gr
            return score


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    img_array = []
    for img in images:
        img = np.cumsum(img, axis=0)
        img = np.cumsum(img, axis=1)
        img_array.append(img)

    return img_array

class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))
    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.iteritems():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print " -- compute all scores --"
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))

        weights = np.hstack((weights_pos, weights_neg))

        print " -- select classifiers --"
        excluded_feats = []
        for i in range(num_classifiers):
            # print 'Current excluded feat ids'
            # print excluded_feats
            weights = np.array([wt/np.sum(weights) for wt in weights])
            h = VJ_Classifier(scores, self.labels, weights=weights, excl_feat=excluded_feats)
            h.train()
            excluded_feats.append(h.feature)
            eps = h.error
            beta = eps/(1. - eps)
            alpha = np.log(1/beta)

            h_j = h.predict(scores.transpose())
            errors = np.array([h_j[i] != self.labels[i] for i in range(0, len(h_j))])
            # err = [int(x) for x in errors]
            self.classifiers.append(h)
            self.alphas.append(alpha)
            weights[errors] = weights[errors]*beta**(1 - (-1))


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)
        scores = np.zeros((len(ii), len(self.haarFeatures)))

        for i, im in enumerate(ii):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        for clf in self.classifiers:
            haar_feature_id = clf.feature
            for i in range(0, len(ii)):
                # print 'This is the haar feature id, after excluding the previous one'
                # print haar_feature_id
                new_score = self.haarFeatures[haar_feature_id].evaluate(ii[i])
                scores[i, haar_feature_id] = new_score

        result = []

        for x in scores:
            if np.sum(np.array([self.alphas[i] * self.classifiers[i].predict(x) for i in range(0, len(self.alphas))])) \
                              >= .5 * np.sum(np.array([self.alphas[i] for i in range(0, len(self.alphas))])):
                result.append(1)
            else:
                result.append(-1)
        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        raise NotImplementedError
