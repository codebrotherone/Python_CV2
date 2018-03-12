"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import itertools

def find_mdpt(pt1, pt2):
    """
    Return the midpoint of two points, used for the markers

    :param pt1: (x, y) tuple
    :param pt2: (x, y) tuple
    :return: (x, y) tuple
    """

    return (pt1+pt2/2)


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    dist = np.sqrt((p1[0] - p0[0])**2 + (p1[1]-p0[1])**2)
    return dist


def get_corners_list(image):
    """Returns a list of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    im = image.copy()

    return [(0, 0), (0, im.shape[0]-1), (im.shape[1]-1, 0), (im.shape[1]-1, im.shape[0]-1)]



def projective_transform_matrices(p1, p2):
    '''
    Matrix Transformation Ax=b --> given A and B find X

    :param p1: list of tuples
    :param p2: list of tuples
    :return: two numpy arrays representing matrices
    '''

    matrixA=np.array((p1[0][0], p1[1][0], p1[2][0], p1[0][1], p1[1][1], p1[2][1], 1, 1, 1)).reshape((3,3))
    matrixb=np.array((p1[3][0], p1[3][1], 1))
    homogenous_coord = np.linalg.solve(matrixA, matrixb)
    scaled_matA= matrixA * homogenous_coord


    matrixB=np.array((p2[0][0], p2[1][0], p2[2][0], p2[0][1], p2[1][1], p2[2][1], 1, 1, 1)).reshape((3,3))
    matrixb2 = np.array((p2[3][0], p2[3][1], 1))
    homogenous_coord2 = np.linalg.solve(matrixB, matrixb2)
    scaled_matB = matrixB * homogenous_coord2

    return scaled_matA, scaled_matB

def find_markers(image, template):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    img=image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)

    dst = cv2.cornerHarris(smooth, 7, 5, 0.05)
    dst = cv2.dilate(dst, None)


    new = np.where(dst >= .1 * dst.max())

    # pts = zip(*new)

    pts = zip(*new[::-1])
    pts = np.array([np.float32(pts)])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, labels, centers = cv2.kmeans(pts, K=4, criteria=criteria, attempts=100, flags=cv2.KMEANS_RANDOM_CENTERS)


    centers = centers.astype(np.int16)

    centers = sorted(centers, key=lambda x: x[0])

    left=sorted(centers[:2], key=lambda x: x[1])
    right=sorted(centers[2:], key=lambda x: x[1])

    topleft, bottomleft = tuple(left[0]), tuple(left[1])
    topright, bottomright = tuple(right[0]), tuple(right[1])

    return [topleft, bottomleft, topright, bottomright]


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """

    im=image.copy()

    print markers

    cv2.line(im, markers[0], markers[1], (0, 0, 255), thickness=thickness)
    cv2.line(im, markers[1], markers[3], (0, 0, 255), thickness=thickness)
    cv2.line(im, markers[3], markers[2], (0, 0, 255), thickness=thickness)
    cv2.line(im, markers[0], markers[2], (0, 0, 255), thickness=thickness)
    return im


    # raise NotImplementedError


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    im1 = imageA.copy()
    im2 = imageB.copy()

    print im1.shape
    print im2.shape


    # #reverse warping
    # try:
    #     markers = find_markers(im2, None)
    #     print 'markers ol chap'
    #     print markers
    #     markers = [[pt[0].astype('int32'), pt[1].astype('int32')] for pt in markers]
    #     markers = [markers[0], markers[2], markers[3], markers[1]]
    # except:
    #     if markers == []:
    #         print 'what a fucking surprise, no markers found'

    markers = get_corners_list(im2)
    markers = [markers[0], markers[2], markers[3], markers[1]]
    markers = [[np.int32(marker[0]), np.int32(marker[1])] for marker in markers]

    grayim2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    cv2.fillPoly(grayim2, np.array([markers]), (255), 8, 0)
    grayim2[grayim2 != 255] = 0

    pts = np.where(grayim2 == (255))
    pts = zip(*pts[::-1])

    pts_array = np.array([(pt[0], pt[1], 1) for pt in pts])

    inv_homography = np.linalg.inv(homography)

    for pt in pts_array:
        src_pt = np.dot(inv_homography, pt)
        src_pt = src_pt/src_pt[2]
        x, y = np.round(src_pt[0]), np.round(src_pt[1])
        if x < 0:
            print x, y
            continue
        if y < 0:
            print x, y
            continue

        if x >=im1.shape[1]:
            print x, y
            continue
            # x=im1.shape[1] - 1
        if y >= im1.shape[0]:
            print x, y
            continue

            # y=im1.shape[0] - 1

        src_val = im1[np.int(np.round(y)), np.int(np.round(x))]
        im2[pt[1], pt[0]] = src_val



    # cv2.imshow('im', im2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return im2

    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    #p1 and p2 represent different planes (4 points on each)
    p1=src_points
    p2=dst_points

    print 'p1 and p2 are '
    print p1, p2

    matA, matB = projective_transform_matrices(p1, p2)

    return np.dot(np.float32(matB), np.linalg.inv(matA)) / np.dot(np.float32(matB), np.linalg.inv(matA))[2][2]



    # raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)


    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None

