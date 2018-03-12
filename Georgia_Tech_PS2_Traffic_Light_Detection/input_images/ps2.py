"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np
import itertools
import pprint as pp

def determine_std(circles):
    traff_x = np.std([x[0] for x in circles])
    traff_y = np.std([x[1] for x in circles])
    traff_rad = np.std([x[2] for x in circles])

    return traff_x + traff_y + traff_rad

def tri_check_lines(pts):
    '''

    :param pts: array of strings representing three line segments.
    :return: bool
    '''
    print 'std dev of pts in yield test'

    x_std = np.std([pt[0] for pt in pts])
    y_std = np.std([pt[1] for pt in pts])

    print x_std + y_std
    return x_std + y_std

def determine_std_lines(lines):
    x1_std = np.std([ x[0] for x in lines ])
    x2_std = np.std([ x[2] for x in lines ])
    y1_std = np.std([ x[3] for x in lines ])
    y2_std = np.std([ x[3] for x in lines ])

    return x1_std + x2_std + y1_std + y2_std

def slope_line(line):
    print line
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    m = (y2-y1)/(x2-x1)
    return m


def check_tri_slopes(pts):
    a=pts[0]
    b=pts[1]
    c=pts[2]
    print 'triangle pts are {} {} {}'.format(a, b, c)
    print a, b, c
    a_slope = np.int64((a[3]-a[1])/(a[2]-a[0]))
    b_slope = np.int64((b[3]-b[1])/(b[2]-b[0]))
    c_slope = np.int64((c[3]-c[1])/(c[2]-c[0]))

    slopes = [a_slope, b_slope, c_slope]
    print 'slopes'
    print slopes
    horizontal_ct = sum([1 for slope in slopes if slope == 0])
    neg_ct = sum([1 for slope in slopes if slope < 0])
    pos_ct = sum([1 for slope in slopes if slope >0])
    if horizontal_ct == 1 and len(set(slopes)) == 3 and neg_ct == 1 and pos_ct==1:
        print 'FOUND OUT TRIANGLE'
        return True
    else:
        'FALSE FALSE FALSE'
        return False

def is_intersection(pt1, pt2):

    print 'TRI'
    print pt1, pt2
    threshold = 5

    if abs(pt1[0] - pt2[0]) <= threshold and abs(pt1[1] - pt2[1]) <= threshold:
        return ((pt1[0]+pt2[0])/2, (pt1[1] + pt1[1])/2) #return the average of two points which are in fact the same
    return ()

def intersecting_pt(line1, line2):

    print 'TRU'
    print line1, line2
    a = is_intersection((line1[0], line1[1]), (line2[0], line2[1]))
    b = is_intersection((line1[0], line1[1]), (line2[2], line2[3]))
    c = is_intersection((line1[2], line1[3]), (line2[2], line2[3]))
    d = is_intersection((line1[2], line1[3]), (line2[0], line2[1]))

    print line1, line2
    output = [a, b, c, d]

    for is_point in output:
        if is_point:
            return is_point

    return None

def find_center(pts):

    x_all = [pt[0] for pt in pts]
    y_all = [pt[1] for pt in pts]

    center = (sum(x_all)/3, sum(y_all)/3)
    return center

def find_dist(line):
    '''
    Find the distance between two points

    param line: list of ints

    return: int
    '''
    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    print ('dist for {} is {}'.format(line, dist))
    return dist

def cmp_dist_lines(line1, line2):
    '''
    Takes two lines and compares them to see if they share the same distance to a certain tolerance level
    param line1: list of ints
    param line2: list of ints

    return: bool
    '''
    #find the distance
    d1 = find_dist(line1)
    d2 = find_dist(line2)

    return abs(d1 - d2)

def cmp_slope_lines(line1, line2):
    '''
    Takes two lines and compares them to see if they share the same slope within a threshold value
    param line1: list of ints
    param line2: list of ints

    return: bool
    '''

    m1 = slope_line(line1[0])
    m2 = slope_line(line2[0])

    print 'HERE ARE YOUR SLOPES'
    print m1, m2

    return abs(m1 - m2)

def find_mirrored_lines(lines):
    '''
    Takes group of lines in [x1, y1, x2, y2] format and checks every pair to see if both x, or y, and slopes
    match up.
    This will determine if the points mirror each other on a polygon's center axis.

    param lines: list of lists (ints)

    return: tuple of lists (ints)
    '''

    thresh = 5
    combos = tuple(itertools.combinations(lines, 2))

    for pair in combos:
        if cmp_dist_lines(pair[0], pair[1], thresh):
            print 'FOUND LINES WITH SAME DIST'
            if cmp_slope_lines(pair[0], pair[1]):
                print 'Found two lines with the same distance and the same slope'
                return pair[0], pair[1]

def find_center_parallel_lines(l1, l2):
    '''
    Takes tuple of two parallel lines and finds the cross midpoint between two lines

    param lines: list of lists representing line segments in [x1, y1, x2, y2]
    return: midpoint between the two lines
    '''


    print l1, l2

    x_sum = l1[0] + l2[2]

    y_sum = l1[1] + l2[3]

    center = (np.around(x_sum/2), np.around(y_sum/2))
    return center

def check_circles(circles, imgColor):

    for circle in circles[0]:
        center = (int(circle[0]), int(circle[1]))
        radius =  int(circle[2])

        #center colors
        b, g, r = imgColor[int(center[1]), int(center[0])][0], imgColor[int(center[1]), int(center[0])][1], imgColor[int(center[1]), int(center[0])][2]
        #outside center white rectangle color
        b2, g2, r2 = imgColor[int(center[1]) + int(radius/2), int(center[0] + int(radius/2))][0], imgColor[int(center[1]) + int(radius/2),int(center[0] + int(radius/2))][1],imgColor[int(center[1]) + int(radius / 2), int(center[0] + int(radius / 2))][2]

        if [b, g, r] == [0, 255, 255]:
            continue
        else:
            if [b, g, r] == [255, 255, 255] and [b2, g2, r2] == [0, 0, 255]:
                center = (int(circle[0]), int(circle[1]), int(circle[2]))
                break
    return center

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    radii = radii_range
    minDistance = min(radii) * 2
    imgColor = img_in.copy()
    #Convert to Gray-Scale
    img = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)

    #Pull Gaussian Blur Image
    blur = cv2.GaussianBlur(img, (9, 9), 0)

    #Hough circle transform
    circles = cv2.HoughCircles(image=blur,
                               method=cv2.cv.CV_HOUGH_GRADIENT,
                               dp=1,
                               minDist=minDistance,
                               param1=30,
                               param2=13,
                               minRadius=min(radii_range),
                               maxRadius=35)

    if circles is not None:
        print 'Found HOUGH CIRCLES'
    else:
        print circles
        print 'NO HOUGH CIRCLES FOUND'

    #round to int values for ease
    circles = np.uint16(np.around(circles))
    sorted_circles = sorted(circles[0, :], key=lambda x: (x[2], x[0], x[1]))

    #pull all combos by groups of 3
    poss_circles = []
    for cnt in range(len(sorted_circles) - 2):
        circ=sorted_circles[cnt: cnt + 3]
        poss_circles.append(circ)

    correct_circles = sorted(poss_circles, key=determine_std)
    correct_circles = correct_circles[0]

    #sort circles by y since x and r are the same, then pull the center for output
    circles_final = sorted(correct_circles, key=lambda x: x[1])
    center = (circles_final[1][0], circles_final[1][1]) #x, y notation for center point

    #find the maximum intensity value for all three circle centers in the original image (this will represent the state)
    max_int_value_and_index = [0, 0]
    for idx, coord in enumerate(circles_final):
        if max(imgColor[coord[1], coord[0]]) > max_int_value_and_index[0]:
            max_int_value_and_index = [max(imgColor[coord[1], coord[0]]), idx]

    #determine state based on index
    if max_int_value_and_index[1] == 0:
        state  = 'red'
    elif max_int_value_and_index[1] == 1:
        state = 'yellow'
    else:
        state = 'green'


    return ((center), state)

def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    imgc = img_in.copy()
    img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)

    #perform canny operation
    canny_edges = cv2.Canny(img, threshold1=90, threshold2=25)
    #params
    minLineLength = 20
    maxGap = 5
    #Hough lines Transform
    #for x in range(60, 80, 5):

    thetas = [30, 90, 330]
    for theta in thetas:
        lines = cv2.HoughLinesP(image=canny_edges,
                                rho=1,
                                theta=np.pi/180*theta,
                                threshold=10,
                                minLineLength=minLineLength,
                                maxLineGap=maxGap)

        if lines is None:
            continue

        else:
            break
        #
        # for line in lines:
        #     line = line[0]
        #     print line
        #     cv2.line(imgc, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 2)
        #
        # cv2.imshow('image with hough lines', imgc)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    #Pull all combinations of lines in groups of 3
    all_lines = itertools.combinations(lines, 3) #this object is a list of tuples which each contain 3 list elements


    center = ()

    for a, b, c, in list(all_lines):
        print a, b, c
        a=a[0]
        b=b[0]
        c=c[0]
        thres = 10
        check1=cmp_dist_lines(a, b, thres)
        check2=cmp_dist_lines(b, c, thres)
        check3=cmp_dist_lines(a, c, thres)

        imgc=img_in.copy()

        if all([check1, check2, check3]):

            print 'found similar distance lines'
            ab = intersecting_pt(a, b)
            bc = intersecting_pt(b, c)
            ac = intersecting_pt(a, c)

            print 'Here are our intersecting points {} {} {}'.format(ab, bc, ac)
            if not all([ab, bc, ac]):
                continue
            else:
                print 'found some points that intersect'
                is_triangle = check_tri_slopes([a, b, c])
                if is_triangle:
                    print 'THIS IS A TRI'
                    approx_tri = (ab, bc, ac)
                    center = find_center(approx_tri)

                    g, bl, r = imgc[center[1], center[0]]

                    if [g, bl, r] != [255, 255, 255]:
                        continue

                    print a, b, c

                    cv2.line(imgc, (a[0], a[1]), (a[2], a[3]), (0, 0, 0), 2)
                    cv2.line(imgc, (b[0], b[1]), (b[2], b[3]), (0, 0, 0), 2)
                    cv2.line(imgc, (c[0], c[1]), (c[2], c[3]), (0, 0, 0), 2)
                    cv2.circle(imgc, center, 5, (0,0,0), 5)

                    cv2.imshow('image with center', imgc)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    print'FOUND OUR CENTER'


        else:
            continue

    print 'center is {}'.format(center)
    return center

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """

    imgc=img_in.copy()
    img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)

    #perform canny operation
    canny_edges = cv2.Canny(img, threshold1=150, threshold2=50)
    #define parameters
    minLineLength = 25
    maxGap = 5
    theta = 45

    lines = cv2.HoughLinesP(image=canny_edges, rho=1, theta=np.pi/180*theta, threshold=30, minLineLength=minLineLength, maxLineGap=maxGap)

    combos = list(itertools.combinations(lines, 2))

    combos.sort(key=lambda x: cmp_dist_lines(x[0][0], x[1][0]))
    print 'after sorting by distance'
    print combos
    centroid = ()

    for pair in combos:
        # if cmp_dist_lines(pair[0], pair[1], threshold=5):
        centroid = find_center_parallel_lines(pair[0][0], pair[1][0])
        b, g, r = imgc[centroid[1], centroid[0]]
        print 'B, G, R'
        print b, g, r

        if (b==0 and g==0 and r>200) or (b==255 and g==255 and r==255):
            return centroid
        else:
            print 'FAIL FAIL FAIL'


    return centroid

def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    imgc=img_in.copy()
    img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)

    print 'image shape is {}'.format(img.shape)
    #perform canny operation
    canny_edges = cv2.Canny(img, threshold1=100, threshold2=80)

    minLineLength = 5
    maxGap = 5


    thetas = [45] #1, 45, 135, 180, 90, 27, 34

    for theta in thetas:
        lines = cv2.HoughLinesP(canny_edges, rho=2, theta=np.pi / 180 * theta, threshold=10,
                                minLineLength=minLineLength,
                                maxLineGap=maxGap)
        print theta
        imgc = img_in.copy()
        if lines is None:
            for num in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[num]:
                    print x1, y1, x2, y2
                    cv2.line(imgc, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            print 'no dice'
            continue

        print lines

    for line in lines:
        if slope_line(line) == 0:
            print 'found a flat line, remove for warning'
            continue
        else:
            print lines

    print 'HELLO'

    l1, l2 = find_mirrored_lines(lines)
    centroid = find_center_parallel_lines(l1, l2)
    #
    # cv2.imshow('imgc', imgc)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    print 'XY'
    print centroid
    return centroid



def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    imgc=img_in.copy()
    img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)

    print 'STARTING CONSTRUCTION'
    #perform canny operation
    canny_edges = cv2.Canny(img, threshold1=100, threshold2=80)
    print img.shape[1]

    minLineLength = 10
    maxGap = 5


    #thetas = [45] #1, 45, 135, 180, 90, 27, 34

    lines = cv2.HoughLinesP(canny_edges, rho=1, theta=np.pi / 180 * 45, threshold=15,
                            minLineLength=minLineLength,
                            maxLineGap=maxGap)

    if lines is not None:
        for num in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[num]:
                print x1, y1, x2, y2
                cv2.line(imgc, (x1, y1), (x2, y2), (255, 0, 0), 2)
    else:
        print 'no dice'

    print 'construction hough lines'
    print lines

    # cv2.imshow(str(45), imgc)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  # cv2.imshow('hough', img)

    l1, l2 = find_mirrored_lines(lines)
    centroid = find_center_parallel_lines(l1, l2)

    imgc=img_in.copy()
    cv2.line(imgc, (l1[0][0], l1[0][1]), (l1[0][2], l1[0][3]), (255, 0, 0), 2)
    cv2.line(imgc, (l2[0][0], l2[0][1]), (l2[0][2], l2[0][3]), (255, 0, 0), 2)
    #
    # print l1, l2
    # cv2.imshow('construction l1 l2', imgc)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  # cv2.imshow('hough', img)

    print centroid
    return centroid

def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """

    minDistance = 10
    imgColor = img_in.copy()
    img = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)

    blur = cv2.blur(img, (5, 5), 0)

    circles = cv2.HoughCircles(blur, cv2.cv.CV_HOUGH_GRADIENT, dp=1, minDist=minDistance, param1=40, param2=20,
                               minRadius=5, maxRadius=400)
    print circles

    for i in circles[0]:
        # draw the outer circle
        cv2.circle(imgColor, (i[0], i[1]), i[2], (0, 0, 0), 1)
        # draw the center of the circle
    #
    # cv2.imshow('img with circles and center', imgColor)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print 'here are circles'
    print circles

    center = check_circles(circles, imgColor)

    return (center[0], center[1])

def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    imgc=img_in.copy()
    tri_Center = yield_sign_detection(imgc)
    temp_dict={}
    if tri_Center == ():
        print 'No Triangles were found'
    else:
        temp_dict['yield'] = tri_Center

    stop_Center = stop_sign_detection(imgc)
    if stop_Center == ():
        print 'NO Stop sign found'
    else:
        print 'Found our stop sign center {}'.format(stop_Center)
        temp_dict['stop'] = stop_Center

    print ('our temp_dict is {}'.format(temp_dict))
    cv2.circle(imgc, temp_dict['stop'], 5, (0, 0, 0), 5)
    cv2.imshow('stop', imgc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    # raise NotImplementedError


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError
