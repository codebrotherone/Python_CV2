"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os
import itertools


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    im = image_in.copy().astype(np.float32)

    new = cv2.normalize(im, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)


    return new


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    im = image.copy()

    grad_x = cv2.Sobel(im, dx=1, dy=0, ddepth=cv2.CV_64F, scale=.125, ksize=3)

    return grad_x


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    im = image.copy()

    grad_y = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=0, dy=1, scale=.125, ksize=3)

    return grad_y


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    im1 = img_a.copy()
    im2 = img_b.copy()

    im1_dx = gradient_x(im1)
    im1_dy = gradient_y(im1)

    I_t = im2 - im1

    pts = list(itertools.product(range(0, im1.shape[1]), range(0, im1.shape[0])))

    K = np.ones((k_size, k_size))/np.float64(k_size**2)

    a = cv2.filter2D(im1_dx * im1_dx, ddepth=-1, kernel=K)
    b = cv2.filter2D(im1_dx * im1_dy, ddepth=-1, kernel=K)
    c = cv2.filter2D(im1_dx * im1_dy, ddepth=-1, kernel=K)
    d = cv2.filter2D(im1_dy * im1_dy, ddepth=-1, kernel=K)

    e = cv2.filter2D(im1_dx * I_t, ddepth=-1, kernel=K)
    f = cv2.filter2D(im1_dy * I_t, ddepth=-1, kernel=K)

    uarr = np.empty(im1.shape)
    varr = np.empty(im1.shape)

    for pt in pts:
        x, y = pt[0], pt[1]
        det = (a[y][x]*d[y][x] - b[y][x]*c[y][x])
        if det == 0:
            det = 1/.001

        invA = 1./det * np.array([d[y][x], -b[y][x], -c[y][x], a[y][x]]).reshape(2, 2)
        matB = np.array([-e[y][x], -f[y][x]])
        u_v = np.dot(invA, matB)
        u = u_v[0]
        v = u_v[1]

        uarr[y][x] = u
        varr[y][x] = v

    return (uarr, varr)






def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    im = image.copy()

    blurred = cv2.GaussianBlur(im, ksize=(5, 5), sigmaX=0, sigmaY=0)

    #return only odd columns, if the image shape is odd then it will round up.
    blurred = blurred[::2, ::2]


    return blurred


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    im=image.copy()
    reduced_list = []

    reduced_list.append(im)
    reduced = im


    while len(reduced_list) != levels:
        reduced_im = reduce_image(reduced)
        reduced = reduced_im
        reduced_list.append(reduced_im)


    return reduced_list



def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    #number of operations
    num_images = len(img_list)
    i=0
    output_width = 0
    while i < num_images:
        im = img_list[i]
        output_width += im.shape[1]
        i+=1

    #create image frame with all 0s
    output_im = np.zeros((img_list[0].shape[0], output_width))
    output_im[output_im == 0.] = 255.


    x_pos = 0
    for img in img_list:
        #insert image values
        output_im[:img.shape[0], x_pos:x_pos +img.shape[1]] = normalize_and_scale(img, (0,255))
        x_pos += img.shape[1]

    # cv2.imshow('combined', normalize_and_scale(output_im, (0, 255)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return output_im



def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    im = image.copy()

    # print 'im shape before expand is {}'.format(im.shape)

    #insert 0 rows for every even row in im
    im = np.insert(arr=im, obj=np.array(range(1, im.shape[0]+1, 1)), values=0, axis=0)
    # insert 0 rows for every even column in im
    im = np.insert(arr=im, obj=np.array(range(1, im.shape[1]+1, 1)), values=0, axis=1)
    #convolve image with 5x5 then multiply by 4
    blurred = cv2.GaussianBlur(im, ksize=(5, 5), sigmaX=0, sigmaY=0)
    blurred = blurred*4.

    # print 'expanded image shape is {}'.format(blurred.shape)

    return blurred



def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    gauss_pyramid = g_pyr
    lp = []
    i = 0
    # print 'len of gauss pyramid is {}'.format(len(gauss_pyramid))
    print [im.shape for im in gauss_pyramid]

    while i + 1 < len(gauss_pyramid):
        print 'the shape of gauss pyramid image is {}'.format(gauss_pyramid[i].shape)
        print '\n'
        print 'first arg is '
        print gauss_pyramid[i].shape
        print 'second arg is '
        print expand_image(gauss_pyramid[i+1]).shape
        try:

            lap_im = gauss_pyramid[i] - expand_image(gauss_pyramid[i+1])
            lp.append(lap_im)
            i+=1
        except ValueError:
            print 'we need to reshape the guass pyramid elements .. '
            arg = gauss_pyramid[i]
            arg2 = expand_image(gauss_pyramid[i+1])

            if arg.shape[0] < arg2.shape[0]:
                arg2 = arg2[:arg.shape[0], :arg.shape[1]]
            else:
                arg = arg[:arg2.shape[0], :arg2.shape[1]]

            lap_im = arg - arg2
            lp.append(lap_im)
            i+=1

    lp.append(gauss_pyramid[-1])

    return lp


    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    im = image.copy()
    #displacements in x-axis and y-axis as U and V for image as float32
    U = U.astype(np.float32)
    V = V.astype(np.float32)

    #meshgrid for image as float32
    im_x, im_y = im.shape[1], im.shape[0]
    x, y = np.meshgrid(range(im_x), range(im_y))
    x = x.astype(np.float32)
    y = y.astype(np.float32)


    x_disp = x + U
    y_disp = y + V

    im_out = cv2.remap(src=im, map1=x_disp, map2=y_disp, interpolation=interpolation, borderMode=border_mode)


    return im_out




def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    # print levels
    im = img_a.copy()
    im2 = img_b.copy()

    gy_A = gaussian_pyramid(im, levels=levels)
    gy_B = gaussian_pyramid(im2, levels=levels)

    c = levels-1
    U, V = optic_flow_lk(gy_A[-1], gy_B[-1], k_size=k_size, k_type=k_type, sigma=sigma)

    while c > 0:
        #expanded flow
        U = expand_image(U) * 2.
        V = expand_image(V) * 2.

        predicted_T1 = warp(gy_A[c-1], U=-U, V=-V, interpolation=interpolation, border_mode=border_mode)
        U_prime, V_prime = optic_flow_lk(predicted_T1, gy_B[c-1], k_size=k_size, k_type=k_type, sigma=sigma)

        U = U + U_prime
        V = V + V_prime

        c -= 1

    return U, V

