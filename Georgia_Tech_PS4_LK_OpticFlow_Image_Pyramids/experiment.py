"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "output"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """
    print u, v

    while level >= 1:
        u = ps4.expand_image(u) * 2
        v = ps4.expand_image(v) * 2
        level -= 1
        if u.shape != pyr[level].shape:
            u = u[:pyr[level].shape[0], :pyr[level].shape[1]]
        if v.shape != pyr[level].shape:
            v = v[:pyr[level].shape[0], :pyr[level].shape[1]]

    return u, v

def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    shift_0 = cv2.blur(src=shift_0, ksize=(15,15))
    shift_r2 = cv2.blur(src=shift_r2, ksize=(15,15))
    shift_r5_u5 = cv2.blur(src=shift_r5_u5, ksize=(15,15))


    k_size = 25  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    shift_0 = cv2.blur(shift_0, (5, 5), )
    shift_r10 = cv2.blur(shift_r10, (5, 5), )
    shift_r20 = cv2.blur(shift_r20, (5, 5), )
    shift_r40 = cv2.blur(shift_r40, (5, 5), )

    k_size = 67  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)


    k_size = 67  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)


    k_size = 67  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r40, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)

    cv2.imshow('combined', yos_img_01_g_pyr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output/ps4-2-a-1.png", yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)


    cv2.imshow('combined', yos_img_01_l_pyr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output/ps4-2-b-1.png", yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 5  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 5  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 5  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 5  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 4  # TODO: Define the number of levels
    k_size = 35  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, 65, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 4  # TODO: Define the number of levels
    k_size = 65  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    #I1(x0) = I0(x0 - tu0)
    I_0 = cv2.imread('input_images/TestSeq/Shift0.png', 0) / 1.
    I_1 = cv2.imread('input_images/TestSeq/ShiftR10.png', 0) / 1.


    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    im_array = []
    t_values = np.arange(0, 1.2, .2)

    U, V = ps4.optic_flow_lk(I_0, I_1, k_size=k_size, k_type=k_type, sigma=sigma)

    for val in t_values:
        print val
        scaled_U = U*val
        scaled_V = V*val
        warped = ps4.warp(I_0, U=-scaled_U, V=-scaled_V, interpolation=interpolation, border_mode=border_mode)
        cv2.imwrite(str(val)+'.png', warped)
        im_array.append(warped)

    r1 = np.concatenate((im_array[0], im_array[1], im_array[2]), axis=1)
    r2 = np.concatenate((im_array[3], im_array[4], im_array[5]), axis=1)

    complete = np.concatenate((r1, r2), axis=0)

    cv2.imwrite('output/ps4-5-1-a-1.png', complete.astype(np.int16))
    print 'FINISHED PART_5A'


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    I_0 = cv2.imread(
        'input_images/MiniCooper/mc01.png', 0) / 1.
    I_1 = cv2.imread(
        'input_images/MiniCooper/mc02.png', 0) / 1.
    # I_0 = cv2.blur(I_0, (15, 15))
    # I_1 = cv2.blur(I_1, (15, 15))
    k_size = 33  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    im_array = []
    t_values = np.arange(0, 1.2, .2)
    # for i in range(15, 85, 2):
    #
    #     U, V = ps4.hierarchical_lk(I_0, I_1, levels=4, k_size=i, k_type=k_type, sigma=sigma,
    #                                interpolation=interpolation, border_mode=border_mode)
    #
    #
    #     print i
    #     u_v = quiver(U, V, scale=3, stride=10)
    #     cv2.imwrite(str(i)+'.png', u_v)
    #
    # return None

    I_temp = I_0
    for val in t_values:
        U, V = ps4.hierarchical_lk(I_temp, I_1, levels=4, k_size=k_size, k_type=k_type, sigma=sigma,
                                   interpolation=interpolation, border_mode=border_mode)
        # Unew, Vnew = cv2.blur(U, (3,3)), cv2.blur(V, (3,3))

        dst = ps4.warp(I_0, U=-U*val, V=-V*val, interpolation=interpolation, border_mode=border_mode)
        I_temp = dst
        # warped = warped.astype('uint8')
        # dst = cv2.fastNlMeansDenoising(warped, None, searchWindowSize=21, templateWindowSize=7, h=10)
        # warped = cv2.fastNlMeansDenoising(warped,10,10,10)
        # warped = cv2.blur(warped, (3,3))
        im_array.append(dst)
        cv2.imwrite(str(val) + '.png', dst)

    im_array[-1] = I_1

    r1 = np.concatenate((im_array[0], im_array[1], im_array[2]), axis=1)
    r2 = np.concatenate((im_array[3], im_array[4], im_array[5]), axis=1)

    complete = np.concatenate((r1, r2), axis=0)


    cv2.imwrite('output/ps4-5-1-b-1.png', complete)

    ##
    ##starting on mc2 --> mc3
    ##
    I_0 = cv2.imread(
        'input_images/MiniCooper/mc02.png', 0) / 1.
    I_1 = cv2.imread(
        'input_images/MiniCooper/mc03.png', 0) / 1.
    # I_0 = cv2.blur(I_0, (15, 15))
    # I_1 = cv2.blur(I_1, (15, 15))
    k_size = 33  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    im_array = []
    t_values = np.arange(0, 1.2, .2)

    I_temp = I_0
    for val in t_values:
        U, V = ps4.hierarchical_lk(I_temp, I_1, levels=4, k_size=k_size, k_type=k_type, sigma=sigma,
                                   interpolation=interpolation, border_mode=border_mode)
        # Unew, Vnew = cv2.blur(U, (3,3)), cv2.blur(V, (3,3))

        dst = ps4.warp(I_0, U=-U * val, V=-V * val, interpolation=interpolation, border_mode=border_mode)
        I_temp = dst
        # warped = warped.astype('uint8')
        # dst = cv2.fastNlMeansDenoising(warped, None, searchWindowSize=21, templateWindowSize=7, h=10)
        # warped = cv2.fastNlMeansDenoising(warped,10,10,10)
        # warped = cv2.blur(warped, (3,3))
        im_array.append(dst)
        cv2.imwrite(str(val) + '.png', dst)

    im_array[-1] = I_1

    r1 = np.concatenate((im_array[0], im_array[1], im_array[2]), axis=1)
    r2 = np.concatenate((im_array[3], im_array[4], im_array[5]), axis=1)

    complete = np.concatenate((r1, r2), axis=0)

    cv2.imwrite('output/ps4-5-1-b-2.png', complete.astype(np.int16))
    # #starting on mc02 --> mc03
    # I_0 = cv2.imread(
    #     'input_images/MiniCooper/mc02.png', 0) / 1.
    # I_1 = cv2.imread(
    #     'input_images/MiniCooper/mc03.png', 0) / 1.
    # I_0 = cv2.blur(I_0, (5, 5))
    # I_1 = cv2.blur(I_1, (5, 5))
    #
    # k_size = 5  # TODO: Select a kernel size
    # k_type = ""  # TODO: Select a kernel type
    # sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    #
    # interpolation = cv2.INTER_CUBIC  # You may try different values
    # border_mode = cv2.BORDER_REFLECT101  # You may try different values
    #
    # im_array = []
    # t_values = np.arange(0, 1.2, .2)
    # # U, V = ps4.hierarchical_lk(I_0, I_1, levels=3, k_size=k_size, k_type=k_type, sigma=sigma, interpolation=interpolation, border_mode=border_mode)
    # U, V = ps4.optic_flow_lk(I_0, I_1, k_size=k_size, k_type=k_type, sigma=sigma)
    # for val in t_values:
    #     scaled_U = U * val
    #     scaled_V = V * val
    #     warped = ps4.warp(I_0, U=-scaled_U, V=-scaled_V, interpolation=interpolation, border_mode=border_mode)
    #     warped = cv2.blur(warped, (5,5))
    #     cv2.imwrite(str(val) + '.png', warped)
    #
    #     im_array.append(warped)
    #
    # r1 = np.concatenate((im_array[0], im_array[1], im_array[2]), axis=1)
    # r2 = np.concatenate((im_array[3], im_array[4], im_array[5]), axis=1)
    #
    # complete = np.concatenate((r1, r2), axis=0)
    #
    # print complete
    #
    # cv2.imwrite('output/ps4-5-1-b-2.png', complete.astype(np.int16))
    # print 'FINISHED PART_5B'



def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    raise NotImplementedError


if __name__ == "__main__":
    # part_1a()
    # part_1b()
    # part_2()
    # part_3a_1()
    # part_3a_2()
    # part_4a()
    # part_4b()
    part_5a()
    # part_5b()