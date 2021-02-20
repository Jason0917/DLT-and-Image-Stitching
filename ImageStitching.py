import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.color import rgb2gray
import random
from math import sqrt, pow
from skimage.transform import ProjectiveTransform


def dlt(src_points, target_points):

    '''Construct A Matrix'''
    A = []
    for i in range(0, len(src_points)):
        a1, a2 = src_points[i][0], src_points[i][1]
        b1, b2 = target_points[i][0], target_points[i][1]
        A.append([-a1, -a2, -1, 0, 0, 0, b1 * a1, b1 * a2, b1])
        A.append([0, 0, 0, -a1, -a2, -1, b2 * a1, b2 * a2, b2])
    A = np.asarray(A)

    '''Compute SVD for A and Get H Matrix'''
    U, S, V = np.linalg.svd(A)
    L = V[-1, :] / V[-1, -1]
    H = L.reshape(3, 3)

    return H


def ransac(src, dst, sample_size, error_t, ratio_t, max_iterations):
    H_robust = np.zeros([3, 3])
    random.seed()
    # random.sample cannot deal with "data" being a numpy array
    # src = list(src)
    for j in range(max_iterations):
        index = random.sample(range(0, len(src)), sample_size)
        # print(index)
        sample_pts = src[index]
        H = dlt(sample_pts, dst[index])
        inlier_index = []
        n_inlier = 0
        n_total = len(src)
        for i in range(len(src)):
            x = src[i][0]
            y = src[i][1]
            x_ = (H[0][0] * x + H[0][1] * y + H[0][2]) / (H[2][0] * x + H[2][1] * y + H[2][2])
            y_ = (H[1][0] * x + H[1][1] * y + H[1][2]) / (H[2][0] * x + H[2][1] * y + H[2][2])
            error = sqrt(pow(dst[i][0] - x_, 2) + pow(dst[i][1] - y_, 2))
            if error < error_t:
                n_inlier += 1
                inlier_index.append(i)
        # print(float(n_inlier) / float(n_total))
        if float(n_inlier) / float(n_total) > ratio_t:
            H_robust = dlt(src[inlier_index], dst[inlier_index])
            break

    return H_robust, inlier_index


def findSmallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index


def lmeds(src, dst, sample_size, max_iterations):
    H_robust = np.zeros([3, 3])
    random.seed()
    # random.sample cannot deal with "data" being a numpy array
    # src = list(src)
    H_list = []
    Med_error_list = []
    sample_index_list = []
    for j in range(max_iterations):
        index = random.sample(range(0, len(src)), sample_size)
        sample_index_list.append(index)
        # print(index)
        sample_pts = src[index]
        H = dlt(sample_pts, dst[index])
        inlier_index = []
        n_inlier = 0
        n_total = len(src)
        error_list = []
        for i in range(len(src)):
            x = src[i][0]
            y = src[i][1]
            x_ = (H[0][0] * x + H[0][1] * y + H[0][2]) / (H[2][0] * x + H[2][1] * y + H[2][2])
            y_ = (H[1][0] * x + H[1][1] * y + H[1][2]) / (H[2][0] * x + H[2][1] * y + H[2][2])
            error = sqrt(pow(dst[i][0] - x_, 2) + pow(dst[i][1] - y_, 2))
            error_list.append(error)

        H_temp = dlt(src[index], dst[index])
        H_list.append(H_temp)
        error_list = np.array(error_list)
        Med_error_list.append(np.median(error_list))

    target_index = findSmallest(Med_error_list)
    H_robust = H_list[target_index]
    inlier_index = sample_index_list[target_index]

    return H_robust, inlier_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--srcdir", help="path to the images directory")
    ap.add_argument("--lmeds", action="store_true")
    args = ap.parse_args()

    use_lmeds = False
    if args.lmeds:
        use_lmeds = True

    files = os.listdir(args.srcdir)
    I0 = io.imread(os.path.join(args.srcdir, files[0]))
    I1 = io.imread(os.path.join(args.srcdir, files[1]))
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(I0, cmap='gray')
    # plt.title("Image 0")
    # plt.subplot(122)
    # plt.imshow(I1, cmap='gray')
    # plt.title("Image 1")
    # plt.show()

    I0 = rgb2gray(I0)
    I1 = rgb2gray(I1)

    # Initialize ORB
    # 800 keypoints is large enough for robust results,
    # but low enough to run within a few seconds.
    orb = ORB(n_keypoints=800, fast_threshold=0.05)

    # Detect keypoints in pano0
    orb.detect_and_extract(I0)
    keypoints0 = orb.keypoints
    descriptors0 = orb.descriptors

    # Detect keypoints in pano1
    orb.detect_and_extract(I1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    matches01 = match_descriptors(descriptors0, descriptors1, cross_check=True)

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Best match subset for pano0 -> pano1
    plot_matches(ax, I0, I1, keypoints0, keypoints1, matches01)
    ax.axis('off');
    # plt.show()

    src = keypoints1[matches01[:, 1]][:, ::-1]
    dst = keypoints0[matches01[:, 0]][:, ::-1]

    if use_lmeds:
        H_robust, inliers_index = lmeds(src, dst, 4, 1000)
    else:
        H_robust, inliers_index = ransac(src, dst, 4, 10, 0.6, 1000)


    Homography = ProjectiveTransform(H_robust)

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    # Best match subset for pano0 -> pano1
    plot_matches(ax, I0, I1, keypoints0, keypoints1, matches01[inliers_index])
    ax.axis('off');
    plt.show()

    from skimage.transform import SimilarityTransform

    # Shape registration target
    r, c = I0.shape[:2]

    # Note that transformations take coordinates in (x, y) format,
    # not (row, column), in order to be consistent with most literature
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # todo compute image corners' new positions
    # Warp the image corners to their new positions
    warped_corners01 = Homography(corners)

    # Find the extents of both the reference image and the warped
    # target image
    all_corners = np.vstack((warped_corners01, corners))

    # The overally output shape will be max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)

    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape[::-1]).astype(int)

    from skimage.transform import warp

    # This in-plane offset is the only necessary transformation for the middle image
    offset1 = SimilarityTransform(translation=-corner_min)

    # print(Homography)
    # print(offset1)
    # Warp pano1 to pano0 using 3rd order interpolation
    transform01 = (Homography + offset1).inverse
    I1_warped = warp(I1, transform01, order=3,
                     output_shape=output_shape, cval=-1)

    I1_mask = (I1_warped != -1)  # Mask == 1 inside image
    I1_warped[~I1_mask] = 0  # Return background values to 0

    # Translate pano0 into place
    I0_warped = warp(I0, offset1.inverse, order=3,
                     output_shape=output_shape, cval=-1)

    I0_mask = (I0_warped != -1)  # Mask == 1 inside image
    I0_warped[~I0_mask] = 0  # Return background values to 0

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(I0_warped, cmap ='gray')
    plt.title("Warped Image 0")
    plt.subplot(122)
    plt.imshow(I1_warped, cmap ='gray')
    plt.title("Warped Image 1")
    plt.show()

    # Add the images together. This could create dtype overflows!
    # We know they are are floating point images after warping, so it's OK.
    merged = (I0_warped + I1_warped)

    # Track the overlap by adding the masks together
    overlap = (I0_mask * 1.0 +  # Multiply by 1.0 for bool -> float conversion
               I1_mask)

    # Normalize through division by `overlap` - but ensure the minimum is 1
    normalized = merged / np.maximum(overlap, 1)
    fig, ax = plt.subplots(figsize=(5, 10))

    ax.imshow(normalized, cmap ='gray')

    plt.tight_layout()
    ax.axis('off');
    plt.show()


if __name__ == "__main__":
    main()
