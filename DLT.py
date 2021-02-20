import numpy as np
import cv2


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def compute_target_points(src_points):
    # Order points and compute src points in the image
    rect = order_points(src_points)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    return dst, maxWidth, maxHeight


def dlt(src_points, norm):

    def Normalization(point):
        m = np.mean(point, 0)
        s = np.std(point)
        Tr = np.array([[s/np.sqrt(2), 0, m[0]], [0, s/np.sqrt(2), m[1]], [0, 0, 1]])
        Tr = np.linalg.inv(Tr)
        trans_point = np.dot(Tr, np.concatenate((point.T, np.ones((1, point.shape[0])))))
        trans_point = trans_point[0:2].T
        return Tr, trans_point

    def denormalization(T1, T2, H_norm):
        H_denorm = np.dot(np.dot(np.linalg.inv(T2), H_norm), T1)
        return H_denorm


    target_points, maxWidth, maxHeight = compute_target_points(src_points)

    if norm:
        T1, src_points = Normalization(src_points)
        T2, target_points = Normalization(target_points)

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

    if norm:
        H = denormalization(T1, T2, H)

    return H, maxWidth, maxHeight
