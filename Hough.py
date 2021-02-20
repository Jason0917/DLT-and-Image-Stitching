import cv2
import numpy as np
from scipy.optimize import leastsq

img = cv2.imread('vanishing_pt/carla.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 80, 50)
# fine tune parameters
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
k = []
b = []
for line in lines:
    rho, theta = line[0]
    # skip near-vertical lines
    if abs(theta - np.pi / 90) < np.pi / 9:
        continue
    cos = np.cos(theta)
    sin = np.sin(theta)
    x0 = cos * rho
    y0 = sin * rho
    x1 = int(x0 + 10000 * (-sin))
    y1 = int(y0 + 10000 * (cos))
    x2 = int(x0 - 10000 * (-sin))
    y2 = int(y0 - 10000 * (cos))
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    k.append((y2 - y1) / (x2 - x1))
    b.append(y1 - (y2 - y1) / (x2 - x1) * x1)
    # print((x1, y1), (x2, y2))


# print(k)
# print(b)


def func(x, y):
    d = 0
    for i in range(len(k)):
        d += abs(k[i] * x - y + b[i]) / ((-1) * (-1) + k[i] * k[i]) ** 0.5
    return d


min_distance = 9999999
vanishing_x = 0
vanishing_y = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        d = func(i, j)
        if d < min_distance:
            min_distance = d
            vanishing_x = i
            vanishing_y = j

print(min_distance)
print(vanishing_x, vanishing_y)
cv2.circle(img, (vanishing_x, vanishing_y), 5, (0, 0, 255), 1)

# vanishing_point = leastsq(func, (0, 0))
# print(vanishing_point)

cv2.imshow("Image with Lines", img);
cv2.waitKey()
