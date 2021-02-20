import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--srcdir", help="path to the images directory")
args = ap.parse_args()

img = cv2.imread(args.srcdir)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 80, 50)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Draw the lines
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


# Calculate the distance from point (x, y) to lines
def func(x, y):
    d = 0
    for i in range(len(k)):
        d += abs(k[i] * x - y + b[i]) / ((-1) * (-1) + k[i] * k[i]) ** 0.5
    return d


# Find the vanishing point with minimum distance to lines
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


# Draw the vanishing point
cv2.circle(img, (vanishing_x, vanishing_y), 5, (0, 0, 255), 2)
cv2.imshow("Image with lines and vanishing point", img);
cv2.waitKey()
