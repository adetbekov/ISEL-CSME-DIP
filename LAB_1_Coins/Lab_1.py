# -*- coding: utf-8 -*-
"""
Lab_1.py: Coin Detection and Counting.
ISEL, Computer Science and Multimedia Engineering, Digital Image processing
November 2017, Lisbon, Portugal
"""
__author__ = "Yeldos Adetbekov"
__email__ = "dosya@inbox.ru"

import numpy as np
import cv2, os

path = "assets/"
files = []
for f in os.listdir(path):
    files.append(cv2.imread(path + f))
imgs = np.array(files)

areas = [7000, 10000, 11000, 13000, 15000, 17000, 18000, 20000]
denominations = [0.01, 0.02, 0.10, 0.05, 0.20, 1, 0.50, 2]

# Get string foramted amount of money
def get_nominal(n):
    n = float(n)
    if n >= 1:
        if n.is_integer():
            return "{} euro".format(int(n))
        else:
            return "{} euros and {} cents".format(int(n), "{:0.2f}".format(np.round(n-int(n), decimals=2))[2:])
    else:
        return "{} cents".format("{:0.2f}".format(np.round(n-int(n), decimals=2))[2:])
    
# Disk structuring element
def disk(r):
    y, x = np.ogrid[-r : r+1, -r : r+1]
    return 1 * ( x ** 2 + y ** 2 <= r ** 2 )


for img in imgs:
    total = 0
    b, g, r = cv2.split(img) # Split to 3 color channels 
    retval, threshold = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    floodfill = threshold.copy()
    h, w = threshold.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8) # Size needs to be 2 pixels than the image.
    cv2.floodFill(floodfill, mask, (0, 0), 255)
    floodfill_inverted = cv2.bitwise_not(floodfill) 
    cleaned_holes = cv2.morphologyEx(floodfill, cv2.MORPH_CLOSE, disk(3)) # Cleaning noises
    out = cv2.bitwise_not(( threshold | floodfill_inverted ) + cleaned_holes)

    erosion = cv2.erode(out, disk(8), iterations = 2)
    dilation = cv2.dilate(erosion, disk(4), iterations = 2)

    im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Cleaning related objects with hole
    cleaned_contours = []
    for (i, c) in enumerate(contours):
        if(hierarchy[0][i][2] == -1 and hierarchy[0][i][3] == -1):
            cleaned_contours.append(c)
    
    for c in cleaned_contours:
        m = cv2.moments(c)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        perimeter = cv2.arcLength(c, True)
        circularity = (np.abs(perimeter) ** 2) / (m['m00'] or 1)
        m = cv2.moments(c)
        s = cv2.contourArea(c)
        top = (int(x) - int(radius), int(y) - int(radius) - 7)

        if(circularity >= 13.5 and circularity <= 18):
            for i, a in enumerate(areas):
                nextElement = areas[i+1] if areas[-1] != a else 22000
                thisElement = a
                currentDenominations = denominations[i]

                if s >= thisElement and s <= nextElement:
                    total += currentDenominations
                    cv2.circle(img, center, radius, (0, 255, 0), 3)
                    cv2.putText(img, get_nominal(currentDenominations), top, cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(img, "Total: " + get_nominal(total), (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow("Coin detection", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

