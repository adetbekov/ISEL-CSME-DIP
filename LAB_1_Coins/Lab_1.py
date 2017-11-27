import numpy as np
import cv2, os

path = 'lab1assets/'
files = []
for f in os.listdir(path):
    if os.path.splitext(f)[1].lower() in ('.jpg'):
        files.append(cv2.imread(path + f))
imgs = np.array(files)

areas = [8100, 11000, 12000, 14000, 16000, 18000, 19000]
denominations = [0.01, 0.02, 0.10, 0.05, 0.20, 1, 0.50]

def get_nominal(n):
    n = float(n)
    if n >= 1:
        if n.is_integer():
            return "{} euro".format(int(n))
        else:
            return "{} euros and {} cents".format(int(n), "{:0.2f}".format(np.round(n-int(n), decimals=2))[2:])
    else:
        return "{} cents".format("{:0.2f}".format(np.round(n-int(n), decimals=2))[2:])

for img in imgs:
    total = 0
    b,g,r = cv2.split(img)
    retval, threshold = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((10,10),np.uint8)
    erosion = cv2.erode(threshold, kernel, iterations = 1)
    im2, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        m = cv2.moments(c)
        (x,y),radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        perimeter = cv2.arcLength(c, True)
        length = (np.abs(perimeter)**2) / m['m00']
        m = cv2.moments(c)
        s = cv2.contourArea(c)
        top = (int(x) - int(radius), int(y) - int(radius) - 7)

        if(m['m00'] <= 2*np.pi*np.square(61) and m['m00'] >= 3322 and length >= 13.5 and length <= 18):
            for i, a in enumerate(areas):
                nextElement = areas[i+1] if areas[-1] != a else 21000
                thisElement = a
                currentDenominations = denominations[i]

                if s >= thisElement and s <= nextElement:
                    if s >= 19300 and s <= 19500:
                        break
                    else:
                        total += currentDenominations
                    cv2.circle(img, center, radius, (0, 255, 0), 3)
                    cv2.putText(img, get_nominal(currentDenominations), top, cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(img, "Total: " + get_nominal(total), (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow("Coin detection", img)
    key = cv2.waitKey(0)
cv2.destroyAllWindows()

