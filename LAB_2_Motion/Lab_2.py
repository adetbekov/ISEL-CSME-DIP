# -*- coding: utf-8 -*-
"""
Lab_2.py: Motion Detection for Object Tracking.
ISEL, Computer Science and Multimedia Engineering, Digital Image processing
January 2017, Lisbon, Portugal
"""
__author__ = "Yeldos Adetbekov"
__email__ = "dosya@inbox.ru"

import cv2, math, argparse
import numpy as np
from datetime import datetime
from time import time
from matplotlib import pyplot as plt
from collections import deque

class Location:
    def __init__(self, i, x, y, c):
        self.id = i
        self.x = x
        self.y = y
        self.c = c
        self.time = time()
        self.pts = deque(maxlen=64)
        self.counter = 0
        (self.dX, self.dY) = (0, 0)
        self.direction = ""
        self.death = 2
        log("{} {} appeared with ({}, {}) coordinates".format(self.c, self.id, self.x, self.y))
        
    def update(self, x, y):
        self.x = x
        self.y = y
        self.time = time()
        
    def ttl(self):
        return time() - self.time
    
    def alive(self):
        return self.ttl() < 2
    
    def dead(self):
        self.death -= 1 if self.death >= 1 and not self.alive() else 0
        return self.death
    
    def add_pts(self, center):
        self.pts.appendleft(center)
    
    
class Pull:
    global result, frame
    def __init__(self, c):
        self.classification = c
        self.pull = []
    
    def add(self, x, y):
        new = Location(self.length(), x, y, self.classification)
        self.pull.append(new)
        return new
    
    def update(self, i, x, y):
        self.pull[i].update(x, y)
        return self.pull[i]
    
    def empty(self):
        return self.length() == 0
    
    def length(self):
        return len(self.pull)
    
    def nearest(self, x, y):
        if self.empty():
            self.add(x, y)
        distances = [self.countDistance(x, loc.x, y, loc.y) for loc in self.pull]
        return self.pull[distances.index(min(distances))]
    
    def nearest_distance(self, nearest, x, y):
        return self.countDistance(x, nearest.x, y, nearest.y)
        
    def countDistance(self, x1, x2, y1, y2):
        return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    
    def add_pts(self, i, center):
        self.pull[i].add_pts(center)
        
    def draw_track(self):
        for o in self.pull:
            for i in np.arange(1, len(o.pts)):
                if o.pts[i - 1] is None or o.pts[i] is None:
                    continue 
                if o.counter >= 10 and i == 1 and o.pts[-1] is not None:
                    o.dX = o.pts[-1][0] - o.pts[i][0]
                    o.dY = o.pts[-1][1] - o.pts[i][1]
                    (dirX, dirY) = ("", "")
                    if np.abs(o.dX) > 25:
                        dirX = "East" if np.sign(o.dX) == 1 else "West"
                    if np.abs(o.dY) > 25:
                        dirY = "North" if np.sign(o.dY) == 1 else "South"
                    if dirX != "" and dirY != "":
                        new_dir = "{}-{}".format(dirY, dirX)
                    else:
                        new_dir = dirX if dirX != "" else dirY
                    if o.direction != new_dir and new_dir != "":
                        log("{} {} directs with ({}, {}) coordinates to {}".format(o.c, o.id, o.x, o.y, o.direction))
                    o.direction = new_dir   
                thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
                cv2.line(frame, o.pts[i - 1], o.pts[i], (255, 255, 0), thickness)
            o.counter += 1  
            if not o.alive():
                o.pts.clear()
                if o.dead():
                    log("{} {} disappeared with ({}, {}) coordinates".format(o.c, o.id, o.x, o.y))
            else:
                cv2.putText(frame, o.direction, (o.x, o.y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    
def check_boundary(x, y, w, h):
    fh, fw = contoursImage.shape[0::1]
    left = x < 10
    right = fw - (x + w) < 10
    bottom = fh - (y + h) < 10
    return not (left or right or bottom) 

def denoise(frame):
    frame = cv2.medianBlur(frame, 3)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

def log(s):
    global result
    s = "{} {}\n".format(datetime.now(), s)
    result += s
    print(s)

def MotionDetection(inVideo, firstFrame, lastFrame, display=True): 
    global contoursImage, result, frame
    result = ""
    counter = 0
    people = Pull("Person")
    cars = Pull("Car")
    
    fgbg = cv2.createBackgroundSubtractorMOG2()
    cap = cv2.VideoCapture(inVideo)
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()
        counter += 1
        key = cv2.waitKey(1) & 0xFF
        if not ret or key == ord("q"):
            break
            
        if counter >= firstFrame and counter <= lastFrame:
            fgmask = fgbg.apply(denoise(frame))

            threshold = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
    
            opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8()), iterations = 1)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8()), iterations = 8)
            dilation = cv2.dilate(closing, np.ones((2, 2), np.uint8()))

            contoursImage = dilation.copy()
            cnts = cv2.findContours(contoursImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img2, contours, hierarchy = cnts
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                (_x, _y), radius = cv2.minEnclosingCircle(c)
                center = (int(_x), int(_y))
                area = cv2.contourArea(c)
                if area > 250:
                    if area > 2000 and w > h and check_boundary(x, y, w, h):
                        nearest = cars.nearest(x, y) 
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        if nearest.alive() and cars.nearest_distance(nearest, x, y) < 30:
                            car = cars.update(nearest.id, x, y)
                            cars.add_pts(nearest.id, center)
                        else:
                            car = cars.add(x, y)
                            cars.add_pts(-1, center)
                        cv2.putText(frame, "C" + str(car.id), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    elif area < 6000 and h > w and check_boundary(x, y, w, h):
                        nearest = people.nearest(x, y) 
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        if nearest.alive() and people.nearest_distance(nearest, x, y) < 15:
                            person = people.update(nearest.id, x, y) 
                            people.add_pts(nearest.id, center)
                        else:
                            person = people.add(x, y)
                            people.add_pts(-1, center)
                        cv2.putText(frame, "P" + str(person.id), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                elif area > 200:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                people.draw_track()
                cars.draw_track()
            if display:
                cv2.imshow("video", frame)
        elif counter >= lastFrame:
            break
    cv2.destroyAllWindows()
    return result 
    
if __name__ == "__main__":
    file = open("Motion.log","w") 
    inVideo = "assets/camera.mp4"
    outVideo = MotionDetection(inVideo, 0, 3064) # Frame range from 0 to 3064
    file.write(outVideo)
    file.close()