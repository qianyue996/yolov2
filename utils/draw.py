import cv2 as cv
import numpy as np

def draw():
    cv.rectangle(self.img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),color=(0,0,255))
    cv.addText(self.img,
                self.objname,
                (int(xmin+5),int(ymax+5)),
                color=(0,0,255))