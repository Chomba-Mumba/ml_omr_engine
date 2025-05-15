import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def preproccess(img,output):
    #binarisation
    assert img is not None, "no input image was provided"
    img = cv.medianBlur(img,5)

    th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    cv.imwrite(output,th)

    #image dewarping

    #staff-line removal

    #layout analysis
    
    #symbol detection

    #symbol classification


if __name__ == "__main__":
    pass