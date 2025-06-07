import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib.pyplot as plt
import os


def deskew_image(image):
    print(os.getcwd())
    #load image in gray scale
    image = cv.imread(image,cv.IMREAD_GRAYSCALE)
    
    #convert image to binary and invert
    _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    #detect horizontal lines, morphological filtering
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (image.shape[1]//2, 1))
    detected_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, horizontal_kernel)

    #hough line transform
    lines = cv.HoughLinesP(detected_lines, 1, np.pi / 180, threshold=100, minLineLength=image.shape[1]//2, maxLineGap=20)

    #compute angles
    angles = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            angles.append(angle)
        
    #average angle
    if angles:
        avg_angle = np.mean(angles)
        print(f"Skew angle: {avg_angle: .2f} degrees")
    else:
        avg_angle = 0
        print("No lines detected")
    
    #deskew image
    (h,w) = image.shape
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, avg_angle, 1.0)
    deskewed = cv.warpAffine(image, M, (w,h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    return deskewed


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