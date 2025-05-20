import cv2
import numpy as np
from pathlib import Path

def remove_barlines(target, output):

    #load image and binarise
    img = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    #detect staff lines (morph open)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    staff_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    #thin to get 1-pixel lines
    thin_lines = cv2.ximgproc.thinning(staff_lines)

    #subtract thin lines from original binary image
    no_staff = cv2.subtract(binary, thin_lines)

    #dilate to repair notes
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    repaired = cv2.dilate(no_staff, dilate_kernel, iterations=1)

    result = cv2.bitwise_not(repaired)
    print(output)
    cv2.imwrite(output, result)

if __name__ == "__main__":
    dir = Path("data/music")
    out = "data/music/removed_staff"
    for file in dir.iterdir():
        if "removed_staff" not in str(file):
            print(file)
            remove_barlines(file, f'{out}/{str(file)[-6:]}')
