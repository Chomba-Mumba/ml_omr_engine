import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def deskew_image(image):
    #load image in gray scale
    image = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("deskewed_out.jpg", image)
    plt.imshow(deskew, cmap='gray')
    plt.title("Deskewed Image")
    plt.axis('off')
    plt.show()

    #convert image to binary and invert
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #detect horizontal lines, morphological filtering
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1]//2, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    #hough line transform
    lines = cv2.HoughLinesP(detected_lines, 1, np.pi / 180, threshold=100, minLineLength=image.shape[1]//2, maxLineGap=20)

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
    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return deskewed

if __name__ == '__main__':
    deskew = deskew_image("../assets/img/sheet_music.png")

    cv2.imwrite("deskewed_out.jpg", deskew)
    plt.imshow(deskew, cmap='gray')
    plt.title("Deskewed Image")
    plt.axis('off')
    plt.show()