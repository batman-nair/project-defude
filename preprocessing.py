# Project: Synthetic Defocusing using Unsupervised Monocular Depth Estimation
# This file contains the function for preprocessing stage

# Input: Image file of any image format extension
# Output: Preprocessed image for input to the model

import cv2

import numpy as np

def preprocess(img):
    # Resizing
    img = cv2.resize(img, (512,512))
    #Denoising
    img = cv2.fastNlMeansDenoisingColored(img)
    #Conversion to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Sharpening 
    blur = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.addWeighted(blur,1.5,img,-0.5,0)
    #Histogram Equalization
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ))
    cv2.imwrite('sample.png',res)


    return img



if __name__ == "__main__":
    img = cv2.imread("./images/sample.png");

    prep_img = preprocess(img)

    cv2.imshow("Original Image", img)
    cv2.imshow("Preprocessed Image", prep_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
