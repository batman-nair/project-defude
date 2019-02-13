# Project: Synthetic Defocusing using Unsupervised Monocular Depth Estimation
# This file contains the function for preprocessing stage

# Input: Image file of any image format extension
# Output: Preprocessed image for input to the model

import cv2
import numpy as np         
import glob                 # Used for file access

def preprocess(img):
    # Denoising
    denoised_img = cv2.fastNlMeansDenoisingColored(img)

    # Conversion to Grayscale
    gray_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

    # Sharpening 
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen_img = cv2.filter2D(gray_img, -1, kernel)
    
    # blur = cv2.GaussianBlur(gray_img,(5,5),0)
    # sharpen_img = cv2.addWeighted(blur,1.5,gray_img,-0.5,0)

    # Histogram Equalization
    equ_img = cv2.equalizeHist(sharpen_img)
    
    # Resizing
    resized_img = cv2.resize(equ_img, (512,256))


    prep_img = resized_img
    return prep_img



if __name__ == "__main__":
    dataset_folder = "/home/haritha/Downloads/2011_09_26_drive_0039_sync/2011_09_26/2011_09_26_drive_0039_sync/image_03/data/*.png"
    for file in glob.glob(dataset_folder):
        img = cv2.imread(file);

        prep_img = preprocess(img)

        cv2.imshow("Original Image", img)
        cv2.imshow("Preprocessed Image", prep_img)

        if cv2.waitKey(2000) == 27: 
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()

