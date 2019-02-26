# Project: Synthetic Defocusing using Unsupervised Monocular Depth Estimation
# This file contains the function for preprocessing stage

# Input: Image file of any image format extension
# Output: Preprocessed image for input to the model

import cv2
import numpy as np
import glob                 # Used for file access

# Performs preprocessing functions on the given image
# Params:
#   img: Image in OpenCV image type (numpy.ndarray)
# Returns: Preprocessed image in OpenCV image type
def preprocess(img):
    # Denoising
    denoised_img = cv2.fastNlMeansDenoisingColored(img)

    # Conversion to Grayscale
    gray_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen_img = cv2.filter2D(gray_img, -1, kernel)

    # Histogram Equalization
    # equ_img = cv2.equalizeHist(sharpen_img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE()
    equ_img = clahe.apply(sharpen_img)

    # Resizing
    resized_img = cv2.resize(equ_img, (512,256), cv2.INTER_AREA)

    prep_img = resized_img
    return prep_img


# Run the preprocessing functions on a single image with before and after
# Escape Key terminates the windows
# Params
#   file: Filename of the image as string
#   waitTime: How long the image should be shown in ms
# Returns: false if terminated using Esc key else true
def preprocess_single(file, waitTime = 0):
    img = cv2.imread(file)
    prep_img = preprocess(img)

    cv2.imshow("Original Image", img)
    cv2.imshow("Preprocessed Image", prep_img)

    if cv2.waitKey(waitTime) == 27:
        cv2.destroyAllWindows()
        return 0
    cv2.destroyAllWindows()

# Performs preprocessing function on multiple images in a folder
# Params:
#     folder_path: Folder name which holds the images with ending slash
def preprocess_multiple(folder_path):
    folder_path = folder_path + ".png"
    for file in glob.glob(folder_path):
        preprocess_single(file, 2000)


if __name__ == "__main__":
    dataset_folder = "/home/haritha/Downloads/2011_09_26_drive_0039_sync/2011_09_26/2011_09_26_drive_0039_sync/image_03/data/"
    preprocess_multiple(dataset_folder)

