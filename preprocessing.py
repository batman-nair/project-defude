# Project: Synthetic Defocusing using Unsupervised Monocular Depth Estimation
# This file contains the function for preprocessing stage

# Input: Image file of any image format extension
# Output: Preprocessed image for input to the model

import cv2

def preprocess(img):
    # Preprocessing steps
    img = cv2.resize(img, (0, 0), 0, fx = 0.5, fy = 0.5)
    img = cv2.fastNlMeansDenoisingColored(img)
    # img = cv2.equalizeHist(img)

    return img



if __name__ == "__main__":
    img = cv2.imread("./images/doge.jpg");

    prep_img = preprocess(img)

    cv2.imshow("Original Image", img)
    cv2.imshow("Preprocessed Image", prep_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
