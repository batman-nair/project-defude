import argparse
import copy
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description='Defocus using depth map.')

parser.add_argument('--image_path', type=str, help='path to input image', default='../images/sample2.png')
parser.add_argument('--blur_method', type=str, help='The type of blur to be applied', default='gaussian')

args = parser.parse_args()

PATH = args.image_path
FILENAME = os.path.basename(PATH).split('.')[0]
IMG_DIR = os.path.dirname(PATH)
BLUR = args.blur_method
blur_function = {
        'avg_blur': lambda img,ker_size: cv2.blur(img, (ker_size,ker_size)),
        'gaussian': lambda img,ker_size: cv2.GaussianBlur(img, (ker_size,ker_size), 0),
        'median': lambda img,ker_size: cv2.medianBlur(img, ker_size),
        'bilateral': lambda img,ker_size: cv2.bilateralFilter(img, ker_size, 75, 75),
        }

class DefocuserObject():

    def __init__(self):
        self.depth_data = np.load(os.path.join(IMG_DIR, FILENAME + "_disp.npy"))
        self.img = cv2.imread(os.path.join(IMG_DIR, FILENAME + ".png"))
        self.blur_imgs = []


        self.blur_images()

    def view_details(self):
        print("Max Depth: ", np.max(self.depth_data))
        print("Min Depth: ", np.min(self.depth_data))

    def normalize_pof(self):
        self.norm_depth_data = self.depth_data - self.point_of_focus
        self.norm_depth_data = np.abs(self.norm_depth_data)
        self.norm_depth_data = self.norm_depth_data / self.norm_depth_data.max()

    # mouse callback function
    def depth_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point_of_focus = self.depth_data[y][x]
            print("Point of focus: ", self.point_of_focus)
            self.normalize_pof()
            print("Normalized depth data around point of focus")

            section_size = 1 / len(self.blur_imgs)
            final_image = np.zeros(self.img.shape)

            for index, blur_img in enumerate(self.blur_imgs):
                mask = (index*section_size <= self.norm_depth_data) & (self.norm_depth_data < (index+1)*section_size)
                masked_img = copy.deepcopy(blur_img)
                # Applying mask on copy of blurred image
                masked_img[mask==0] = [0, 0, 0]
                final_image = final_image + masked_img

            final_image = np.uint8(final_image)
            cv2.imshow("Final", final_image)
            cv2.imwrite(os.path.join(IMG_DIR, FILENAME + "_defocus.png"), final_image)

            #     print("x: ", x, ", y: ", y, "Depth value: ", self.norm_depth_data[y][x])

    def view_image(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.depth_callback)

        cv2.imshow("image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def blur_images(self):
        print("Generating blurred versions for image")
        self.blur_imgs.append(self.img)
        for ker_size in range(5, 22, 4):
            self.blur_imgs.append(blur_function[BLUR](self.img, ker_size))

            # Average blur
            # self.blur_imgs.append(cv2.blur(self.img, (ker_size,ker_size)))

            # Gaussian blur
            # self.blur_imgs.append(cv2.GaussianBlur(self.img, (ker_size,ker_size), 0))
            # Median blur
            # self.blur_imgs.append(cv2.medianBlur(self.img, ker_size))

            # Bilateral blur
            # self.blur_imgs.append(cv2.bilateralFilter(self.img, ker_size, 75, 75))


        # Show blurred images
        # for index, blur_img in enumerate(self.blur_imgs):
        #     cv2.imshow("blur " + str(index), blur_img)


if __name__ == "__main__":
    defocuser = DefocuserObject()
    defocuser.view_image()


