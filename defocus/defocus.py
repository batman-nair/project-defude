import argparse
import copy
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description='Defocus using depth map.')

parser.add_argument('--image_path', type=str, help='path to input image', default='../images/sample2.png')
parser.add_argument('--blur_method', type=str, help='The type of blur to be applied', default='gaussian')

args = parser.parse_args()

# This class holds functions that handle defocusing of an image using a depth map
# It takes in the path to the image to be defocused and the blur method to be used for defocusing
# It requires the depth map to be saved in the same folder as the image, with the suffix and extension _disp.npy
# Parameters:
#   image_path: Path to the image which is to be defocused
#   blur_method: The blur function to be used for blurring.
#       Available values are: gaussian, avg_blur, median, bilateral
class DefocuserObject():

    def __init__(self, image_path = "../images/sample2.png", blur_method = "gaussian"):
        self.img_name = os.path.basename(image_path).split('.')[0]
        self.img_ext = os.path.basename(image_path).split('.')[-1]
        self.img_dir = os.path.dirname(image_path)
        self.blur_method = blur_method

        # Lambda functions to call the appropriate blur function according to set method
        # All the functions takes 2 arguments: the image and the kernel size to be used in the function
        # As the kernel size increases the amount of blur increases
        self.blur_function = {
            'avg_blur': lambda img,ker_size: cv2.blur(img, (ker_size,ker_size)),
            'gaussian': lambda img,ker_size: cv2.GaussianBlur(img, (ker_size,ker_size), 0),
            'median': lambda img,ker_size: cv2.medianBlur(img, ker_size),
            'bilateral': lambda img,ker_size: cv2.bilateralFilter(img, ker_size, 75, 75),
        }


        self.depth_data = np.load(os.path.join(self.img_dir, self.img_name + "_disp.npy"))
        self.img = cv2.imread(os.path.join(self.img_dir, self.img_name + "." + self.img_ext))
        self.blur_imgs = []

        # The blurred versions of the images can be precalculated
        self.blur_images()

    # Normalizes the depth values based on the depth of focus
    # The depth of focus is saved in point_of_focus
    # norm_depth_data holds the final normalized depth of focus
    # 0 value corresponds to the point which is focused
    # 1 value corresponds to the point furthest from the point of focus
    def normalize_pof(self):
        self.norm_depth_data = self.depth_data - self.point_of_focus
        self.norm_depth_data = np.abs(self.norm_depth_data)
        self.norm_depth_data = self.norm_depth_data / self.norm_depth_data.max()

    # Mouse callback function
    # The point of the click is taken as the point of focus and defocusing is performed around it
    def depth_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point_of_focus = self.depth_data[y][x]
            self.defocus_with_pof()

    # The depth map is normalized around the point of focus after which the defocusing of the image is done by sectioning and masking
    def defocus_with_pof(self):
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
        # cv2.imshow("Final", final_image)
        cv2.imwrite(os.path.join(self.img_dir, self.img_name + "_defocus.png"), final_image)
        cv2.imshow("Project Defude", final_image)


    def set_pof_from_coord(self, norm_x, norm_y):
        h, w = self.depth_data.shape[:2]
        self.point_of_focus = self.depth_data[int(h * norm_y)][int(w * norm_x)]
        self.defocus_with_pof()


    # Creates a window to display the original image
    # The callback function is attached to this window
    def view_image_for_blur(self):
        cv2.namedWindow("Project Defude", flags = (cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE))
        cv2.setMouseCallback("Project Defude", self.depth_callback)

        cv2.imshow("Project Defude", self.img)

        # cv2.namedWindow("Project Defude", flags=cv2.WINDOW_GUI_NORMAL)
        while(cv2.waitKey() != 27):
            pass
        cv2.destroyAllWindows()

    # Generated different blurred versions of the original image
    # The blurred versions are stored in the list blur_imgs
    def blur_images(self):
        print("Generating blurred versions for image")
        self.blur_imgs.append(self.img)
        for ker_size in range(5, 22, 4):
            self.blur_imgs.append(self.blur_function[self.blur_method](self.img, ker_size))

        # Show blurred images
        # for index, blur_img in enumerate(self.blur_imgs):
        #     cv2.imshow("blur " + str(index), blur_img)


if __name__ == "__main__":
    PATH = args.image_path
    BLUR = args.blur_method

    defocuser = DefocuserObject(image_path = PATH, blur_method = BLUR)
    # defocuser.set_pof_from_coord(0.9, 0.9)
    defocuser.view_image_for_blur()
