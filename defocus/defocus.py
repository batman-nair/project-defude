import scipy.io as scio
import numpy as np
import cv2

IMG_DIR = "../images/"
FILENAME = "sample2"

class DefocuserObject():

    def __init__(self):
        self.depth_data = np.load(IMG_DIR + FILENAME + "_disp.npy")
        self.depth_img = cv2.imread(IMG_DIR + FILENAME + "_disp.png")
        self.img = cv2.imread(IMG_DIR + FILENAME + ".png")
        self.blur_imgs = []

        self.first_click = True

        self.blur_images()

    def view_details(self):
        print("Max Depth: ", np.max(self.depth_data))
        print("Min Depth: ", np.min(self.depth_data))

    def normalize_pof(self):
        self.depth_data = self.depth_data - self.point_of_focus
        self.depth_data = np.abs(self.depth_data)
        self.depth_data = self.depth_data / self.depth_data.max()

    # mouse callback function
    def depth_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.first_click:
                self.first_click = False
                self.point_of_focus = self.depth_data[y][x]
                print("Point of focus: ", self.point_of_focus)
                self.normalize_pof()
                print("Normalized depth data around point of focus")

                section_size = 1 / len(self.blur_imgs)
                final_image = np.zeros(self.img.shape)

                print("final shape: ", final_image.shape)
                print("img shape ", self.img.shape)

                for index, blur_img in enumerate(self.blur_imgs):
                    mask = (index*section_size <= self.depth_data) & (self.depth_data < (index+1)*section_size)
                    masked_img = blur_img
                    # Applying mask on copy of blurred image
                    masked_img[mask==0] = [0, 0, 0]
                    final_image = final_image + masked_img

                final_image = np.uint8(final_image)
                cv2.imwrite(IMG_DIR + FILENAME + "_defocus.png", final_image)
                cv2.imshow("Final", final_image)

            else:
                print("x: ", x, ", y: ", y, "Depth value: ", self.depth_data[y][x])

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
            self.blur_imgs.append(cv2.GaussianBlur(self.img, (ker_size,ker_size), 0))

        # Show blurred images
        # for index, blur_img in enumerate(self.blur_imgs):
        #     cv2.imshow("blur " + str(index), blur_img)


if __name__ == "__main__":
    defocuser = DefocuserObject()
    defocuser.view_image()


