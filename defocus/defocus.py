import scipy.io as scio
import numpy as np
import cv2

IMG_DIR = "../images/"

class DefocuserObject():

    def __init__(self):
        self.depth_data = np.load(IMG_DIR + "sample_disp.npy")
        self.depth_img = cv2.imread(IMG_DIR + "sample_disp.png")
        self.img = cv2.imread(IMG_DIR + "sample.png")

        self.first_click = True

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
            else:
                print("x: ", x, ", y: ", y, "Depth value: ", self.depth_data[y][x])

    def view_image(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.depth_callback)

        cv2.imshow("image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    defocuser = DefocuserObject()
    defocuser.view_image()


