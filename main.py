import argparse
import os

parser = argparse.ArgumentParser(description='Synthetic Defocussing Using Depth Estimation')

parser.add_argument('--image_path', type=str, help='path to input image', default='images/sample2.png')
parser.add_argument('--model_path', type=str, help='path to saved model', default='blah')
parser.add_argument('--blur_method', type=str, help='the type of blur to be applied', default='gaussian')

args = parser.parse_args()

img_path = os.path.abspath(args.image_path)
model_path = os.path.abspath(args.model_path)
blur_method = args.blur_method

os.system("python ./depth/depth_simple.py --model_path " + model_path + " --image_path " + img_path)
os.system("python ./defocus/defocus.py --image_path " + img_path + " --blur_method " + blur_method)
