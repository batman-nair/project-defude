import argparse
import os

parser = argparse.ArgumentParser(description='Synthetic Defocussing Using Depth Estimation')

parser.add_argument('--image_path', type=str, help='path to input image', default='/home/arjun/works/project-defude/images/sample2.png')
parser.add_argument('--checkpoint_path', type=str, help='path to saved model', default='/home/arjun/works/project-defude/depth/trained_models/model_city2kitti_resnet')
parser.add_argument('--blur_method', type=str, help='the type of blur to be applied', default='gaussian')

args = parser.parse_args()

img_path = args.image_path
checkpoint_path = args.checkpoint_path
blur_method = args.blur_method

os.system("python ./depth/depth_simple.py --checkpoint_path " + checkpoint_path + " --image_path " + img_path)
os.system("python ./defocus/defocus.py --image_path " + img_path + " --blur_method " + blur_method)
