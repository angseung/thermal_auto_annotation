import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from auto_annotation import annotation
from utils import draw_bbox_on_img

# plt.style.use('dark_background')

# img = cv2.imread("images/2023-03-08 082312.png")
# img = cv2.imread("images/2023-03-08 104749.png")
img = cv2.imread("images/frame_00010.jpg")

image_dir = "./images/standing_a_person"
target_dir = "./images/standing_a_person_bboxed"

if not os.path.isdir(target_dir):
    os.makedirs(target_dir, exist_ok=True)

for fname in os.listdir(image_dir):
    img = cv2.imread(f"{image_dir}/{fname}")
    bboxes = annotation(img, binary_thresh=100, mode="normal", output_type="yolo", draw_plot=False, morp_iters=0)
    bboxed_img = draw_bbox_on_img(img, bboxes, color=(255, 255, 255))
    cv2.imwrite(f"{target_dir}/{fname}", bboxed_img)

# fig_fin = plt.figure()
# plt.imshow(bboxed_img)
# plt.show()

# save to gif
path = [f"{i}" for i in os.listdir(target_dir)]
images = [Image.open(f"{target_dir}/{i}") for i in path]
imageio.mimsave(f"./annotation.gif", images)
