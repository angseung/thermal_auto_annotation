import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from auto_annotation import annotation
from utils import draw_bbox_on_img, write_label

# plt.style.use('dark_background')

# img = cv2.imread("images/2023-03-08 082312.png")
# img = cv2.imread("images/2023-03-08 104749.png")
# img = cv2.imread("images/frame_00010.jpg")

# image_dir = "./images/standing_a_person"
# image_dir = "./images/standing_people"
image_dir = "./images/long_take"
# image_dir = "./images/lie"
# target_dir = "./images/standing_a_person_labeled"
# target_dir = "./images/standing_people_labeled"
target_dir = "./images/long_take_labeled"
# target_dir = "./images/lie_labeled"

if not os.path.isdir(target_dir):
    os.makedirs(target_dir, exist_ok=True)

if not os.path.isdir(f"{target_dir}/labels"):
    os.makedirs(f"{target_dir}/labels")

for fname in os.listdir(image_dir):
    img = cv2.imread(f"{image_dir}/{fname}")
    bboxes = annotation(img, binary_thresh=100, mode="normal", output_type="yolo", draw_plot=False, morp_iters=0)

    if bboxes.shape[0] == 3:
        bboxed_img = draw_bbox_on_img(img, bboxes, color=(255, 255, 255))
        # cv2.imwrite(f"{target_dir}/{fname}", bboxed_img)
        cv2.imwrite(f"{target_dir}/{fname}", img)
        write_label(target_dir=f"{target_dir}/labels", fname=fname, bboxes=bboxes)


# fig_fin = plt.figure()
# plt.imshow(bboxed_img)
# plt.show()

# save to gif
# path = [f"{i}" for i in os.listdir(target_dir)]
# images = [Image.open(f"{target_dir}/{i}") for i in path]
# imageio.mimsave(f"./annotation_2.gif", images)
