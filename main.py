import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from auto_annotation import annotation

# plt.style.use('dark_background')

img = cv2.imread("images/2023-03-08 082312.png")
bboxes = annotation(img, binary_thresh=100, mode="normal", draw_plot=True)
