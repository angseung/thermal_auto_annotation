from typing import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import write_label, label_voc2yolo


def annotation(
    img: np.ndarray,
    binary_thresh: int = 100,
    output_type: str = "yolo",
    mode: Optional[str] = "normal",
    morp_iters: Optional[int] = 3,
    draw_plot: Optional[bool] = False,
) -> Union[np.ndarray, Dict, List]:
    height, width, channel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    if mode == "adaptive":
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

        img_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9,
        )
    elif mode == "normal":
        img_thresh = cv2.threshold(gray, binary_thresh, 255, cv2.THRESH_BINARY)[1]

    else:
        raise ValueError

    mask = np.ones((5, 17), dtype=np.uint8)

    if morp_iters:
        morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, mask, iterations=morp_iters)

    else:
        morph = img_thresh.copy()

    if draw_plot:
        plt.figure(figsize=(12, 10))
        plt.subplot(221)
        plt.imshow(gray, cmap="gray")
        plt.subplot(222)
        plt.imshow(img_thresh, cmap="gray")
        plt.subplot(223)
        plt.imshow(morph, cmap="gray")

    # find contours
    contours, _ = cv2.findContours(
        img_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        contours_dict.append(
            {
                "contour": contour,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + (w / 2),
                "cy": y + (h / 2),
            }
        )

    # filtering small contours (any < pixel_thresh pixels)
    possible_contours = []
    pixel_thresh = 15
    ratio_thresh = 1.8

    for contour in contours_dict:
        ratio = contour["w"] / contour["h"] if contour["w"] > contour["h"] else contour["h"] / contour["w"]
        if contour["w"] > pixel_thresh and contour["h"] > pixel_thresh and ratio > ratio_thresh:
            possible_contours.append(contour)

    for d in possible_contours:
        cv2.rectangle(
            morph,
            pt1=(d["x"], d["y"]),
            pt2=(d["x"] + d["w"], d["y"] + d["h"]),
            color=(255, 255, 255),
            thickness=1,
        )

    if draw_plot:
        plt.subplot(224)
        plt.imshow(morph, cmap="gray")
        plt.show()

    # convert contours to yolo format
    bboxes = np.zeros((len(possible_contours), 5), dtype=np.uint32)
    for i, contour in enumerate(possible_contours):
        xtl, ytl = contour["x"], contour["y"]
        xbr, ybr = contour["x"] + contour["w"], contour["y"] + contour["h"]
        bboxes[i, 1:] = (xtl, ytl, xbr, ybr)

    bboxes_yolo = label_voc2yolo(bboxes, h=height, w=width)

    if output_type == "yolo":
        return bboxes_yolo

    else:
        return possible_contours
