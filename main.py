import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL

# plt.style.use('dark_background')

img = cv2.imread('images/2023-03-08 082312.png')
height, width, channel = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

# img_thresh = cv2.adaptiveThreshold(
#     img_blurred,
#     maxValue=255.0,
#     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     thresholdType=cv2.THRESH_BINARY_INV,
#     blockSize=19,
#     C=9
# )

img_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
# img_thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)[1]

mask = np.ones((5, 17), dtype=np.uint8)
morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, mask, iterations=3)

plt.figure(figsize=(12, 10))
plt.subplot(221)
plt.imshow(gray, cmap='gray')
plt.subplot(222)
plt.imshow(img_thresh, cmap='gray')
plt.subplot(223)
plt.imshow(morph, cmap='gray')

## find contours
contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_TREE,
                               method=cv2.CHAIN_APPROX_SIMPLE)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1,
                 color=(255, 255, 255))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h),
                  color=(255, 255, 255), thickness=2)

    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

plt.subplot(224)
plt.imshow(temp_result, cmap='gray')
plt.show()
