import math
import random
from datetime import datetime
from typing import List, Union, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw


def remove_bg_from_img(img: np.ndarray, bg_color: str = "yellow") -> np.ndarray:
    if img.shape[2] == 4:
        raise ValueError

    # Convert image to image gray
    tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Applying thresholding technique
    if bg_color in ["white", "green"]:
        alpha = cv2.bitwise_not(cv2.inRange(tmp, 25, 30))
    elif bg_color in ["yellow", "blue"]:
        _, alpha = cv2.threshold(tmp, 250, 255, cv2.THRESH_BINARY_INV)  # done

    # Using cv2.split() to split channels
    # of coloured image
    b, g, r = cv2.split(img)

    # Making list of Blue, Green, Red
    # Channels and alpha
    bgra = [b, g, r, alpha]

    return cv2.merge(bgra, 4)


def warp_point(x: int, y: int, M: np.ndarray) -> Tuple[int, int]:
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    return (
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    )


def random_perspective(
    img: np.ndarray,
    labels: np.ndarray,
    mode: str = "auto",
    bg_color: str = "yellow",
    max_pad_order: Tuple[int, int] = (4, 8),
    return_mat: Optional[bool] = False,
    pads: Optional[Union[None, Tuple[int, int]]] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    random.seed(datetime.now().timestamp())
    if pads is not None:
        assert len(pads) == 2

    is_alpha = True if img.shape[2] == 4 else False

    if bg_color in ["yellow", "blue"]:
        color = (255, 255, 255)  # (B, G, R)
    elif bg_color in ["white", "green"]:
        color = (0, 0, 255)  # (B, G, R)
    else:
        raise ValueError

    H, W = img.shape[:2]
    max_pad_h = H // max_pad_order[0]
    max_pad_w = W // max_pad_order[1]

    if mode == "auto":
        mode_list = ["top", "bottom", "left", "right"]
        selected = random.randint(0, 3)
        mode = mode_list[selected]

    labels = label_yolo2voc(labels, H, W)

    if mode in ["top", "bottom"]:
        pad_l, pad_r = (random.randint(1, max_pad_w), random.randint(1, max_pad_w))

        if pads is not None:
            pad_l, pad_r = pads

        if is_alpha:
            img_padded = np.zeros([H, W + pad_l + pad_r, 4], dtype=np.uint8)
            img_padded[:, :, 0] = color[0]  # B
            img_padded[:, :, 1] = color[1]  # G
            img_padded[:, :, 2] = color[2]  # R
            img_padded[:, pad_l:-pad_r, 3] = img[:, :, 3]  # alpha
            img_padded[:, pad_l:-pad_r, :3] = img[:, :, :3]
        else:
            img_padded = np.zeros([H, W + pad_l + pad_r, 3], dtype=np.uint8)
            img_padded[:, :, 0] = color[0]  # B
            img_padded[:, :, 1] = color[1]  # G
            img_padded[:, :, 2] = color[2]  # R
            img_padded[:, pad_l:-pad_r, :] = img

        if mode == "top":
            point_before = np.float32(
                [
                    [2 * pad_l, 0],
                    [W + pad_l - pad_r, 0],
                    [2 * pad_l, H],
                    [W + pad_l - pad_r, H],
                ]
            )
            point_after = np.float32(
                [[pad_l, 0], [W + pad_l, 0], [2 * pad_l, H], [W + pad_l - pad_r, H]]
            )

        elif mode == "bottom":
            point_before = np.float32(
                [
                    [2 * pad_l, 0],
                    [W + pad_l - pad_r, 0],
                    [2 * pad_l, H],
                    [W + pad_l - pad_r, H],
                ]
            )
            point_after = np.float32(
                [[2 * pad_l, 0], [W + pad_l - pad_r, 0], [pad_l, H], [W + pad_l, H]]
            )

    elif mode in ["left", "right"]:
        pad_top, pad_bottom = (
            random.randint(1, max_pad_h),
            random.randint(1, max_pad_h),
        )
        if pads is not None:
            pad_top, pad_bottom = pads

        if is_alpha:
            img_padded = np.zeros([H + pad_top + pad_bottom, W, 4], dtype=np.uint8)
            img_padded[:, :, 0] = color[0]  # B
            img_padded[:, :, 1] = color[1]  # G
            img_padded[:, :, 2] = color[2]  # R
            img_padded[pad_top:-pad_bottom, :, 3] = img[:, :, 3]  # alpha
            img_padded[pad_top:-pad_bottom, :, :3] = img[:, :, :3]
        else:
            img_padded = np.zeros([H + pad_top + pad_bottom, W, 3], dtype=np.uint8)
            img_padded[:, :, 0] = color[0]  # B
            img_padded[:, :, 1] = color[1]  # G
            img_padded[:, :, 2] = color[2]  # R
            img_padded[pad_top:-pad_bottom, :, :] = img

        # img_padded = np.zeros([H + pad_top + pad_bottom, W, 3], dtype=np.uint8)
        # img_padded[:, :, :] = 255
        # img_padded[pad_top:-pad_bottom, :, :] = img

        if mode == "left":
            point_before = np.float32(
                [
                    [0, 2 * pad_top],
                    [W, pad_top],
                    [0, H + pad_top - pad_bottom],
                    [W, H + pad_top],
                ]
            )
            point_after = np.float32(
                [[0, pad_top], [W, pad_top], [0, H + pad_top], [W, H + pad_top]]
            )

        elif mode == "right":
            point_before = np.float32(
                [
                    [0, pad_top],
                    [W, 2 * pad_top],
                    [0, H + pad_top],
                    [W, H + pad_top - pad_bottom],
                ]
            )
            point_after = np.float32(
                [[0, pad_top], [W, pad_top], [0, H + pad_top], [W, H + pad_top]]
            )

    mtrx = cv2.getPerspectiveTransform(point_before, point_after)
    result = cv2.warpPerspective(
        img_padded,
        mtrx,
        img_padded.shape[:2][::-1],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=color,
    )

    for i, label in enumerate(labels):
        xtl, ytl, xbr, ybr = label.tolist()[1:]

        if mode in ["top", "bottom"]:
            xtl += pad_l
            xbr += pad_l

        elif mode in ["left", "right"]:
            ytl += pad_top
            ybr += pad_top

        xtl_new, ytl_new = warp_point(xtl, ytl, mtrx)
        xtr_new, ytr_new = warp_point(xbr, ytl, mtrx)
        xbl_new, ybl_new = warp_point(xtl, ybr, mtrx)
        xbr_new, ybr_new = warp_point(xbr, ybr, mtrx)

        labels[i, 1:] = np.uint32(
            [
                (xtl_new + xbl_new) // 2,
                (ytl_new + ytr_new) // 2,
                (xtr_new + xbr_new) // 2,
                (ybl_new + ybr_new) // 2,
            ]
        )

    labels = label_voc2yolo(labels, *result.shape[:2])

    if return_mat:
        return result, labels, mtrx

    else:
        return result, labels


def parse_label(fname: str) -> np.ndarray:
    """
    parses the label file, then converts it to np.ndarray type
    Args:
        fname: label file name

    Returns: label as np.ndarray

    """
    with open(fname, encoding="utf-8") as f:
        bboxes = f.readlines()
        label = []

    for bbox in bboxes:
        label.append(bbox.split())

    return np.array(label, dtype=np.float64)


def random_bright(img: np.ndarray, offset: Optional[float] = 0.25) -> np.ndarray:
    random.seed(datetime.now().timestamp())
    is_alpha = False

    if img.shape[2] == 4:
        is_alpha = True
        alpha = img[:, :, 3]
        img = img[:, :, :3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = offset + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if is_alpha:
        blue, green, red = cv2.split(img)
        bgra = [blue, green, red, alpha]
        img = cv2.merge(bgra, 4)

    return img


def blend_bgra_on_bgr(fg: np.ndarray, bg: np.ndarray, row: int, col: int) -> np.ndarray:
    assert fg.shape[2] == 4 and bg.shape[2] == 3
    _, mask = cv2.threshold(fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.cvtColor(fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = bg[row : row + h, col : col + w]

    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    blended = masked_fg + masked_bg

    return blended


def blend_bgra_on_bgra(
    fg: np.ndarray, bg: np.ndarray, row: int, col: int
) -> np.ndarray:
    assert fg.shape[2] == 4 and bg.shape[2] == 4

    padded_fg = np.zeros_like(bg, dtype=np.uint8)
    h, w = fg.shape[:2]
    padded_fg[row : row + h, col : col + w, :] = fg

    _, mask_fg = cv2.threshold(bg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    _, mask_bg = cv2.threshold(padded_fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    alpha = cv2.bitwise_or(mask_fg, mask_bg)

    bg[:, :, :3] = blend_bgra_on_bgr(bg=bg[:, :, :3], fg=padded_fg, row=0, col=0)

    blue, green, red = cv2.split(bg[:, :, :3])
    bgra = [blue, green, red, alpha]

    return cv2.merge(bgra)


def blend_bgr_on_bgra(fg: np.ndarray, bg: np.ndarray, row: int, col: int) -> np.ndarray:
    assert fg.shape[2] == 3 and bg.shape[2] == 4
    h, w = fg.shape[:2]
    bg[row : row + h, col : col + w, :3] = fg

    return bg


def make_bboxes(
    img: np.ndarray, obj: np.ndarray, label: int, ytl: int, xtl: int
) -> str:
    h, w = obj.shape[:2]
    xbr = xtl + w
    ybr = ytl + h
    center_x = (xtl + xbr) / 2.0
    center_y = (ytl + ybr) / 2.0

    h_bg, w_bg = img.shape[:2]

    # yolo format (x_center, y_center, width, height)
    return f"{label} {center_x / w_bg} {center_y / h_bg} {w / w_bg} {h / h_bg}"


def convert_bbox_to_label(bboxes: List[str]) -> np.ndarray:
    labels = np.zeros((len(bboxes), 5))

    for i, bbox in enumerate(bboxes):
        labels[i, :] = np.array(bbox.split())

    return labels


def write_label_from_str(target_dir: str, fname: str, *bboxes: List[str]) -> None:
    num_boxes = len(bboxes)

    with open(f"{target_dir}/{fname}.txt", "w") as f:
        for i in range(num_boxes):
            f.write(f"{bboxes[i]}\n")


def write_label(target_dir: str, fname: str, bboxes: np.ndarray) -> None:
    """
    exports np.ndarray label to txt file
    Args:
        target_dir: save dir for label file
        fname: file name of label
        bboxes: annotation information, np.ndarray type

    Returns: None

    """
    num_boxes = bboxes.shape[0]

    with open(f"{target_dir}/{fname}.txt", "w") as f:
        for i in range(num_boxes):
            target_str = f"{int(bboxes[i][0])} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]} {bboxes[i][4]}"
            f.write(f"{target_str}\n")


def random_resize(
    img: np.ndarray,
    label: Optional[Union[np.ndarray, None]] = None,
    scale_min: Union[int, float] = 0.75,
    scale_max: Union[int, float] = 2.5,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    random.seed(datetime.now().timestamp())
    scaled = random.uniform(scale_min, scale_max)
    h, w = img.shape[:2]

    if h > w:
        ratio = h / w
        w_scaled = w * scaled
        h_scaled = w_scaled * ratio

    else:
        ratio = w / h
        h_scaled = h * scaled
        w_scaled = h_scaled * ratio

    size = int(w_scaled), int(h_scaled)

    if label is not None:
        label = label_yolo2voc(label, h, w).astype(np.float64)
        label[:, 1:] *= scaled
        label = label_voc2yolo(label, h_scaled, w_scaled)

        return cv2.resize(img, size, interpolation=cv2.INTER_AREA), label

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def label_yolo2voc(label_yolo: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    converts label format from yolo to voc
    Args:
        label_yolo: (x_center, y_center, w, h), normalized
        h: img height
        w: img width

    Returns: (xtl, ytl, xbr, ybr)

    """
    label_voc = np.zeros(label_yolo.shape, dtype=np.float64)
    label_voc[:, 0] = label_yolo[:, 0]

    label_yolo_temp = label_yolo.copy()
    label_yolo_temp[:, [1, 3]] *= w
    label_yolo_temp[:, [2, 4]] *= h

    # convert x_center, y_center to xtl, ytl
    label_voc[:, 1] = label_yolo_temp[:, 1] - 0.5 * label_yolo_temp[:, 3]
    label_voc[:, 2] = label_yolo_temp[:, 2] - 0.5 * label_yolo_temp[:, 4]

    # convert width, height to xbr, ybr
    label_voc[:, 3] = label_voc[:, 1] + label_yolo_temp[:, 3]
    label_voc[:, 4] = label_voc[:, 2] + label_yolo_temp[:, 4]

    return label_voc.astype(np.uint32)


def label_voc2yolo(label_voc: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    converts label format from voc to yolo
    Args:
        label_voc: (xtl, ytl, xbr, ybr)
        h: img heights
        w: img width

    Returns: (x_center, y_center, w, h), normalized

    """
    label_yolo = np.zeros(label_voc.shape, dtype=np.float64)
    label_yolo[:, 0] = label_voc[:, 0]

    # convert xtl, ytl to x_center, y_center
    label_yolo[:, 1] = 0.5 * (label_voc[:, 1] + label_voc[:, 3])
    label_yolo[:, 2] = 0.5 * (label_voc[:, 2] + label_voc[:, 4])

    # convert xbr, ybr to width, height
    label_yolo[:, 3] = label_voc[:, 3] - label_voc[:, 1]
    label_yolo[:, 4] = label_voc[:, 4] - label_voc[:, 2]

    # normalize
    label_yolo[:, [1, 3]] /= w
    label_yolo[:, [2, 4]] /= h

    return label_yolo


def draw_bbox_on_img(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    if label.dtype != np.uint8:
        label = label_yolo2voc(label, *(img.shape[:2]))

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i in range(label.shape[0]):
        pos = tuple(label[i][1:].tolist())
        draw.rectangle(pos, outline=(0, 0, 0), width=3)

    return np.asarray(img)


def augment_img_label_and_save(
    img: np.ndarray,
    labels: np.ndarray,
    target_dir: str,
    fname: str,
    resize: bool = True,
    resize_scale: Tuple[float, float] = (1.0, 3.0),
    bright: bool = True,
    bright_offset: float = 0.25,
    perspective: bool = True,
    rotate: bool = False,
    angle: Union[int, str] = "auto",
    mode: str = "auto",
    remove_bg: bool = False,
    bg_color: str = "yellow",
    debug: bool = False,
):
    if bright:
        # BGRA -> BGRA
        img = random_bright(img, offset=bright_offset)

    if resize:
        # BGR -> BGR
        img, labels = random_resize(
            img, labels, scale_min=resize_scale[0], scale_max=resize_scale[1]
        )

    if perspective:
        # BGR -> BGR
        img, labels = random_perspective(img, labels, mode=mode, bg_color=bg_color)

    if rotate:
        # BGR -> BGR
        if angle == "auto":
            random.seed(datetime.now().timestamp())
            angle = random.randint(-10, 10)

        img, labels = rotate_img_and_bboxes(img, labels, angle=angle, bg_color=bg_color)

    if debug:
        # BGR -> BGR
        img = draw_bbox_on_img(img=img, label=labels)

    if remove_bg:
        # BGR -> BGRA
        img = remove_bg_from_img(img, bg_color=bg_color)

    cv2.imwrite(target_dir + "/images/train/" + fname + ".png", img)
    write_label(target_dir + "/labels/train", fname, labels)


def get_angle_from_warp(rqmtx: np.ndarray) -> Tuple[float, float, float]:
    phi = math.atan2(rqmtx[2][1], rqmtx[2][2])
    psi = math.atan2(rqmtx[1][0], rqmtx[0][0])

    if math.cos(psi) == 0:
        thetaa = math.atan2(-rqmtx[2][0], (rqmtx[1][0] / math.sin(psi)))
    else:
        thetaa = math.atan2(-rqmtx[2][0], (rqmtx[0][0] / math.cos(psi)))

    s = math.atan2(
        rqmtx[0][0], -math.sqrt(rqmtx[2][1] * rqmtx[2][1] + rqmtx[2][2] * rqmtx[2][2])
    )

    pi = math.pi

    phid = phi * (180 / pi)
    thetaad = thetaa * (180 / pi)
    psid = psi * (180 / pi)

    return phid, thetaad, psid


def rotate_point(
    point_x: Union[int, float], point_y: Union[int, float], angle: int
) -> Tuple[float, float]:
    angle = math.radians(angle)
    cos_coef = math.cos(angle)
    sin_coef = math.sin(angle)

    point_x_re = cos_coef * point_x - sin_coef * point_y
    point_y_re = sin_coef * point_x + cos_coef * point_y

    return point_x_re, point_y_re


def get_corners(bboxes: np.ndarray) -> np.ndarray:
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(
    corners: np.ndarray, angle: Union[float, int], cx: int, cy: int, h: int, w: int
):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0])))
    )

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners: np.ndarray) -> np.ndarray:
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def rotate_img_and_bboxes(
    img: np.ndarray,
    bboxes: Union[np.ndarray, str],
    angle: int,
    bg_color: str = "yellow",
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = img.shape[:2]
    (cX, cY) = (width // 2, height // 2)
    mat = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # adjust the rotation matrix to take into account translation
    mat[0, 2] += (bound_w / 2) - cX
    mat[1, 2] += (bound_h / 2) - cY

    if bg_color in ["yellow", "blue"]:
        color = (255, 255, 255)
    elif bg_color in ["white", "green"]:
        color = (0, 0, 255)
    else:
        raise ValueError

    rotated_img = cv2.warpAffine(
        img,
        mat,
        (bound_w, bound_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=color,
    )

    if isinstance(bboxes, str):
        bboxes = parse_label(bboxes)

    labels_voc = label_yolo2voc(bboxes, h=height, w=width)

    points = get_corners(labels_voc[:, 1:])
    calculated = rotate_box(points, angle=angle, cx=cX, cy=cY, h=height, w=width)
    calculated = get_enclosing_box(calculated)
    labels_voc[:, 1:] = calculated

    bboxes = label_voc2yolo(labels_voc, h=bound_h, w=bound_w)

    return rotated_img, bboxes


if __name__ == "__main__":
    fg = cv2.imread("EL04wn2084.png", cv2.IMREAD_UNCHANGED)
    bg = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
    aa = blend_bgra_on_bgra(fg, bg, 30, 30)
    cv2.imwrite("aa.png", aa)
