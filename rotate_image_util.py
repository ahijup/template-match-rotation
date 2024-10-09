import cv2
import numpy as np
import math

def point_point_affine(x1, y1, r1, x2, y2, r2):
    dr = r2 - r1
    a = math.cos(dr)
    b = math.sin(dr)
    tx = x2 - a * x1 + b * y1
    ty = y2 - b * x1 - a * y1
    return a, b, tx, ty

def rigid_warp_image(src, dstSz, srcX, srcY, srcR, borderValue=0):
    n1, n2, tx, ty = point_point_affine(
        srcX, srcY, math.radians(srcR), 
        (dstSz[0] / 2), (dstSz[1] / 2), 0)
    rot_mat = np.array([[n1, -n2, tx], [n2, n1, ty]], dtype=np.float32)
    return cv2.warpAffine(src, rot_mat, dstSz, None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, borderValue)

def righd_warp_image_mask(src, mask, dstSz, srcX, srcY, srcR, borderValue=0):
    n1, n2, tx, ty = point_point_affine(
        srcX, srcY, math.radians(srcR), 
        (dstSz[0] / 2), (dstSz[1] / 2), 0)
    rot_mat = np.array([[n1, -n2, tx], [n2, n1, ty]], dtype=np.float32)
    return cv2.warpAffine(src, rot_mat, dstSz, None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, borderValue), cv2.warpAffine(mask, rot_mat, dstSz, None, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 0), rot_mat

def match_template_pixel(src, tmpl, ext_sz = 1):
    ncc_map = cv2.matchTemplate(src, tmpl, cv2.TM_CCORR_NORMED)

    # Find the maximum value and its index
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc_map)

    # Print the maximum value and its index
    # print('Maximum value:', max_val)
    # print('Maximum index:', max_loc)
    sz = ext_sz * 2 + 1
    if ncc_map.shape[0] <= sz or ncc_map.shape[1] <= sz:
        x, y, w, h = 0, 0, ncc_map.shape[1], ncc_map.shape[0]
    else:
        # Define the ROI coordinates
        x, y, w, h = max_loc[0] - ext_sz, max_loc[1] - ext_sz, sz, sz

    if x < 0:
        w = w + x
        x = 0
    if y < 0:
        h = h + y
        y = 0
    if x + w > ncc_map.shape[1]:
        w = ncc_map.shape[1] - x
    if y + h > ncc_map.shape[0]:
        h = ncc_map.shape[0] - y
    # Crop the ROI using numpy slicing
    ncc_max_ampl = ncc_map[y:y+h, x:x+w].copy()
    # print('ncc_max_ampl:', ncc_max_ampl)

    return max_loc, ncc_max_ampl

def match_template_pixel_mask(src, tmpl, mask, ext_sz = 1):
    ncc_map = cv2.matchTemplate(src, tmpl, cv2.TM_CCORR_NORMED, mask=mask)

    # Find the maximum value and its index
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc_map)

    # Print the maximum value and its index
    # print('Maximum value:', max_val)
    # print('Maximum index:', max_loc)
    sz = ext_sz * 2 + 1
    if ncc_map.shape[0] <= sz or ncc_map.shape[1] <= sz:
        x, y, w, h = 0, 0, ncc_map.shape[1], ncc_map.shape[0]
    else:
        # Define the ROI coordinates
        x, y, w, h = max_loc[0] - ext_sz, max_loc[1] - ext_sz, sz, sz

    if x < 0:
        w = w + x
        x = 0
    if y < 0:
        h = h + y
        y = 0
    if x + w > ncc_map.shape[1]:
        w = ncc_map.shape[1] - x
    if y + h > ncc_map.shape[0]:
        h = ncc_map.shape[0] - y
    # Crop the ROI using numpy slicing
    ncc_max_ampl = ncc_map[y:y+h, x:x+w].copy()
    # print('ncc_max_ampl:', ncc_max_ampl)

    return max_loc, ncc_max_ampl

def calculate_size_after_rotation(w, h, angle):
    cos_angle = np.abs(np.cos(np.radians(angle)))
    sin_angle = np.abs(np.sin(np.radians(angle)))
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)
    return new_w, new_h

def rotate_template(template, mask, angle):
    h, w = template.shape
    # Calculate the new bounding dimensions of the image
    new_w, new_h = calculate_size_after_rotation(w, h, angle)
    return righd_warp_image_mask(template, mask,(new_w, new_h), w / 2, h / 2, angle, 0)
