import cv2
import numpy as np
import time
import math
import polygon_utils

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

def rotate_template_to_dict(template, angle_start, angle_sweep, angle_step):
    
    rotated_template_map = {}
    rotated_template_mask_map = {}
    rotated_template_vertecies_map = {}


    template_mask = np.ones(template.shape, dtype=np.uint8) * 255


    angle_list = np.arange(angle_start, angle_start + angle_sweep + angle_step, angle_step)
    # print('Angle list:', angle_list)

    for angle in angle_list:
        rotated_template, template_mask_rotated, rot_mat = rotate_template(template, template_mask, angle)
        template_mask_rotated[template_mask_rotated > 0] = 255

        rotated_template_map[angle] = rotated_template
        rotated_template_mask_map[angle] = template_mask_rotated
        un_rotated_vertecies = np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]], [0, template.shape[0]]], dtype=np.float32)
        rotated_vertecies = cv2.transform(un_rotated_vertecies.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
        cx, cy = np.mean(rotated_vertecies[:, 0]), np.mean(rotated_vertecies[:, 1])
        rotated_vertecies[:, 0] -= cx
        rotated_vertecies[:, 1] -= cy
        rotated_template_vertecies_map[angle] = rotated_vertecies

        # print('Angle:', angle)
        # cv2.imshow('rotated_template', rotated_template)
        # cv2.imshow('rotated_template_mask', template_mask_rotated)
        # cv2.waitKey(0)

    return rotated_template_map, rotated_template_mask_map, rotated_template_vertecies_map

def match_template_rotate(image, rotated_template_map, rotated_template_mask_map, min_score):
    candidate_poses = []
    for angle, rotated_template in rotated_template_map.items():
        if rotated_template_mask_map.get(angle) is None:
            mask = None
        else:
            mask = rotated_template_mask_map[angle]
        
        tmpl_w, tmpl_h = rotated_template.shape[1], rotated_template.shape[0]
        # if angle == -32:
        #     cv2.imwrite(f'images/ncc_rotate/rotated_{angle}_template.png', rotated_template)
        #     cv2.imwrite(f'images/ncc_rotate/rotated_{angle}_mask.png', mask)
        #     cv2.imwrite(f'images/ncc_rotate/target.png', image)
            
        #     print('Angle:', angle)
        #     cv2.imshow('rotated_template', rotated_template)
        #     cv2.imshow('rotated_template_mask', mask)
        #     cv2.waitKey(0)
        # cv2.imshow('rotated_template', rotated_template)
        # cv2.imshow('rotated_template_mask', mask)
        # cv2.waitKey(0)
        ncc_loc, ncc_scores = match_template_pixel_mask(image, rotated_template, mask, 1)
        if ncc_scores.max() > min_score:
            ncc_loc = (ncc_loc[0] + tmpl_w / 2, ncc_loc[1] + tmpl_h / 2)
            x, max_x = least_squared_fitting(np.array([ncc_loc[0] - 1, ncc_loc[0], ncc_loc[0] + 1], dtype=np.float64), ncc_scores[1, :])
            y, max_y = least_squared_fitting(np.array([ncc_loc[1] - 1, ncc_loc[1], ncc_loc[1] + 1], dtype=np.float64), ncc_scores[:, 1])
            # offset_x, offset_y, max_s, _ = pf.poly_fitting_subpix_offset(ncc_scores)
            candidate_poses.append((angle, (x, y), (max_x + max_y) / 2))
            # candidate_poses.append((angle, ncc_loc, ncc_scores.max()))
        #print('Angle:', angle, 'Location:', ncc_loc, 'Score:', ncc_scores.max())

    return candidate_poses


def least_squared_fitting(x, y_vals):
    A = np.zeros((x.shape[0], 3), dtype=np.float64)
    A[:, 0] = x * x
    A[:, 1] = x
    A[:, 2] = 1
    P = np.linalg.lstsq(A, y_vals, rcond=None)
    a0, a1, a2 = P[0]
    # peak position: -a1 / 2a0
    x0 = -a1 / (2 * a0)
    # peak value: a0 * x0 * x0 + a1 * x0 + a2
    max_val = a0 * x0 * x0 + a1 * x0 + a2
    return x0, max_val

def least_squared_fitting_3D(x, y, t, w):
    A = np.zeros((27, 10), dtype=np.float64)
    S = np.zeros((27, 1), dtype=np.float64)

    row_idx = 0
    rot_idx = 0
    for r in t:
        rad = np.deg2rad(r)
        y_idx = 0
        for yy in y:
            x_idx = 0
            for xx in x:
                A[row_idx, 0] = xx * xx
                A[row_idx, 1] = yy * yy
                A[row_idx, 2] = rad * rad
                A[row_idx, 3] = xx * yy
                A[row_idx, 4] = xx * rad
                A[row_idx, 5] = yy * rad
                A[row_idx, 6] = xx
                A[row_idx, 7] = yy
                A[row_idx, 8] = rad
                A[row_idx, 9] = 1
                S[row_idx] = w[rot_idx][y_idx][x_idx]
                row_idx += 1
                x_idx += 1
            y_idx += 1
        rot_idx += 1

    A_t = np.transpose(A)
    A_squared = np.matmul(A_t, A)
    A_inv = np.linalg.inv(A_squared)
    A_inv_A_t = np.matmul(A_inv, A_t)
    P = np.matmul(A_inv_A_t, S)
    a, b, c, d, e, f, g, h, i, j = P.flatten()


    # do partial different
    # f(x, y, r) / dx = 2 * a * x + d * y + e * r + g = 0
    # f(x, y, r) / dy = 2 * b * y + d * x + f * r + h = 0
    # f(x, y, r) / dr = 2 * c * r + e * x + f * y + i = 0
    # [[2 * a, d, e], [d, 2 * b, f], [e, f, c]] * [x, y, r] = [-g, -h, -i]
    MM = np.array([[2 * a, d, e], [d, 2 * b, f], [e, f, 2 * c]], dtype=np.float64)
    N = np.array([-g, -h, -i], dtype=np.float64)

    MM_inv = np.linalg.inv(MM)
    Sol = np.matmul(MM_inv, N)

    # Sol = np.linalg.solve(MM, N)
    x0, y0, r0 = Sol.flatten()
    max_value = a * x0 * x0 + b * y0 * y0 + c * r0 * r0 + d * x0 * y0 + e * x0 * r0 + f * y0 * r0 + g * x0 + h * y0 + i * r0 + j
    r0 = np.rad2deg(r0)
    return x0, y0, r0, max_value

def find_subpix_1D_peak(loc, cur_angle, angle_step, ncc_scores, ncc_scores_l, ncc_scores_r):
    a0, a_max = least_squared_fitting(np.array([-angle_step + cur_angle, cur_angle, angle_step + cur_angle], dtype=np.float64), 
                          np.array([ncc_scores_l[1, 1], ncc_scores[1, 1], ncc_scores_r[1, 1]], dtype=np.float64))
    x0, x_max = least_squared_fitting(np.array([loc[0] - 1, loc[0], loc[0] + 1], dtype=np.float64),
                                      np.array([ncc_scores[1, 0], ncc_scores[1, 1], ncc_scores[1, 2]], dtype=np.float64))
    y0, y_max = least_squared_fitting(np.array([loc[1] - 1, loc[1], loc[1] + 1], dtype=np.float64),
                                      np.array([ncc_scores[0, 1], ncc_scores[1, 1], ncc_scores[2, 1]], dtype=np.float64))
    
    scores = np.array([a_max, x_max, y_max], dtype=np.float64)
    s = np.mean(scores)
    return a0, (x0, y0), s
def find_subpix_3D_peak(loc, cur_angle, angle_step, ncc_scores, ncc_scores_l, ncc_scores_r):
    if ncc_scores[1, 1] > ncc_scores_l[1, 1] and ncc_scores[1, 1] > ncc_scores_r[1, 1]:
        x0, y0, r, s = least_squared_fitting_3D(
                             np.array([loc[0] - 1, loc[0], loc[0] + 1], dtype=np.float64), 
                             np.array([loc[1] - 1, loc[1], loc[1] + 1], dtype=np.float64), 
                             np.array([-angle_step + cur_angle, cur_angle, angle_step + cur_angle], dtype=np.float64),
                             (ncc_scores_l, ncc_scores, ncc_scores_r))
        # print('x0, y0, r:', x0, y0, r)
    else:
        x0, y0, r, s = loc[0], loc[1], cur_angle, ncc_scores[1, 1]
    return r, (x0, y0), s


def match_template_refine_subpix(image, template, loc, cur_angle, angle_step = 1):
    tmpl_w, tmpl_h = template.shape[1], template.shape[0]

    l_image = rigid_warp_image(image, (tmpl_w + 2, tmpl_h + 2), loc[0], loc[1], -(cur_angle - angle_step), 0)
    c_image = rigid_warp_image(image, (tmpl_w + 2, tmpl_h + 2), loc[0], loc[1], -cur_angle, 0)
    r_image = rigid_warp_image(image, (tmpl_w + 2, tmpl_h + 2), loc[0], loc[1], -(cur_angle + angle_step), 0)

    # cv2.imshow('l_image', l_image)
    # cv2.imshow('c_image', c_image)
    # cv2.imshow('r_image', r_image)
    # cv2.waitKey(0)
    ncc_scores = cv2.matchTemplate(c_image, template, cv2.TM_CCORR_NORMED)
    ncc_scores_l = cv2.matchTemplate(l_image, template, cv2.TM_CCORR_NORMED)
    ncc_scores_r = cv2.matchTemplate(r_image, template, cv2.TM_CCORR_NORMED)

    return find_subpix_1D_peak(loc, cur_angle, angle_step, ncc_scores, ncc_scores_l, ncc_scores_r)


def match_template_refine_iter(image, template, loc, angle, angle_step, max_iter):
    max_score = 0
    ret_candidate = []

    angle_step_new = angle_step
    iter = 0
    while iter < max_iter:
        cur_angle, cur_loc, cur_score = match_template_refine_subpix(image, template, loc, angle, angle_step_new)
        angle_step_new = angle_step_new / 2
        if angle_step_new < 0.1:
            break
        # if max_score < cur_score:
        max_score = cur_score
        ret_candidate = (cur_angle, cur_loc, cur_score)
        # else:
        #     break
        loc = cur_loc
        angle = cur_angle

        iter += 1

    return ret_candidate, iter

def nms_candidate_poses(candidate_poses, max_overlap, rotated_template_vertices_map):
    # filter out the overlapped poses
    candidate_poses.sort(key=lambda x: x[2], reverse=True)
    if len(candidate_poses) > 1:
        filtered_candidate = []
        filtered = np.array([False] * len(candidate_poses))
        filtered_candidate.append(candidate_poses[0])
        for i in range(1, len(candidate_poses)):
            if filtered[i]:
                continue
            ang1 = candidate_poses[i][0]
            x, y = candidate_poses[i][1]
            vertices = rotated_template_vertices_map[ang1].copy()
            vertices[:, 0] += x
            vertices[:, 1] += y
            # print('Vertices:', vertices)
            area = polygon_utils.polygon_area(vertices)
            vertices = [tuple(pt) for pt in vertices]

            overlapped = False
            for j in range(len(filtered_candidate)):
                ang2 = filtered_candidate[j][0]
                x2, y2 = filtered_candidate[j][1]
                vertices2 = rotated_template_vertices_map[ang2].copy()
                vertices2[:, 0] += x2
                vertices2[:, 1] += y2
                vertices2 = [tuple(pt) for pt in vertices2]
                intersection_polygon = polygon_utils.sutherland_hodgman(vertices, vertices2)
                area2 = polygon_utils.polygon_area(intersection_polygon)
                if (area2 / area) > max_overlap:
                    overlapped = True
                    filtered[j] = True
                    break
            if not overlapped:
                filtered_candidate.append(candidate_poses[i])
        candidate_poses = filtered_candidate
    return candidate_poses

def match_template(image, template, 
                   rotated_template_map, rotated_template_mask_map, rotated_template_vertices_map, 
                   template_angle_step,
                   min_score, max_overlap):
    
    tic = time.time()
    candidate_poses = match_template_rotate(image, rotated_template_map, rotated_template_mask_map, min_score)
    toc = time.time()
    print('Elapsed time(phase-1):', toc - tic)
    
    tic = time.time()
    candidate_poses = nms_candidate_poses(candidate_poses, max_overlap, rotated_template_vertices_map)
    toc = time.time()
    print('Elapsed time(phase-2):', toc - tic)

    # ret_match = candidate_poses
    tic = time.time()
    ret_match = []
    for angle, loc, score in candidate_poses:
        # print('pre-refine:', angle, loc, score)
        # refine the location
        # final_match = match_template_refine_iter(image, template, (loc[0], loc[1]), angle, angle_step, min_score, 10)
        final_match, iter = match_template_refine_iter(image, template, loc, angle, template_angle_step, 3)
        ret_match.append(final_match)
        # print('Final match:', iter, final_match)
    toc = time.time()
    print('Elapsed time(phase-3):', toc - tic)
    return ret_match

def draw_matches(image, tmpl_size, matches):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    vertices = np.array([
        [-tmpl_size[0] / 2, -tmpl_size[1] / 2], 
        [tmpl_size[0] / 2, -tmpl_size[1] / 2], 
        [tmpl_size[0] / 2, tmpl_size[1] / 2], 
        [-tmpl_size[0] / 2, tmpl_size[1] / 2]], dtype=np.float32)
    # rotated_vertecies = cv2.transform(un_rotated_vertecies.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
    for angle, loc, score in matches:
        x, y = loc
        n1, n2, tx, ty = point_point_affine(0, 0, math.radians(angle), x, y, 0)
        rot_mat = np.array([[n1, -n2, tx], [n2, n1, ty]], dtype=np.float32)
        template_vertice = cv2.transform(vertices.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
        for i in range(4):
            cv2.line(image_rgb, 
                    (int(template_vertice[i][0]) , int(template_vertice[i][1])), 
                    (int(template_vertice[(i + 1) % 4][0]), int(template_vertice[(i + 1) % 4][1])), 
                    (0, 0, 255), 2)
        cv2.putText(image_rgb, f'{score:.2f}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image_rgb
