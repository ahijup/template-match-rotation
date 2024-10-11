import template_match_core
import rotate_image_util
import polygon_utils

import math
import time
import numpy as np
import cv2


def get_auto_pyrmaid_levels(template):
    tmpl_w, tmpl_h = template.shape[1], template.shape[0]
    sz = min(tmpl_w, tmpl_h)
    levels = 0
    while sz > 16:
        sz = sz // 2
        levels += 1
    return levels + 1

def create_pyramid_rotated_template_map(template, angle_start, angle_sweep, angle_step, num_levels):
    tmpl_w, tmpl_h = template.shape[1], template.shape[0]
    if num_levels == 0:
        num_levels = get_auto_pyrmaid_levels(template)
    if angle_step <= 0:
        angle_step = np.rad2deg(math.atan (2.0 / max (tmpl_w, tmpl_h)))
    ret_maps = []
    for level in range(num_levels):
        rotated_template_map, rotated_template_mask_map, rotated_template_vertecies_map = template_match_core.rotate_template_to_dict(template, angle_start, angle_sweep, angle_step)
        ret_maps.append((rotated_template_map, rotated_template_mask_map, rotated_template_vertecies_map, template, angle_step))
        template = cv2.pyrDown(template)
        angle_step = angle_step * 2
    return ret_maps


def crop_image(image, loc, sz):
    cx, cy = loc
    return rotate_image_util.rigid_warp_image(image, sz, cx, cy, 0)

def find_nearest_angle(angle, angles):
    angles = list(angles)
    min_diff = 180
    nearest_angle = 0
    angle_idx = -1
    idx = 0
    for ang in angles:
        diff = abs(ang - angle)
        if diff < min_diff:
            min_diff = diff
            nearest_angle = ang
            angle_idx = idx
        idx += 1
    if angle_idx <= 0 or angle_idx >= len(angles) - 1:
        if angle_idx == 0:
            return nearest_angle, nearest_angle, angles[angle_idx + 1]
        else:
            return nearest_angle, angles[angle_idx - 1], nearest_angle
        #print('Warning: angle not found')
        #raise Exception('Error: angle not found')
    return nearest_angle, angles[angle_idx - 1], angles[angle_idx + 1]

def get_rotated_template_size(cur_level_template, angle):
    cur_level_tmpl_img = cur_level_template[0][angle]
    return cur_level_tmpl_img.shape[1], cur_level_tmpl_img.shape[0]

def match_rotate_template(image, template, angle):
    cur_templ = template[0][angle]
    cur_templ_mask = template[1][angle]
    scores = cv2.matchTemplate(image, cur_templ, cv2.TM_CCORR_NORMED, mask=cur_templ_mask)
    return scores

def match_template_pyramid(test_image, rotated_template_model, min_score, max_overlap):
    tic = time.time()
    # build the pyramid
    target_pyramid = []
    min_score_pyramid = []

    for level in range(len(rotated_template_model)):
        min_score_pyramid.append(min_score)
        target_pyramid.append(test_image)
        test_image = cv2.pyrDown(test_image)
        min_score = min_score * 0.9

    top_level_template = rotated_template_model[-1]
    top_level_image = target_pyramid[-1]
    top_min_score = min_score_pyramid[-1]

    bottom_level_template = rotated_template_model[0]
    bottom_level_image = target_pyramid[0]


    candidates = template_match_core.match_template_rotate(top_level_image, top_level_template[0], top_level_template[1], top_min_score, False)
    toc = time.time()
    print('Elapsed time(phase-1):', toc - tic)
    # print('Candidates:', candidates)

    # tic = time.time()
    best_candidates = candidates #template_match_core.nms_candidate_poses(candidates, max_overlap, top_level_template[2])
    # toc = time.time()
    # print('Elapsed time(phase-2):', toc - tic)


    tic = time.time()
    ext_size = 2
    subpix_count = 0

    ret_matches = []
    for angle, loc, score in best_candidates:
        discard_match = False
        # restore the location to bottom level
        for level in range(len(rotated_template_model) - 2, -1, -1):
            cur_level_image = target_pyramid[level]
            new_x = loc[0] * 2
            new_y = loc[1] * 2
            # print('Level:', level, ', loc = ', new_x, new_y)
            cur_level_template = rotated_template_model[level]
            
            # find the nearest rotation angle
            cur_angle, prev_angle, next_angle = find_nearest_angle(angle, cur_level_template[0].keys())

            # get the rotated template size
            tmpl_c_w, tmpl_c_h = get_rotated_template_size(cur_level_template, cur_angle)
            tmpl_p_w, tmpl_p_h = get_rotated_template_size(cur_level_template, prev_angle)
            tmpl_n_w, tmpl_n_h = get_rotated_template_size(cur_level_template, next_angle)

            tmpl_w = max(tmpl_c_w, tmpl_p_w, tmpl_n_w)
            tmpl_h = max(tmpl_c_h, tmpl_p_h, tmpl_n_h)

            new_w = tmpl_w + ext_size * 2 # +-3 pixels
            new_h = tmpl_h + ext_size * 2 # +-3 pixels

            x1 = new_x - ext_size 
            y1 = new_y - ext_size 
            x2 = x1 + new_w 
            y2 = y1 + new_h
            tmpl_sec_w = 0
            tmpl_sec_h = 0


            # crop the image
            # cropped_image = crop_image(cur_level_image, (new_x, new_y), (new_w, new_h))
            
            if x1 < 0 or y1 < 0 or x2 >= cur_level_image.shape[1] or y2 >= cur_level_image.shape[0]:
                # zero padding
                cropped_image = np.zeros((new_h, new_w), dtype=np.uint8)
                dst_x1, dst_y1 = 0, 0
                dst_x2, dst_y2 = new_w, new_h
                src_x1, src_y1 = x1, y1
                src_x2, src_y2 = x2, y2
                

                if x1 < 0:
                    src_x1 = 0
                    dst_x1 = -x1
                if y1 < 0:
                    src_y1 = 0
                    dst_y1 = -y1

                if x2 >= cur_level_image.shape[1]:
                    src_x2 = cur_level_image.shape[1]
                    dst_x2 = new_w - (x2 - cur_level_image.shape[1])
                if y2 >= cur_level_image.shape[0]:
                    src_y2 = cur_level_image.shape[0]
                    dst_y2 = new_h - (y2 - cur_level_image.shape[0])
                cropped_image[dst_y1:dst_y2, dst_x1:dst_x2] = cur_level_image[src_y1:src_y2, src_x1:src_x2]
            else:
                cropped_image = cur_level_image[int(y1):int(y2), int(x1):int(x2)]

            

            cur_angle_scores = match_rotate_template(cropped_image, cur_level_template, cur_angle)
            prev_angle_scores = match_rotate_template(cropped_image, cur_level_template, prev_angle)
            next_angle_scores = match_rotate_template(cropped_image, cur_level_template, next_angle)
            cur_max_score = np.max(cur_angle_scores)
            prev_max_score = np.max(prev_angle_scores)
            next_max_score = np.max(next_angle_scores)
            

            max_score_map = []

            if cur_max_score >= prev_max_score and cur_max_score >= next_max_score:
                angle = cur_angle
                max_score_map = cur_angle_scores
                tmpl_sec_w = tmpl_c_w
                tmpl_sec_h = tmpl_c_h
            elif prev_max_score >= cur_max_score and prev_max_score >= next_max_score:
                angle = prev_angle
                max_score_map = prev_angle_scores
                tmpl_sec_w = tmpl_p_w
                tmpl_sec_h = tmpl_p_h
            elif next_max_score >= cur_max_score and next_max_score >= prev_max_score:
                angle = next_angle
                max_score_map = next_angle_scores
                tmpl_sec_w = tmpl_n_w
                tmpl_sec_h = tmpl_n_h
            else:
                print('Error: no maximum score found')
                raise Exception('Error: no maximum score found')
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(max_score_map)

            if score > max_val:
                discard_match = True
                break

            score = max_val
            loc = (x1 + max_loc[0], y1 + max_loc[1])
            if score < min_score_pyramid[level]:
                print('Score too low:', score)
                discard_match = True
                break
                # raise Exception('Score too low:', score)
            if level == 0:
                loc = (loc[0] + tmpl_sec_w / 2, loc[1] + tmpl_sec_h / 2)
                vertices = cur_level_template[2][angle].copy()
                cx, cy = np.mean(vertices[:, 0]), np.mean(vertices[:, 1])
                vertices[:, 0] += loc[0]
                vertices[:, 1] += loc[1]

        if discard_match:
            continue
        cur_match_area = polygon_utils.polygon_area(vertices)
        vertices = [tuple(pt) for pt in vertices]

        remove_idx = -1
        match_row_idx = 0
        # check the overlap
        for match, match_vertices, match_score in ret_matches:
            overlap_vertices = polygon_utils.sutherland_hodgman(vertices, match_vertices)
            overlap_area = polygon_utils.polygon_area(overlap_vertices)
            if overlap_area / cur_match_area > max_overlap:
                if score > match_score:
                    remove_idx = match_row_idx
                else:
                    discard_match = True
                break
            match_row_idx += 1

        if discard_match:
            continue

        if remove_idx != -1:
            del ret_matches[remove_idx]

        # print('pre-refine:', angle, loc, score)
        final_match, iter = template_match_core.match_template_refine_iter(bottom_level_image, bottom_level_template[3], loc, angle, 1.0, 3)
        subpix_count = subpix_count + 1
        # print('Final match:', iter, final_match)
        ret_matches.append((final_match, vertices, score))
    

    if len(ret_matches) > 0:
        ret = []
        for match, vertices, score in ret_matches:
            ret.append(match)
        ret_matches = ret
    toc = time.time()
    print('Elapsed time(phase-3):', toc - tic)
    print('Subpix count:', subpix_count)
    return ret_matches
