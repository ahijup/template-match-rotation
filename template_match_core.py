import cv2
import numpy as np
import time
import math
import polygon_utils
import rotate_image_util
import subpix_utils

def rotate_template_to_dict(template, angle_start, angle_sweep, angle_step):
    
    rotated_template_map = {}
    rotated_template_mask_map = {}
    rotated_template_vertecies_map = {}


    template_mask = np.ones(template.shape, dtype=np.uint8) * 255


    angle_list = np.arange(angle_start, angle_start + angle_sweep + angle_step, angle_step)
    # print('Angle list:', angle_list)

    for angle in angle_list:
        rotated_template, template_mask_rotated, rot_mat = rotate_image_util.rotate_template(template, template_mask, angle)
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

def match_template_rotate(image, rotated_template_map, rotated_template_mask_map, min_score, is_center = True):
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
        ncc_loc, ncc_scores = rotate_image_util.match_template_pixel_mask(image, rotated_template, mask, 1)
        if ncc_scores.max() > min_score:
            if is_center:
                ncc_loc = (ncc_loc[0] + tmpl_w / 2, ncc_loc[1] + tmpl_h / 2)
                x, max_x = subpix_utils.least_squared_fitting(np.array([ncc_loc[0] - 1, ncc_loc[0], ncc_loc[0] + 1], dtype=np.float64), ncc_scores[1, :])
                y, max_y = subpix_utils.least_squared_fitting(np.array([ncc_loc[1] - 1, ncc_loc[1], ncc_loc[1] + 1], dtype=np.float64), ncc_scores[:, 1])
                candidate_poses.append((angle, (x, y), (max_x + max_y) / 2))
            else:
                # offset_x, offset_y, max_s, _ = pf.poly_fitting_subpix_offset(ncc_scores)
                #
                candidate_poses.append((angle, ncc_loc, ncc_scores.max()))
        #print('Angle:', angle, 'Location:', ncc_loc, 'Score:', ncc_scores.max())

    return candidate_poses

def match_template_refine_subpix(image, template, loc, cur_angle, angle_step = 1):
    tmpl_w, tmpl_h = template.shape[1], template.shape[0]

    l_image = rotate_image_util.rigid_warp_image(image, (tmpl_w + 2, tmpl_h + 2), loc[0], loc[1], -(cur_angle - angle_step), 0)
    c_image = rotate_image_util.rigid_warp_image(image, (tmpl_w + 2, tmpl_h + 2), loc[0], loc[1], -cur_angle, 0)
    r_image = rotate_image_util.rigid_warp_image(image, (tmpl_w + 2, tmpl_h + 2), loc[0], loc[1], -(cur_angle + angle_step), 0)

    # cv2.imshow('l_image', l_image)
    # cv2.imshow('c_image', c_image)
    # cv2.imshow('r_image', r_image)
    # cv2.waitKey(0)
    ncc_scores = cv2.matchTemplate(c_image, template, cv2.TM_CCORR_NORMED)
    ncc_scores_l = cv2.matchTemplate(l_image, template, cv2.TM_CCORR_NORMED)
    ncc_scores_r = cv2.matchTemplate(r_image, template, cv2.TM_CCORR_NORMED)

    # return subpix_utils.find_subpix_1D_peak(loc, cur_angle, angle_step, ncc_scores, ncc_scores_l, ncc_scores_r)
    return subpix_utils.find_subpix_1D_peak(loc, cur_angle, angle_step, ncc_scores, ncc_scores_l, ncc_scores_r)


def match_template_refine_iter(image, template, loc, angle, angle_step, max_iter):
    max_score = 0
    ret_candidate = []

    angle_step_new = angle_step
    iter = 0
    while iter < max_iter:
        cur_angle, cur_loc, cur_score = match_template_refine_subpix(image, template, loc, angle, angle_step_new)
        # print('Iter:', iter, 'Cur angle:', cur_angle, 'Cur loc:', cur_loc, 'Cur score:', cur_score, 'angle_step:', angle_step_new)
        angle_step_new = angle_step_new / 2
        
        if max_score < cur_score:
            max_score = cur_score
            ret_candidate = (cur_angle, cur_loc, cur_score)
        # else:
        #     break
        loc = cur_loc
        angle = cur_angle

        iter += 1
        if angle_step_new < 0.05:
            break

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
                if len(intersection_polygon) < 3:
                    continue
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
    print('Candidate poses:', candidate_poses)
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
        n1, n2, tx, ty = rotate_image_util.point_point_affine(0, 0, math.radians(angle), x, y, 0)
        rot_mat = np.array([[n1, -n2, tx], [n2, n1, ty]], dtype=np.float32)
        template_vertice = cv2.transform(vertices.reshape(1, -1, 2), rot_mat).reshape(-1, 2)
        for i in range(4):
            cv2.line(image_rgb, 
                    (int(template_vertice[i][0]) , int(template_vertice[i][1])), 
                    (int(template_vertice[(i + 1) % 4][0]), int(template_vertice[(i + 1) % 4][1])), 
                    (0, 0, 255), 2)
        cv2.putText(image_rgb, f'{score:.2f}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image_rgb
