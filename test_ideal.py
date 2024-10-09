import cv2
import numpy as np
import template_match_core
import rotate_image_util
import math


# test with a simple ideal pattern
pattern = np.ones((200, 300), dtype=np.uint8) * 255
pattern[50:150, 100:200] = 0
pat_w, pat_h = pattern.shape[1], pattern.shape[0]

angle_start = -45
angle_sweep = 90
angle_step = np.rad2deg(math.atan (2.0 / max (pat_w, pat_h)))

rotated_template_map, rotated_template_mask_map, rotated_template_vertices_map = template_match_core.rotate_template_to_dict(pattern, angle_start, angle_sweep, angle_step)

min_score = 0.9
max_overlap = 0.1

p_x, p_y, p_a = 637.55, 360.4, 0.25

## 1280*720
n1, n2, tx, ty = rotate_image_util.point_point_affine(
            (pattern.shape[1]) / 2, (pattern.shape[0]) / 2, 0,
            p_x, p_y, math.radians(-p_a))
rot_mat = np.array([[n1, -n2, tx], [n2, n1, ty]], dtype=np.float32)
test_image = cv2.warpAffine(pattern, rot_mat, (1280, 720), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)  

cv2.imwrite('images/ideal_test_image.png', test_image)
cv2.imwrite('images/ideal_pattern.png', pattern)

result = template_match_core.match_template(test_image, pattern, rotated_template_map, rotated_template_mask_map, rotated_template_vertices_map, angle_step, min_score, max_overlap)
print('Transformation Params:', p_a, (p_x, p_y))
print('Matches:', result)
print('--------------------------------------')

