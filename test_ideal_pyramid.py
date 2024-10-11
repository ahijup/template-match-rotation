import template_match_pyramid
import rotate_image_util

import numpy as np
import cv2
import math
import time
import csv
import os

# test with a simple ideal pattern
pattern = np.ones((200, 300), dtype=np.uint8) * 255
pattern[50:150, 100:200] = 0
pat_w, pat_h = pattern.shape[1], pattern.shape[0]

angle_start = -45
angle_sweep = 90
angle_step = 1
num_pyramid_levels = 0

rotated_template_model = template_match_pyramid.create_pyramid_rotated_template_map(pattern, angle_start, angle_sweep, angle_step, num_pyramid_levels)

min_score = 0.9
max_overlap = 0.1
# 637.1	359	0.4
p_x, p_y, p_a = 637.1	,359	,0.4

n1, n2, tx, ty = rotate_image_util.point_point_affine(
            (pattern.shape[1]) / 2, (pattern.shape[0]) / 2, 0,
            p_x, p_y, math.radians(-p_a))
rot_mat = np.array([[n1, -n2, tx], [n2, n1, ty]], dtype=np.float32)
test_image = cv2.warpAffine(pattern, rot_mat, (1280, 720), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)  
print('Ideal:', p_x, p_y, p_a)
tic = time.time()
matches = template_match_pyramid.match_template_pyramid(test_image, rotated_template_model, min_score, max_overlap)
toc = time.time()
proc_time = (toc - tic) * 1000

if os.path.exists('temp/matches.csv'):
    os.remove('temp/matches.csv')


# 
for t in range(20):
    for y in range(20):
        for x in range(20):
            xx = (x - 10) / 10.0
            yy = (y - 10) / 10.0
            tt = (t - 10) / 10.0

            p_x, p_y, p_a = 638.0 + xx, 360.0 + yy, tt

            ## 1280*720
            n1, n2, tx, ty = rotate_image_util.point_point_affine(
                        (pattern.shape[1]) / 2, (pattern.shape[0]) / 2, 0,
                        p_x, p_y, math.radians(-p_a))
            rot_mat = np.array([[n1, -n2, tx], [n2, n1, ty]], dtype=np.float32)
            test_image = cv2.warpAffine(pattern, rot_mat, (1280, 720), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)  
            print('Ideal:', p_x, p_y, p_a)
            tic = time.time()
            matches = template_match_pyramid.match_template_pyramid(test_image, rotated_template_model, min_score, max_overlap)
            toc = time.time()
            proc_time = (toc - tic) * 1000

            # print('Elapsed time:', toc - tic)
            
            # print('Matches:', matches)
            with open('temp/matches.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                for match in matches:
                    writer.writerow([p_x, p_y, p_a, match[1][0], match[1][1], match[0], proc_time])
            print('---------------------------------')
print('test done')
# cv2.imshow('test_image', test_image)
# cv2.imshow('pattern', pattern)
# cv2.waitKey(0)
