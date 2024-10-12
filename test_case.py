import cv2
import numpy as np
import template_match_core
import math
import time

template = cv2.imread('temp/tmpl_crop.png', cv2.IMREAD_GRAYSCALE)
tmpl_w, tmpl_h = template.shape[1], template.shape[0]
image_list = ['temp/[570 240 200 180](670 330).bmp', 
              'temp/from_graduate/0708\LLevel2\9_3_10_260.bmp', 
              ]

angle_start = -45
angle_sweep = 90
angle_step = np.rad2deg(math.atan (2.0 / max (tmpl_w, tmpl_h)))

rotated_template_map, rotated_template_mask_map, rotated_template_vertices_map = template_match_core.rotate_template_to_dict(template, angle_start, angle_sweep, angle_step)

min_score = 0.9
max_overlap = 0.1

for image_path in image_list:
    print('Image:', image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tic = time.time()
    result = template_match_core.match_template(image, template, rotated_template_map, rotated_template_mask_map, rotated_template_vertices_map, angle_step, min_score, max_overlap)
    toc = time.time()
    print('Elapsed time:', toc - tic)
    print('Matches:', result)
    # image_rgb = draw_matches(image, (tmpl_w, tmpl_h), result)
    # cv2.imshow('image', image_rgb)
    # cv2.waitKey(0)
    print('--------------------------------------')
