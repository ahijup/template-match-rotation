import cv2
import numpy as np
import template_match_pyramid
import time

template = cv2.imread('images/pattern.png', cv2.IMREAD_GRAYSCALE)
tmpl_w, tmpl_h = template.shape[1], template.shape[0]
image_list = [
              'images/source.png', 
              'images/target_rot15.png', 
              'images/target_rot25.png', 
              'images/target_rot35.png', 
              'images/target_rotn15.png', 
              'images/target_rotn25.png', 
              'images/target_rotn35.png']

angle_start = -45
angle_sweep = 90
angle_step = 1

num_pyramid_levels = 0

rotated_template_model = template_match_pyramid.create_pyramid_rotated_template_map(template, angle_start, angle_sweep, angle_step, num_pyramid_levels)


min_score = 0.9
max_overlap = 0.1

for image_path in image_list:
    print('Image:', image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tic = time.time()
    result = template_match_pyramid.match_template_pyramid(image, rotated_template_model, min_score, max_overlap)
    toc = time.time()
    print('Elapsed time:', toc - tic)
    print('Matches:', result)
    # image_rgb = draw_matches(image, (tmpl_w, tmpl_h), result)
    # cv2.imshow('image', image_rgb)
    # cv2.waitKey(0)
    print('--------------------------------------')
