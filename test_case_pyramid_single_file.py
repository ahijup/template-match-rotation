import cv2
import template_match_pyramid
import template_match_core
import time

template = cv2.imread('temp/tmpl_crop.png', cv2.IMREAD_GRAYSCALE)
tmpl_w, tmpl_h = template.shape[1], template.shape[0]


angle_start = -45
angle_sweep = 90
angle_step = 0

num_pyramid_levels = 0

rotated_template_model = template_match_pyramid.create_pyramid_rotated_template_map(template, angle_start, angle_sweep, angle_step, num_pyramid_levels)

image_list = ['temp/[570 240 200 180](670 330).bmp', 
              'temp/from_graduate/0708\LLevel2\9_3_10_260.bmp', 
              'temp/from_graduate/0708\LLevel2\9_3_10_758.bmp'
              ]

min_score = 0.8
max_overlap = 0.1

for image_path in image_list:
    print('Image:', image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tic = time.time()
    result = template_match_pyramid.match_template_pyramid(image, rotated_template_model, min_score, max_overlap)
    toc = time.time()
    print('Elapsed time:', toc - tic)
    print('Matches:', result)
    image_rgb = template_match_core.draw_matches(image, (tmpl_w, tmpl_h), result)
    cv2.imshow('image', image_rgb)
    cv2.waitKey(0)
    print('--------------------------------------')


