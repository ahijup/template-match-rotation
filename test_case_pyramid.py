import cv2
import template_match_pyramid
import time
import os
import csv

def get_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, filename))
    return image_files

def get_all_subdirectories(directory):
    subdirectories = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subdirectories.append(os.path.join(root, dir))
    return subdirectories

template = cv2.imread('temp/tmpl_crop.png', cv2.IMREAD_GRAYSCALE)
tmpl_w, tmpl_h = template.shape[1], template.shape[0]


angle_start = -45
angle_sweep = 90
angle_step = 1

num_pyramid_levels = 0

rotated_template_model = template_match_pyramid.create_pyramid_rotated_template_map(template, angle_start, angle_sweep, angle_step, num_pyramid_levels)

min_score = 0.8
max_overlap = 0.1

f = get_all_subdirectories('temp/from_graduate/0708')

for folder in f:
    image_list = get_image_files(folder)
    f_name = os.path.basename(folder)
    
    if os.path.exists('temp/' + f_name + '.csv'):
        os.remove('temp/' + f_name + '.csv')
    for image_path in image_list:
        print('Image:', image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        tic = time.time()
        matches = template_match_pyramid.match_template_pyramid(image, rotated_template_model, min_score, max_overlap)
        toc = time.time()
        print('Elapsed time:', toc - tic)
        proc_time = (toc - tic) * 1000
        
        with open('temp/' + f_name + '.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if len(matches) > 0:
                for match in matches:
                    writer.writerow([image_path, match[1][0], match[1][1], match[0], match[2], proc_time])
            else:
                writer.writerow([image_path, 0, 0, 0, proc_time])
        # image_rgb = draw_matches(image, (tmpl_w, tmpl_h), result)
        # cv2.imshow('image', image_rgb)
        # cv2.waitKey(0)
        print('--------------------------------------')

