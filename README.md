### Template Match with rotation using NCC
This project is a proof of concept about how to match a template in an image containing a single target at arbitrary angles by using normalized cross-correlation (NCC) 
1. template_match_core.py: core function of matching template
2. test_ideal.py: Test the matching function on an ideal image to verify the accuracy of the algorithm
3. test_images.py Test the matching function on a template image at arbitrary angles.
4. polygon_utils.py: provide a function to calculate the area of overlap between two polygons 
5. rotate_image_util.py: provide a function to rotate an image and obtain the rigid transformation matrix 
6. subpix_utils.py: provide a function to calculate subpixel-accurate positions via least-squares fitting to a second-degree polynomial 
7. template_match_pyramid.py: optimized matching template by image pyramid 
8. test_ideal_pyramid.py: Test the pyramid optimized version of matching function on an ideal image
9. test_images_pyramid.py: Test the pyramid optimized version of matching function on a template image at arbitrary angles

### Required package
```
numpy
opencv-python
```
