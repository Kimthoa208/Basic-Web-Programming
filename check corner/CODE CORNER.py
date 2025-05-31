import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny, corner_harris, corner_peaks
from skimage import color
from skimage.io import imread
from skimage.io import imsave

BUILDING2 = imread('BUILDING2.jpg')
plt.imshow(BUILDING2)
plt.title('BUILDING2')
plt.axis('off')
plt.show()
BUILDING2_gray = color.rgb2gray(BUILDING2)
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
def show_image_with_corners(image, coords, title="Corners detected"):
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=7) 
    plt.axis('off')
    plt.show()
canny_edges = canny(BUILDING2_gray)
show_image(canny_edges, "Edges with Canny")
canny_edges_0_5 = canny(BUILDING2_gray, sigma=0.5)
show_image(canny_edges_0_5, "Edges with Canny (Sigma 0.5)")
measure_image = corner_harris(BUILDING2_gray)
show_image(measure_image, "Harris Response")
coords = corner_peaks(corner_harris(BUILDING2_gray), min_distance=59)
print("A total of", len(coords), "corners were detected.")
show_image_with_corners(BUILDING2_gray, coords)
output_image = np.copy(BUILDING2)  
output_image[coords[:, 0], coords[:, 1]] = [255, 0, 0] 
imsave('output_image_with_corners.jpg', output_image)  