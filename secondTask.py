import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis



def segment_fruit(nir_image, rgb_image):

    #Application of a Gaussian Blur
    blur_nir = cv2.GaussianBlur(nir_image,(3,3),0)

    otsu_value = plot_nir_histogram_with_threshold(nir_image)
    print('otsu_value',otsu_value)

    #Thresholding to segment the image
    th, thresh = cv2.threshold(blur_nir, 50, 255, cv2.THRESH_BINARY)

    #filling of the holes inside the fruit blob using a flood-fill approach
    h, w = thresh.shape[:2]
    m1 = np.zeros((h+2, w+2), np.uint8)
    ff1 = thresh.copy()
    cv2.floodFill(ff1, m1, (0,0), 255)
    #we then invert the result obtained by the floodfill operation in order to highlight the holes
    holes = cv2.bitwise_not(ff1)

    #Mask of the apples
    mask_nir = holes | thresh
    mask_rgb = cv2.cvtColor(mask_nir, cv2.COLOR_GRAY2RGB)

    #Application of the masks to infrared images
    seg_nir = nir_image * (mask_nir/255)

    #Application of the masks to colored images
    ones = np.ones(rgb_image.shape, dtype=int)
    bool_mask_nir = ones & mask_rgb
    app_rgb = rgb_image * bool_mask_nir.astype(np.uint8)
    cv2.imshow('MaskedImage', app_rgb)

    return app_rgb

def kmeans_fruit(app_rgb):
    # Convert the color image to the LAB color space
    color_lab = cv2.cvtColor(app_rgb, cv2.COLOR_RGB2LAB)
    # Extract the A and B channels
    lab_ab = color_lab[:, :, 1:3]
    height, width = lab_ab.shape[:2]
    lab_ab_2d = lab_ab.reshape((-1, 2))

    # K-means clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(lab_ab_2d)
    labels = kmeans.labels_

    # Reshape labels to 2D
    segmented_img = labels.reshape((height, width))

    # Find the russet cluster based on LAB color space properties
    centers = kmeans.cluster_centers_
    
    # Analyze the L channel mean value for each cluster
    cluster_means = []
    for i in range(3):
        mask = (segmented_img == i)
        l_values = color_lab[:, :, 0][mask]
        cluster_means.append(np.mean(l_values))

    russet_cluster_idx = np.argmax(cluster_means)

    # Create a mask for the russet area
    russet_mask = (segmented_img == russet_cluster_idx).astype(np.uint8)

    return russet_cluster_idx, lab_ab, segmented_img

def mahalanobis_fruit(russet_cluster_idx, lab_ab, segmented_img, rgb_image):

    # Compute the mean and covariance of the russet cluster
    russet_pixels = lab_ab[segmented_img == russet_cluster_idx]
    mean_russet = np.mean(russet_pixels, axis=0)
    cov_russet = np.cov(russet_pixels, rowvar=False)
    inv_cov_russet = np.linalg.inv(cov_russet)

    # Calculate Mahalanobis distance for each pixel
    mahal_dist = np.apply_along_axis(lambda x: mahalanobis(x, mean_russet, inv_cov_russet), 2, lab_ab)
    threshold = 5  # Threshold can be adjusted
    final_russet_mask = (mahal_dist < threshold).astype(np.uint8)

    # Apply the final mask to the original image
    final_russet_img = cv2.bitwise_and(rgb_image, rgb_image, mask=final_russet_mask)

    # Convert the final image to BGR for display with OpenCV
    final_russet_img_bgr = cv2.cvtColor(final_russet_img, cv2.COLOR_RGB2BGR)

    return final_russet_img_bgr

def color_space_comparsion(app_rgb):
    hsv = cv2.cvtColor(app_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(app_rgb, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(app_rgb, cv2.COLOR_RGB2LUV)
    hls = cv2.cvtColor(app_rgb, cv2.COLOR_RGB2HLS)

    plot_color_histograms(app_rgb, "RGB", ["R", "G", "B"])
    plot_color_histograms(hsv, "HSV", ["H", "S", "V"])
    plot_color_histograms(lab, "LAB", ["L", "A", "B"])
    plot_color_histograms(luv, "LUV", ["L", "U", "V"])
    plot_color_histograms(hls, "HLS", ['H', 'L', 'S'])

    plot_3d_color_space(app_rgb, app_rgb, "RGB", ["R", "G", "B"])
    plot_3d_color_space(hsv, app_rgb, "HSV", ["H", "S", "V"])
    plot_3d_color_space(lab, app_rgb, "LAB", ["L", "A", "B"])
    plot_3d_color_space(luv, app_rgb, "luv", ["L", "U", "V"])
    plot_3d_color_space(hls, app_rgb, "HLS", ['H', 'L', 'S'])

def plot_color_histograms(image, color_space_name, channel_labels):
    plt.figure(figsize=(10, 4))
    for i in range(image.shape[2]):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, label=channel_labels[i])
    plt.title(f'{color_space_name} Color Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()   

def plot_3d_color_space(image, original_rgb, color_space_name, channel_labels):
    # Reshape image to a list of pixels
    pixels = image.reshape((-1, 3))
    pixels_rgb = original_rgb.reshape((-1, 3))
    
    # Optional: Sample to speed up plotting
    pixels = pixels  # take every 100th pixel
    pixels_rgb = pixels_rgb

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = pixels[:, 0]
    ys = pixels[:, 1]
    zs = pixels[:, 2]

    ax.scatter(xs, ys, zs, c=pixels_rgb / 255.0, s=1)
    ax.set_xlabel(channel_labels[0])
    ax.set_ylabel(channel_labels[1])
    ax.set_zlabel(channel_labels[2])
    ax.set_title(f'{color_space_name} 3D Color Space (Colored with RGB palette)')

    plt.tight_layout()
    plt.show()

def plot_nir_histogram_with_threshold(nir_image):
    pixel_values = nir_image.flatten()

    otsu_value, _ = cv2.threshold(nir_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values, bins=256, range=(0, 256), color='gray', alpha=0.7, label='Pixel Intensity')
    plt.axvline(otsu_value, color='red', linestyle='--', label=f'Otsu Threshold = {int(otsu_value)}')
    plt.title("NIR Image Histogram with Otsu Threshold")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return otsu_value


def run2():
    picNo = input("Enter the picture number(4 or 5): ")
    # Load NIR and color images
    nir_image = cv2.imread('Task2pics/C0_00000' + picNo +'.png', cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread('Task2pics/C1_00000' + picNo +'.png')
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    app_rgb = segment_fruit(nir_image, rgb_image)

    #just for comparative study and to be run only once to have an understanding of colorspaces and choosing the best 
    #color_space_comparsion(app_rgb) 

    russet_cluster_idx, lab_ab, segmented_img = kmeans_fruit(app_rgb)

    final_russet_img_bgr = mahalanobis_fruit(russet_cluster_idx, lab_ab, segmented_img, rgb_image)    



    # Show the results
    cv2.imshow('Original Image', color_image)
    cv2.imshow('Russet Detection', final_russet_img_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#run2()
