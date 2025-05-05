import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis

def run2():
    picNo = input("Enter the picture number(4 or 5): ")
    # Load NIR and color images
    nir_image = cv2.imread('Task2pics/C0_00000' + picNo +'.png', cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread('Task2pics/C1_00000' + picNo +'.png')
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    #Application of a Gaussian Blur
    blur_nir = cv2.GaussianBlur(nir_image,(3,3),0)

    #Otsu Thresholding to segment the image
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

    # Show the results
    cv2.imshow('Original Image', color_image)
    cv2.imshow('Russet Detection', final_russet_img_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

run2()
