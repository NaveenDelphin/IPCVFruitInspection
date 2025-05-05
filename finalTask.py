import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis

def run3():
    picNo = input("Enter the picture number(6 to 10): ")
    # Load NIR and color images
    nir_image = cv2.imread('Task3pics/C0_00000' + picNo +'.png', cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread('Task3pics/C1_00000' + picNo +'.png')
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    #Application of a Gaussian Blur
    blur_nir = cv2.GaussianBlur(nir_image,(3,3),0)

    #Otsu Thresholding to segment the image
    th, thresh = cv2.threshold(blur_nir, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

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

    # Convert the color image to the LAB color space
    lab_image = cv2.cvtColor(app_rgb, cv2.COLOR_RGB2LAB)

    # Extract the A and B channels
    lab_ab = lab_image[:, :, 1:3]
    height, width = lab_ab.shape[:2]
    lab_ab_2d = lab_ab.reshape((-1, 2))

    # K-means clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(lab_ab_2d)
    labels = kmeans.labels_

    # Reshape labels to 2D
    segmented_img = labels.reshape((height, width))

    # Find the kiwi cluster based on LAB color space properties
    centers = kmeans.cluster_centers_
    
    # Assuming the kiwi is the largest object in the image, we can find the largest cluster
    kiwi_cluster_idx = np.argmin(np.bincount(labels))

    # Create a mask for the kiwi area
    kiwi_mask = (segmented_img == kiwi_cluster_idx).astype(np.uint8)

    # Apply the mask to the original image
    kiwi_segmented = cv2.bitwise_and(rgb_image, rgb_image, mask=kiwi_mask)

    # Convert the segmented image back to BGR for display with OpenCV
    kiwi_segmented_bgr = cv2.cvtColor(kiwi_segmented, cv2.COLOR_RGB2BGR)

    # Compute the mean and covariance of the russet cluster
    russet_pixels = lab_ab[segmented_img == kiwi_cluster_idx]
    mean_russet = np.mean(russet_pixels, axis=0)
    cov_russet = np.cov(russet_pixels, rowvar=False)
    inv_cov_russet = np.linalg.inv(cov_russet)

    # Calculate Mahalanobis distance for each pixel
    mahal_dist = np.apply_along_axis(lambda x: mahalanobis(x, mean_russet, inv_cov_russet), 2, lab_ab)
    threshold = 2.5  # Threshold can be adjusted
    final_russet_mask = (mahal_dist < threshold).astype(np.uint8)
    
    # Apply the final mask to the original image
    final_russet_img = cv2.bitwise_and(rgb_image, rgb_image, mask=final_russet_mask)

    #Apply the final mask to nir image
    final_nir_russet_img = cv2.bitwise_and(nir_image, nir_image, mask=final_russet_mask)
    

    # Convert the final image to BGR for display with OpenCV
    final_russet_img_bgr = cv2.cvtColor(final_russet_img, cv2.COLOR_RGB2BGR)
    result = final_russet_img_bgr
    result[nir_image == 0] = [0, 0, 0]
    grayImage = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Thresholding to segment the fruit
    blurred_nir = cv2.GaussianBlur(grayImage, (5, 5), cv2.BORDER_DEFAULT)
    _, thresh = cv2.threshold(blurred_nir, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill holes inside the fruit blob
    thresh_filled = thresh.copy()
    cv2.floodFill(thresh_filled, None, (0, 0), 255)
    thresh_filled = cv2.bitwise_not(thresh_filled)

    #Masking the original color image

    # Erosion followed by Minkowsky subtraction
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(thresh_filled, kernel, iterations=1)
    minkowski_subtraction = cv2.subtract(thresh_filled, eroded)

    # Find contours
    contours, _ = cv2.findContours(minkowski_subtraction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on color image
    for contour in contours:
        # Calculate center and radius of the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        # Draw the circle around the contour
        cv2.circle(color_image, center, radius, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Result', color_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

run3()