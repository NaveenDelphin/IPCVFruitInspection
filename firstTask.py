import cv2
import numpy as np

def minkowski_subtraction(img, kernel_size=5):

    img_blur = cv2.bilateralFilter(img, 5, 20, 20)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(img_blur, kernel, iterations=1)
    dilated = cv2.dilate(img_blur, kernel, iterations=1)
    diff = cv2.absdiff(dilated, eroded)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_contrast = cv2.convertScaleAbs(diff_norm, alpha=2.5, beta=0)
    return diff_contrast

def segment_fruit(nir_image):

    #Application of a Gaussian Blur
    blur_temp = cv2.GaussianBlur(nir_image,(3,3),0)

    #Otsu Thresholding to segment the image
    th, otsu_temp = cv2.threshold(blur_temp, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    #filling of the holes inside the fruit blob using a flood-fill approach
    h, w = otsu_temp.shape[:2]
    m1 = np.zeros((h+2, w+2), np.uint8)
    ff1 = otsu_temp.copy()
    cv2.floodFill(ff1, m1, (0,0), 255)
    #we then invert the result obtained by the floodfill operation in order to highlight the holes
    holes_temp = cv2.bitwise_not(ff1)

    #Mask of the apples
    mask_temp = holes_temp | otsu_temp
    mask_c_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)

    #Application of the masks to infrared images
    app_nir_temp = nir_image * (mask_temp/255)

    return app_nir_temp.astype(np.uint8), mask_temp

def detect_defects(fruit_mask, mask_temp):

    # Apply Minkowski Subtraction for edge extraction
    edges_temp = minkowski_subtraction(fruit_mask, kernel_size=3)
    # Sharpen the edges
    sharpen_kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])
    sharpened_edges = cv2.filter2D(edges_temp, -1, sharpen_kernel)
    cv2.imshow("sharpened_edges", sharpened_edges.astype(np.uint8))
    
    _, high_thresh = cv2.threshold(sharpened_edges, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow(" high_thresh", high_thresh)
    
    #Perform a closing operation to have better defects' edges
    clo_ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    high_thresh = high_thresh.astype(np.uint8)
    closing_temp=cv2.morphologyEx(high_thresh, cv2.MORPH_CLOSE, clo_ker)
    cv2.imshow("closing_temp", closing_temp)

    #Fill of the undefected parts of the apples
    h_ff, w_ff= closing_temp.shape
    closing_temp = closing_temp.astype(np.uint8)
    row_ff=np.zeros((2, w_ff))
    col_ff=np.zeros((h_ff+2, 2))
    n_mask= ~mask_temp
    mask_ff_temp=np.vstack((n_mask, row_ff))
    mask_ff_temp=np.hstack((mask_ff_temp, col_ff))
    mask_ff_temp = mask_ff_temp.astype(np.uint8)
    cv2.floodFill(closing_temp, mask_ff_temp, (175,150), 255)
    holes_ff_temp= closing_temp
    cv2.imshow("filled", holes_ff_temp)

    #Subtract the previous image to the fully filled mask of the apple to higlight the defects
    open_ker=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    holes_ff_temp = holes_ff_temp.astype(np.uint8)    
    holes_ff_temp=~holes_ff_temp
    sub_temp=mask_temp & holes_ff_temp
    open_temp=cv2.morphologyEx(sub_temp, cv2.MORPH_OPEN, open_ker)
    cv2.imshow('defectsAlone', open_temp)

    return open_temp


def contour_defects(open_temp, color_image):

    # Find contours of the defects in open_temp
    contours, _ = cv2.findContours(open_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw rounded contours
    rounded_contours_img = np.zeros_like(color_image)

    # Draw rounded contours around the defects
    for contour in contours:
        # Calculate center and radius of the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        # Draw the circle around the contour
        cv2.circle(rounded_contours_img, center, radius, (0, 255, 0), 2)

    # Superimpose the rounded contours onto the original color image
    result_image = cv2.addWeighted(color_image, 1, rounded_contours_img, 0.5, 0)

    # Display the result
    cv2.imshow('Defects on Color Image', result_image)

    return result_image

def run1():

    picNo = input("Enter the picture number(1, 2 or 3): ")

    # Load NIR and color images
    nir_image = cv2.imread('Task1pics/C0_00000' + picNo +'.png', cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread('Task1pics/C1_00000' + picNo +'.png')

    # rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    fruit_mask, mask_temp = segment_fruit(nir_image)
    edges = detect_defects(fruit_mask, mask_temp)
    detected_image = contour_defects(edges, color_image)

    
    cv2.imshow("Segmented Fruit", fruit_mask)
    cv2.imshow("Edges (Minkowski Subtraction)", edges.astype(np.uint8))
    # cv2.imshow("Defects", defects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run1()

# import cv2
# import numpy as np

# def run1():
   
#     picNo = input("Enter the picture number(1, 2 or 3): ")
#     # Load NIR and color images
#     nir_image = cv2.imread('Task1pics/C0_00000' + picNo +'.png', cv2.IMREAD_GRAYSCALE)
#     color_image = cv2.imread('Task1pics/C1_00000' + picNo +'.png')
#     rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

#     #Application of a Gaussian Blur
#     blur_temp = cv2.GaussianBlur(nir_image,(3,3),0)

#     #Otsu Thresholding to segment the image
#     th, otsu_temp = cv2.threshold(blur_temp, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    
#     #filling of the holes inside the fruit blob using a flood-fill approach
#     h, w = otsu_temp.shape[:2]
#     m1 = np.zeros((h+2, w+2), np.uint8)
#     ff1 = otsu_temp.copy()
#     cv2.floodFill(ff1, m1, (0,0), 255)
#     #we then invert the result obtained by the floodfill operation in order to highlight the holes
#     holes_temp = cv2.bitwise_not(ff1)

#     #Mask of the apples
#     mask_temp = holes_temp | otsu_temp
#     mask_c_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB)

#     #Application of the masks to infrared images
#     app_nir_temp = nir_image * (mask_temp/255)
#     fruit_mask = app_nir_temp.astype(np.uint8)

#     #Minkowski Subtraction
#     kernel_size=5
#     img_blur = cv2.bilateralFilter(fruit_mask, 5, 20, 20)
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     eroded = cv2.erode(img_blur, kernel, iterations=1)
#     dilated = cv2.dilate(img_blur, kernel, iterations=1)
#     diff = cv2.absdiff(dilated, eroded)
#     diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
#     diff_contrast = cv2.convertScaleAbs(diff_norm, alpha=2.5, beta=0)

#      # Sharpen the edges
#     sharpen_kernel = np.array([[-1, -1, -1], 
#                                [-1,  9, -1], 
#                                [-1, -1, -1]])
#     sharpened_edges = cv2.filter2D(diff_contrast, -1, sharpen_kernel)
#     cv2.imshow("sharpened_edges", sharpened_edges.astype(np.uint8))
    
#     _, high_thresh = cv2.threshold(sharpened_edges, 180, 255, cv2.THRESH_BINARY)
#     cv2.imshow(" high_thresh", high_thresh)

#     #Perform a closing operation to have better defects' edges
#     clo_ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
#     high_thresh = high_thresh.astype(np.uint8)
#     closing_temp=cv2.morphologyEx(high_thresh, cv2.MORPH_CLOSE, clo_ker)
#     cv2.imshow("closing_temp", closing_temp)

#     cv2.imshow("Segmented Fruit", fruit_mask)
#     #cv2.imshow("Edges (Minkowski Subtraction)", edges.astype(np.uint8))
#     # cv2.imshow("Defects", defects)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # #Application of a Gaussian Blur
#     # blur_temp = cv2.GaussianBlur(nir_image,(3,3),0)
#     # #cv2.imshow('bluredImage', blur_temp)

#     # #Otsu Thresholding to segment the image
#     # th, otsu_temp = cv2.threshold(blur_temp, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
#     # #cv2.imshow('ThresholdedImage', otsu_temp)

#     # #filling of the holes inside the fruit blob using a flood-fill approach
#     # h, w = otsu_temp.shape[:2]
#     # m1 = np.zeros((h+2, w+2), np.uint8)
#     # ff1 = otsu_temp.copy()
#     # cv2.floodFill(ff1, m1, (0,0), 255)
#     # #we then invert the result obtained by the floodfill operation in order to highlight the holes
#     # holes_temp = cv2.bitwise_not(ff1)
#     # cv2.imshow('FilledHoles', holes_temp)

#     # #Mask of the apples
#     # mask_temp = holes_temp | otsu_temp
#     # mask_c_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB) 
#     # cv2.imshow('Mask', mask_c_temp)

#     # #Application of the masks to infrared images
#     # app_nir_temp = nir_image * (mask_temp/255)
#     # cv2.imshow('nirMasked', app_nir_temp.astype(np.uint8))

#     # #Application of a Bilateral Filter before Edge detection
#     # app_nir_temp = app_nir_temp.astype(np.uint8)
#     # img_bilateral = cv2.bilateralFilter(app_nir_temp,5, 20, 20)
#     # cv2.imshow('bilat',img_bilateral)

#     # #Minkowski Subtraction
#     # kernel =  np.ones((5, 5), np.uint8)
#     # eroded = cv2.erode(img_bilateral, kernel, iterations=1)
#     # dilated = cv2.dilate(img_bilateral, kernel, iterations= 1)
#     # diff = cv2.absdiff(dilated, eroded)
#     # diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
#     # diff_contrast = cv2.convertScaleAbs(diff_norm, alpha= 2.5, beta= 0)
#     # sharpen_kernel = np.array([[-1, -1, -1], 
#     #                            [-1,  9, -1], 
#     #                            [-1, -1, -1]])
#     # sharpened_edges = cv2.filter2D(diff_contrast, -1, sharpen_kernel)
#     # cv2.imshow("sharpened_edges", sharpened_edges.astype(np.uint8))
#     # _, high_thresh = cv2.threshold(sharpened_edges, 180, 255, cv2.THRESH_BINARY)
#     # cv2.imshow(" high_thresh", high_thresh)

#     # #Perform a closing operation to have better defects' edges
#     # clo_ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
#     # high_thresh = high_thresh.astype(np.uint8)
#     # closing_temp=cv2.morphologyEx(high_thresh, cv2.MORPH_CLOSE, clo_ker)
#     # cv2.imshow("closing_temp", closing_temp)


#     # #Fill of the undefected parts of the apples
#     # h_ff, w_ff= closing_temp.shape
#     # closing_temp = closing_temp.astype(np.uint8)
#     # row_ff=np.zeros((2, w_ff))
#     # col_ff=np.zeros((h_ff+2, 2))
#     # n_mask=~mask_temp
#     # mask_ff_temp=np.vstack((n_mask, row_ff))
#     # mask_ff_temp=np.hstack((mask_ff_temp, col_ff))
#     # mask_ff_temp = mask_ff_temp.astype(np.uint8)
#     # cv2.floodFill(closing_temp, mask_ff_temp, (175,150), 255)
#     # holes_ff_temp= closing_temp
#     # cv2.imshow("filled", holes_ff_temp)

#     # #Subtract the previous image to the fully filled mask of the apple to higlight the defects
#     # open_ker=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     # holes_ff_temp = holes_ff_temp.astype(np.uint8)    
#     # holes_ff_temp=~holes_ff_temp
#     # sub_temp=mask_temp & holes_ff_temp
#     # open_temp=cv2.morphologyEx(sub_temp, cv2.MORPH_OPEN, open_ker)
#     # cv2.imshow('defectsAlone', open_temp)

#     # # Find contours of the defects in open_temp
#     # contours, _ = cv2.findContours(open_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # # Create a blank image to draw rounded contours
#     # rounded_contours_img = np.zeros_like(color_image)

#     # # Draw rounded contours around the defects
#     # for contour in contours:
#     #     # Calculate center and radius of the minimum enclosing circle
#     #     (x, y), radius = cv2.minEnclosingCircle(contour)
#     #     center = (int(x), int(y))
#     #     radius = int(radius)
#     #     # Draw the circle around the contour
#     #     cv2.circle(rounded_contours_img, center, radius, (0, 255, 0), 2)

#     # # Superimpose the rounded contours onto the original color image
#     # result_image = cv2.addWeighted(color_image, 1, rounded_contours_img, 0.5, 0)

#     # # Display the result
#     # cv2.imshow('Defects on Color Image', result_image)

#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()