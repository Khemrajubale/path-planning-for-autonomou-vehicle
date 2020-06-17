import cv2
import numpy as np
from scipy.stats import itemfreq

# REFER DOCUMENTATION for opencv k-means clustering:
# Using K-means clustering algorithm to extract the dominant color in the desired area (referenced by "image" below):
def get_dominant_color(image, n_colors):# "n_colors" is the desired number of clusters
    '''We use cv2.kmeans() function which takes a 2D array as input, and since our original image is 3D (width, height and depth
       of 3 RGB values), we need to flatten the height and width into a single vector of 3 pixels (R, G and B):2'''
    pixels = np.float32(image).reshape((-1, 3))# -1 means that the no. of rows would be automatically calculated given the no. of cols. (which is 3 here)
    #3 above means 3 columns, one each for R, G and B channels. We are not concerned about the coordinates of pixels..
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    #print(palette)
    # refer "random practice" under "numpy" in jupyter notebook for "argmax" and "itemfreq":
    return palette[np.argmax(itemfreq(labels)[:, -1])]
    '''We are basically listing the labels and their frequency. The centroid corresponding to the label with maximum frequency
       is returned. For eg, there could be pixels with maximum "red" label frequency'''



cameraCapture = cv2.VideoCapture(0)# "cameraCapture" is the name given to the object of "VideoCapture"..
cv2.namedWindow('camera')

# Read and process frames in loop
'''"read" method accessed via the defined object returns a tuple (return_value,image). Here the return_value is named "success" which
    is true(reading successful) or false(reading not successful):'''
success, frame = cameraCapture.read()


while success:
    cv2.waitKey(1)
    success, frame = cameraCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 37)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1, 50, param1=120, param2=40)# refer documentation


    if not circles is None:
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            '''Basically finding the circle which has the largest radius as that would basically be the traffic sign's shape:'''
            # refer "random practice" under "numpy" in jupyter notebook:
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            square = frame[y-r:y+r, x-r:x+r]


            '''The "get_dominant_color" function defined above returns the coordinates of the centroid (B,G,R) of the cluster
               containing the pixels with the dominant color. This is an array [blue_value, green_value, red_value].'''
            #asking to use 2 clusters, the "get_dominant_color" will return only the centroid of the one which has maximum pixels:
            dominant_color = get_dominant_color(square, 2)
            print(dominant_color)
            '''Thus, "dominant_color[0]" = blue_value, "dominant_color[1]" = green_value and "dominant_color[2]" = red_value'''
            if dominant_color[2] > 100: # ie., if red_value > 100, then the image is dominantly red in color.
                print("STOP") # As the "Stop" traffic sign is dominantly red in colour..
            elif dominant_color[0] > 80:
                # "//" gives integer value of the quotient..
                zone_0 = square[square.shape[0]*3//8:square.shape[0]*5//8, square.shape[1]*1//8:square.shape[1]*3//8]
                cv2.imshow('Zone0', zone_0)
                zone_0_color = get_dominant_color(zone_0, 1)

                zone_1 = square[square.shape[0]*1//8:square.shape[0]*3//8, square.shape[1]*3//8:square.shape[1]*5//8]
                cv2.imshow('Zone1', zone_1)
                zone_1_color = get_dominant_color(zone_1, 1)

                zone_2 = square[square.shape[0]*3//8:square.shape[0]*5//8, square.shape[1]*5//8:square.shape[1]*7//8]
                cv2.imshow('Zone2', zone_2)
                zone_2_color = get_dominant_color(zone_2, 1)

                if zone_1_color[2] < 60:
                    if sum(zone_0_color) > sum(zone_2_color):
                        print("LEFT")
                    else:
                        print("RIGHT")
                else:
                    if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                        print("FORWARD")

            else:
                print("N/A")


        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            #cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('camera', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cv2.destroyAllWindows()
cameraCapture.release()
