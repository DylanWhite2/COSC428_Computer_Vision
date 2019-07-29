import sys
import cv2
import numpy as np


'''
Changing any of these params will have an effect on other params and the
behaviour of the program.  These should look to be modified to all be relative
to say the width and height of the image for example.  Similarly, the blob
detector params should be scaled based off of these.... future project.
'''
# image scaling params.  Changing these will effect the blob detector and morphology
width = 1080
height = 720

# morphology params.  Same erosion size for both iterations.  Smaller dilation
# for second iteration to get smaller blobs
erodeSize = 8
dilateSize = 30
dilateSize2 = 10

# used to determine how close points can be to be classified as dupliate
# between iterations.
duplicate_thresh = 20
duplicate_keypoints = []


def main(image):
    # load image specified in command line
    img = cv2.imread(image, 1)

    # increase the 180x160 px image to 1080x720px for ease of seeing.  This also changes how the morphology and blob detector function. Set as globals
    scaled_img = cv2.resize(img, dsize=(width, height),
                            interpolation=cv2.INTER_CUBIC)

    # nomalize the image
    norm_img = cv2.normalize(scaled_img, scaled_img, 0,
                             255, cv2.NORM_MINMAX, cv2.CV_8U)

    # convert to HSV colour space as this will detect the coloured hot spots more easily
    hsv_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)

    # only look at the value channel of HSV as it will ignore the greyscale bg.
    v_img = hsv_img[:, :, 2]

    # remove noise while retaining edges
    blurred_img = cv2.bilateralFilter(v_img, 9, 150, 150)

    # erode away noise such as legs under tables and other small hot objects
    eroded_img = cv2.erode(blurred_img, np.ones((erodeSize, erodeSize)))

    # some people end up being broken into multiple blobs, merge these together.  This also causes farfield close objects to merge.... not good.
    dilated_img = cv2.dilate(eroded_img, np.ones((dilateSize, dilateSize)))

    # threshold the image so are only left with points of interest
    retval, thresh_img = cv2.threshold(
        dilated_img, 220, 255, cv2.THRESH_BINARY)

    # invert the image for the blob detector
    invert_img = cv2.bitwise_not(thresh_img)

    # setup the first pass params of the blob detector
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 255

    # helps remove some of the merging between rows of seats
    params.filterByInertia = True
    params.minInertiaRatio = 0.25

    params.filterByArea = False
    params.filterByCircularity = False
    params.filterByConvexity = False

    # run the detector and get the keypoints
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(invert_img)

    #--- Second Pass Through Blob Detector ---#

    # keep the same erosion as first pass as we just want to adjust dilation now
    eroded_img2 = cv2.erode(blurred_img, np.ones((erodeSize, erodeSize)))

    # use a smaller dilation to reduce some of the merging of multiple hotspots into 1 hotspot.
    dilated_img2 = cv2.dilate(eroded_img2, np.ones((dilateSize2, dilateSize2)))

    # threshold the image same as above
    retval2, thresh_img2 = cv2.threshold(
        dilated_img2, 220, 255, cv2.THRESH_BINARY)

    # invert the image same as above
    invert_img2 = cv2.bitwise_not(thresh_img2)

    # setup the second pass params of the blob detector
    params2 = cv2.SimpleBlobDetector_Params()

    params2.minThreshold = 0
    params2.maxThreshold = 255

    # as now using a smaller dilation, large blobs in nearfield are broken into
    # multiple smaller blobs.  minArea is a type of noise filer to remove these
    # smaller blobs
    params2.filterByArea = True
    params2.minArea = 800
    params2.maxArea = 6000

    params2.filterByCircularity = False
    params2.filterByConvexity = False
    params2.filterByInertia = False

    # run the deteto and get the keypoints of the image with new dilation
    detector2 = cv2.SimpleBlobDetector_create(params2)
    new_keypoints = detector2.detect(invert_img2)

    # algorithm to detect duplicate/estimated duplicate keypoints between both
    # images.  Simple points in square detection, might be nicer to change it
    # to be within a circle rather than a square.
    for new_keypoint in new_keypoints:
        for original_keypoint in keypoints:
            x_upper = original_keypoint.pt[0] + duplicate_thresh
            x_lower = original_keypoint.pt[0] - duplicate_thresh
            y_upper = original_keypoint.pt[1] + duplicate_thresh
            y_lower = original_keypoint.pt[1] - duplicate_thresh

            # if the new keypoint is within an area of the old keypoint, it is
            # duplicate. Add it to the list of duplicates.
            if x_lower <= new_keypoint.pt[0] < x_upper and y_lower <= new_keypoint.pt[1] < y_upper:
                duplicate_keypoints.append(new_keypoint)

    # if any of the new keypoints aren't duplicates, add them to the original keypoints.
    for new_keypoint in new_keypoints:
        if new_keypoint not in duplicate_keypoints:
            keypoints.append(new_keypoint)

    # the amout of people found in the scene.
    people = str(len(keypoints))

    # draw the keypoints on the image
    img_keypoints = cv2.drawKeypoints(invert_img, keypoints, np.array(
        []), (255, 0, 255),  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # put text number of Contour Keypoints on Screen in Blue
    cv2.putText(img_keypoints, people, (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    # show the keypoints on the image
    cv2.imshow('People Counter', img_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # allows to pass a valid image filename to the program
    main(sys.argv[1])

'''
Notes:

1. openCV keypoints are ordered in decending y value.  This may be helpful.

2. I'm only doing 2 passes through blob detector.  This could be made recurssive
    if everything is set to scale off each otherself.

3. A CNN may help if there is enough training data.

4. A different camera position would help I think

5.
'''
